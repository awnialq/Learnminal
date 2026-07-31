//! Per-terminal-session command history recorded by the shell integration hooks.
//!
//! Each PTY gets a session id and a JSONL file under `~/.ai-cli-learning/sessions/`.
//! The shell hook (`extra/shell-integration/learnminal.{zsh,bash}`, materialised into
//! `~/.ai-cli-learning/shell/`) appends one record per command with its exact exit code;
//! the terminal supplies the output text, which only exists on screen. When the hook is
//! not installed the history degrades to grid-only parsing.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Once;
use std::time::{Duration, SystemTime};

use log::warn;
use serde::{Deserialize, Serialize};

use crate::learnminal::grid_extractor::GridBlock;
use crate::learnminal::settings::{atomic_write, state_dir};
use crate::learnminal::types::{CommandEntry, HistorySource};

const SESSIONS_DIR_NAME: &str = "sessions";
/// Directory the shell integration scripts are published to, and the name of each
/// script in it. The overlay's install hint is built from these.
pub const SHELL_DIR_NAME: &str = "shell";
pub const ZSH_SCRIPT_NAME: &str = "learnminal.zsh";
pub const BASH_SCRIPT_NAME: &str = "learnminal.bash";

const ENV_SESSION_ID: &str = "LEARNMINAL_SESSION_ID";
const ENV_SESSION_FILE: &str = "LEARNMINAL_SESSION_FILE";
const ENV_SESSION_VERSION: &str = "LEARNMINAL_SESSION_VERSION";

/// Record schema version. The shell hooks refuse to run against a different one.
const SCHEMA_VERSION: u32 = 1;

/// Bytes read from the end of a session file; roughly 100 records.
const TAIL_BYTES: u64 = 16 * 1024;
/// Session files untouched for this long are removed by the background sweep.
const STALE_AFTER: Duration = Duration::from_secs(24 * 60 * 60);
/// Hard cap on retained session files, regardless of age.
const MAX_SESSION_FILES: usize = 64;
/// How far back in the block list a single record may reach for its output.
const MATCH_WINDOW: usize = 3;
/// Shortest grid command text allowed to prefix-match a longer record command.
const MIN_PREFIX_MATCH_CHARS: usize = 8;

const ZSH_SCRIPT: &str = include_str!("../../../extra/shell-integration/learnminal.zsh");
const BASH_SCRIPT: &str = include_str!("../../../extra/shell-integration/learnminal.bash");

/// One command as recorded by the shell integration hook.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandRecord {
    #[serde(default)]
    pub v: u32,
    #[serde(default)]
    pub seq: u64,
    #[serde(default)]
    pub pid: u32,
    #[serde(default)]
    pub cmd: String,
    #[serde(default)]
    pub exit: Option<i32>,
    #[serde(default)]
    pub cwd: String,
    #[serde(default)]
    pub start: Option<i64>,
    #[serde(default)]
    pub end: Option<i64>,
}

/// Identity of one terminal session, shared with its shell through the environment.
#[derive(Debug, Default)]
pub struct Session {
    id: String,
    path: Option<PathBuf>,
}

impl Session {
    /// Create a session with a fresh id. Does not touch the filesystem; the shell hook
    /// creates the file on the first command.
    pub fn new() -> Self {
        let id = uuid::Uuid::new_v4().simple().to_string();
        let path = sessions_dir().map(|dir| dir.join(format!("{id}.jsonl")));
        Self { id, path }
    }

    /// Session file path, or `None` when the home directory could not be resolved.
    pub fn path(&self) -> Option<&Path> {
        self.path.as_deref()
    }

    /// Environment passed to the shell. `LEARNMINAL_SESSION_FILE` doubles as the hook's
    /// enable switch, so it is absent when there is nowhere to write.
    pub fn env_vars(&self) -> Vec<(String, String)> {
        let mut vars = vec![(ENV_SESSION_ID.to_owned(), self.id.clone())];
        if let Some(path) = &self.path {
            vars.push((ENV_SESSION_FILE.to_owned(), path.to_string_lossy().into_owned()));
            vars.push((ENV_SESSION_VERSION.to_owned(), SCHEMA_VERSION.to_string()));
        }
        vars
    }

    /// Newest `max` records for this session, oldest first. Empty when the hook is not
    /// installed or the file cannot be read.
    pub fn recent_commands(&self, max: usize) -> Vec<CommandRecord> {
        match &self.path {
            Some(path) => read_tail_records(path, max),
            None => Vec::new(),
        }
    }

    /// Whether the shell integration has recorded anything for this session.
    ///
    /// A stat, not a parse: `/info` only needs to know whether the hook is writing, and
    /// the hook refuses to write at all unless it agrees with [`SCHEMA_VERSION`].
    pub fn is_active(&self) -> bool {
        self.path
            .as_ref()
            .and_then(|path| std::fs::metadata(path).ok())
            .is_some_and(|meta| meta.len() > 0)
    }
}

fn sessions_dir() -> Option<PathBuf> {
    state_dir().map(|dir| dir.join(SESSIONS_DIR_NAME))
}

pub fn shell_dir() -> Option<PathBuf> {
    state_dir().map(|dir| dir.join(SHELL_DIR_NAME))
}

/// Read the newest `max` records from the end of `path`, oldest first.
///
/// Bounded: only the last [`TAIL_BYTES`] are read, so cost is independent of file size.
/// Never panics; any error yields an empty vector.
fn read_tail_records(path: &Path, max: usize) -> Vec<CommandRecord> {
    let read = || -> std::io::Result<Vec<CommandRecord>> {
        let mut file = File::open(path)?;
        let len = file.metadata()?.len();
        let start = len.saturating_sub(TAIL_BYTES);
        if start > 0 {
            file.seek(SeekFrom::Start(start))?;
        }
        let mut buf = Vec::with_capacity(TAIL_BYTES as usize);
        file.take(TAIL_BYTES).read_to_end(&mut buf)?;
        Ok(parse_tail(&buf, start > 0, max))
    };

    match read() {
        Ok(records) => records,
        Err(err) => {
            if err.kind() != std::io::ErrorKind::NotFound {
                warn!("learnminal session: reading {}: {err}", path.display());
            }
            Vec::new()
        },
    }
}

/// Parse JSONL records out of a tail buffer, oldest first.
///
/// `partial_head` says the buffer starts mid-file, in which case the first line is a
/// fragment and is dropped. A torn final line simply fails to parse and is skipped.
///
/// Scans newest-first and stops at `max`, so the ~100 records a full tail holds are not
/// deserialized just to throw all but a dozen away.
fn parse_tail(bytes: &[u8], partial_head: bool, max: usize) -> Vec<CommandRecord> {
    // Lossy: the tail may begin or end inside a multi-byte character.
    let text = String::from_utf8_lossy(bytes);
    let text = if partial_head {
        match text.find('\n') {
            Some(i) => &text[i + 1..],
            None => return Vec::new(),
        }
    } else {
        text.as_ref()
    };

    // Several shells (tmux panes, subshells) can share one file. Keep only the shell
    // that wrote last, so the history reads as one coherent sequence.
    let mut newest_pid: Option<u32> = None;
    let mut records: Vec<CommandRecord> = Vec::with_capacity(max.min(16));

    for line in text.lines().rev() {
        if records.len() == max {
            break;
        }
        let Ok(record) = serde_json::from_str::<CommandRecord>(line) else { continue };
        if record.v != SCHEMA_VERSION || record.cmd.trim().is_empty() {
            continue;
        }
        let pid = *newest_pid.get_or_insert(record.pid);
        if record.pid == pid || record.pid == 0 {
            records.push(record);
        }
    }

    records.reverse();
    records
}

/// Join shell records with grid-recovered output.
///
/// Records are authoritative for command text, exit code, and cwd; blocks contribute
/// output only. Returns at most `max` entries, oldest first.
pub fn merge_history(
    records: &[CommandRecord],
    blocks: &[GridBlock],
    max: usize,
) -> (Vec<CommandEntry>, HistorySource) {
    if max == 0 {
        return (Vec::new(), HistorySource::None);
    }

    if records.is_empty() {
        let entries = grid_only_entries(blocks, max);
        let source = if entries.is_empty() { HistorySource::None } else { HistorySource::GridOnly };
        return (entries, source);
    }

    // Walk newest to oldest with a cursor that only ever moves backwards through the
    // blocks. That ordering constraint is what keeps repeated identical commands
    // matched to their own output instead of all sharing the newest block.
    let mut cursor = blocks.len();
    let mut entries: Vec<CommandEntry> = Vec::with_capacity(max);

    for record in records.iter().rev() {
        let lo = cursor.saturating_sub(MATCH_WINDOW);
        let hit = (lo..cursor).rev().find(|&i| commands_equal(&blocks[i].command, &record.cmd));

        match hit {
            Some(i) => {
                entries.push(entry_from(record, Some(&blocks[i])));
                cursor = i;
            },
            None => entries.push(entry_from(record, None)),
        }

        if entries.len() == max {
            break;
        }
    }

    entries.reverse();
    (entries, HistorySource::ShellHook)
}

fn entry_from(record: &CommandRecord, block: Option<&GridBlock>) -> CommandEntry {
    CommandEntry {
        command: record.cmd.trim().to_owned(),
        exit_code: record.exit,
        output: block.map(|block| block.output.clone()).unwrap_or_default(),
        cwd: record.cwd.clone(),
    }
}

/// Newest `max` grid blocks as entries, oldest first. Exit codes are unavailable.
fn grid_only_entries(blocks: &[GridBlock], max: usize) -> Vec<CommandEntry> {
    let start = blocks.len().saturating_sub(max);
    blocks[start..]
        .iter()
        .map(|block| CommandEntry {
            command: block.command.clone(),
            exit_code: None,
            output: block.output.clone(),
            cwd: String::new(),
        })
        .collect()
}

/// Whether a command read off the grid refers to the same command as a shell record.
///
/// The grid copy can be mangled in two ways: cut short, because a long command wrapped
/// onto the next row and only the first row was read; or given a prefix, because prompt
/// decoration was misparsed as part of the command. Anything looser than that risks
/// pairing neighbouring commands — `git status` and `git stash` share a program and a
/// length, but their outputs must never be swapped. The record is the truth.
fn commands_equal(grid_cmd: &str, record_cmd: &str) -> bool {
    let grid = normalize_command(grid_cmd);
    let record = normalize_command(record_cmd);

    if grid.is_empty() || record.is_empty() {
        return false;
    }
    if grid == record {
        return true;
    }
    // A partial match is only believable when the overlapping text — the shorter of
    // the two strings — is long enough not to be a coincidence.
    let long_enough = |text: &str| text.chars().count() >= MIN_PREFIX_MATCH_CHARS;

    // Grid line cut off at the terminal width, or carrying leftover prompt text.
    (record.starts_with(&grid) && long_enough(&grid))
        || (grid.ends_with(&record) && long_enough(&record))
}

fn normalize_command(command: &str) -> String {
    command.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// Delete session files that no live terminal is using.
pub fn cleanup_stale_sessions(current: Option<&Path>) {
    let Some(dir) = sessions_dir() else { return };
    cleanup_stale_sessions_in(&dir, current, SystemTime::now());
}

fn cleanup_stale_sessions_in(dir: &Path, current: Option<&Path>, now: SystemTime) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };

    let mut fresh: Vec<(SystemTime, PathBuf)> = Vec::new();

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().and_then(|ext| ext.to_str()) != Some("jsonl") {
            continue;
        }
        if Some(path.as_path()) == current {
            continue;
        }

        let modified = entry.metadata().and_then(|meta| meta.modified()).ok();
        let stale = modified
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age > STALE_AFTER);

        if stale {
            let _ = std::fs::remove_file(&path);
        } else {
            fresh.push((modified.unwrap_or(SystemTime::UNIX_EPOCH), path));
        }
    }

    if fresh.len() > MAX_SESSION_FILES {
        fresh.sort_by_key(|(modified, _)| *modified);
        for (_, path) in &fresh[..fresh.len() - MAX_SESSION_FILES] {
            let _ = std::fs::remove_file(path);
        }
    }
}

/// Materialise the shell integration scripts into `~/.ai-cli-learning/shell/`.
///
/// Users source them from there rather than from the install location, which is not
/// stable across a Homebrew install, a `.app` bundle, and a cargo build.
///
/// Runs once per process: every window would otherwise redo the same disk work.
pub fn ensure_shell_scripts() {
    static PUBLISHED: Once = Once::new();
    PUBLISHED.call_once(publish_shell_scripts);
}

fn publish_shell_scripts() {
    let Some(dir) = shell_dir() else { return };
    if let Err(err) = std::fs::create_dir_all(&dir) {
        warn!("learnminal session: creating {}: {err}", dir.display());
        return;
    }

    for (name, contents) in [(ZSH_SCRIPT_NAME, ZSH_SCRIPT), (BASH_SCRIPT_NAME, BASH_SCRIPT)] {
        let path = dir.join(name);
        // Compare the whole file, not a version header: a fix to the hook's body that
        // does not bump a version would otherwise never reach anyone who already has it.
        if std::fs::read_to_string(&path).is_ok_and(|existing| existing == contents) {
            continue;
        }
        if let Err(err) = atomic_write(&dir, &path, contents.as_bytes()) {
            warn!("learnminal session: writing {}: {err}", path.display());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    fn record(seq: u64, cmd: &str, exit: i32) -> String {
        format!(
            r#"{{"v":1,"seq":{seq},"pid":42,"cmd":"{cmd}","exit":{exit},"cwd":"/tmp","start":1,"end":2}}"#
        )
    }

    fn block(command: &str, output: &str) -> GridBlock {
        GridBlock { command: command.into(), output: output.into(), prompt_row: 0 }
    }

    fn rec(cmd: &str, exit: i32) -> CommandRecord {
        CommandRecord {
            v: 1,
            seq: 0,
            pid: 1,
            cmd: cmd.into(),
            exit: Some(exit),
            cwd: String::new(),
            start: None,
            end: None,
        }
    }

    #[test]
    fn parse_tail_reads_records_written_by_the_shell_hooks() {
        // Copied verbatim from `learnminal.zsh` and `learnminal.bash` output. If the
        // scripts' record format drifts from these field names, this test fails rather
        // than the history silently going empty at runtime.
        let zsh = r#"{"v":1,"seq":3,"pid":36676,"cmd":"echo \"he said \\\"hi\\\"\"","exit":0,"cwd":"/tmp","start":1785443349,"end":1785443349}"#;
        let bash = r#"{"v":1,"seq":5,"pid":36676,"cmd":"ls /nonexistent-path-xyz","exit":1,"cwd":"/tmp","start":1785443349,"end":1785443349}"#;
        let text = format!("{zsh}\n{bash}\n");

        let records = parse_tail(text.as_bytes(), false, 5);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].cmd, r#"echo "he said \"hi\"""#);
        assert_eq!(records[0].exit, Some(0));
        assert_eq!(records[1].cmd, "ls /nonexistent-path-xyz");
        assert_eq!(records[1].exit, Some(1));
        assert_eq!(records[1].cwd, "/tmp");
    }

    #[test]
    fn parse_tail_reads_last_n_records() {
        let text: String =
            (0..10).map(|i| format!("{}\n", record(i, &format!("cmd{i}"), 0))).collect();
        let records = parse_tail(text.as_bytes(), false, 5);
        assert_eq!(records.len(), 5);
        assert_eq!(records[0].cmd, "cmd5");
        assert_eq!(records[4].cmd, "cmd9");
    }

    #[test]
    fn parse_tail_drops_partial_head_when_offset_nonzero() {
        // Buffer begins in the middle of a record, as a mid-file seek would leave it.
        let text = format!("d\":\"pwd\",\"exit\":0}}\n{}\n", record(2, "ls", 0));
        let records = parse_tail(text.as_bytes(), true, 5);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].cmd, "ls");
    }

    #[test]
    fn parse_tail_partial_head_without_newline_is_empty() {
        assert!(parse_tail(b"d\":\"pwd\",\"exit\":0}", true, 5).is_empty());
    }

    #[test]
    fn parse_tail_keeps_first_line_when_offset_zero() {
        let text = format!("{}\n{}\n", record(1, "pwd", 0), record(2, "ls", 0));
        let records = parse_tail(text.as_bytes(), false, 5);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].cmd, "pwd");
    }

    #[test]
    fn parse_tail_tolerates_torn_last_line() {
        let text = format!("{}\n{{\"v\":1,\"seq\":9,\"cmd\":\"gi", record(1, "ls", 0));
        let records = parse_tail(text.as_bytes(), false, 5);
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].cmd, "ls");
    }

    #[test]
    fn parse_tail_skips_garbage_and_blank_lines() {
        let text = format!("not json\n\n{}\n   \n", record(1, "ls", 0));
        let records = parse_tail(text.as_bytes(), false, 5);
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn parse_tail_rejects_wrong_version_and_empty_command() {
        let text = concat!(
            r#"{"v":2,"seq":1,"pid":1,"cmd":"ls","exit":0}"#,
            "\n",
            r#"{"v":1,"seq":2,"pid":1,"cmd":"   ","exit":0}"#,
            "\n",
        );
        assert!(parse_tail(text.as_bytes(), false, 5).is_empty());
    }

    #[test]
    fn parse_tail_filters_records_from_other_pids() {
        let text = concat!(
            r#"{"v":1,"seq":1,"pid":100,"cmd":"a","exit":0}"#,
            "\n",
            r#"{"v":1,"seq":2,"pid":100,"cmd":"b","exit":0}"#,
            "\n",
            r#"{"v":1,"seq":3,"pid":200,"cmd":"c","exit":0}"#,
            "\n",
            r#"{"v":1,"seq":4,"pid":200,"cmd":"d","exit":0}"#,
            "\n",
        );
        let records = parse_tail(text.as_bytes(), false, 10);
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].cmd, "c");
        assert_eq!(records[1].cmd, "d");
    }

    #[test]
    fn parse_tail_with_zero_max_is_empty() {
        let text = format!("{}\n", record(1, "ls", 0));
        assert!(parse_tail(text.as_bytes(), false, 0).is_empty());
    }

    #[test]
    fn read_tail_records_missing_file_returns_empty() {
        let dir = tempfile::tempdir().unwrap();
        assert!(read_tail_records(&dir.path().join("nope.jsonl"), 5).is_empty());
    }

    #[test]
    fn read_tail_records_handles_file_larger_than_tail_bytes() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session.jsonl");
        let padding = "x".repeat(200);
        let text: String =
            (0..400).map(|i| format!("{}\n", record(i, &format!("cmd{i} {padding}"), 0))).collect();
        assert!(text.len() > TAIL_BYTES as usize);
        std::fs::write(&path, text).unwrap();

        let records = read_tail_records(&path, 5);
        assert_eq!(records.len(), 5);
        assert!(records[4].cmd.starts_with("cmd399"));
    }

    #[test]
    fn cleanup_removes_stale_but_keeps_current_fresh_and_foreign_files() {
        let dir = tempfile::tempdir().unwrap();
        let stale = dir.path().join("stale.jsonl");
        let fresh = dir.path().join("fresh.jsonl");
        let current = dir.path().join("current.jsonl");
        let other = dir.path().join("notes.txt");
        for path in [&stale, &fresh, &current, &other] {
            std::fs::write(path, "").unwrap();
        }

        // Rather than backdating mtimes, run the sweep from a future "now" so that
        // every file looks older than STALE_AFTER except the ones we exempt.
        let later = SystemTime::now() + STALE_AFTER + Duration::from_secs(60);
        cleanup_stale_sessions_in(dir.path(), Some(&current), later);
        assert!(!stale.exists() && !fresh.exists(), "aged-out sessions are swept");
        assert!(current.exists(), "the live session must never be swept");
        assert!(other.exists(), "non-session files must be left alone");

        // And with a present-day "now", nothing is old enough to remove.
        for path in [&stale, &fresh] {
            std::fs::write(path, "").unwrap();
        }
        cleanup_stale_sessions_in(dir.path(), Some(&current), SystemTime::now());
        assert!(stale.exists() && fresh.exists());
    }

    #[test]
    fn merge_pairs_records_to_blocks_in_order() {
        let records = [rec("ls", 0), rec("cargo build", 101)];
        let blocks = [block("ls", "a.txt"), block("cargo build", "error[E0433]")];
        let (entries, source) = merge_history(&records, &blocks, 5);

        assert_eq!(source, HistorySource::ShellHook);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].command, "ls");
        assert_eq!(entries[0].output, "a.txt");
        assert_eq!(entries[1].exit_code, Some(101));
        assert_eq!(entries[1].output, "error[E0433]");
    }

    #[test]
    fn merge_duplicate_commands_map_to_distinct_blocks() {
        let records = [rec("ls", 0), rec("ls", 0), rec("ls", 0)];
        let blocks = [block("ls", "a"), block("ls", "b"), block("ls", "c")];
        let (entries, _) = merge_history(&records, &blocks, 5);

        let outputs: Vec<&str> = entries.iter().map(|e| e.output.as_str()).collect();
        assert_eq!(outputs, ["a", "b", "c"]);
    }

    #[test]
    fn merge_record_without_block_gets_empty_output() {
        let records = [rec("clear", 0), rec("echo hi", 0)];
        let blocks = [block("echo hi", "hi")];
        let (entries, _) = merge_history(&records, &blocks, 5);

        assert_eq!(entries[0].command, "clear");
        assert!(entries[0].output.is_empty());
        assert_eq!(entries[1].output, "hi");
    }

    #[test]
    fn merge_extra_blocks_without_records_are_dropped() {
        let records = [rec("echo hi", 0)];
        let blocks = [block("older", "x"), block("echo hi", "hi")];
        let (entries, _) = merge_history(&records, &blocks, 5);

        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].output, "hi");
    }

    #[test]
    fn merge_match_window_prevents_distant_pairing() {
        let records = [rec("target", 0)];
        let blocks = [
            block("target", "wanted"),
            block("noise1", ""),
            block("noise2", ""),
            block("noise3", ""),
            block("noise4", ""),
        ];
        let (entries, _) = merge_history(&records, &blocks, 5);

        assert_eq!(entries.len(), 1);
        assert!(entries[0].output.is_empty(), "match should not reach past MATCH_WINDOW");
    }

    #[test]
    fn merge_empty_records_falls_back_to_grid_only() {
        let blocks = [block("ls", "a"), block("pwd", "/tmp")];
        let (entries, source) = merge_history(&[], &blocks, 5);

        assert_eq!(source, HistorySource::GridOnly);
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[1].command, "pwd");
        assert!(entries[1].exit_code.is_none());
    }

    #[test]
    fn merge_with_nothing_at_all_reports_no_history() {
        let (entries, source) = merge_history(&[], &[], 5);
        assert!(entries.is_empty());
        assert_eq!(source, HistorySource::None);
    }

    #[test]
    fn merge_respects_max_and_keeps_newest() {
        let records = [rec("a", 0), rec("b", 0), rec("c", 0), rec("d", 0)];
        let (entries, _) = merge_history(&records, &[], 2);

        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].command, "c");
        assert_eq!(entries[1].command, "d");
    }

    #[test]
    fn commands_equal_accepts_exact_whitespace_and_mangled_variants() {
        assert!(commands_equal("git status", "git  status"), "whitespace is normalized");
        assert!(commands_equal("cargo build --rel", "cargo build --release"), "grid line wrapped");
        assert!(commands_equal("(venv) cargo build", "cargo build"), "prompt text leaked in");
    }

    #[test]
    fn commands_equal_rejects_neighbouring_and_short_commands() {
        // Sibling subcommands must never be paired: their outputs would be swapped.
        assert!(!commands_equal("git status", "git stash"));
        assert!(!commands_equal("ls", "ls -la /very/long/path"), "too little shared text");
        assert!(!commands_equal("cd /a", "cd /b"));
        assert!(!commands_equal("", "ls"));
        assert!(!commands_equal("ls", ""));
    }

    proptest! {
        #[test]
        fn parse_tail_never_panics_on_arbitrary_bytes(
            bytes in prop::collection::vec(any::<u8>(), 0..4096),
            partial in any::<bool>(),
        ) {
            let _ = parse_tail(&bytes, partial, 5);
        }

        #[test]
        fn parse_tail_length_bounded(max in 0usize..20) {
            let text: String =
                (0..30).map(|i| format!("{}\n", record(i, &format!("cmd{i}"), 0))).collect();
            prop_assert!(parse_tail(text.as_bytes(), false, max).len() <= max);
        }

        #[test]
        fn parse_tail_preserves_seq_order(count in 1usize..20) {
            let text: String =
                (0..count).map(|i| format!("{}\n", record(i as u64, "ls", 0))).collect();
            let records = parse_tail(text.as_bytes(), false, count);
            for pair in records.windows(2) {
                prop_assert!(pair[0].seq <= pair[1].seq);
            }
        }

        #[test]
        fn merge_identical_lists_pair_index_wise(count in 1usize..6) {
            let records: Vec<CommandRecord> =
                (0..count).map(|i| rec(&format!("cmd{i}"), 0)).collect();
            let blocks: Vec<GridBlock> =
                (0..count).map(|i| block(&format!("cmd{i}"), &format!("out{i}"))).collect();
            let (entries, _) = merge_history(&records, &blocks, count);

            prop_assert_eq!(entries.len(), count);
            for (i, entry) in entries.iter().enumerate() {
                prop_assert_eq!(&entry.output, &format!("out{i}"));
            }
        }

        #[test]
        fn merge_never_panics(
            record_cmds in prop::collection::vec(".*", 0..8),
            block_cmds in prop::collection::vec(".*", 0..8),
            max in 0usize..8,
        ) {
            let records: Vec<CommandRecord> =
                record_cmds.iter().map(|c| rec(c, 0)).collect();
            let blocks: Vec<GridBlock> =
                block_cmds.iter().map(|c| block(c, "out")).collect();
            let (entries, _) = merge_history(&records, &blocks, max);
            prop_assert!(entries.len() <= max);
        }
    }
}
