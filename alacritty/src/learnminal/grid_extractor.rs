use std::panic::{self, AssertUnwindSafe};

use alacritty_terminal::grid::{Dimensions, Grid, GridCell};
use alacritty_terminal::index::{Column, Line, Point};
use alacritty_terminal::selection::SelectionRange;
use alacritty_terminal::term::cell::{Cell, Flags};
use alacritty_terminal::term::viewport_to_point;

use crate::learnminal::session::{self, CommandRecord};
use crate::learnminal::types::{HistorySource, TerminalContext};

pub const PREFIX_LINES: usize = 40;
pub const SUFFIX_LINES: usize = 40;
pub const MAX_CHARS: usize = 8000;

/// Grid lines (scrollback + screen) scanned when reconstructing command blocks.
pub const HISTORY_SCAN_LINES: usize = 500;
/// Command blocks reconstructed from the grid before matching against shell records.
pub const HISTORY_MAX_BLOCKS: usize = 12;
/// Commands reported in [`TerminalContext::command_history`].
pub const HISTORY_MAX_ENTRIES: usize = 5;

const PROMPT_CHARS: &[char] = &['$', '#', '%', '❯'];

fn learnminal_state_dir() -> Option<std::path::PathBuf> {
    session::state_dir()
}

/// Read the exit code written by the shell hook at `~/.ai-cli-learning/last_exit_code`.
///
/// Shells configure a PRECMD/PROMPT_COMMAND hook that writes `$?` to this file
/// before each prompt so the terminal can report the actual last exit code.
/// Returns `None` if the file is missing, unreadable, or contains non-integer text.
pub fn read_last_exit_code() -> Option<i32> {
    let path = learnminal_state_dir()?.join("last_exit_code");
    std::fs::read_to_string(path).ok()?.trim().parse().ok()
}

/// Read the last executed command from `~/.ai-cli-learning/last_command`.
///
/// Written by the shell precmd hook (see "Shell integration" in README.md). More
/// reliable than parsing the terminal grid, which can false-match `$` in command
/// output. Process-global, so prefer [`session::Session::recent_commands`] when it
/// has records: every window overwrites this file.
pub fn read_last_command() -> Option<String> {
    let path = learnminal_state_dir()?.join("last_command");
    let raw = std::fs::read_to_string(path).ok()?.trim().to_owned();
    if raw.is_empty() {
        return None;
    }
    // Strip zsh EXTENDED_HISTORY prefix format: ": <timestamp>:<duration>;<command>"
    let command = if let Some(rest) = raw.strip_prefix(": ") {
        match rest.find(';') {
            Some(i) => rest[i + 1..].trim().to_owned(),
            None => raw,
        }
    } else {
        raw
    };
    if command.is_empty() {
        None
    } else {
        Some(command)
    }
}

/// Extract terminal context from the visible grid, with middle-truncation and panic safety.
///
/// `records` are the commands the shell integration hook wrote for this session, oldest
/// first. They are authoritative for command text and exit codes; the grid supplies the
/// output each command produced. Pass an empty slice when the hook is not installed.
pub fn extract_context(
    grid: &Grid<Cell>,
    selection: Option<SelectionRange>,
    cwd: &str,
    last_exit_code: Option<i32>,
    records: &[CommandRecord],
) -> TerminalContext {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        extract_context_inner(grid, selection, cwd, last_exit_code, records)
    }));

    result.unwrap_or_default()
}

fn extract_context_inner(
    grid: &Grid<Cell>,
    selection: Option<SelectionRange>,
    cwd: &str,
    last_exit_code: Option<i32>,
    records: &[CommandRecord],
) -> TerminalContext {
    let all_lines = collect_visible_lines(grid);
    let visible_text = truncate_lines(&all_lines);

    // Extract the last command block (command + output) from the grid first so we can
    // use the block command as an improved fallback when the shell hook file is absent.
    let (block_command, last_command_output) = extract_command_block(&all_lines);

    // Reconstruct the session history from the scrollback, which reaches further back
    // than the viewport and is unaffected by how far the user has scrolled.
    let recent_lines = collect_recent_lines(grid, HISTORY_SCAN_LINES);
    let blocks = extract_command_blocks(&recent_lines, HISTORY_MAX_BLOCKS);
    let (mut command_history, history_source) =
        session::merge_history(records, &blocks, HISTORY_MAX_ENTRIES);

    // On the grid-only path the newest command's exit code is the one thing the shell
    // still tells us, via the legacy state file.
    if let Some(newest) = command_history.last_mut() {
        if newest.exit_code.is_none() {
            newest.exit_code = last_exit_code;
        }
    }

    // A shell record's command text beats every grid heuristic. Without records the
    // fallback chain is unchanged: the legacy state file, then the grid.
    let recorded_command = match history_source {
        HistorySource::ShellHook => command_history
            .last()
            .map(|entry| entry.command.clone())
            .filter(|command| !command.is_empty()),
        HistorySource::GridOnly | HistorySource::None => None,
    };
    let last_command = recorded_command.or_else(read_last_command).unwrap_or_else(|| {
        if !block_command.is_empty() { block_command } else { extract_last_command(&all_lines) }
    });

    // Prefer the session's own newest exit code over the process-global state file,
    // which every Learnminal window writes to.
    let exit_code = command_history.last().and_then(|entry| entry.exit_code).or(last_exit_code);

    let selected_text =
        selection.and_then(|range| extract_selection(grid, range).filter(|s| !s.is_empty()));

    TerminalContext {
        visible_text,
        selected_text,
        last_command,
        last_command_output,
        command_history,
        history_source,
        cwd: cwd.to_owned(),
        exit_code,
        rows: grid.screen_lines() as u16,
        cols: grid.columns() as u16,
    }
}

fn collect_visible_lines(grid: &Grid<Cell>) -> Vec<String> {
    let display_offset = grid.display_offset();
    (0..grid.screen_lines())
        .map(|row| {
            let grid_line = viewport_to_point(display_offset, Point::new(row, Column(0))).line;
            line_to_string(grid, grid_line)
        })
        .collect()
}

/// Inclusive `[top, bottom]` grid-line range covering the last `max_lines` lines.
///
/// Buffer-relative, unlike [`collect_visible_lines`]: it always ends at the bottom of the
/// buffer so scrolling up does not change which commands are recovered.
fn scan_range(topmost: i32, bottommost: i32, max_lines: usize) -> (i32, i32) {
    let max = max_lines.max(1) as i64;
    let start = (bottommost as i64 - max + 1).max(topmost as i64);
    (start as i32, bottommost)
}

fn collect_recent_lines(grid: &Grid<Cell>, max_lines: usize) -> Vec<String> {
    let (start, end) = scan_range(grid.topmost_line().0, grid.bottommost_line().0, max_lines);
    (start..=end).map(|line| line_to_string(grid, Line(line))).collect()
}

fn line_to_string(grid: &Grid<Cell>, line: Line) -> String {
    let mut result = String::new();
    for col in 0..grid.columns() {
        let cell = &grid[line][Column(col)];
        if !cell.flags().contains(Flags::WIDE_CHAR_SPACER) {
            result.push(cell.c);
        }
    }
    result.trim_end().to_owned()
}

fn truncate_lines(all_lines: &[String]) -> String {
    let visible_text = if all_lines.len() <= PREFIX_LINES + SUFFIX_LINES {
        all_lines.join("\n")
    } else {
        let skipped = all_lines.len() - PREFIX_LINES - SUFFIX_LINES;
        let prefix = &all_lines[..PREFIX_LINES];
        let suffix = &all_lines[all_lines.len() - SUFFIX_LINES..];
        let marker = format!("\n... [truncated {skipped} lines] ...\n");
        format!("{}{}{}", prefix.join("\n"), marker, suffix.join("\n"))
    };

    truncate_chars(visible_text)
}

fn truncate_chars(text: String) -> String {
    truncate_with_marker(&text, MAX_CHARS, "\n... [char limit reached]")
}

/// Truncate to `max_chars` *characters* and append `marker` when anything was cut.
///
/// Counts characters, not bytes: slicing at a byte offset panics when it lands inside a
/// multi-byte character, which would discard the whole context via the `catch_unwind`
/// in [`extract_context`].
fn truncate_with_marker(text: &str, max_chars: usize, marker: &str) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max_chars).collect();
    format!("{kept}{marker}")
}

/// Extract the most recent command and its output from the visible grid.
///
/// Scans from the bottom to find the two most recent prompt-bearing lines, then returns
/// (command, output) where command is what was typed and output is the text between the
/// two prompts. Both strings are empty when fewer than two prompt lines are visible or the
/// earlier prompt has no command (e.g. user just opened the shell).
pub fn extract_command_block(lines: &[String]) -> (String, String) {
    extract_command_blocks(lines, 1)
        .pop()
        .map(|block| (block.command, block.output))
        .unwrap_or_default()
}

/// A command and the output it produced, reconstructed from the grid.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GridBlock {
    pub command: String,
    pub output: String,
    /// Index into the scanned line slice of the prompt row the command was typed on.
    pub prompt_row: usize,
}

/// Cap on a single block's output, to stay within the LLM context budget.
const MAX_BLOCK_OUTPUT_CHARS: usize = 3_000;

/// Reconstruct up to `max_blocks` command blocks from `lines`, oldest first.
///
/// A block is the text between two consecutive prompt-bearing lines, so the bottommost
/// prompt acts as a delimiter only — a command the user is still typing is excluded.
/// Bare prompts (no command after the marker) are delimiters too.
pub fn extract_command_blocks(lines: &[String], max_blocks: usize) -> Vec<GridBlock> {
    if max_blocks == 0 {
        return Vec::new();
    }

    // Scan bottom-up for prompt rows; N blocks need N + 1 delimiters.
    let mut prompt_rows: Vec<usize> = Vec::with_capacity(max_blocks + 1);
    for (i, line) in lines.iter().enumerate().rev() {
        if line_has_prompt(line) {
            prompt_rows.push(i);
            if prompt_rows.len() == max_blocks + 1 {
                break;
            }
        }
    }
    prompt_rows.reverse();

    let mut blocks = Vec::with_capacity(prompt_rows.len().saturating_sub(1));
    for pair in prompt_rows.windows(2) {
        let (cmd_row, next_row) = (pair[0], pair[1]);
        let command = match command_after_last_prompt(&lines[cmd_row]) {
            Some(command) if !command.is_empty() => command,
            _ => continue,
        };

        let output = lines[cmd_row + 1..next_row].join("\n").trim().to_owned();
        let output =
            truncate_with_marker(&output, MAX_BLOCK_OUTPUT_CHARS, "\n... [output truncated]");

        blocks.push(GridBlock { command, output, prompt_row: cmd_row });
    }

    blocks
}

/// Returns `true` if `line` contains a prompt character at a plausible prompt position.
///
/// Unlike `command_after_last_prompt`, this accepts an empty tail (end-of-line) so
/// bare prompts like `user@host $ ` are also detected.
fn line_has_prompt(line: &str) -> bool {
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    for i in 0..chars.len() {
        let (idx, ch) = chars[i];
        if !PROMPT_CHARS.contains(&ch) {
            continue;
        }
        let next = chars.get(i + 1).map(|(_, c)| *c);
        // Accept space/tab after the prompt, or end of line (bare prompt).
        if !matches!(next, Some(' ') | Some('\t') | None) {
            continue;
        }
        if is_likely_prompt_position(line, idx, ch) {
            return true;
        }
    }
    false
}

/// Returns the most recent non-empty command after a prompt in the last 10 lines.
pub fn extract_last_command(lines: &[String]) -> String {
    if lines.is_empty() {
        return String::new();
    }

    let search_start = lines.len().saturating_sub(10);
    for line in lines[search_start..].iter().rev() {
        if let Some(command) = command_after_last_prompt(line) {
            return command;
        }
    }

    String::new()
}

fn command_after_last_prompt(line: &str) -> Option<String> {
    // Use the LAST prompt on the line (per design Property 4), not the first.
    // `$ echo $HOME` must yield `echo $HOME`, not stop at interior `$` tokens.
    let chars: Vec<(usize, char)> = line.char_indices().collect();
    let mut last_command = None;

    for i in 0..chars.len() {
        let (idx, ch) = chars[i];
        if !PROMPT_CHARS.contains(&ch) {
            continue;
        }
        let next = chars.get(i + 1).map(|(_, c)| *c);
        if !matches!(next, Some(' ') | Some('\t')) {
            continue;
        }
        if !is_likely_prompt_position(line, idx, ch) {
            continue;
        }
        let after = &line[idx + ch.len_utf8()..];
        let cmd = after.trim().to_owned();
        if !cmd.is_empty() {
            last_command = Some(cmd);
        }
    }

    last_command
}

/// True when `ch` at `idx` is plausibly a shell prompt marker, not `$` inside output text.
fn is_likely_prompt_position(line: &str, idx: usize, ch: char) -> bool {
    if idx == 0 {
        return true;
    }

    let prefix = line[..idx].trim_end();
    if prefix.is_empty() {
        return true;
    }

    // `❯` (U+276F) is used almost exclusively as a shell prompt arrow and is
    // virtually never found in command output.  Accept any occurrence followed
    // by a space as a prompt position without further heuristics.
    if ch == '❯' {
        return true;
    }

    // Character immediately before the prompt marker (not trim_end — that drops the space
    // before `$` on lines like `user@host $ cmd`).
    let before = line[..idx].chars().last();

    match before {
        Some(']' | ')' | ':' | '-' | '─' | '»' | '~') => true,
        Some(' ') => {
            // Space before `$` is common on real prompts (`user@host $`, `(venv) $`)
            // but also in prose output (`use $ git`). Require prompt-like prefix.
            prefix.contains('@')
                || prefix.ends_with(']')
                || prefix.ends_with(')')
                || prefix.contains('─')
                || prefix.contains('❯')
        },
        _ => false,
    }
}

fn extract_selection(grid: &Grid<Cell>, range: SelectionRange) -> Option<String> {
    let mut text = String::new();
    let mut point = range.start;

    loop {
        if point > range.end {
            break;
        }

        if range.contains(point) {
            let cell = &grid[point.line][point.column];
            if !cell.flags().contains(Flags::WIDE_CHAR_SPACER) {
                text.push(cell.c);
            }
        }

        if point == range.end {
            break;
        }

        if point.column >= grid.last_column() {
            point.column = Column(0);
            point.line += 1;
            if !text.is_empty() && !text.ends_with('\n') {
                text.push('\n');
            }
        } else {
            point.column += Column(1);
        }
    }

    Some(text.trim().to_owned()).filter(|s| !s.is_empty())
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    const TRUNCATION_SUFFIX_LEN: usize = "\n... [char limit reached]".len();

    // ---- Unit tests (Task 2.3) ----

    #[test]
    fn no_truncation_at_boundary() {
        let lines: Vec<String> =
            (0..PREFIX_LINES + SUFFIX_LINES).map(|i| format!("line{i}")).collect();
        let text = truncate_lines(&lines);
        assert!(!text.contains("truncated"));
        assert_eq!(text.lines().count(), PREFIX_LINES + SUFFIX_LINES);
    }

    #[test]
    fn truncation_kicks_in_one_over_boundary() {
        let lines: Vec<String> =
            (0..PREFIX_LINES + SUFFIX_LINES + 1).map(|i| format!("line{i}")).collect();
        let text = truncate_lines(&lines);
        assert!(text.contains("truncated 1 lines"));
    }

    #[test]
    fn last_command_extracts_after_dollar() {
        let lines = vec!["noise".into(), "$ git status".into()];
        assert_eq!(extract_last_command(&lines), "git status");
    }

    #[test]
    fn last_command_empty_without_prompt() {
        let lines = vec!["no prompt here".into()];
        assert!(extract_last_command(&lines).is_empty());
    }

    // Regression: old "last occurrence" logic returned "HOME" for this input.
    #[test]
    fn last_command_preserves_dollar_in_command() {
        let lines = vec!["$ echo $HOME".into()];
        assert_eq!(extract_last_command(&lines), "echo $HOME");
    }

    // Regression: old logic returned `H"` for --format="%H".
    #[test]
    fn last_command_preserves_percent_in_format_flag() {
        let lines = vec!["$ git log --format=\"%H\"".into()];
        assert_eq!(extract_last_command(&lines), "git log --format=\"%H\"");
    }

    // Regression: hash in command should not be treated as a prompt.
    #[test]
    fn last_command_preserves_hash_in_command() {
        let lines = vec!["$ git log --grep=\"#123\"".into()];
        assert_eq!(extract_last_command(&lines), "git log --grep=\"#123\"");
    }

    #[test]
    fn empty_grid_produces_empty_visible_text_and_command() {
        let lines: Vec<String> = Vec::new();
        assert_eq!(truncate_lines(&lines), "");
        assert_eq!(extract_last_command(&lines), "");
    }

    #[test]
    fn last_command_handles_all_four_prompt_chars() {
        for prompt in &['$', '#', '%', '❯'] {
            let lines = vec![format!("user@host {prompt} echo hi")];
            assert_eq!(extract_last_command(&lines), "echo hi", "prompt char: {prompt}");
        }
    }

    #[test]
    fn last_command_skips_prompts_outside_last_10_lines() {
        // 11 lines: prompt only on the first line, which is outside the 10-line window.
        let mut lines: Vec<String> = vec!["$ first cmd".into()];
        lines.extend((0..10).map(|i| format!("noprompt{i}")));
        assert!(extract_last_command(&lines).is_empty());
    }

    #[test]
    fn last_command_ignores_dollar_signs_in_output_prose() {
        let lines = vec![
            "$ git status".into(),
            "On branch main".into(),
            "You can run $ git diff next".into(),
            "user@host $ ".into(),
        ];
        assert_eq!(extract_last_command(&lines), "git status");
    }

    #[test]
    fn last_command_skips_empty_current_prompt() {
        let lines = vec![
            "$ cargo build".into(),
            "   Compiling learnminal".into(),
            "awni@mbp ~/proj $ ".into(),
        ];
        assert_eq!(extract_last_command(&lines), "cargo build");
    }

    #[test]
    fn last_command_uses_last_prompt_on_line() {
        assert_eq!(command_after_last_prompt("$ echo $HOME").as_deref(), Some("echo $HOME"));
    }

    // Regression: ❯ directly after a path segment (no @ or other anchor) must be detected.
    #[test]
    fn last_command_detects_chevron_after_path() {
        let lines = vec!["~/projects/learnminal❯ cargo build".into()];
        assert_eq!(extract_last_command(&lines), "cargo build");
    }

    #[test]
    fn last_command_detects_chevron_after_short_path() {
        let lines = vec!["~/src❯ git status".into()];
        assert_eq!(extract_last_command(&lines), "git status");
    }

    // read_last_command should strip zsh EXTENDED_HISTORY prefix ": ts:0;cmd".
    #[test]
    fn read_last_command_strips_extended_history_prefix() {
        // Simulate what EXTENDED_HISTORY writes: ": 1716000000:0;git rebase -i HEAD~3"
        let raw = ": 1716000000:0;git rebase -i HEAD~3";
        // Strip manually using the same logic as read_last_command.
        let command = if let Some(rest) = raw.strip_prefix(": ") {
            match rest.find(';') {
                Some(i) => rest[i + 1..].trim().to_owned(),
                None => raw.to_owned(),
            }
        } else {
            raw.to_owned()
        };
        assert_eq!(command, "git rebase -i HEAD~3");
    }

    // ---- extract_command_block tests ----

    #[test]
    fn command_block_extracts_command_and_output() {
        let lines: Vec<String> = vec![
            "user@host $ git status".into(),
            "On branch main".into(),
            "nothing to commit".into(),
            "user@host $ ".into(),
        ];
        let (cmd, out) = extract_command_block(&lines);
        assert_eq!(cmd, "git status");
        assert_eq!(out, "On branch main\nnothing to commit");
    }

    #[test]
    fn command_block_empty_when_only_one_prompt() {
        let lines: Vec<String> = vec!["user@host $ git status".into()];
        let (cmd, out) = extract_command_block(&lines);
        assert!(cmd.is_empty());
        assert!(out.is_empty());
    }

    #[test]
    fn command_block_empty_output_when_command_produced_no_output() {
        let lines: Vec<String> = vec!["user@host $ clear".into(), "user@host $ ".into()];
        let (cmd, out) = extract_command_block(&lines);
        assert_eq!(cmd, "clear");
        assert!(out.is_empty());
    }

    #[test]
    fn command_block_works_with_chevron_prompt() {
        let lines: Vec<String> = vec![
            "~/proj❯ cargo test".into(),
            "running 5 tests".into(),
            "test result: ok".into(),
            "~/proj❯ ".into(),
        ];
        let (cmd, out) = extract_command_block(&lines);
        assert_eq!(cmd, "cargo test");
        assert!(out.contains("running 5 tests"));
    }

    #[test]
    fn visible_text_exactly_at_max_chars_has_no_marker() {
        let text = truncate_chars("a".repeat(MAX_CHARS));
        assert_eq!(text.len(), MAX_CHARS);
        assert!(!text.contains("[char limit reached]"));
    }

    #[test]
    fn visible_text_one_over_max_chars_adds_marker() {
        let text = truncate_chars("a".repeat(MAX_CHARS + 1));
        assert!(text.contains("[char limit reached]"));
        assert_eq!(text.len(), MAX_CHARS + TRUNCATION_SUFFIX_LEN);
    }

    #[test]
    fn visible_text_respects_max_chars() {
        let text = truncate_chars("a".repeat(MAX_CHARS + 100));
        assert!(text.len() <= MAX_CHARS + TRUNCATION_SUFFIX_LEN);
    }

    #[test]
    fn truncate_chars_is_char_boundary_safe() {
        // Byte-based truncation would slice through a multi-byte character and panic,
        // which `extract_context`'s catch_unwind would turn into an empty context.
        let text = truncate_chars("é".repeat(MAX_CHARS + 10));
        assert!(text.contains("[char limit reached]"));
        assert_eq!(text.chars().count(), MAX_CHARS + "\n... [char limit reached]".chars().count());
    }

    #[test]
    fn command_block_output_truncation_is_char_safe() {
        let long_output = "日".repeat(4_000);
        let lines: Vec<String> = vec!["$ cat notes.txt".into(), long_output, "$ ".into()];
        let (cmd, out) = extract_command_block(&lines);
        assert_eq!(cmd, "cat notes.txt");
        assert!(out.ends_with("[output truncated]"));
        assert!(out.chars().count() <= 3_000 + "\n... [output truncated]".chars().count());
    }

    // ---- extract_command_blocks tests ----

    fn sample_session() -> Vec<String> {
        vec![
            "user@host $ cd /tmp".into(),
            "user@host $ ls".into(),
            "a.txt".into(),
            "b.txt".into(),
            "user@host $ false".into(),
            "user@host $ ".into(),
        ]
    }

    #[test]
    fn extract_command_blocks_returns_oldest_to_newest() {
        let blocks = extract_command_blocks(&sample_session(), 5);
        let commands: Vec<&str> = blocks.iter().map(|b| b.command.as_str()).collect();
        assert_eq!(commands, ["cd /tmp", "ls", "false"]);
        assert!(blocks[0].output.is_empty());
        assert_eq!(blocks[1].output, "a.txt\nb.txt");
    }

    #[test]
    fn extract_command_blocks_respects_max_blocks() {
        let blocks = extract_command_blocks(&sample_session(), 2);
        let commands: Vec<&str> = blocks.iter().map(|b| b.command.as_str()).collect();
        assert_eq!(commands, ["ls", "false"], "the newest blocks must be the ones kept");

        assert!(extract_command_blocks(&sample_session(), 0).is_empty());
    }

    #[test]
    fn extract_command_blocks_skips_bare_prompt_rows() {
        let lines: Vec<String> = vec![
            "user@host $ ".into(),
            "user@host $ echo hi".into(),
            "hi".into(),
            "user@host $ ".into(),
        ];
        let blocks = extract_command_blocks(&lines, 5);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].command, "echo hi");
    }

    #[test]
    fn extract_command_block_wrapper_matches_legacy_behaviour() {
        // The single-block wrapper must keep returning exactly what the old bottom-two-
        // prompts scan did, since `last_command` still falls back to it.
        assert_eq!(extract_command_block(&sample_session()), ("false".to_owned(), String::new()));
        assert_eq!(extract_command_block(&[]), (String::new(), String::new()));

        // Bare prompt directly above the current one: no command, so no block.
        let bare: Vec<String> = vec!["user@host $ ".into(), "user@host $ ".into()];
        assert_eq!(extract_command_block(&bare), (String::new(), String::new()));
    }

    // ---- scan_range tests ----

    #[test]
    fn scan_range_clamps_to_topmost() {
        assert_eq!(scan_range(-10, 23, 500), (-10, 23));
        assert_eq!(scan_range(-10_000, 23, 500), (-476, 23));
    }

    #[test]
    fn scan_range_handles_empty_history_and_degenerate_limits() {
        assert_eq!(scan_range(0, 23, 500), (0, 23));
        assert_eq!(scan_range(-100, 23, 1), (23, 23));
        // A zero limit would produce an inverted range; clamp it to a single line.
        assert_eq!(scan_range(-100, 23, 0), (23, 23));
    }

    // ---- Property tests (Task 2.2) ----

    fn prompt_char_strategy() -> impl Strategy<Value = char> {
        prop_oneof![Just('$'), Just('#'), Just('%'), Just('❯')]
    }

    proptest! {
        // Property 2: visible_text length is bounded for any grid size.
        // The schema cap is MAX_CHARS, but our implementation appends a fixed-length
        // suffix marker on overflow; assert the strict upper bound that includes it.
        #[test]
        fn property2_visible_text_length_bounded(
            line_count in 0usize..200,
            line_len in 0usize..120,
        ) {
            let line = "a".repeat(line_len);
            let lines: Vec<String> = (0..line_count).map(|_| line.clone()).collect();
            let text = truncate_lines(&lines);
            prop_assert!(
                text.len() <= MAX_CHARS + TRUNCATION_SUFFIX_LEN,
                "len={} exceeds bound {}", text.len(), MAX_CHARS + TRUNCATION_SUFFIX_LEN,
            );
        }

        // Property 3: middle-truncation preserves exactly PREFIX_LINES + SUFFIX_LINES
        // input lines, plus the marker, when input exceeds the threshold.
        // Keep total chars well under MAX_CHARS so char-truncation does not interfere.
        #[test]
        fn property3_middle_truncation_preserves_prefix_and_suffix(extra in 1usize..50) {
            let total = PREFIX_LINES + SUFFIX_LINES + extra;
            // Bracket-wrapped tags ensure unique substrings (e.g. [L1] is not in [L10]).
            let lines: Vec<String> = (0..total).map(|i| format!("[L{i}]")).collect();
            let text = truncate_lines(&lines);

            for i in 0..PREFIX_LINES {
                prop_assert!(
                    text.contains(&format!("[L{i}]")),
                    "prefix line [L{i}] missing from output",
                );
            }
            for i in (total - SUFFIX_LINES)..total {
                prop_assert!(
                    text.contains(&format!("[L{i}]")),
                    "suffix line [L{i}] missing from output",
                );
            }
            // A line strictly between prefix and suffix windows must be dropped.
            let middle_idx = PREFIX_LINES + (extra / 2);
            if middle_idx >= PREFIX_LINES && middle_idx < total - SUFFIX_LINES {
                prop_assert!(
                    !text.contains(&format!("[L{middle_idx}]")),
                    "middle line [L{middle_idx}] should be truncated",
                );
            }
            let marker = format!("truncated {extra} lines");
            prop_assert!(text.contains(&marker));
        }

        // Property 4 (positive): for a single line `<prefix><prompt> <cmd>` with no
        // other prompt chars, last_command returns the trimmed cmd.
        #[test]
        fn property4_last_command_returns_text_after_last_prompt(
            prefix in prop_oneof![
                Just(String::new()),
                Just("user@host ".to_owned()),
                Just("(venv) ".to_owned()),
            ],
            prompt in prompt_char_strategy(),
            spaces in "[ ]{1,3}",
            cmd in "[a-zA-Z0-9_./]+",
        ) {
            let line = format!("{prefix}{prompt}{spaces}{cmd}");
            let lines = vec![line];
            prop_assert_eq!(extract_last_command(&lines), cmd);
        }

        // Property 4 (negative): no prompt chars anywhere → empty result.
        #[test]
        fn property4_last_command_empty_without_any_prompt_char(
            lines in prop::collection::vec("[a-zA-Z0-9 ]*", 0..15),
        ) {
            // Strategy excludes prompt chars by construction; assert as a sanity guard.
            for line in &lines {
                prop_assume!(!line.chars().any(|c| PROMPT_CHARS.contains(&c)));
            }
            prop_assert!(extract_last_command(&lines).is_empty());
        }

        // Multi-block extraction must survive arbitrary screen content: it runs on
        // scrollback, which contains anything the user ever ran.
        #[test]
        fn extract_command_blocks_is_bounded_and_well_formed(
            lines in prop::collection::vec(".*", 0..80),
            max_blocks in 0usize..10,
        ) {
            let blocks = extract_command_blocks(&lines, max_blocks);
            prop_assert!(blocks.len() <= max_blocks);
            for pair in blocks.windows(2) {
                prop_assert!(pair[0].prompt_row < pair[1].prompt_row, "blocks must be ordered");
            }
            for block in &blocks {
                prop_assert!(!block.command.is_empty(), "bare prompts are delimiters only");
                prop_assert!(block.prompt_row < lines.len());
            }
        }

        // The single-block wrapper is the fallback for `last_command`; it must stay
        // equivalent to asking for the newest of many blocks.
        #[test]
        fn extract_command_block_agrees_with_multi_block(
            lines in prop::collection::vec(".*", 0..40),
        ) {
            let single = extract_command_block(&lines);
            let newest = extract_command_blocks(&lines, 1)
                .pop()
                .map(|b| (b.command, b.output))
                .unwrap_or_default();
            prop_assert_eq!(single, newest);
        }
    }
}
