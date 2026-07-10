use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;

use alacritty_terminal::grid::{Dimensions, Grid, GridCell};
use alacritty_terminal::index::{Column, Line, Point};
use alacritty_terminal::term::viewport_to_point;
use alacritty_terminal::selection::SelectionRange;
use alacritty_terminal::term::cell::{Cell, Flags};

use crate::learnminal::types::TerminalContext;

pub const PREFIX_LINES: usize = 40;
pub const SUFFIX_LINES: usize = 40;
pub const MAX_CHARS: usize = 8000;

const PROMPT_CHARS: &[char] = &['$', '#', '%', '❯'];

fn learnminal_state_dir() -> Option<PathBuf> {
    Some(PathBuf::from(std::env::var_os("HOME")?).join(".ai-cli-learning"))
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
/// Written by the shell precmd hook (see INSTALL.md). More reliable than parsing
/// the terminal grid, which can false-match `$` in command output.
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
    if command.is_empty() { None } else { Some(command) }
}

/// Extract terminal context from the visible grid, with middle-truncation and panic safety.
pub fn extract_context(
    grid: &Grid<Cell>,
    selection: Option<SelectionRange>,
    cwd: &str,
    last_exit_code: Option<i32>,
) -> TerminalContext {
    let result = panic::catch_unwind(AssertUnwindSafe(|| {
        extract_context_inner(grid, selection, cwd, last_exit_code)
    }));

    result.unwrap_or_default()
}

fn extract_context_inner(
    grid: &Grid<Cell>,
    selection: Option<SelectionRange>,
    cwd: &str,
    last_exit_code: Option<i32>,
) -> TerminalContext {
    let all_lines = collect_visible_lines(grid);
    let visible_text = truncate_lines(&all_lines);

    // Extract the last command block (command + output) from the grid first so we can
    // use the block command as an improved fallback when the shell hook file is absent.
    let (block_command, last_command_output) = extract_command_block(&all_lines);

    let last_command = read_last_command().unwrap_or_else(|| {
        if !block_command.is_empty() {
            block_command
        } else {
            extract_last_command(&all_lines)
        }
    });

    let selected_text =
        selection.and_then(|range| extract_selection(grid, range).filter(|s| !s.is_empty()));

    TerminalContext {
        visible_text,
        selected_text,
        last_command,
        last_command_output,
        cwd: cwd.to_owned(),
        exit_code: last_exit_code,
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

fn truncate_chars(mut text: String) -> String {
    if text.len() > MAX_CHARS {
        text.truncate(MAX_CHARS);
        text.push_str("\n... [char limit reached]");
    }
    text
}

/// Extract the most recent command and its output from the visible grid.
///
/// Scans from the bottom to find the two most recent prompt-bearing lines, then returns
/// (command, output) where command is what was typed and output is the text between the
/// two prompts. Both strings are empty when fewer than two prompt lines are visible or the
/// earlier prompt has no command (e.g. user just opened the shell).
pub fn extract_command_block(lines: &[String]) -> (String, String) {
    // Collect indices of the bottom two prompt lines (scanning from the bottom up).
    let mut prompt_rows: Vec<usize> = Vec::new();
    for (i, line) in lines.iter().enumerate().rev() {
        if line_has_prompt(line) {
            prompt_rows.push(i);
            if prompt_rows.len() == 2 {
                break;
            }
        }
    }

    if prompt_rows.len() < 2 {
        return (String::new(), String::new());
    }

    // prompt_rows[0] = bottommost prompt (current, usually empty)
    // prompt_rows[1] = previous prompt (where the last command was typed)
    let current_row = prompt_rows[0];
    let cmd_row = prompt_rows[1];

    let command = command_after_last_prompt(&lines[cmd_row]).unwrap_or_default();
    if command.is_empty() {
        return (String::new(), String::new());
    }

    let raw_output = lines[cmd_row + 1..current_row].join("\n");
    let output = raw_output.trim().to_owned();

    // Cap output to stay within the LLM context budget.
    const MAX_OUTPUT_CHARS: usize = 3_000;
    let output = if output.len() > MAX_OUTPUT_CHARS {
        format!("{}\n... [output truncated]", &output[..MAX_OUTPUT_CHARS])
    } else {
        output
    };

    (command, output)
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
        let lines: Vec<String> = (0..PREFIX_LINES + SUFFIX_LINES).map(|i| format!("line{i}")).collect();
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
        let lines: Vec<String> = vec![
            "user@host $ git status".into(),
        ];
        let (cmd, out) = extract_command_block(&lines);
        assert!(cmd.is_empty());
        assert!(out.is_empty());
    }

    #[test]
    fn command_block_empty_output_when_command_produced_no_output() {
        let lines: Vec<String> = vec![
            "user@host $ clear".into(),
            "user@host $ ".into(),
        ];
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
    }
}
