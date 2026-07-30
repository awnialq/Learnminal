//! Native chat prompt construction.
//!
//! Combines environment, last command + output + exit code, past journal notes,
//! a budgeted reference excerpt, and the user's question. Plain text only.

use std::path::Path;

use crate::learnminal::journal::JournalNote;
use crate::learnminal::settings::ExperienceLevel;
use crate::learnminal::types::{HistorySource, ReferenceContext, TerminalContext};

const CONTEXT_MAX_CHARS: usize = 1_000;
const EXCERPT_MAX_CHARS: usize = 2_000;
/// The excerpt largely restates what the history block already shows, so it shrinks
/// when history is present to keep the overall prompt roughly the same size.
const EXCERPT_MAX_CHARS_WITH_HISTORY: usize = 1_200;
const JOURNAL_BUDGET_CHARS: usize = 1_500;
const HISTORY_BUDGET_CHARS: usize = 1_500;
const HISTORY_COMMAND_MAX_CHARS: usize = 200;
/// Output allowance per entry, newest first. Recent commands get the most room.
const HISTORY_OUTPUT_BUDGETS: [usize; 5] = [600, 350, 200, 150, 100];

/// Build the full chat prompt sent to Ollama.
pub fn build_chat_prompt(
    ctx: &TerminalContext,
    reference: Option<&ReferenceContext>,
    journal_notes: &[JournalNote],
    message: &str,
    experience_level: ExperienceLevel,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(&system_instructions(experience_level));
    prompt.push_str("\n\n");

    if let Some(env) = env_line() {
        prompt.push_str(&env);
        prompt.push_str("\n\n");
    }

    // The history block already carries the newest command, its exit code, and its
    // output, so the single-command blocks would only repeat it.
    let history_block = format_command_history(ctx);
    match &history_block {
        Some(history) => {
            prompt.push_str(history);
            prompt.push_str("\n\n");

            if let Some(selection) = selection_block(ctx) {
                prompt.push_str(&selection);
                prompt.push_str("\n\n");
            }
        },
        None => {
            if !ctx.last_command.trim().is_empty() {
                prompt.push_str("Last command:\n");
                prompt.push_str(ctx.last_command.trim());
                prompt.push_str("\n\n");
            }

            let context_block = context_block(ctx);
            if !context_block.is_empty() {
                prompt.push_str(&context_block);
                prompt.push_str("\n\n");
            }
        },
    }

    if let Some(notes_block) = format_journal_notes(journal_notes) {
        prompt.push_str(&notes_block);
        prompt.push_str("\n\n");
    }

    match reference {
        Some(reference) if reference.has_body() => {
            prompt.push_str(&format!("Reference ({}):\n", reference.source.label()));
            prompt.push_str(reference.body.trim());
            prompt.push_str("\n\n");
        },
        Some(reference) if !reference.program.is_empty() => {
            prompt.push_str(&format!(
                "Context status: No local man/--help (or fallback docs) for {}.\n\n",
                reference.program
            ));
        },
        _ => {},
    }

    let excerpt_budget =
        if history_block.is_some() { EXCERPT_MAX_CHARS_WITH_HISTORY } else { EXCERPT_MAX_CHARS };
    let excerpt = truncate(&ctx.visible_text, excerpt_budget);
    if !excerpt.trim().is_empty() {
        prompt.push_str("Terminal excerpt:\n");
        prompt.push_str(&excerpt);
        prompt.push_str("\n\n");
    }

    prompt.push_str("User question:\n");
    prompt.push_str(message.trim());
    prompt
}

fn system_instructions(level: ExperienceLevel) -> String {
    let mut instructions = String::from(
        "You are an expert command-line educator helping a developer understand their shell.\n\
         Answer in clear conversational plain text. Do not use markdown.\n\
         Prefer the Reference and Past notes sections over remembered training data.\n\
         Do not invent flags or options that are not present in the Reference.\n\
         If Reference is missing, say so rather than guessing flags.\n\
         You may call the web_search tool for current events, version changes, changelogs,\n\
         or facts not covered by Reference. Prefer local Reference for flags and options.\n\
         When using search results, briefly cite titles or URLs in plain text.\n",
    );
    instructions.push_str(&experience_guidance(level));
    instructions
}

fn experience_guidance(level: ExperienceLevel) -> String {
    match level {
        ExperienceLevel::Beginner => {
            "User experience level: Beginner.\n\
             Assume little or no terminal experience. Define terminology in plain language,\n\
             explain what each command and flag does step by step, avoid jargon unless you\n\
             briefly explain it, and include safe next actions the user can try.\n"
                .into()
        },
        ExperienceLevel::Novice => {
            "User experience level: Novice.\n\
             The user has some shell experience. Keep explanations approachable but lighter\n\
             than for beginners. Focus on non-obvious flags, common failure modes, and why\n\
             a command behaves the way it does without over-explaining basics.\n"
                .into()
        },
        ExperienceLevel::Professional => {
            "User experience level: Professional.\n\
             The user is a comfortable daily CLI user. Be concise. Focus on root cause,\n\
             tradeoffs, exact commands, and useful references. Skip elementary shell concepts.\n"
                .into()
        },
        ExperienceLevel::Expert => {
            "User experience level: Expert.\n\
             The user has deep terminal knowledge. Be terse and precise. Avoid basics.\n\
             Emphasize edge cases, internals, and high-signal diagnostics.\n"
                .into()
        },
    }
}

fn format_journal_notes(notes: &[JournalNote]) -> Option<String> {
    if notes.is_empty() {
        return None;
    }
    let program = &notes[0].program;
    let mut block = format!("Past notes for {program}:\n");
    let mut used = block.chars().count();
    for (i, note) in notes.iter().enumerate() {
        let q = truncate(&note.question, 200);
        let a = truncate(&note.reply, 400);
        let entry = format!("{}. Q: {q}\n   A: {a}\n", i + 1);
        let entry_len = entry.chars().count();
        if used + entry_len > JOURNAL_BUDGET_CHARS {
            break;
        }
        block.push_str(&entry);
        used += entry_len;
    }
    Some(block)
}

/// Best-effort "Environment: <os>, <shell>" line (no subprocesses).
fn env_line() -> Option<String> {
    let os = match std::env::consts::OS {
        "macos" => "macOS",
        "linux" => "Linux",
        "windows" => "Windows",
        other => other,
    };
    let shell = std::env::var("SHELL")
        .ok()
        .and_then(|p| Path::new(&p).file_name().map(|n| n.to_string_lossy().into_owned()))
        .unwrap_or_default();
    if shell.is_empty() {
        Some(format!("Environment: {os}"))
    } else {
        Some(format!("Environment: {os}, {shell}"))
    }
}

/// Recent commands with their exit codes and output, oldest first.
///
/// Returns `None` when no history was recovered, in which case the caller falls back to
/// the single-command blocks.
fn format_command_history(ctx: &TerminalContext) -> Option<String> {
    let entries: Vec<_> =
        ctx.command_history.iter().filter(|entry| !entry.command.trim().is_empty()).collect();
    if entries.is_empty() {
        return None;
    }

    // Build newest first so that running out of budget drops the oldest commands
    // rather than truncating the most relevant one.
    let mut rendered: Vec<String> = Vec::with_capacity(entries.len());
    let mut used = 0;

    for (age, entry) in entries.iter().rev().enumerate() {
        let index = entries.len() - 1 - age;
        let mut text = format!("$ {}", truncate(entry.command.trim(), HISTORY_COMMAND_MAX_CHARS));
        // A zero exit code is the uninteresting case; matches `context_block`.
        if let Some(code) = entry.exit_code.filter(|code| *code != 0) {
            text.push_str(&format!("    [exit {code}]"));
        }
        text.push('\n');

        // Show the directory only where it changed, so `cd` stays legible without
        // repeating the same path on every line.
        let cwd_changed = index == 0 || entries[index - 1].cwd != entry.cwd;
        if !entry.cwd.is_empty() && cwd_changed {
            text.push_str(&format!("  (cwd: {})\n", entry.cwd));
        }

        let output_budget = HISTORY_OUTPUT_BUDGETS.get(age).copied().unwrap_or(0);
        let output = entry.output.trim();
        if output_budget > 0 && !output.is_empty() {
            for line in truncate_middle(output, output_budget).lines() {
                text.push_str("  ");
                text.push_str(line);
                text.push('\n');
            }
        }

        let len = text.chars().count();
        if !rendered.is_empty() && used + len > HISTORY_BUDGET_CHARS {
            break;
        }
        used += len;
        rendered.push(text);
    }

    rendered.reverse();
    let mut block =
        String::from("Recent commands (oldest first; the last one is the most recent):\n");
    for entry in rendered {
        block.push_str(&entry);
    }

    if ctx.history_source == HistorySource::GridOnly {
        block.push_str(
            "Note: exit codes for earlier commands are unavailable (shell integration not \
             installed).\n",
        );
    }

    Some(truncate(block.trim_end(), HISTORY_BUDGET_CHARS))
}

/// Keep the head and tail of `text`, dropping the middle.
///
/// Compiler and package-manager output puts the actual error at the end, which a
/// head-only truncation would throw away.
fn truncate_middle(text: &str, max_chars: usize) -> String {
    let total = text.chars().count();
    if total <= max_chars {
        return text.to_owned();
    }

    let head_len = max_chars * 3 / 5;
    let tail_len = max_chars - head_len;
    let head: String = text.chars().take(head_len).collect();
    let tail: String = text.chars().skip(total - tail_len).collect();
    let omitted = total - max_chars;

    format!("{}\n... [{omitted} chars omitted] ...\n{}", head.trim_end(), tail.trim_start())
}

fn selection_block(ctx: &TerminalContext) -> Option<String> {
    let selection = ctx.selected_text.as_ref().filter(|s| !s.is_empty())?;
    Some(format!("Selection:\n{}", truncate(selection, CONTEXT_MAX_CHARS)))
}

fn context_block(ctx: &TerminalContext) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(selection) = selection_block(ctx) {
        parts.push(selection);
    }
    if let Some(code) = ctx.exit_code {
        if code != 0 {
            parts.push(format!("Exit code: {code}"));
        }
    }
    if !ctx.last_command_output.is_empty() {
        parts.push(format!("Output:\n{}", truncate(&ctx.last_command_output, CONTEXT_MAX_CHARS)));
    }
    parts.join("\n")
}

fn truncate(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max_chars).collect();
    format!("{}\n... [truncated]", kept.trim_end())
}

#[cfg(test)]
mod tests {
    use proptest::prelude::*;

    use super::*;
    use crate::learnminal::types::{CommandEntry, ReferenceSource};

    fn ctx() -> TerminalContext {
        TerminalContext {
            last_command: "git push origin main".into(),
            last_command_output: "error: failed to push".into(),
            exit_code: Some(1),
            visible_text: "$ git push origin main\nerror: failed to push".into(),
            ..TerminalContext::default()
        }
    }

    #[test]
    fn prompt_includes_command_output_and_question() {
        let prompt =
            build_chat_prompt(&ctx(), None, &[], "why did this fail?", ExperienceLevel::Beginner);
        assert!(prompt.contains("Last command:\ngit push origin main"));
        assert!(prompt.contains("Exit code: 1"));
        assert!(prompt.contains("Output:\nerror: failed to push"));
        assert!(prompt.contains("User question:\nwhy did this fail?"));
        assert!(prompt.contains("Prefer the Reference and Past notes"));
        assert!(prompt.contains("web_search"));
    }

    #[test]
    fn prompt_includes_reference_when_present() {
        let reference = ReferenceContext {
            program: "git".into(),
            source: ReferenceSource::Man,
            body: "NAME\n git - tracker".into(),
        };
        let prompt =
            build_chat_prompt(&ctx(), Some(&reference), &[], "explain", ExperienceLevel::Novice);
        assert!(prompt.contains("Reference (man):"));
        assert!(prompt.contains("git - tracker"));
    }

    #[test]
    fn prompt_includes_missing_reference_notice() {
        let reference = ReferenceContext::empty("obscuretool");
        let prompt =
            build_chat_prompt(&ctx(), Some(&reference), &[], "help", ExperienceLevel::Beginner);
        assert!(prompt.contains("No local man/--help"));
        assert!(prompt.contains("obscuretool"));
    }

    #[test]
    fn prompt_includes_journal_notes() {
        let notes = vec![JournalNote {
            id: 1,
            program: "git".into(),
            question: "how do I rebase?".into(),
            reply: "Use git rebase -i".into(),
            last_command: String::new(),
            reference_source: "man".into(),
            verified: Some(true),
            created_at: 1,
        }];
        let prompt =
            build_chat_prompt(&ctx(), None, &notes, "again?", ExperienceLevel::Professional);
        assert!(prompt.contains("Past notes for git:"));
        assert!(prompt.contains("how do I rebase?"));
        assert!(prompt.contains("Use git rebase -i"));
    }

    #[test]
    fn prompt_omits_zero_exit_code() {
        let mut c = ctx();
        c.exit_code = Some(0);
        let prompt = build_chat_prompt(&c, None, &[], "q", ExperienceLevel::Beginner);
        assert!(!prompt.contains("Exit code:"));
    }

    #[test]
    fn prompt_includes_beginner_guidance() {
        let prompt = build_chat_prompt(&ctx(), None, &[], "q", ExperienceLevel::Beginner);
        assert!(prompt.contains("User experience level: Beginner."));
        assert!(prompt.contains("step by step"));
    }

    #[test]
    fn prompt_includes_expert_guidance() {
        let prompt = build_chat_prompt(&ctx(), None, &[], "q", ExperienceLevel::Expert);
        assert!(prompt.contains("User experience level: Expert."));
        assert!(prompt.contains("terse and precise"));
        assert!(!prompt.contains("step by step"));
    }

    #[test]
    fn truncate_caps_long_text() {
        let long = "a".repeat(600);
        let out = truncate(&long, 500);
        assert!(out.contains("[truncated]"));
        assert!(out.chars().count() <= 500 + "\n... [truncated]".chars().count());
    }

    // ---- Session command history ----

    fn entry(command: &str, exit_code: Option<i32>, output: &str) -> CommandEntry {
        CommandEntry {
            command: command.into(),
            exit_code,
            output: output.into(),
            cwd: String::new(),
        }
    }

    fn ctx_with_history(entries: Vec<CommandEntry>) -> TerminalContext {
        TerminalContext {
            command_history: entries,
            history_source: HistorySource::ShellHook,
            ..ctx()
        }
    }

    #[test]
    fn history_block_replaces_last_command_and_output() {
        let history = ctx_with_history(vec![
            entry("cd /tmp", Some(0), ""),
            entry("git push origin main", Some(1), "error: failed to push"),
        ]);
        let prompt = build_chat_prompt(&history, None, &[], "why?", ExperienceLevel::Beginner);

        assert!(prompt.contains("Recent commands (oldest first"));
        assert!(prompt.contains("$ cd /tmp"));
        assert!(prompt.contains("$ git push origin main    [exit 1]"));
        assert!(prompt.contains("error: failed to push"));

        // The newest entry already carries all three, so the legacy blocks must go.
        assert!(!prompt.contains("Last command:"));
        assert!(!prompt.contains("Exit code:"));
        assert!(!prompt.contains("Output:"));
    }

    #[test]
    fn history_absent_keeps_legacy_blocks_untouched() {
        let prompt = build_chat_prompt(&ctx(), None, &[], "why?", ExperienceLevel::Beginner);
        assert!(!prompt.contains("Recent commands"));
        assert!(prompt.contains("Last command:\ngit push origin main"));
        assert!(prompt.contains("Exit code: 1"));
        assert!(prompt.contains("Output:\nerror: failed to push"));
    }

    #[test]
    fn history_keeps_selection_block() {
        let mut history = ctx_with_history(vec![entry("ls", Some(0), "a.txt")]);
        history.selected_text = Some("a.txt".into());
        let prompt = build_chat_prompt(&history, None, &[], "what?", ExperienceLevel::Beginner);
        assert!(prompt.contains("Recent commands"));
        assert!(prompt.contains("Selection:\na.txt"));
    }

    #[test]
    fn history_omits_zero_exit_codes() {
        let history = ctx_with_history(vec![entry("ls", Some(0), "a.txt")]);
        let block = format_command_history(&history).unwrap();
        assert!(block.contains("$ ls\n"));
        assert!(!block.contains("[exit"));
    }

    #[test]
    fn history_shows_cwd_only_when_it_changes() {
        let mut entries = vec![
            entry("cd /a", Some(0), ""),
            entry("ls", Some(0), ""),
            entry("cd /b", Some(0), ""),
        ];
        entries[0].cwd = "/a".into();
        entries[1].cwd = "/a".into();
        entries[2].cwd = "/b".into();

        let block = format_command_history(&ctx_with_history(entries)).unwrap();
        assert_eq!(block.matches("(cwd: /a)").count(), 1);
        assert_eq!(block.matches("(cwd: /b)").count(), 1);
    }

    #[test]
    fn history_grid_only_adds_degraded_note() {
        let mut history = ctx_with_history(vec![entry("ls", None, "a.txt")]);
        history.history_source = HistorySource::GridOnly;
        let block = format_command_history(&history).unwrap();
        assert!(block.contains("shell integration not installed"));
    }

    #[test]
    fn history_is_none_when_every_command_is_blank() {
        let history = ctx_with_history(vec![entry("   ", Some(0), "junk")]);
        assert!(format_command_history(&history).is_none());
    }

    #[test]
    fn history_newest_entry_gets_the_largest_output_budget() {
        let noise = "x".repeat(2_000);
        let history =
            ctx_with_history(vec![entry("old", Some(0), &noise), entry("new", Some(0), &noise)]);
        let block = format_command_history(&history).unwrap();

        let old_len = block.split("$ new").next().unwrap().chars().count();
        let new_len = block.split("$ new").nth(1).unwrap().chars().count();
        assert!(new_len > old_len, "newest output should get more room: {new_len} vs {old_len}");
    }

    #[test]
    fn history_drops_oldest_entries_when_over_budget() {
        let noise = "x".repeat(5_000);
        let entries: Vec<CommandEntry> =
            (0..5).map(|i| entry(&format!("cmd{i}"), Some(0), &noise)).collect();
        let block = format_command_history(&ctx_with_history(entries)).unwrap();

        assert!(block.chars().count() <= HISTORY_BUDGET_CHARS + "\n... [truncated]".len());
        assert!(block.contains("$ cmd4"), "the newest command must always survive");
        assert!(!block.contains("$ cmd0"), "the oldest commands are dropped first");
    }

    #[test]
    fn history_single_huge_entry_still_respects_budget() {
        let history = ctx_with_history(vec![entry("cmd", Some(0), &"x".repeat(50_000))]);
        let block = format_command_history(&history).unwrap();
        assert!(block.chars().count() <= HISTORY_BUDGET_CHARS + "\n... [truncated]".len());
    }

    #[test]
    fn history_shrinks_the_terminal_excerpt() {
        let mut history = ctx_with_history(vec![entry("ls", Some(0), "a.txt")]);
        history.visible_text = "y".repeat(5_000);
        let with_history = build_chat_prompt(&history, None, &[], "q", ExperienceLevel::Beginner);

        let mut legacy = ctx();
        legacy.visible_text = "y".repeat(5_000);
        let without_history = build_chat_prompt(&legacy, None, &[], "q", ExperienceLevel::Beginner);

        assert!(with_history.matches('y').count() < without_history.matches('y').count());
    }

    #[test]
    fn truncate_middle_keeps_head_and_tail() {
        let text = format!("{}MIDDLE{}", "H".repeat(100), "T".repeat(100));
        let out = truncate_middle(&text, 60);
        assert!(out.starts_with('H'));
        assert!(out.ends_with('T'));
        assert!(!out.contains("MIDDLE"));
        assert!(out.contains("chars omitted"));
    }

    #[test]
    fn truncate_middle_is_char_boundary_safe() {
        let text = "日".repeat(200);
        let out = truncate_middle(&text, 50);
        assert!(out.contains("chars omitted"));
        assert!(out.starts_with('日') && out.ends_with('日'));
    }

    proptest! {
        #[test]
        fn history_block_length_is_always_bounded(
            commands in prop::collection::vec("[a-z ]{0,40}", 0..12),
            output_len in 0usize..3_000,
        ) {
            let entries: Vec<CommandEntry> = commands
                .iter()
                .map(|command| entry(command, Some(1), &"z".repeat(output_len)))
                .collect();
            if let Some(block) = format_command_history(&ctx_with_history(entries)) {
                prop_assert!(
                    block.chars().count() <= HISTORY_BUDGET_CHARS + "\n... [truncated]".len()
                );
            }
        }
    }
}
