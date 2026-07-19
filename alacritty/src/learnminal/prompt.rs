//! Native chat prompt construction.
//!
//! Combines environment, last command + output + exit code, past journal notes,
//! a budgeted reference excerpt, and the user's question. Plain text only.

use std::path::Path;

use crate::learnminal::journal::JournalNote;
use crate::learnminal::types::{ReferenceContext, TerminalContext};

const CONTEXT_MAX_CHARS: usize = 1_000;
const EXCERPT_MAX_CHARS: usize = 2_000;
const JOURNAL_BUDGET_CHARS: usize = 1_500;

/// Build the full chat prompt sent to Ollama.
pub fn build_chat_prompt(
    ctx: &TerminalContext,
    reference: Option<&ReferenceContext>,
    journal_notes: &[JournalNote],
    message: &str,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(
        "You are an expert command-line educator helping a developer understand their shell.\n\
         Answer in clear conversational plain text. Do not use markdown.\n\
         Prefer the Reference and Past notes sections over remembered training data.\n\
         Do not invent flags or options that are not present in the Reference.\n\
         If Reference is missing, say so rather than guessing flags.\n\n",
    );

    if let Some(env) = env_line() {
        prompt.push_str(&env);
        prompt.push_str("\n\n");
    }

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

    let excerpt = truncate(&ctx.visible_text, EXCERPT_MAX_CHARS);
    if !excerpt.trim().is_empty() {
        prompt.push_str("Terminal excerpt:\n");
        prompt.push_str(&excerpt);
        prompt.push_str("\n\n");
    }

    prompt.push_str("User question:\n");
    prompt.push_str(message.trim());
    prompt
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

fn context_block(ctx: &TerminalContext) -> String {
    let mut parts: Vec<String> = Vec::new();
    if let Some(sel) = ctx.selected_text.as_ref().filter(|s| !s.is_empty()) {
        parts.push(format!("Selection:\n{}", truncate(sel, CONTEXT_MAX_CHARS)));
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
    use super::*;
    use crate::learnminal::types::ReferenceSource;

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
        let prompt = build_chat_prompt(&ctx(), None, &[], "why did this fail?");
        assert!(prompt.contains("Last command:\ngit push origin main"));
        assert!(prompt.contains("Exit code: 1"));
        assert!(prompt.contains("Output:\nerror: failed to push"));
        assert!(prompt.contains("User question:\nwhy did this fail?"));
        assert!(prompt.contains("Prefer the Reference and Past notes"));
    }

    #[test]
    fn prompt_includes_reference_when_present() {
        let reference = ReferenceContext {
            program: "git".into(),
            source: ReferenceSource::Man,
            body: "NAME\n git - tracker".into(),
        };
        let prompt = build_chat_prompt(&ctx(), Some(&reference), &[], "explain");
        assert!(prompt.contains("Reference (man):"));
        assert!(prompt.contains("git - tracker"));
    }

    #[test]
    fn prompt_includes_missing_reference_notice() {
        let reference = ReferenceContext::empty("obscuretool");
        let prompt = build_chat_prompt(&ctx(), Some(&reference), &[], "help");
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
        let prompt = build_chat_prompt(&ctx(), None, &notes, "again?");
        assert!(prompt.contains("Past notes for git:"));
        assert!(prompt.contains("how do I rebase?"));
        assert!(prompt.contains("Use git rebase -i"));
    }

    #[test]
    fn prompt_omits_zero_exit_code() {
        let mut c = ctx();
        c.exit_code = Some(0);
        let prompt = build_chat_prompt(&c, None, &[], "q");
        assert!(!prompt.contains("Exit code:"));
    }

    #[test]
    fn truncate_caps_long_text() {
        let long = "a".repeat(600);
        let out = truncate(&long, 500);
        assert!(out.contains("[truncated]"));
        assert!(out.chars().count() <= 500 + "\n... [truncated]".chars().count());
    }
}
