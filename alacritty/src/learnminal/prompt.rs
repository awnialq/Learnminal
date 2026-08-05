//! Native chat prompt construction.
//!
//! Combines environment, last command + output + exit code, past journal notes,
//! a budgeted reference excerpt, and the user's question. Plain text only.

use std::path::Path;

use crate::learnminal::journal::JournalNote;
use crate::learnminal::ollama::ToolSet;
use crate::learnminal::settings::ExperienceLevel;
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
    experience_level: ExperienceLevel,
    tools: &ToolSet,
) -> String {
    let mut prompt = String::new();
    prompt.push_str(&system_instructions(experience_level, tools));
    prompt.push_str("\n\n");

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

/// Base instructions plus guidance for whichever tools are actually enabled.
///
/// A tool is only described when it is offered; advertising a disabled tool
/// makes small models call it and then stall on the "unknown tool" reply.
fn system_instructions(level: ExperienceLevel, tools: &ToolSet) -> String {
    let mut instructions = String::from(
        "You are an expert command-line educator helping a developer understand their shell.\n\
         Answer in clear conversational plain text. Do not use markdown.\n\
         Prefer the Reference and Past notes sections over remembered training data.\n\
         Do not invent flags or options that are not present in the Reference.\n\
         If Reference is missing, say so rather than guessing flags.\n",
    );
    if tools.web_search {
        instructions.push_str(
            "You may call the web_search tool for current events, version changes, changelogs,\n\
             or facts not covered by Reference. Prefer local Reference for flags and options.\n\
             When using search results, briefly cite titles or URLs in plain text.\n",
        );
    }
    if let Some(root) = tools.read_exec.as_deref() {
        instructions.push_str(
            "You may call the run_command tool to inspect this machine with read-only commands\n\
             such as ls, cat, head, wc, stat, find, grep, which, env, uname, ps, df, and\n\
             git status/log/diff. Use it whenever the answer depends on the user's actual files,\n\
             directories, git state, or installed versions rather than on general knowledge.\n\
             Investigate before you answer, and keep going until you have a real answer.\n\
             One command is rarely enough: list a directory, then look inside the promising\n\
             entries; if you do not know where something lives, search for it with find, and\n\
             widen or loosen the search (case-insensitive, partial names, more depth) when the\n\
             first attempt finds nothing. Related names count — a request for \"movies\" may be\n\
             served by a directory called Videos, Media, or Film.\n\
             You have ample tool calls available. Do not ration them, do not estimate when\n\
             you could measure, and never cite limited steps as a reason to stop early — if a\n\
             complete answer needs several more commands, run them. When asked which item is\n\
             largest, newest, or most numerous, measure it (du, wc, stat, ls -l) rather than\n\
             inferring from names.\n\
             Run one command per call: pipes, redirects, and chaining are unavailable.\n\
             A leading ~ is expanded to the home directory. find's grouping syntax works, so\n\
             prefer one \\( -iname \"*.a\" -o -iname \"*.b\" \\) search over one call per pattern.\n\
             A refused or failed command is information, not a dead end. Read the error text\n\
             before deciding anything. \"No such file or directory\" usually means the name was\n\
             wrong, not that the thing is missing: check the spelling, the capitalisation, and\n\
             whether the user's word for it differs from the actual name. When the tool lists\n\
             similarly named entries, treat them as the likely answer and look at them.\n\
             Then try a different command that respects the error — do not repeat a refused\n\
             command unchanged, and do not stop at the first failure. Only give up once you\n\
             have genuinely run out of readable places to look; then say what you tried and\n\
             what stopped you, rather than implying the thing does not exist.\n",
        );
        instructions.push_str(&format!("You can read files under {}.\n", root.display()));
    }
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

    /// The pre-existing default: web search on, environment inspection off.
    fn search_only() -> ToolSet {
        ToolSet { web_search: true, read_exec: None }
    }

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
            build_chat_prompt(
            &ctx(),
            None,
            &[],
            "why did this fail?",
            ExperienceLevel::Beginner,
            &search_only(),
        );
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
            build_chat_prompt(
            &ctx(),
            Some(&reference),
            &[],
            "explain",
            ExperienceLevel::Novice,
            &search_only(),
        );
        assert!(prompt.contains("Reference (man):"));
        assert!(prompt.contains("git - tracker"));
    }

    #[test]
    fn prompt_includes_missing_reference_notice() {
        let reference = ReferenceContext::empty("obscuretool");
        let prompt =
            build_chat_prompt(
            &ctx(),
            Some(&reference),
            &[],
            "help",
            ExperienceLevel::Beginner,
            &search_only(),
        );
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
            build_chat_prompt(
            &ctx(),
            None,
            &notes,
            "again?",
            ExperienceLevel::Professional,
            &search_only(),
        );
        assert!(prompt.contains("Past notes for git:"));
        assert!(prompt.contains("how do I rebase?"));
        assert!(prompt.contains("Use git rebase -i"));
    }

    #[test]
    fn prompt_omits_zero_exit_code() {
        let mut c = ctx();
        c.exit_code = Some(0);
        let prompt = build_chat_prompt(&c, None, &[], "q", ExperienceLevel::Beginner, &search_only());
        assert!(!prompt.contains("Exit code:"));
    }

    #[test]
    fn prompt_includes_beginner_guidance() {
        let prompt = build_chat_prompt(&ctx(), None, &[], "q", ExperienceLevel::Beginner, &search_only());
        assert!(prompt.contains("User experience level: Beginner."));
        assert!(prompt.contains("step by step"));
    }

    #[test]
    fn prompt_includes_expert_guidance() {
        let prompt = build_chat_prompt(&ctx(), None, &[], "q", ExperienceLevel::Expert, &search_only());
        assert!(prompt.contains("User experience level: Expert."));
        assert!(prompt.contains("terse and precise"));
        assert!(!prompt.contains("step by step"));
    }

    #[test]
    fn tool_guidance_follows_enabled_tools() {
        // Only enabled tools are described: a model told about a tool it was
        // not given will call it and stall on the "unknown tool" reply.
        let none = build_chat_prompt(
            &ctx(),
            None,
            &[],
            "q",
            ExperienceLevel::Expert,
            &ToolSet::default(),
        );
        assert!(!none.contains("web_search"));
        assert!(!none.contains("run_command"));

        let search =
            build_chat_prompt(&ctx(), None, &[], "q", ExperienceLevel::Expert, &search_only());
        assert!(search.contains("web_search"));
        assert!(!search.contains("run_command"));

        let both = build_chat_prompt(
            &ctx(),
            None,
            &[],
            "q",
            ExperienceLevel::Expert,
            &ToolSet { web_search: true, read_exec: Some(std::env::temp_dir()) },
        );
        assert!(both.contains("web_search"));
        assert!(both.contains("run_command"));
        assert!(both.contains("git status/log/diff"));
        assert!(both.contains("refused"));
    }

    #[test]
    fn truncate_caps_long_text() {
        let long = "a".repeat(600);
        let out = truncate(&long, 500);
        assert!(out.contains("[truncated]"));
        assert!(out.chars().count() <= 500 + "\n... [truncated]".chars().count());
    }
}
