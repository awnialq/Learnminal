//! Fetch `man`/`--help` output natively and extract a budgeted context block.
//!
//! Also builds a [`ReferenceContext`] with package/docs fallbacks when manuals
//! are missing. Used as hidden context for the chat model.

use std::io::Read;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::learnminal::docs_fallback;
use crate::learnminal::types::{ReferenceContext, ReferenceSource};

const SUBPROCESS_TIMEOUT: Duration = Duration::from_secs(5);

const CORE_SECTIONS: &[&str] = &["NAME", "SYNOPSIS", "DESCRIPTION", "USAGE"];
const ARGUMENT_SECTIONS: &[&str] = &["POSITIONAL ARGUMENTS", "ARGUMENTS"];
const EXAMPLE_SECTIONS: &[&str] = &["EXAMPLES", "FLAGS", "COMMANDS"];
const OPTIONS_CONTEXT_BUDGET: usize = 12_000;

/// Default character budget for the hidden man/`--help` context excerpt.
pub const DEFAULT_CONTEXT_BUDGET: usize = 4_000;

/// A fetched manual or `--help` body.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManualResult {
    /// `"man"` or `"help"`.
    pub source: String,
    pub body: String,
}

/// Fetch a manual for `program`, trying `man` then `--help`/`-h`.
///
/// Returns `None` when nothing usable is available.
pub fn fetch_manual(program: &str) -> Option<ManualResult> {
    let program = program.trim();
    if program.is_empty() {
        return None;
    }

    if let Some(body) = fetch_man(program) {
        return Some(ManualResult { source: "man".to_owned(), body });
    }
    if let Some(body) = fetch_help(program) {
        return Some(ManualResult { source: "help".to_owned(), body });
    }
    None
}

/// Fetch and extract a budgeted context excerpt for `program`.
pub fn manual_context(program: &str, budget: usize) -> Option<String> {
    let ctx = reference_context(program, budget);
    if ctx.has_body() {
        Some(ctx.body)
    } else {
        None
    }
}

/// Resolve the best available reference for `program` (man → help → package → docs).
pub fn reference_context(program: &str, budget: usize) -> ReferenceContext {
    let program = program.trim();
    if program.is_empty() {
        return ReferenceContext::empty(String::new());
    }

    if let Some(manual) = fetch_manual(program) {
        let extracted = extract_manual_context(&manual.body, budget);
        if !extracted.trim().is_empty() {
            let source = if manual.source == "man" {
                ReferenceSource::Man
            } else {
                ReferenceSource::Help
            };
            return ReferenceContext {
                program: program.to_owned(),
                source,
                body: extracted,
            };
        }
    }

    if let Some(pkg) = docs_fallback::package_info(program) {
        let body = truncate_chars(&pkg, budget);
        if !body.trim().is_empty() {
            return ReferenceContext {
                program: program.to_owned(),
                source: ReferenceSource::Package,
                body,
            };
        }
    }

    if let Some(docs) = docs_fallback::official_docs(program) {
        let body = truncate_chars(&docs, budget);
        if !body.trim().is_empty() {
            return ReferenceContext {
                program: program.to_owned(),
                source: ReferenceSource::Docs,
                body,
            };
        }
    }

    ReferenceContext::empty(program.to_owned())
}

/// Truncate to at most `max_chars` unicode characters.
pub fn truncate_chars(text: &str, max_chars: usize) -> String {
    if text.chars().count() <= max_chars {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max_chars.saturating_sub(16)).collect();
    format!("{}\n... [truncated]", kept.trim_end())
}

fn fetch_man(program: &str) -> Option<String> {
    let mut cmd = Command::new("man");
    cmd.args(["-P", "cat", program]);
    cmd.env("MANPAGER", "cat");
    cmd.env("PAGER", "cat");
    cmd.env("MAN_POSIXLY_CORRECT", "1");
    let (stdout, _stderr) = run_with_timeout(cmd)?;
    let text = clean(&String::from_utf8_lossy(&stdout));
    let text = text.trim();
    if text.chars().count() > 100 {
        Some(text.to_owned())
    } else {
        None
    }
}

fn fetch_help(program: &str) -> Option<String> {
    for flag in ["--help", "-h"] {
        let mut cmd = Command::new(program);
        cmd.arg(flag);
        let Some((stdout, stderr)) = run_with_timeout(cmd) else {
            continue;
        };
        let mut combined = String::from_utf8_lossy(&stdout).into_owned();
        combined.push_str(&String::from_utf8_lossy(&stderr));
        let text = clean(&combined);
        let text = text.trim();
        if text.chars().count() > 50 {
            return Some(text.to_owned());
        }
    }
    None
}

/// Run a command with a hard timeout, capturing stdout/stderr.
///
/// Returns `None` if the command fails to spawn or exceeds the timeout.
pub(crate) fn run_with_timeout_public(cmd: &mut Command) -> Option<(Vec<u8>, Vec<u8>)> {
    run_with_timeout_inner(cmd)
}

fn run_with_timeout(mut cmd: Command) -> Option<(Vec<u8>, Vec<u8>)> {
    run_with_timeout_inner(&mut cmd)
}

fn run_with_timeout_inner(cmd: &mut Command) -> Option<(Vec<u8>, Vec<u8>)> {
    cmd.stdin(Stdio::null()).stdout(Stdio::piped()).stderr(Stdio::piped());
    let mut child = cmd.spawn().ok()?;

    // Drain pipes on dedicated threads to avoid deadlocking on full buffers.
    let mut stdout_pipe = child.stdout.take();
    let mut stderr_pipe = child.stderr.take();
    let out_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(pipe) = stdout_pipe.as_mut() {
            let _ = pipe.read_to_end(&mut buf);
        }
        buf
    });
    let err_handle = std::thread::spawn(move || {
        let mut buf = Vec::new();
        if let Some(pipe) = stderr_pipe.as_mut() {
            let _ = pipe.read_to_end(&mut buf);
        }
        buf
    });

    let deadline = Instant::now() + SUBPROCESS_TIMEOUT;
    loop {
        match child.try_wait() {
            Ok(Some(_status)) => {
                let stdout = out_handle.join().unwrap_or_default();
                let stderr = err_handle.join().unwrap_or_default();
                return Some((stdout, stderr));
            },
            Ok(None) => {
                if Instant::now() >= deadline {
                    let _ = child.kill();
                    let _ = child.wait();
                    let _ = out_handle.join();
                    let _ = err_handle.join();
                    return None;
                }
                std::thread::sleep(Duration::from_millis(25));
            },
            Err(_) => return None,
        }
    }
}

/// Strip ANSI SGR/erase sequences and overstrike backspaces (man formatting).
fn clean(text: &str) -> String {
    strip_backspaces(&strip_ansi(text))
}

fn strip_ansi(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '\x1b' && i + 1 < chars.len() && chars[i + 1] == '[' {
            let mut j = i + 2;
            while j < chars.len() && (chars[j].is_ascii_digit() || chars[j] == ';') {
                j += 1;
            }
            if j < chars.len() && (chars[j] == 'm' || chars[j] == 'K') {
                i = j + 1;
                continue;
            }
        }
        out.push(chars[i]);
        i += 1;
    }
    out
}

fn strip_backspaces(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for character in text.chars() {
        if character == '\x08' {
            out.pop();
        } else {
            out.push(character);
        }
    }
    out
}

// ── Section parsing / budgeting ───────────────────────────────────────────────

/// Match a man-page section header, e.g. `NAME`, `SEE ALSO`, `OPTIONS:`.
fn parse_section_header(line: &str) -> Option<String> {
    let trimmed = line.trim();
    if trimmed.is_empty() {
        return None;
    }
    let name_part = trimmed.strip_suffix(':').unwrap_or(trimmed);
    let chars: Vec<char> = name_part.chars().collect();
    if chars.len() < 2 || chars.len() > 42 {
        return None;
    }
    if !chars[0].is_ascii_uppercase() {
        return None;
    }
    if !(chars[1].is_ascii_uppercase() || chars[1].is_ascii_digit()) {
        return None;
    }
    for &c in &chars[2..] {
        if !(c.is_ascii_uppercase() || c.is_ascii_digit() || c == ' ' || c == '-') {
            return None;
        }
    }
    let name = name_part.trim();
    if name.is_empty() {
        None
    } else {
        Some(name.to_owned())
    }
}

fn split_man_sections(body: &str) -> Vec<(String, Vec<String>)> {
    let mut sections: Vec<(String, Vec<String>)> = Vec::new();
    let mut current_name = "Reference".to_owned();
    let mut current_lines: Vec<String> = Vec::new();

    for raw in body.lines() {
        let line = raw.trim_end().to_owned();
        if let Some(name) = parse_section_header(line.trim()) {
            if !current_lines.is_empty() {
                sections.push((current_name.clone(), std::mem::take(&mut current_lines)));
            }
            current_name = name;
        } else {
            current_lines.push(line);
        }
    }
    if !current_lines.is_empty() {
        sections.push((current_name, current_lines));
    }
    sections
}

fn section_text(lines: &[String]) -> String {
    lines.join("\n").trim().to_owned()
}

fn truncate_section(name: &str, text: &str, max_chars: usize) -> String {
    let suffix = format!("\n\n[... {name} truncated ...]");
    let suffix_len = suffix.chars().count();
    if text.chars().count() + suffix_len <= max_chars {
        return text.to_owned();
    }
    let keep = max_chars.saturating_sub(suffix_len);
    let kept: String = text.chars().take(keep).collect();
    format!("{}{suffix}", kept.trim_end())
}

fn is_options_section(name: &str) -> bool {
    name == "OPTIONS" || name.contains("OPTIONS")
}

fn section_text_of(raw_sections: &[(String, Vec<String>)], name: &str) -> Option<String> {
    raw_sections.iter().find(|(n, _)| n == name).map(|(_, lines)| section_text(lines))
}

fn context_section_order(raw_sections: &[(String, Vec<String>)]) -> Vec<String> {
    let mut ordered: Vec<String> = Vec::new();
    let mut seen = std::collections::HashSet::new();

    let mut add = |name: &str| {
        if seen.contains(name) {
            return;
        }
        if let Some(text) = section_text_of(raw_sections, name) {
            if !text.is_empty() {
                ordered.push(name.to_owned());
                seen.insert(name.to_owned());
            }
        }
    };

    if let Some(ref_text) = section_text_of(raw_sections, "Reference") {
        if !ref_text.is_empty() && ref_text.chars().count() <= 800 {
            add("Reference");
        }
    }
    for name in CORE_SECTIONS {
        add(name);
    }
    for name in ARGUMENT_SECTIONS {
        add(name);
    }
    for name in EXAMPLE_SECTIONS {
        add(name);
    }
    for (name, _) in raw_sections {
        if name.ends_with(" COMMANDS") {
            add(name);
        }
    }
    for (name, _) in raw_sections {
        if is_options_section(name) {
            add(name);
        }
    }
    ordered
}

/// Extract high-signal manual sections for hidden chat context.
pub fn extract_manual_context(body: &str, budget: usize) -> String {
    let body = body.trim();
    if body.is_empty() {
        return String::new();
    }

    let raw_sections = split_man_sections(body);
    if raw_sections.len() == 1 && raw_sections[0].0 == "Reference" && body.chars().count() <= budget
    {
        return body.to_owned();
    }
    let ordered = context_section_order(&raw_sections);
    if ordered.is_empty() {
        if body.chars().count() <= budget {
            return body.to_owned();
        }
        return truncate_section("Reference", body, budget);
    }

    let lookup = |name: &str| -> String {
        raw_sections
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, lines)| section_text(lines))
            .unwrap_or_default()
    };

    let mut parts: Vec<String> = Vec::new();
    let mut used = 0usize;
    let options_names: Vec<&String> = ordered.iter().filter(|n| is_options_section(n)).collect();
    let non_options: Vec<&String> = ordered.iter().filter(|n| !is_options_section(n)).collect();

    for name in non_options {
        let text = lookup(name);
        let remaining = budget.saturating_sub(used);
        if remaining <= name.len() + 20 {
            break;
        }
        let text = truncate_section(name, &text, remaining - name.len() - 1);
        let block = format!("{name}\n{text}");
        used += block.chars().count() + 2;
        parts.push(block);
        if used >= budget {
            break;
        }
    }

    let mut options_budget = OPTIONS_CONTEXT_BUDGET.min(budget.saturating_sub(used));
    if options_budget > 300 && !options_names.is_empty() {
        let per_section = (options_budget / options_names.len()).max(400);
        for name in options_names {
            let text = lookup(name);
            if text.is_empty() {
                continue;
            }
            let remaining = per_section.min(options_budget);
            if remaining < 200 {
                break;
            }
            let text = truncate_section(name, &text, remaining.saturating_sub(name.len() + 1));
            let block = format!("{name}\n{text}");
            options_budget = options_budget.saturating_sub(block.chars().count() + 2);
            parts.push(block);
            if options_budget <= 200 {
                break;
            }
        }
    }

    let extracted = parts.join("\n\n").trim().to_owned();
    if extracted.is_empty() {
        return truncate_section("Reference", body, budget);
    }
    if extracted.chars().count() > budget {
        return truncate_section("Summary", &extracted, budget);
    }
    extracted
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clean_strips_ansi_and_backspaces() {
        let raw = "\x1b[1mNAME\x1b[0m\ng\x08git \x08\x08helper";
        let cleaned = clean(raw);
        assert!(!cleaned.contains('\x1b'));
        assert!(!cleaned.contains('\x08'));
        assert!(cleaned.contains("NAME"));
    }

    #[test]
    fn parse_section_header_matches_uppercase_titles() {
        assert_eq!(parse_section_header("NAME"), Some("NAME".to_owned()));
        assert_eq!(parse_section_header("SEE ALSO"), Some("SEE ALSO".to_owned()));
        assert_eq!(parse_section_header("OPTIONS:"), Some("OPTIONS".to_owned()));
        assert_eq!(parse_section_header("Description"), None);
        assert_eq!(parse_section_header("git status is nice"), None);
        assert_eq!(parse_section_header(""), None);
    }

    #[test]
    fn split_man_sections_groups_by_header() {
        let body = "NAME\n  git - stupid content tracker\n\nSYNOPSIS\n  git <cmd>\n";
        let sections = split_man_sections(body);
        let names: Vec<&str> = sections.iter().map(|(n, _)| n.as_str()).collect();
        assert!(names.contains(&"NAME"));
        assert!(names.contains(&"SYNOPSIS"));
    }

    #[test]
    fn extract_prioritizes_name_and_synopsis() {
        let body = "NAME\n  git - tracker\n\nSYNOPSIS\n  git <cmd>\n\nBUGS\n  none\n";
        let extracted = extract_manual_context(body, 1000);
        assert!(extracted.contains("NAME"));
        assert!(extracted.contains("SYNOPSIS"));
    }

    #[test]
    fn extract_truncates_when_over_budget() {
        let big = "x".repeat(5000);
        let body = format!("DESCRIPTION\n{big}\n");
        let extracted = extract_manual_context(&body, 500);
        assert!(extracted.chars().count() <= 500);
        assert!(extracted.contains("truncated"));
    }

    #[test]
    fn extract_returns_whole_body_when_no_sections_and_small() {
        let body = "just some free-form help text without headers";
        let extracted = extract_manual_context(body, 1000);
        assert_eq!(extracted, body);
    }

    #[test]
    fn extract_empty_body_is_empty() {
        assert_eq!(extract_manual_context("   ", 1000), "");
    }

    #[test]
    fn manual_context_missing_program_is_none() {
        assert!(manual_context("", DEFAULT_CONTEXT_BUDGET).is_none());
        assert!(fetch_manual("definitely-not-a-real-binary-xyz-123").is_none());
    }
}
