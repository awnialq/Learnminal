//! Package-metadata and allowlisted official-docs fallbacks when man/--help is missing.

use std::process::Command;
use std::time::Duration;

use reqwest::blocking::Client;

use crate::learnminal::manpage::{run_with_timeout_public, truncate_chars};

const DOCS_TIMEOUT: Duration = Duration::from_secs(5);
const PACKAGE_MIN_CHARS: usize = 40;

/// Run a package-manager info command and return trimmed stdout when usable.
pub fn package_info(program: &str) -> Option<String> {
    let program = program.trim();
    if program.is_empty() {
        return None;
    }

    #[cfg(target_os = "macos")]
    {
        if which("brew") {
            if let Some(text) = run_info(&mut Command::new("brew").args(["info", program])) {
                return Some(text);
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        if which("apt-cache") {
            if let Some(text) =
                run_info(&mut Command::new("apt-cache").args(["show", program]))
            {
                return Some(text);
            }
        }
        if which("pacman") {
            if let Some(text) = run_info(&mut Command::new("pacman").args(["-Si", program])) {
                return Some(text);
            }
        }
    }

    // Cross-platform extras.
    if which("npm") {
        if let Some(text) =
            run_info(&mut Command::new("npm").args(["view", program, "description", "version"]))
        {
            return Some(text);
        }
    }

    None
}

fn run_info(cmd: &mut Command) -> Option<String> {
    let (stdout, stderr) = run_with_timeout_public(cmd)?;
    let mut combined = String::from_utf8_lossy(&stdout).into_owned();
    if combined.trim().is_empty() {
        combined.push_str(&String::from_utf8_lossy(&stderr));
    }
    let text = combined.trim();
    if text.chars().count() >= PACKAGE_MIN_CHARS && !looks_like_not_found(text) {
        Some(truncate_chars(text, 4_000))
    } else {
        None
    }
}

fn looks_like_not_found(text: &str) -> bool {
    let lower = text.to_ascii_lowercase();
    lower.contains("no available formula")
        || lower.contains("unable to locate package")
        || (lower.contains("error: package") && lower.contains("was not found"))
        || lower.contains("404 not found")
        || lower.contains("npm error code e404")
}

fn which(name: &str) -> bool {
    #[cfg(unix)]
    {
        Command::new("which")
            .arg(name)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
    #[cfg(not(unix))]
    {
        let _ = name;
        false
    }
}

/// Map a small set of programs to allowlisted official doc URLs.
fn docs_url(program: &str) -> Option<&'static str> {
    match program {
        "git" => Some("https://git-scm.com/docs/git"),
        "ssh" => Some("https://man7.org/linux/man-pages/man1/ssh.1.html"),
        "curl" => Some("https://man7.org/linux/man-pages/man1/curl.1.html"),
        "tar" => Some("https://man7.org/linux/man-pages/man1/tar.1.html"),
        "grep" => Some("https://man7.org/linux/man-pages/man1/grep.1.html"),
        "find" => Some("https://man7.org/linux/man-pages/man1/find.1.html"),
        "npm" => Some("https://docs.npmjs.com/cli/v10/commands/npm"),
        "cargo" => Some("https://doc.rust-lang.org/cargo/commands/cargo.html"),
        "rustc" => Some("https://doc.rust-lang.org/rustc/index.html"),
        _ => None,
    }
}

/// Fetch allowlisted official docs as plain text.
pub fn official_docs(program: &str) -> Option<String> {
    let program = program.trim();
    let url = docs_url(program)?;
    let client = Client::builder().timeout(DOCS_TIMEOUT).build().ok()?;
    let response = client.get(url).send().ok()?;
    if !response.status().is_success() {
        return None;
    }
    let html = response.text().ok()?;
    let text = strip_html(&html);
    let text = text.trim();
    if text.chars().count() < 80 {
        None
    } else {
        Some(truncate_chars(text, 4_000))
    }
}

fn strip_html(html: &str) -> String {
    let mut out = String::with_capacity(html.len() / 2);
    let mut in_tag = false;
    let mut in_script = false;
    let chars: Vec<char> = html.chars().collect();
    let mut i = 0;
    while i < chars.len() {
        if chars[i] == '<' {
            // Detect script/style blocks roughly.
            let rest: String = chars[i..].iter().take(10).collect::<String>().to_ascii_lowercase();
            if rest.starts_with("<script") || rest.starts_with("<style") {
                in_script = true;
            } else if rest.starts_with("</script") || rest.starts_with("</style") {
                in_script = false;
            }
            in_tag = true;
            i += 1;
            continue;
        }
        if chars[i] == '>' {
            in_tag = false;
            i += 1;
            continue;
        }
        if !in_tag && !in_script {
            let c = chars[i];
            if c == '\n' || c == '\r' || !c.is_whitespace() {
                out.push(c);
            } else if !out.ends_with(' ') && !out.ends_with('\n') {
                out.push(' ');
            }
        }
        i += 1;
    }
    // Collapse blank lines.
    let mut collapsed = String::new();
    for line in out.lines() {
        let t = line.trim();
        if t.is_empty() {
            if !collapsed.ends_with("\n\n") {
                collapsed.push('\n');
            }
        } else {
            collapsed.push_str(t);
            collapsed.push('\n');
        }
    }
    collapsed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn docs_url_allowlist_only() {
        assert!(docs_url("git").is_some());
        assert!(docs_url("cargo").is_some());
        assert!(docs_url("totally-unknown-xyz").is_none());
    }

    #[test]
    fn strip_html_keeps_text() {
        let html = "<html><head><style>a{}</style></head><body><h1>Git</h1><p>Docs</p></body></html>";
        let text = strip_html(html);
        assert!(text.contains("Git"));
        assert!(text.contains("Docs"));
        assert!(!text.contains("<h1>"));
    }

    #[test]
    fn looks_like_not_found_detects_common_errors() {
        assert!(looks_like_not_found("Error: No available formula with the name \"foo\"."));
        assert!(!looks_like_not_found("git: Distributed version control"));
    }
}
