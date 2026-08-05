use serde::{Deserialize, Serialize};

/// Snapshot of terminal state gathered for the chat prompt.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalContext {
    pub visible_text: String,
    pub selected_text: Option<String>,
    pub last_command: String,
    /// Output produced by the last command, extracted from the visible grid.
    /// Empty when the output has scrolled off screen or no command was detected.
    #[serde(default)]
    pub last_command_output: String,
    /// Recent commands from this terminal session, oldest first. The final entry is
    /// the same command as [`Self::last_command`]. Empty when nothing was recovered.
    #[serde(default)]
    pub command_history: Vec<CommandEntry>,
    /// Where [`Self::command_history`] came from, which determines how much of it
    /// can be trusted (grid-only history has no exit codes).
    #[serde(default)]
    pub history_source: HistorySource,
    pub cwd: String,
    pub exit_code: Option<i32>,
    pub rows: u16,
    pub cols: u16,
}

impl Default for TerminalContext {
    fn default() -> Self {
        Self {
            visible_text: String::new(),
            selected_text: None,
            last_command: String::new(),
            last_command_output: String::new(),
            command_history: Vec::new(),
            history_source: HistorySource::None,
            cwd: String::new(),
            exit_code: None,
            rows: 0,
            cols: 0,
        }
    }
}

/// One command from the session history, as shown to the model.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandEntry {
    pub command: String,
    /// `None` when the exit code could not be recovered (grid-only history).
    #[serde(default)]
    pub exit_code: Option<i32>,
    /// Output recovered from the terminal grid. Empty when the command's output was
    /// cleared, scrolled away, or never matched to a grid block.
    #[serde(default)]
    pub output: String,
    /// Working directory the command ran in. Empty on the grid-only path.
    #[serde(default)]
    pub cwd: String,
}

/// Provenance of [`TerminalContext::command_history`].
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum HistorySource {
    /// No history could be recovered.
    #[default]
    None,
    /// Reconstructed from the terminal grid alone; exit codes are unavailable.
    GridOnly,
    /// Joined with records written by the shell integration hook; exit codes exact.
    ShellHook,
}

/// System environment snapshot for the overlay `/info` command.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SystemInfo {
    #[serde(default)]
    pub os: String,
    #[serde(default)]
    pub arch: String,
    #[serde(default)]
    pub shell: String,
    #[serde(default)]
    pub package_managers: Vec<String>,
    /// `{ "pacman": ["vim", "git", ...] }` — always empty now (inventory dropped).
    #[serde(default)]
    pub installed_packages: std::collections::HashMap<String, Vec<String>>,
    #[serde(default)]
    pub installed_packages_total: Option<u64>,
    #[serde(default)]
    pub installed_tools: Vec<String>,
    #[serde(default)]
    pub collected_at: Option<i64>,
    #[serde(default)]
    pub collected_at_display: Option<String>,
}

impl SystemInfo {
    /// Whether enough data was collected for `/info`.
    pub fn is_complete(&self) -> bool {
        !self.os.is_empty() && self.collected_at.is_some()
    }
}

/// Provenance of the reference text injected into the chat prompt.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReferenceSource {
    Man,
    Help,
    Package,
    Docs,
    None,
}

impl ReferenceSource {
    pub fn as_str(self) -> &'static str {
        match self {
            ReferenceSource::Man => "man",
            ReferenceSource::Help => "help",
            ReferenceSource::Package => "package",
            ReferenceSource::Docs => "docs",
            ReferenceSource::None => "none",
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            ReferenceSource::Man => "man",
            ReferenceSource::Help => "--help",
            ReferenceSource::Package => "package info",
            ReferenceSource::Docs => "official docs",
            ReferenceSource::None => "none",
        }
    }
}

/// Budgeted reference text for a program (man/help/package/docs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReferenceContext {
    pub program: String,
    pub source: ReferenceSource,
    pub body: String,
}

impl ReferenceContext {
    pub fn empty(program: impl Into<String>) -> Self {
        Self { program: program.into(), source: ReferenceSource::None, body: String::new() }
    }

    pub fn has_body(&self) -> bool {
        !self.body.trim().is_empty()
    }
}

/// Truncate `text` to at most `max_chars` *characters*, appending `marker` if anything
/// was cut. Pass an empty marker for a silent cap.
///
/// Counts characters, not bytes: slicing at a byte offset panics when it lands inside a
/// multi-byte character, which in the extraction path would discard the whole context.
/// Scans at most `max_chars + 1` characters, so the cost does not grow with how far the
/// text overshoots its budget.
pub fn truncate_with_marker(text: &str, max_chars: usize, marker: &str) -> String {
    match text.char_indices().nth(max_chars) {
        None => text.to_owned(),
        Some((cut, _)) => format!("{}{marker}", text[..cut].trim_end()),
    }
}

/// First token of a shell command line (program name).
pub fn extract_program_name(last_command: &str) -> String {
    let trimmed = last_command.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    let token = trimmed.split_whitespace().next().unwrap_or("");
    // Strip path: `/usr/bin/git` → `git`
    std::path::Path::new(token)
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| token.to_owned())
}

/// Resolve the program from the last command, else a shell-looking token in the user message.
pub fn resolve_program(last_command: &str, user_message: &str) -> String {
    let from_cmd = extract_program_name(last_command);
    if !from_cmd.is_empty() {
        return from_cmd;
    }
    extract_program_from_message(user_message)
}

fn extract_program_from_message(message: &str) -> String {
    // Prefer backtick/`$ ` command snippets, then bare tokens that look like programs.
    for segment in message.split('`') {
        let candidate = extract_program_name(segment.trim_start_matches('$'));
        if looks_like_program(&candidate) {
            return candidate;
        }
    }
    for token in message.split_whitespace() {
        let cleaned = token.trim_matches(|c: char| {
            matches!(c, ',' | '.' | ';' | ':' | '?' | '!' | '"' | '\'' | '(' | ')')
        });
        let candidate = extract_program_name(cleaned);
        if looks_like_program(&candidate) {
            return candidate;
        }
    }
    String::new()
}

fn looks_like_program(name: &str) -> bool {
    if name.is_empty() || name.len() > 40 {
        return false;
    }
    // Skip common English words that are not programs.
    const STOP: &[&str] = &[
        "how", "do", "i", "the", "a", "an", "to", "for", "with", "what", "why", "is", "my", "this",
        "that", "and", "or", "of", "in", "on", "it", "me", "can", "you", "please", "help", "explain",
        "about", "command", "error", "failed", "using", "fix", "something", "specific", "again",
        "does", "did", "not", "work", "working", "want", "need", "should", "could", "would",
        "when", "where", "which", "who", "from", "into", "just", "like", "make", "get", "set",
        "run", "use", "used", "try", "tell", "show", "give",
    ];
    let lower = name.to_ascii_lowercase();
    if STOP.contains(&lower.as_str()) {
        return false;
    }
    name.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn system_info_complete_requires_os_and_collected_at() {
        let mut info = SystemInfo {
            os: "Linux".into(),
            arch: String::new(),
            shell: String::new(),
            package_managers: Vec::new(),
            installed_packages: std::collections::HashMap::new(),
            installed_packages_total: None,
            installed_tools: Vec::new(),
            collected_at: Some(1),
            collected_at_display: None,
        };
        assert!(info.is_complete());
        info.collected_at = None;
        assert!(!info.is_complete());
        info.collected_at = Some(1);
        info.os = String::new();
        assert!(!info.is_complete());
    }

    #[test]
    fn truncate_with_marker_cuts_on_char_boundaries() {
        assert_eq!(truncate_with_marker("abc", 3, "!"), "abc", "exact fit is left alone");
        assert_eq!(truncate_with_marker("abcd", 3, "!"), "abc!");
        assert_eq!(truncate_with_marker("ab  cd", 4, "!"), "ab!", "kept text is right-trimmed");
        assert_eq!(truncate_with_marker("ééé", 2, ""), "éé", "multi-byte chars must not panic");
    }

    #[test]
    fn extract_program_name_takes_first_token() {
        assert_eq!(extract_program_name("git rebase -i HEAD~3"), "git");
        assert_eq!(extract_program_name("   ls -la"), "ls");
        assert_eq!(extract_program_name(""), "");
        assert_eq!(extract_program_name("/usr/bin/git status"), "git");
    }

    #[test]
    fn resolve_program_falls_back_to_message() {
        assert_eq!(resolve_program("git status", "why?"), "git");
        assert_eq!(resolve_program("", "how do I use `kubectl get pods`?"), "kubectl");
        assert_eq!(resolve_program("", "explain docker compose"), "docker");
        assert!(resolve_program("", "how do I fix this error?").is_empty());
    }

    #[test]
    fn terminal_context_deserializes_legacy_json_without_history() {
        // Contexts serialized before command history existed must still load.
        let legacy = r#"{
            "visible_text": "hello",
            "selected_text": null,
            "last_command": "ls",
            "cwd": "/tmp",
            "exit_code": 0,
            "rows": 24,
            "cols": 80
        }"#;
        let ctx: TerminalContext = serde_json::from_str(legacy).unwrap();
        assert_eq!(ctx.last_command, "ls");
        assert!(ctx.command_history.is_empty());
        assert_eq!(ctx.history_source, HistorySource::None);
        assert!(ctx.last_command_output.is_empty());
    }

    fn arb_command_entry() -> impl Strategy<Value = CommandEntry> {
        (".*", prop::option::of(any::<i32>()), ".*", ".*").prop_map(
            |(command, exit_code, output, cwd)| CommandEntry { command, exit_code, output, cwd },
        )
    }

    fn arb_history_source() -> impl Strategy<Value = HistorySource> {
        prop_oneof![
            Just(HistorySource::None),
            Just(HistorySource::GridOnly),
            Just(HistorySource::ShellHook),
        ]
    }

    fn arb_terminal_context() -> impl Strategy<Value = TerminalContext> {
        (
            (
                ".*",
                prop::option::of(".*"),
                ".*",
                ".*",
                ".*",
                prop::option::of(any::<i32>()),
                any::<u16>(),
                any::<u16>(),
            ),
            (prop::collection::vec(arb_command_entry(), 0..4), arb_history_source()),
        )
            .prop_map(
                |(
                    (
                        visible_text,
                        selected_text,
                        last_command,
                        last_command_output,
                        cwd,
                        exit_code,
                        rows,
                        cols,
                    ),
                    (command_history, history_source),
                )| {
                    TerminalContext {
                        visible_text,
                        selected_text,
                        last_command,
                        last_command_output,
                        command_history,
                        history_source,
                        cwd,
                        exit_code,
                        rows,
                        cols,
                    }
                },
            )
    }

    proptest! {
        #[test]
        fn terminal_context_round_trips(ctx in arb_terminal_context()) {
            let json = serde_json::to_string(&ctx).unwrap();
            let parsed: TerminalContext = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(ctx, parsed);
        }
    }
}
