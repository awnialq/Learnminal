//! Read-only command execution tool (`run_command`).
//!
//! Lets the chat model inspect the user's machine without being able to change
//! it. Safety is structural rather than advisory:
//!
//! 1. Shell metacharacters are refused outright.
//! 2. The command is tokenized here — no shell is ever spawned, so `;`, `|`,
//!    and `$(…)` cannot chain or expand into a second command.
//! 3. Only binaries on [`ALLOWLIST`] may run, with per-binary subcommand and
//!    denied-flag rules.
//! 4. Path arguments are canonicalized and must resolve inside the working
//!    directory subtree or a small read-safe global set, and must never touch
//!    a known-secret path.
//! 5. Execution is capped by the shared subprocess timeout, and output is
//!    ANSI-stripped, secret-redacted, and truncated.
//!
//! Every failure is returned as a plain string for the model to read, mirroring
//! [`crate::learnminal::web_search::search_tool_result`]; a refusal is a
//! conversational turn, not a program error.

use std::fmt;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

use crate::learnminal::manpage;

/// Character budget for command output handed back to the model.
const OUTPUT_BUDGET_CHARS: usize = 4_000;

/// Cap on names offered when a path is not found.
const MAX_SUGGESTIONS: usize = 12;

/// Prefix of the progress status emitted for each executed command.
///
/// The UI strips this to recover the command, so both sides share the constant.
pub const RUNNING_STATUS_PREFIX: &str = "Running: ";

/// Characters that signal an attempt to chain, redirect, or expand.
///
/// None of these can actually do anything here — there is no shell to
/// interpret them — but their presence means the model wants behavior the tool
/// does not provide, and a clear refusal beats silently passing `|` to `ls` as
/// a filename.
///
/// Deliberately absent: `(`, `)` and `\`. Those carry no chaining meaning to
/// `execve`, and `find . \( -iname a -o -iname b \)` is the standard way to
/// group predicates. Refusing them forced the model into one `find` per
/// pattern, burning a tool round each time.
const SHELL_METACHARS: &[char] = &['|', '&', ';', '<', '>', '$', '`', '\n', '\r', '{', '}'];

/// Absolute prefixes readable regardless of the working directory.
const READ_SAFE_GLOBALS: &[&str] = &[
    "/proc",
    "/sys/devices",
    "/etc/os-release",
    "/etc/hostname",
    "/etc/shells",
    "/usr/share",
    "/usr/lib/os-release",
];

/// Path components that are never readable, even inside the working directory.
///
/// The subtree rule alone is not enough: when the user's cwd *is* their home
/// directory, `~/.ssh` sits inside the allowed subtree.
const SECRET_PATH_COMPONENTS: &[&str] = &[
    ".ssh",
    ".aws",
    ".gnupg",
    ".gpg",
    ".kube",
    ".docker",
    ".netrc",
    ".env",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    "credentials",
    "secrets",
];

/// Extensions that usually indicate private key material.
const SECRET_EXTENSIONS: &[&str] = &["pem", "key", "p12", "pfx", "keystore"];

/// Substrings marking an environment variable whose value must be hidden.
const SECRET_ENV_MARKERS: &[&str] = &[
    "TOKEN",
    "SECRET",
    "PASSWORD",
    "PASSWD",
    "CREDENTIAL",
    "APIKEY",
    "_KEY",
    "AUTH",
    "SESSION",
    "PRIVATE",
];

/// A binary the model is allowed to run, plus its restrictions.
struct SafeCommand {
    /// Executable name, matched exactly (no paths).
    bin: &'static str,
    /// When set, the first non-flag argument must be one of these.
    subcommands: Option<&'static [&'static str]>,
    /// Arguments that flip this binary from read-only to destructive.
    denied_flags: &'static [&'static str],
}

/// Subcommands of `git` that only read repository state.
const GIT_SUBCOMMANDS: &[&str] = &[
    "status",
    "log",
    "diff",
    "show",
    "branch",
    "remote",
    "rev-parse",
    "describe",
    "tag",
    "blame",
    "shortlog",
    "ls-files",
    "ls-remote",
    "count-objects",
];

/// `find` predicates that write, delete, or execute.
const FIND_DENIED: &[&str] =
    &["-delete", "-exec", "-execdir", "-ok", "-okdir", "-fprintf", "-fls", "-fprint", "-fprint0"];

/// Version-only probes: running these binaries with anything else would
/// execute user code (`python3 script.py`) or mutate state (`npm install`).
const VERSION_ONLY: &[&str] = &["--version", "-V", "-v", "version"];

const ALLOWLIST: &[SafeCommand] = &[
    // Filesystem inspection.
    SafeCommand { bin: "ls", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "pwd", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "cat", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "head", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "tail", subcommands: None, denied_flags: &["-f", "--follow"] },
    SafeCommand { bin: "wc", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "file", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "stat", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "du", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "df", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "tree", subcommands: None, denied_flags: &["-o"] },
    SafeCommand { bin: "realpath", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "basename", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "dirname", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "readlink", subcommands: None, denied_flags: &[] },
    // Search.
    SafeCommand { bin: "grep", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "rg", subcommands: None, denied_flags: &["--pre", "--hostname-bin"] },
    SafeCommand { bin: "find", subcommands: None, denied_flags: FIND_DENIED },
    // System introspection.
    SafeCommand { bin: "uname", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "hostname", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "whoami", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "id", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "uptime", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "date", subcommands: None, denied_flags: &["-s", "--set"] },
    SafeCommand { bin: "env", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "printenv", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "which", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "ps", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "free", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "lscpu", subcommands: None, denied_flags: &[] },
    SafeCommand { bin: "sw_vers", subcommands: None, denied_flags: &[] },
    // Git, restricted to read-only subcommands. `git config` is excluded
    // because `git config --global k v` writes.
    SafeCommand {
        bin: "git",
        subcommands: Some(GIT_SUBCOMMANDS),
        denied_flags: &["-c", "--exec-path"],
    },
    // Toolchain version probes only.
    SafeCommand { bin: "node", subcommands: Some(VERSION_ONLY), denied_flags: &[] },
    SafeCommand { bin: "python3", subcommands: Some(VERSION_ONLY), denied_flags: &[] },
    SafeCommand { bin: "python", subcommands: Some(VERSION_ONLY), denied_flags: &[] },
    SafeCommand { bin: "rustc", subcommands: Some(VERSION_ONLY), denied_flags: &[] },
    SafeCommand { bin: "cargo", subcommands: Some(VERSION_ONLY), denied_flags: &[] },
    SafeCommand { bin: "go", subcommands: Some(VERSION_ONLY), denied_flags: &[] },
    SafeCommand { bin: "npm", subcommands: Some(VERSION_ONLY), denied_flags: &[] },
    SafeCommand {
        bin: "ollama",
        subcommands: Some(&["list", "--version", "-v"]),
        denied_flags: &[],
    },
];

/// Why a command was refused. Rendered into text the model can act on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolicyError {
    Empty,
    ShellMetachar(char),
    UnterminatedQuote,
    NotAllowed(String),
    DeniedSubcommand { bin: String, sub: String },
    DeniedFlag { bin: String, flag: String },
    PathOutsideScope(String),
    SecretPath(String),
    /// Path did not resolve; `similar` holds near-matches from its parent so
    /// the model can recover from a misspelling instead of concluding the
    /// thing does not exist.
    PathNotFound { token: String, similar: Vec<String> },
    SpawnFailed(String),
    TimedOut,
}

impl PolicyError {
    /// Refusal text for the model. Phrased instructionally so the next attempt
    /// is better rather than a repeat.
    pub fn as_tool_message(&self) -> String {
        match self {
            Self::Empty => "run_command refused: no command was provided.".to_owned(),
            Self::ShellMetachar(ch) => format!(
                "run_command refused: '{ch}' is not supported. Pipes, redirects, subshells, and \
                 chaining are unavailable — run one simple command per call."
            ),
            Self::UnterminatedQuote => {
                "run_command refused: unterminated quote in the command.".to_owned()
            },
            Self::NotAllowed(bin) => format!(
                "run_command refused: '{bin}' is not permitted. Only read-only inspection \
                 commands are allowed, such as ls, cat, head, wc, stat, find, grep, which, env, \
                 uname, ps, df, and git status/log/diff."
            ),
            Self::DeniedSubcommand { bin, sub } => format!(
                "run_command refused: '{bin} {sub}' is not permitted. Only read-only {bin} \
                 subcommands are allowed."
            ),
            Self::DeniedFlag { bin, flag } => format!(
                "run_command refused: the '{flag}' option of '{bin}' can modify the system."
            ),
            Self::PathOutsideScope(path) => format!(
                "run_command refused: '{path}' is outside the working directory. Only files under \
                 the current directory can be read."
            ),
            Self::SecretPath(path) => {
                format!("run_command refused: '{path}' may contain credentials and cannot be read.")
            },
            Self::PathNotFound { token, similar } => {
                let mut message = format!("run_command: '{token}' does not exist.");
                if !similar.is_empty() {
                    message.push_str(&format!(
                        " These names do exist there: {}. The intended item may be one of them \
                         under a different name or spelling.",
                        similar.join(", ")
                    ));
                }
                message
            },
            Self::SpawnFailed(bin) => {
                format!("run_command error: '{bin}' could not be started (is it installed?).")
            },
            Self::TimedOut => {
                "run_command error: the command took too long and was stopped.".to_owned()
            },
        }
    }
}

impl fmt::Display for PolicyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.as_tool_message())
    }
}

/// Whether the read-only command tool is available (`LEARNMINAL_READ_EXEC` not
/// `0`/`false`/`off`/`no`).
pub fn read_exec_enabled() -> bool {
    enabled_from(std::env::var("LEARNMINAL_READ_EXEC").ok().as_deref())
}

/// Kill-switch parsing, split out so it is testable without mutating the
/// process environment (tests share one environment across threads).
fn enabled_from(value: Option<&str>) -> bool {
    match value {
        Some(value) => {
            let v = value.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no")
        },
        None => true,
    }
}

/// Directory the tool should be confined to for a request.
///
/// Prefers the shell's reported working directory, falling back to the
/// process's own. `None` means no usable root, which disables the tool.
pub fn inspect_root(cwd: &str) -> Option<PathBuf> {
    let cwd = cwd.trim();
    if !cwd.is_empty() {
        if let Ok(path) = Path::new(cwd).canonicalize() {
            if path.is_dir() {
                return Some(path);
            }
        }
    }
    std::env::current_dir().ok().and_then(|path| path.canonicalize().ok())
}

/// Run one read-only command rooted at `root` and format the result.
///
/// Never returns `Err`: refusals and failures become short strings so a denied
/// request does not abort the chat.
pub fn run_command_tool_result(command: &str, root: &Path) -> String {
    match run_checked(command, root) {
        Ok(output) => output,
        Err(err) => err.as_tool_message(),
    }
}

fn run_checked(command: &str, root: &Path) -> Result<String, PolicyError> {
    let tokens = expand_tildes(parse_command(command)?);
    let spec = check_policy(&tokens)?;
    check_paths(&tokens, root)?;
    execute(&tokens, spec, root)
}

/// Expand a leading `~` in each argument to the user's home directory.
///
/// A shell would normally do this. Since we deliberately never spawn one, a
/// model writing the natural `~/Videos` would otherwise pass `~` through as a
/// literal filename and get a bare "No such file" — a confusing dead end that
/// looks like the directory is missing. Expansion happens before the scope and
/// secret-path checks, so it widens nothing.
fn expand_tildes(tokens: Vec<String>) -> Vec<String> {
    let Some(home) = home::home_dir() else {
        return tokens;
    };
    tokens
        .into_iter()
        .enumerate()
        .map(|(i, token)| {
            // Index 0 is the binary name; `~` is never valid there.
            if i == 0 {
                return token;
            }
            if token == "~" {
                return home.to_string_lossy().into_owned();
            }
            match token.strip_prefix("~/") {
                Some(rest) => home.join(rest).to_string_lossy().into_owned(),
                None => token,
            }
        })
        .collect()
}

/// Reject shell metacharacters, then split into argv without shell semantics.
fn parse_command(command: &str) -> Result<Vec<String>, PolicyError> {
    let trimmed = command.trim();
    if trimmed.is_empty() {
        return Err(PolicyError::Empty);
    }
    if let Some(ch) = trimmed.chars().find(|ch| SHELL_METACHARS.contains(ch)) {
        return Err(PolicyError::ShellMetachar(ch));
    }
    let tokens = tokenize(trimmed)?;
    if tokens.is_empty() {
        return Err(PolicyError::Empty);
    }
    Ok(tokens)
}

/// Quote-aware split into argv. Quotes group and backslash escapes the next
/// character; nothing else is interpreted.
///
/// Unescaping matters because a shell would normally strip the backslash from
/// `\(` before `find` ever sees it. Passing `\(` through literally makes find
/// fail with "unknown predicate", so the tool has to do that one job itself.
fn tokenize(input: &str) -> Result<Vec<String>, PolicyError> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut started = false;
    let mut quote: Option<char> = None;
    let mut escaped = false;

    for ch in input.chars() {
        if escaped {
            current.push(ch);
            started = true;
            escaped = false;
            continue;
        }
        match quote {
            // Inside quotes a backslash is literal, as in single quotes.
            Some(q) if ch == q => quote = None,
            Some(_) => current.push(ch),
            None if ch == '\\' => {
                escaped = true;
                started = true;
            },
            None if ch == '\'' || ch == '"' => {
                quote = Some(ch);
                started = true;
            },
            None if ch.is_whitespace() => {
                if started {
                    tokens.push(std::mem::take(&mut current));
                    started = false;
                }
            },
            None => {
                current.push(ch);
                started = true;
            },
        }
    }
    if escaped {
        return Err(PolicyError::UnterminatedQuote);
    }

    if quote.is_some() {
        return Err(PolicyError::UnterminatedQuote);
    }
    if started {
        tokens.push(current);
    }
    Ok(tokens)
}

/// Resolve the binary against the allowlist and enforce its restrictions.
fn check_policy(tokens: &[String]) -> Result<&'static SafeCommand, PolicyError> {
    let bin = tokens.first().ok_or(PolicyError::Empty)?;
    // Reject paths: only bare binary names resolved via PATH are permitted.
    if bin.contains('/') {
        return Err(PolicyError::NotAllowed(bin.clone()));
    }
    let spec = ALLOWLIST
        .iter()
        .find(|entry| entry.bin == bin.as_str())
        .ok_or_else(|| PolicyError::NotAllowed(bin.clone()))?;

    let args = &tokens[1..];

    for arg in args {
        // Match `--flag=value` as well as bare `--flag`.
        let flag = arg.split('=').next().unwrap_or(arg);
        if spec.denied_flags.contains(&flag) {
            return Err(PolicyError::DeniedFlag { bin: bin.clone(), flag: flag.to_owned() });
        }
    }

    if let Some(allowed) = spec.subcommands {
        let sub = args
            .iter()
            .find(|arg| !arg.starts_with('-') || VERSION_ONLY.contains(&arg.as_str()))
            .ok_or_else(|| PolicyError::DeniedSubcommand {
                bin: bin.clone(),
                sub: "(none)".to_owned(),
            })?;
        if !allowed.contains(&sub.as_str()) {
            return Err(PolicyError::DeniedSubcommand { bin: bin.clone(), sub: sub.clone() });
        }
    }

    Ok(spec)
}

/// Every path-like argument must resolve inside the allowed scope.
fn check_paths(tokens: &[String], root: &Path) -> Result<(), PolicyError> {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());

    for token in &tokens[1..] {
        if !is_path_candidate(token, &root) {
            continue;
        }
        let raw = Path::new(token);
        let joined = if raw.is_absolute() { raw.to_path_buf() } else { root.join(raw) };

        // canonicalize resolves `..` and symlinks; comparing before it would
        // let `../../etc` or a symlink escape the subtree.
        let resolved = joined.canonicalize().map_err(|_| PolicyError::PathNotFound {
            token: token.clone(),
            similar: suggestions_within_scope(&joined, &root),
        })?;

        if is_secret_path(&resolved) {
            return Err(PolicyError::SecretPath(token.clone()));
        }
        if !resolved.starts_with(&root) && !is_read_safe_global(&resolved) {
            return Err(PolicyError::PathOutsideScope(token.clone()));
        }
    }
    Ok(())
}

/// [`similar_entries`], but only when the containing directory is itself
/// readable under the current scope.
///
/// Suggestions are computed for paths that failed to resolve, which includes
/// paths pointing outside the sandbox — listing their neighbours would leak
/// filenames the tool is not allowed to read.
fn suggestions_within_scope(target: &Path, root: &Path) -> Vec<String> {
    let Some(parent) = target.parent() else {
        return Vec::new();
    };
    let Ok(parent) = parent.canonicalize() else {
        return Vec::new();
    };
    if is_secret_path(&parent) {
        return Vec::new();
    }
    if !parent.starts_with(root) && !is_read_safe_global(&parent) {
        return Vec::new();
    }
    similar_entries(target)
}

/// Names in `target`'s parent directory that plausibly match its final
/// component, so a near-miss reads as a spelling problem rather than absence.
///
/// A user asking about "Movies" on a machine that has "Videos" should not be
/// told the thing does not exist; the model needs the candidates to reason
/// about. Matching is deliberately loose — case differences, one name
/// containing the other, or a shared prefix.
fn similar_entries(target: &Path) -> Vec<String> {
    let (Some(parent), Some(name)) = (target.parent(), target.file_name()) else {
        return Vec::new();
    };
    let needle = name.to_string_lossy().to_ascii_lowercase();
    if needle.is_empty() {
        return Vec::new();
    }

    let siblings = visible_entries(parent);
    let lexical: Vec<String> = siblings
        .iter()
        .filter(|raw| {
            let candidate = raw.to_ascii_lowercase();
            let shares_prefix = {
                let n = needle.len().min(candidate.len()).min(4);
                n >= 3 && needle.is_char_boundary(n) && candidate.is_char_boundary(n)
                    && needle[..n] == candidate[..n]
            };
            candidate.contains(&needle) || needle.contains(&candidate) || shares_prefix
        })
        .take(5)
        .cloned()
        .collect();

    // A lexical near-miss is a spelling problem. When there is none, the wanted
    // name may simply differ from the real one — "Movies" on a machine that
    // calls it "Videos" — so hand back what is actually there and let the model
    // make the connection.
    if lexical.is_empty() {
        siblings.into_iter().take(MAX_SUGGESTIONS).collect()
    } else {
        lexical
    }
}

/// Non-hidden names directly inside `dir`, sorted for stable output.
fn visible_entries(dir: &Path) -> Vec<String> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut names: Vec<String> = entries
        .flatten()
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .filter(|name| !name.starts_with('.'))
        .collect();
    names.sort();
    names
}

/// Whether an argument should be treated as a path to validate.
fn is_path_candidate(token: &str, root: &Path) -> bool {
    if token.starts_with('-') || token.is_empty() {
        return false;
    }
    token.contains('/') || token.starts_with('.') || root.join(token).exists()
}

fn is_read_safe_global(path: &Path) -> bool {
    READ_SAFE_GLOBALS.iter().any(|prefix| path.starts_with(prefix))
}

/// Whether any component or extension marks this path as credential material.
fn is_secret_path(path: &Path) -> bool {
    for component in path.components() {
        let Component::Normal(part) = component else {
            continue;
        };
        let name = part.to_string_lossy().to_ascii_lowercase();
        if SECRET_PATH_COMPONENTS.iter().any(|secret| name == *secret) {
            return true;
        }
    }
    path.extension()
        .map(|ext| {
            let ext = ext.to_string_lossy().to_ascii_lowercase();
            SECRET_EXTENSIONS.contains(&ext.as_str())
        })
        .unwrap_or(false)
}

/// Spawn the command with a neutral environment and format its output.
fn execute(tokens: &[String], spec: &SafeCommand, root: &Path) -> Result<String, PolicyError> {
    let mut cmd = Command::new(&tokens[0]);
    cmd.args(&tokens[1..]);
    cmd.current_dir(root);
    // Keep output plain and non-interactive: a pager would block until the
    // timeout, and color escapes would waste the model's context budget.
    cmd.env("PAGER", "cat");
    cmd.env("GIT_PAGER", "cat");
    cmd.env("GIT_TERMINAL_PROMPT", "0");
    cmd.env("NO_COLOR", "1");
    cmd.env("CLICOLOR", "0");
    cmd.env("TERM", "dumb");

    let (status, stdout, stderr) = manpage::run_with_timeout_status(&mut cmd).ok_or_else(|| {
        // The runner collapses spawn failure and timeout into None; a
        // missing binary is by far the likelier of the two.
        if which_exists(&tokens[0]) {
            PolicyError::TimedOut
        } else {
            PolicyError::SpawnFailed(tokens[0].clone())
        }
    })?;

    let mut body = String::from_utf8_lossy(&stdout).into_owned();
    let errors = String::from_utf8_lossy(&stderr);
    if !errors.trim().is_empty() {
        if !body.is_empty() && !body.ends_with('\n') {
            body.push('\n');
        }
        body.push_str(errors.trim_end());
        body.push('\n');
    }

    let mut body = manpage::clean(&body);
    if spec.bin == "env" || spec.bin == "printenv" {
        body = redact_env_output(&body);
    }

    let command_line = tokens.join(" ");
    let exit = status.code().unwrap_or(-1);
    let body = manpage::truncate_chars(body.trim_end(), OUTPUT_BUDGET_CHARS);

    // A failed command usually means a name was wrong, not that the thing is
    // absent. Offer the near-misses so the model can correct the spelling
    // rather than reporting "not found" to the user.
    let hint = if status.success() { String::new() } else { spelling_hint(tokens, root) };

    if body.trim().is_empty() {
        return Ok(format!("$ {command_line}\n[exit {exit}] (no output){hint}"));
    }
    Ok(format!("$ {command_line}\n[exit {exit}]\n{body}{hint}"))
}

/// "Did you mean" line for arguments of a failed command that do not exist.
fn spelling_hint(tokens: &[String], root: &Path) -> String {
    let root = root.canonicalize().unwrap_or_else(|_| root.to_path_buf());
    for token in &tokens[1..] {
        if token.starts_with('-') || token.is_empty() {
            continue;
        }
        let raw = Path::new(token);
        let joined = if raw.is_absolute() { raw.to_path_buf() } else { root.join(raw) };
        if joined.exists() {
            continue;
        }
        let similar = suggestions_within_scope(&joined, &root);
        if !similar.is_empty() {
            return format!(
                "\n[hint] '{token}' does not exist, but these do: {}. The intended item may be \
                 one of them under a different name or spelling — check before concluding it is \
                 missing.",
                similar.join(", ")
            );
        }
    }
    String::new()
}

fn which_exists(bin: &str) -> bool {
    let Ok(path) = std::env::var("PATH") else {
        return false;
    };
    std::env::split_paths(&path).any(|dir| dir.join(bin).is_file())
}

/// Replace values of environment variables whose names look sensitive.
///
/// `env` is genuinely useful for debugging and is also the fastest route for a
/// cloud credential to end up inside an LLM prompt.
fn redact_env_output(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        match line.split_once('=') {
            Some((key, _)) if is_secret_env_key(key) => {
                out.push_str(key);
                out.push_str("=<redacted>");
            },
            _ => out.push_str(line),
        }
        out.push('\n');
    }
    out
}

fn is_secret_env_key(key: &str) -> bool {
    let upper = key.trim().to_ascii_uppercase();
    SECRET_ENV_MARKERS.iter().any(|marker| upper.contains(marker))
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    fn refuse(command: &str, root: &Path) -> PolicyError {
        run_checked(command, root).expect_err("expected refusal")
    }

    #[test]
    fn tokenize_handles_quotes() {
        assert_eq!(tokenize(r#"cat "my file.txt""#).unwrap(), vec!["cat", "my file.txt"]);
        assert_eq!(tokenize("ls   -la  src").unwrap(), vec!["ls", "-la", "src"]);
        assert_eq!(tokenize("grep 'two words' f").unwrap(), vec!["grep", "two words", "f"]);
        // An empty quoted string is still an argument.
        assert_eq!(tokenize(r#"grep "" f"#).unwrap(), vec!["grep", "", "f"]);
    }

    #[test]
    fn tokenize_rejects_unterminated_quote() {
        assert_eq!(tokenize("cat \"oops").unwrap_err(), PolicyError::UnterminatedQuote);
    }

    #[test]
    fn find_grouping_syntax_survives_tokenizing() {
        // The shell strips these backslashes before find sees them; with no
        // shell in the loop the tokenizer has to do it.
        let tokens = tokenize(r#"find . \( -iname "*.mp4" -o -iname "*.mkv" \)"#).unwrap();
        assert_eq!(tokens, vec![
            "find", ".", "(", "-iname", "*.mp4", "-o", "-iname", "*.mkv", ")"
        ]);
        assert!(check_policy(&tokens).is_ok());
    }

    #[test]
    fn grouped_find_actually_matches_both_patterns() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("a.mp4"), "x").unwrap();
        fs::write(dir.path().join("b.mkv"), "x").unwrap();
        fs::write(dir.path().join("c.txt"), "x").unwrap();

        let out = run_command_tool_result(
            r#"find . -type f \( -iname "*.mp4" -o -iname "*.mkv" \)"#,
            dir.path(),
        );
        assert!(out.contains("a.mp4"), "{out}");
        assert!(out.contains("b.mkv"), "{out}");
        assert!(!out.contains("c.txt"), "{out}");
    }

    #[test]
    fn escaping_does_not_smuggle_a_second_command() {
        let dir = tempfile::tempdir().unwrap();
        // Escaping a metacharacter must not slip it past the scan.
        assert!(matches!(refuse(r"ls \; rm -rf /", dir.path()), PolicyError::ShellMetachar(';')));
        assert!(matches!(refuse(r"ls \| cat", dir.path()), PolicyError::ShellMetachar('|')));
        // A trailing backslash is malformed rather than silently dropped.
        assert!(matches!(refuse("ls foo\\", dir.path()), PolicyError::UnterminatedQuote));
    }

    #[test]
    fn rejects_shell_metacharacters() {
        let dir = tempfile::tempdir().unwrap();
        for command in [
            "ls | rm -rf /",
            "ls; rm -rf /",
            "ls && rm -rf /",
            "echo $(rm -rf /)",
            "cat file > out.txt",
            "cat `whoami`",
        ] {
            assert!(
                matches!(refuse(command, dir.path()), PolicyError::ShellMetachar(_)),
                "expected metachar refusal for {command}"
            );
        }
    }

    #[test]
    fn rejects_unknown_binary() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(refuse("rm -rf .", dir.path()), PolicyError::NotAllowed("rm".into()));
        assert_eq!(refuse("sed -i s/a/b/ f", dir.path()), PolicyError::NotAllowed("sed".into()));
        // A path-qualified binary bypasses PATH lookup, so it is refused too.
        assert_eq!(refuse("/bin/ls", dir.path()), PolicyError::NotAllowed("/bin/ls".into()));
    }

    #[test]
    fn rejects_denied_git_subcommand() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            refuse("git config --global user.name x", dir.path()),
            PolicyError::DeniedSubcommand { bin: "git".into(), sub: "config".into() }
        );
        assert_eq!(
            refuse("git push origin main", dir.path()),
            PolicyError::DeniedSubcommand { bin: "git".into(), sub: "push".into() }
        );
        assert!(check_policy(&tokenize("git status --short").unwrap()).is_ok());
    }

    #[test]
    fn rejects_find_delete() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            refuse("find . -name x -delete", dir.path()),
            PolicyError::DeniedFlag { bin: "find".into(), flag: "-delete".into() }
        );
        assert_eq!(refuse("find . -exec rm {} +", dir.path()), PolicyError::ShellMetachar('{'));
    }

    #[test]
    fn version_only_binaries_reject_other_args() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(
            refuse("python3 script.py", dir.path()),
            PolicyError::DeniedSubcommand { bin: "python3".into(), sub: "script.py".into() }
        );
        assert_eq!(
            refuse("npm install left-pad", dir.path()),
            PolicyError::DeniedSubcommand { bin: "npm".into(), sub: "install".into() }
        );
        assert!(check_policy(&tokenize("python3 --version").unwrap()).is_ok());
    }

    #[test]
    fn rejects_path_escape() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("sub");
        fs::create_dir(&nested).unwrap();
        assert_eq!(
            refuse("cat ../secret.txt", &nested),
            PolicyError::PathNotFound { token: "../secret.txt".into(), similar: Vec::new() },
            "must not leak names from outside the root"
        );

        // An existing file outside the root is refused for being outside it.
        fs::write(dir.path().join("outside.txt"), "data").unwrap();
        assert_eq!(
            refuse("cat ../outside.txt", &nested),
            PolicyError::PathOutsideScope("../outside.txt".into())
        );
    }

    #[test]
    fn rejects_symlink_escape() {
        let dir = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let target = outside.path().join("target.txt");
        fs::write(&target, "data").unwrap();

        let root = dir.path().join("root");
        fs::create_dir(&root).unwrap();
        let link = root.join("link.txt");
        #[cfg(unix)]
        std::os::unix::fs::symlink(&target, &link).unwrap();

        assert_eq!(refuse("cat link.txt", &root), PolicyError::PathOutsideScope("link.txt".into()));
    }

    #[test]
    fn rejects_secret_paths_inside_root() {
        let dir = tempfile::tempdir().unwrap();
        let ssh = dir.path().join(".ssh");
        fs::create_dir(&ssh).unwrap();
        fs::write(ssh.join("id_rsa"), "PRIVATE KEY").unwrap();
        // Inside the allowed subtree, yet still refused.
        assert_eq!(
            refuse("cat .ssh/id_rsa", dir.path()),
            PolicyError::SecretPath(".ssh/id_rsa".into())
        );

        fs::write(dir.path().join("server.pem"), "cert").unwrap();
        assert_eq!(
            refuse("cat server.pem", dir.path()),
            PolicyError::SecretPath("server.pem".into())
        );
    }

    #[test]
    fn allows_paths_inside_root() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("notes.txt"), "hello").unwrap();
        let tokens = tokenize("cat notes.txt").unwrap();
        assert!(check_paths(&tokens, dir.path()).is_ok());
    }

    #[test]
    fn allows_read_safe_global() {
        let dir = tempfile::tempdir().unwrap();
        if Path::new("/etc/os-release").exists() {
            let tokens = tokenize("cat /etc/os-release").unwrap();
            assert!(check_paths(&tokens, dir.path()).is_ok());
        }
    }

    #[test]
    fn redacts_secret_env_vars() {
        let input = "HOME=/home/u\nAWS_SECRET_ACCESS_KEY=abc123\nGITHUB_TOKEN=ghp_x\nLANG=C\n";
        let out = redact_env_output(input);
        assert!(out.contains("HOME=/home/u"));
        assert!(out.contains("LANG=C"));
        assert!(out.contains("AWS_SECRET_ACCESS_KEY=<redacted>"));
        assert!(out.contains("GITHUB_TOKEN=<redacted>"));
        assert!(!out.contains("abc123"));
        assert!(!out.contains("ghp_x"));
    }

    #[test]
    fn runs_allowed_command_and_reports_exit_code() {
        let dir = tempfile::tempdir().unwrap();
        fs::write(dir.path().join("notes.txt"), "hello world").unwrap();
        let out = run_command_tool_result("cat notes.txt", dir.path());
        assert!(out.starts_with("$ cat notes.txt"), "{out}");
        assert!(out.contains("[exit 0]"), "{out}");
        assert!(out.contains("hello world"), "{out}");
    }

    #[test]
    fn truncates_long_output() {
        let dir = tempfile::tempdir().unwrap();
        let big = "x".repeat(50_000);
        fs::write(dir.path().join("big.txt"), &big).unwrap();
        let out = run_command_tool_result("cat big.txt", dir.path());
        assert!(out.chars().count() < OUTPUT_BUDGET_CHARS + 200, "len {}", out.chars().count());
        assert!(out.contains("[truncated]"), "{out}");
    }

    #[test]
    fn failed_command_suggests_similarly_named_entries() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("Videos")).unwrap();
        fs::create_dir(dir.path().join("Music")).unwrap();

        // The user says "Movies"; the machine has "Videos".
        let out = run_command_tool_result("ls Movies", dir.path());
        assert!(out.contains("[hint]"), "{out}");
        // "Movies" is not a misspelling of "Videos" — no lexical rule can link
        // them — so the hint must surface what is actually there instead.
        assert!(out.contains("Videos"), "{out}");
        assert!(out.contains("Music"), "{out}");
    }

    #[test]
    fn case_mismatch_is_offered_as_a_suggestion() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("Videos")).unwrap();
        let out = run_command_tool_result("ls videos", dir.path());
        assert!(out.contains("Videos"), "{out}");
    }

    #[test]
    fn successful_command_has_no_hint() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("Videos")).unwrap();
        let out = run_command_tool_result("ls Videos", dir.path());
        assert!(!out.contains("[hint]"), "{out}");
    }

    #[test]
    fn missing_path_refusal_carries_suggestions() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("Videos")).unwrap();
        // A slash makes this a path candidate, so it is refused before running.
        let out = run_command_tool_result("ls ./Movies", dir.path());
        assert!(out.contains("Videos"), "{out}");
        assert!(out.contains("does not exist"), "{out}");
    }

    #[test]
    fn suggestions_refuse_a_parent_outside_the_root() {
        let outside = tempfile::tempdir().unwrap();
        fs::create_dir(outside.path().join("SecretProject")).unwrap();
        let root = outside.path().join("root");
        fs::create_dir(&root).unwrap();

        // Target sits in the root's *parent*, which is out of scope.
        let target = outside.path().join("Movies");
        assert!(
            suggestions_within_scope(&target, &root).is_empty(),
            "must not list names from outside the root"
        );
        // The unguarded helper would happily have listed them.
        assert!(
            similar_entries(&target).contains(&"SecretProject".to_owned()),
            "precondition: the raw helper does list them, which is why the guard exists"
        );
    }

    #[test]
    fn suggestions_refuse_a_secret_parent() {
        let root = tempfile::tempdir().unwrap();
        let ssh = root.path().join(".ssh");
        fs::create_dir(&ssh).unwrap();
        fs::write(ssh.join("id_rsa"), "key").unwrap();

        // Inside the root, but naming what lives in ~/.ssh is still a leak.
        let target = ssh.join("id_rsa_missing");
        assert!(suggestions_within_scope(&target, root.path()).is_empty());
    }

    #[test]
    fn failed_command_hint_never_names_files_outside_root() {
        let outside = tempfile::tempdir().unwrap();
        fs::write(outside.path().join("SIBLING_SECRET.txt"), "x").unwrap();
        let root = outside.path().join("root");
        fs::create_dir(&root).unwrap();
        fs::create_dir(root.join("Videos")).unwrap();

        // Reaches the execute path: no slash, so it is not a path candidate.
        let out = run_command_tool_result("ls Movies", &root);
        assert!(out.contains("Videos"), "in-scope names should still be offered: {out}");
        assert!(!out.contains("SIBLING_SECRET"), "leaked a name from outside the root: {out}");

        // And a path candidate pointing outside is refused with no listing.
        let out = run_command_tool_result("cat ../SIBLING_SECRET.txt", &root);
        assert!(!out.contains("SIBLING_SECRET.txt\n"), "leaked content: {out}");
        assert!(out.contains("outside the working directory"), "{out}");
    }

    #[test]
    fn suggestions_skip_hidden_entries() {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join(".moviecache")).unwrap();
        let hits = similar_entries(&dir.path().join("movies"));
        assert!(hits.is_empty(), "hidden entries should not be suggested: {hits:?}");
    }

    #[test]
    fn expands_leading_tilde_to_home() {
        let home = home::home_dir().expect("home dir");
        let out = expand_tildes(vec![
            "ls".into(),
            "-la".into(),
            "~".into(),
            "~/Videos".into(),
            "not~here".into(),
        ]);
        assert_eq!(out[0], "ls");
        assert_eq!(out[1], "-la");
        assert_eq!(out[2], home.to_string_lossy());
        assert_eq!(out[3], home.join("Videos").to_string_lossy());
        // A tilde that is not leading is an ordinary character.
        assert_eq!(out[4], "not~here");
    }

    #[test]
    fn tilde_expansion_still_obeys_scope_and_secrets() {
        let dir = tempfile::tempdir().unwrap();
        // ~ resolves outside the sandbox root, so scope still refuses it.
        let err = refuse("ls ~", dir.path());
        assert!(
            matches!(err, PolicyError::PathOutsideScope(_) | PolicyError::PathNotFound { .. }),
            "expected scope refusal, got {err:?}"
        );
        // And an expanded path into a secret directory is still refused.
        let err = refuse("cat ~/.ssh/id_rsa", dir.path());
        assert!(
            matches!(err, PolicyError::SecretPath(_) | PolicyError::PathNotFound { .. }),
            "expected secret refusal, got {err:?}"
        );
    }

    #[test]
    fn kill_switch_disables_only_on_falsey_values() {
        // Unset means available — the tool is on by default.
        assert!(enabled_from(None));
        for off in ["0", "false", "off", "no", "OFF", " False "] {
            assert!(!enabled_from(Some(off)), "{off} should disable the tool");
        }
        for on in ["1", "true", "on", "yes", ""] {
            assert!(enabled_from(Some(on)), "{on} should leave the tool enabled");
        }
    }

    #[test]
    fn empty_command_refused() {
        let dir = tempfile::tempdir().unwrap();
        assert_eq!(refuse("   ", dir.path()), PolicyError::Empty);
    }

    #[test]
    fn refusals_render_as_readable_text() {
        let dir = tempfile::tempdir().unwrap();
        let message = run_command_tool_result("rm -rf /", dir.path());
        assert!(message.starts_with("run_command refused:"), "{message}");
        assert!(message.contains("read-only"), "{message}");
    }
}


