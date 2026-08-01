//! Actionable items extracted from a finished assistant reply.
//!
//! After the chat answer has fully streamed, the model is asked to re-read its
//! own answer and register the commands it mentioned via a `list_actions` tool
//! call. The result is a short list of `(command, goal)` pairs shown in the
//! top-right Actions panel.
//!
//! Every failure path here is best-effort and yields an empty list — the
//! feature must never break the chat flow.

use serde_json::Value;

use crate::learnminal::ollama::{self, OllamaClient};

/// Tool the model calls to register runnable commands.
pub const TOOL_NAME: &str = "list_actions";

/// Maximum items shown in the panel.
pub const MAX_ACTIONS: usize = 5;

/// Commands longer than this are almost certainly prose, not a command.
const MAX_COMMAND_LEN: usize = 120;

/// Blurbs are meant to be a couple of words.
const MAX_GOAL_LEN: usize = 32;

/// Cap on how much of the reply is fed back for extraction.
const REPLY_BUDGET_CHARS: usize = 4_000;

/// A single runnable command plus a few words on what it is for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActionItem {
    /// Single-line shell command, ready to run.
    pub command: String,
    /// Two-to-four word summary of the command's goal.
    pub goal: String,
}

/// Whether action extraction is enabled (`LEARNMINAL_ACTIONS` not `0`/`false`/`off`/`no`).
pub fn actions_enabled() -> bool {
    match std::env::var("LEARNMINAL_ACTIONS") {
        Ok(value) => {
            let v = value.trim().to_ascii_lowercase();
            !(v == "0" || v == "false" || v == "off" || v == "no")
        },
        Err(_) => true,
    }
}

/// Ask the model to list the runnable commands contained in `reply`.
pub fn extract_actions(
    client: &OllamaClient,
    model: &str,
    reply: &str,
    last_command: &str,
) -> Vec<ActionItem> {
    if reply.trim().is_empty() {
        return Vec::new();
    }

    let prompt = build_extraction_prompt(reply, last_command);
    match client.call_tool_once(model, &prompt, ollama::actions_tool_schema(), TOOL_NAME) {
        Ok(Some(arguments)) => parse_actions(&arguments),
        Ok(None) => Vec::new(),
        Err(err) => {
            log::debug!("learnminal: action extraction failed: {}", err.user_message());
            Vec::new()
        },
    }
}

fn build_extraction_prompt(reply: &str, last_command: &str) -> String {
    let mut prompt = String::with_capacity(REPLY_BUDGET_CHARS + 512);
    prompt.push_str(
        "You are extracting runnable shell commands from an answer that was just given to a \
         terminal user.\n\nCall the list_actions tool with the commands the answer tells the user \
         to run.\n\nRules:\n- Only include commands that appear in the answer, or that the answer \
         directly instructs the user to run. Do not invent commands or flags.\n- Each command must \
         be a single line that can be pasted into a shell as-is. No placeholders in angle brackets \
         unless the answer used them.\n- Each goal must be a 2-4 word lowercase phrase describing \
         what the command achieves, for example \"show as a list\" or \"check dir sizes\".\n- At \
         most 5 commands, most useful first.\n- If the answer contains no runnable command, call \
         list_actions with an empty list.\n\n",
    );

    if !last_command.trim().is_empty() {
        prompt.push_str("The user's last command was: ");
        prompt.push_str(last_command.trim());
        prompt.push_str("\n\n");
    }

    prompt.push_str("Answer to extract from:\n");
    prompt.push_str(&truncate_chars(reply.trim(), REPLY_BUDGET_CHARS));
    prompt.push('\n');
    prompt
}

/// Tolerant parse of the tool arguments into action items.
///
/// Accepts a bare array, `{"actions": [...]}` (or `items`/`commands`), and any
/// of those nested inside a JSON string, since small models are inconsistent.
fn parse_actions(raw: &Value) -> Vec<ActionItem> {
    let mut out: Vec<ActionItem> = Vec::new();
    for value in actions_array(raw, 0) {
        let Some(item) = parse_item(&value) else {
            continue;
        };
        if out.iter().any(|existing| existing.command == item.command) {
            continue;
        }
        out.push(item);
        if out.len() == MAX_ACTIONS {
            break;
        }
    }
    out
}

/// Dig the list of items out of whatever shape the model produced.
fn actions_array(raw: &Value, depth: usize) -> Vec<Value> {
    if depth > 2 {
        return Vec::new();
    }
    match raw {
        Value::Array(items) => items.clone(),
        Value::Object(map) => {
            for key in ["actions", "items", "commands", "list"] {
                if let Some(nested) = map.get(key) {
                    let items = actions_array(nested, depth + 1);
                    if !items.is_empty() {
                        return items;
                    }
                }
            }
            // A single item passed as a bare object.
            if map.contains_key("command") || map.contains_key("cmd") {
                return vec![raw.clone()];
            }
            Vec::new()
        },
        Value::String(text) => match ollama::parse_embedded_json(text) {
            Some(parsed) => actions_array(&parsed, depth + 1),
            None => Vec::new(),
        },
        _ => Vec::new(),
    }
}

fn parse_item(value: &Value) -> Option<ActionItem> {
    let map = value.as_object()?;
    let command = sanitize_command(&first_str(map, &["command", "cmd", "action"])?)?;
    let goal =
        first_str(map, &["goal", "description", "purpose", "blurb", "summary"]).unwrap_or_default();
    Some(ActionItem { command, goal: sanitize_goal(&goal) })
}

fn first_str(map: &serde_json::Map<String, Value>, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|key| map.get(*key).and_then(Value::as_str))
        .map(|s| s.trim().to_owned())
        .filter(|s| !s.is_empty())
}

/// Normalize a candidate command, rejecting anything that is not a single runnable line.
fn sanitize_command(raw: &str) -> Option<String> {
    let mut command = raw.trim();
    if command.contains('\n') || command.contains('\r') {
        return None;
    }
    // Models often echo the prompt marker along with the command.
    for prefix in ["$ ", "> ", "% ", "# "] {
        if let Some(stripped) = command.strip_prefix(prefix) {
            command = stripped.trim_start();
        }
    }
    command = command.trim_matches('`').trim();
    if command.is_empty() || command.chars().count() > MAX_COMMAND_LEN {
        return None;
    }
    Some(command.to_owned())
}

/// Collapse whitespace and clip the blurb to a couple of words' worth of room.
fn sanitize_goal(raw: &str) -> String {
    let collapsed = raw.split_whitespace().collect::<Vec<_>>().join(" ");
    let trimmed = collapsed.trim_end_matches('.').trim();
    if trimmed.chars().count() <= MAX_GOAL_LEN {
        return trimmed.to_owned();
    }
    format!("{}…", truncate_chars(trimmed, MAX_GOAL_LEN.saturating_sub(1)).trim_end())
}

fn truncate_chars(text: &str, max: usize) -> String {
    text.chars().take(max).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_wrapper_object() {
        let raw = json!({
            "actions": [
                { "command": "ls -l", "goal": "show as a list" },
                { "command": "du -sh *", "goal": "check dir sizes" },
            ]
        });
        let actions = parse_actions(&raw);
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0].command, "ls -l");
        assert_eq!(actions[0].goal, "show as a list");
        assert_eq!(actions[1].command, "du -sh *");
    }

    #[test]
    fn parses_bare_array() {
        let raw = json!([{ "command": "git status", "goal": "see changes" }]);
        let actions = parse_actions(&raw);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].command, "git status");
    }

    #[test]
    fn parses_stringified_arguments() {
        let raw = json!(r#"{"actions": [{"command": "pwd", "goal": "print cwd"}]}"#);
        let actions = parse_actions(&raw);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].command, "pwd");
    }

    #[test]
    fn parses_single_bare_object() {
        let raw = json!({ "command": "whoami", "goal": "show user" });
        assert_eq!(parse_actions(&raw).len(), 1);
    }

    #[test]
    fn rejects_multiline_and_overlong_commands() {
        let long = "a".repeat(MAX_COMMAND_LEN + 1);
        let raw = json!([
            { "command": "ls\nrm -rf /", "goal": "bad" },
            { "command": long, "goal": "bad" },
            { "command": "   ", "goal": "bad" },
            { "command": "ls", "goal": "fine" },
        ]);
        let actions = parse_actions(&raw);
        assert_eq!(actions.len(), 1);
        assert_eq!(actions[0].command, "ls");
    }

    #[test]
    fn strips_prompt_markers_and_backticks() {
        let raw = json!([{ "command": "$ `ls -a`", "goal": "show hidden files" }]);
        let actions = parse_actions(&raw);
        assert_eq!(actions[0].command, "ls -a");
    }

    #[test]
    fn dedupes_and_caps() {
        let mut items = Vec::new();
        for i in 0..MAX_ACTIONS + 3 {
            items.push(json!({ "command": format!("cmd{i}"), "goal": "go" }));
        }
        items.push(json!({ "command": "cmd0", "goal": "dupe" }));
        let actions = parse_actions(&Value::Array(items));
        assert_eq!(actions.len(), MAX_ACTIONS);
    }

    #[test]
    fn truncates_long_goal() {
        let raw = json!([{
            "command": "ls",
            "goal": "this is a very long explanation that goes well past the limit",
        }]);
        let goal = &parse_actions(&raw)[0].goal;
        assert!(goal.chars().count() <= MAX_GOAL_LEN);
        assert!(goal.ends_with('…'));
    }

    #[test]
    fn accepts_missing_goal() {
        let raw = json!([{ "command": "ls" }]);
        let actions = parse_actions(&raw);
        assert_eq!(actions.len(), 1);
        assert!(actions[0].goal.is_empty());
    }

    #[test]
    fn garbage_yields_empty() {
        assert!(parse_actions(&json!("not json at all")).is_empty());
        assert!(parse_actions(&json!(42)).is_empty());
        assert!(parse_actions(&json!({ "unrelated": true })).is_empty());
        assert!(parse_actions(&json!([])).is_empty());
    }

    #[test]
    fn extract_actions_end_to_end_via_tool_call() {
        let mut server = mockito::Server::new();
        let _mock = server
            .mock("POST", "/api/chat")
            .with_status(200)
            .with_body(
                r#"{"message":{"role":"assistant","tool_calls":[{"function":{"name":"list_actions","arguments":{"actions":[{"command":"ls -l","goal":"show as a list"},{"command":"du -sh *","goal":"check dir sizes"}]}}}]}}"#,
            )
            .create();

        let client = OllamaClient::new(&server.url());
        let actions = extract_actions(&client, "m", "Use ls -l to list files.", "ls");
        assert_eq!(actions.len(), 2);
        assert_eq!(actions[0], ActionItem {
            command: "ls -l".into(),
            goal: "show as a list".into()
        });
    }

    #[test]
    fn extract_actions_returns_empty_when_backend_is_down() {
        // Port 1 has nothing listening; the call must degrade to no actions.
        let client = OllamaClient::new("http://127.0.0.1:1");
        assert!(extract_actions(&client, "m", "some answer", "ls").is_empty());
    }

    #[test]
    fn extract_actions_skips_empty_reply_without_calling_out() {
        let client = OllamaClient::new("http://127.0.0.1:1");
        assert!(extract_actions(&client, "m", "   ", "ls").is_empty());
    }

    #[test]
    fn extraction_prompt_includes_reply_and_last_command() {
        let prompt = build_extraction_prompt("Run ls -l to list files.", "cd /tmp");
        assert!(prompt.contains("list_actions"));
        assert!(prompt.contains("Run ls -l to list files."));
        assert!(prompt.contains("cd /tmp"));
    }

    #[test]
    fn env_toggle_defaults_on_when_unset() {
        // Only assert the parsing branch; the env var itself is process-global.
        assert!(actions_enabled() || std::env::var("LEARNMINAL_ACTIONS").is_ok());
    }
}
