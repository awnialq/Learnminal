use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Snapshot of terminal state sent to the Python backend.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TerminalContext {
    pub visible_text: String,
    pub selected_text: Option<String>,
    pub last_command: String,
    /// Output produced by the last command, extracted from the visible grid.
    /// Empty when the output has scrolled off screen or no command was detected.
    #[serde(default)]
    pub last_command_output: String,
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
            cwd: String::new(),
            exit_code: None,
            rows: 0,
            cols: 0,
        }
    }
}

/// JSON payload POSTed to `http://127.0.0.1:8765/explain`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExplainRequest {
    pub request_id: String,
    pub timestamp: i64,
    pub terminal: TerminalContext,
    /// User follow-up question from the overlay input field; omitted on first explain.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub follow_up_question: Option<String>,
}

impl ExplainRequest {
    pub fn from_context(
        ctx: TerminalContext,
        request_id: String,
        timestamp: i64,
        follow_up_question: Option<String>,
    ) -> Self {
        Self { request_id, timestamp, terminal: ctx, follow_up_question }
    }
}

/// System environment snapshot from `GET /system-info` (overlay `/info`).
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
    /// `{ "pacman": ["vim", "git", ...], "pip3": ["numpy", ...] }` from last system scan.
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
    /// Whether the backend returned enough data for `/info` and agent prompts.
    pub fn is_complete(&self) -> bool {
        !self.os.is_empty() && self.collected_at.is_some()
    }
}

/// Per-flag breakdown in the structured response.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FlagExplanation {
    pub flag: String,
    pub meaning: String,
    pub example: String,
}

/// Structured educational response from the ReAct agent.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExplainResponse {
    pub command_name: String,
    pub flags_explained: Vec<FlagExplanation>,
    pub general_utility: String,
    pub contextual_usage: String,
    pub error_fix: Option<String>,
    #[serde(default)]
    pub similar_commands: Vec<String>,
    #[serde(default)]
    pub tool_calls_made: Vec<String>,
    /// Shell-ready commands for the top-right actionable HUD (max 5).
    #[serde(default)]
    pub actionable_items: Vec<String>,
}

/// Incremental SSE text chunk.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StreamChunk {
    pub text: String,
    pub chunk_index: u32,
}

/// Section in a command reference (Command mode).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReferenceSection {
    pub name: String,
    pub lines: Vec<String>,
}

/// Formatted man/--help from `POST /command-reference`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CommandReferenceResponse {
    pub program: String,
    pub source: String,
    pub title: String,
    pub sections: Vec<ReferenceSection>,
}

/// JSON payload POSTed to `http://127.0.0.1:8765/chat`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChatRequest {
    pub request_id: String,
    pub timestamp: i64,
    pub terminal: TerminalContext,
    pub message: String,
}

impl ChatRequest {
    pub fn from_context(ctx: TerminalContext, request_id: String, timestamp: i64, message: String) -> Self {
        Self { request_id, timestamp, terminal: ctx, message }
    }
}

/// Parsed chat SSE done event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatDoneEvent {
    pub reply: String,
    pub actionable_items: Vec<String>,
}

/// Parse chat SSE done event `{"reply": "...", "actionable_items": [...], "done": true}`.
pub fn parse_chat_done_event(data: &str) -> Option<ChatDoneEvent> {
    let envelope: serde_json::Value = serde_json::from_str(data).ok()?;
    if !envelope.get("done").and_then(serde_json::Value::as_bool).unwrap_or(false) {
        return None;
    }
    let reply = envelope.get("reply").and_then(|v| v.as_str())?.to_owned();
    let actionable_items = string_list(envelope.get("actionable_items"));
    Some(ChatDoneEvent { reply, actionable_items })
}

/// First token of a shell command line (program name).
pub fn extract_program_name(last_command: &str) -> String {
    let trimmed = last_command.trim();
    if trimmed.is_empty() {
        return String::new();
    }
    // Simple split; Python uses shlex — good enough for common cases.
    trimmed.split_whitespace().next().unwrap_or("").to_owned()
}

/// Strip optional markdown code fences from model JSON output.
pub fn strip_json_fences(text: &str) -> String {
    let text = text.trim();
    if !text.starts_with("```") {
        return text.to_owned();
    }
    let mut lines: Vec<&str> = text.lines().skip(1).collect();
    if lines.last().is_some_and(|l| l.trim().starts_with("```")) {
        lines.pop();
    }
    lines.join("\n")
}

fn string_field(obj: &serde_json::Map<String, Value>, key: &str) -> String {
    obj.get(key)
        .and_then(|v| match v {
            Value::String(s) => Some(s.clone()),
            Value::Number(n) => Some(n.to_string()),
            Value::Bool(b) => Some(b.to_string()),
            _ => None,
        })
        .unwrap_or_default()
}

fn optional_string_field(obj: &serde_json::Map<String, Value>, key: &str) -> Option<String> {
    let s = string_field(obj, key);
    if s.is_empty() { None } else { Some(s) }
}

fn string_list(value: Option<&Value>) -> Vec<String> {
    match value {
        Some(Value::Array(arr)) => arr
            .iter()
            .filter_map(|v| v.as_str().map(str::to_owned))
            .collect(),
        Some(Value::String(s)) => vec![s.clone()],
        _ => Vec::new(),
    }
}

fn parse_flags(value: Option<&Value>) -> Vec<FlagExplanation> {
    let Some(Value::Array(arr)) = value else { return Vec::new() };
    arr.iter()
        .filter_map(|item| {
            let obj = item.as_object()?;
            Some(FlagExplanation {
                flag: obj.get("flag").and_then(|v| v.as_str())?.to_owned(),
                meaning: obj.get("meaning").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
                example: obj.get("example").and_then(|v| v.as_str()).unwrap_or("").to_owned(),
            })
        })
        .collect()
}

/// Parse LLM / backend JSON into `ExplainResponse`, tolerating minor schema drift.
pub fn parse_explain_response_lenient(value: &serde_json::Value) -> Option<ExplainResponse> {
    let obj = value.as_object()?;
    let command_name = string_field(obj, "command_name");
    Some(ExplainResponse {
        command_name: if command_name.is_empty() { "unknown".into() } else { command_name },
        flags_explained: parse_flags(obj.get("flags_explained")),
        general_utility: string_field(obj, "general_utility"),
        contextual_usage: string_field(obj, "contextual_usage"),
        error_fix: optional_string_field(obj, "error_fix"),
        similar_commands: string_list(obj.get("similar_commands")),
        tool_calls_made: string_list(obj.get("tool_calls_made")),
        actionable_items: string_list(obj.get("actionable_items")),
    })
}

/// Parse an SSE `data:` payload containing `{"structured": {...}, "done": true}`.
pub fn parse_structured_done_event(data: &str) -> Option<ExplainResponse> {
    let envelope: serde_json::Value = serde_json::from_str(data).ok()?;
    if !envelope.get("done").and_then(serde_json::Value::as_bool).unwrap_or(false) {
        return None;
    }
    let structured = envelope.get("structured")?;
    parse_explain_response_lenient(structured)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn system_info_complete_when_os_and_collected_at_present() {
        let info = SystemInfo {
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
        let empty = SystemInfo {
            os: String::new(),
            arch: String::new(),
            shell: String::new(),
            package_managers: Vec::new(),
            installed_packages: std::collections::HashMap::new(),
            installed_packages_total: None,
            installed_tools: Vec::new(),
            collected_at: None,
            collected_at_display: None,
        };
        assert!(!empty.is_complete());
    }

    #[test]
    fn parse_structured_done_tolerates_malformed_flags() {
        let data = r#"{"structured":{"command_name":"git","flags_explained":["-a"],"general_utility":"utility","contextual_usage":"context","error_fix":null,"similar_commands":[],"tool_calls_made":[]},"done":true}"#;
        let response = parse_structured_done_event(data).expect("parses leniently");
        assert_eq!(response.command_name, "git");
        assert!(response.flags_explained.is_empty());
        assert_eq!(response.general_utility, "utility");
    }

    #[test]
    fn parse_explain_response_lenient_from_json_object() {
        let json = r#"{"command_name":"ls","flags_explained":[],"general_utility":"list","contextual_usage":"here","error_fix":null,"similar_commands":[],"tool_calls_made":[]}"#;
        let value: Value = serde_json::from_str(json).unwrap();
        let response = parse_explain_response_lenient(&value).unwrap();
        assert_eq!(response.command_name, "ls");
    }

    fn arb_terminal_context() -> impl Strategy<Value = TerminalContext> {
        (
            ".*",
            prop::option::of(".*"),
            ".*",
            ".*",
            ".*",
            prop::option::of(any::<i32>()),
            any::<u16>(),
            any::<u16>(),
        )
            .prop_map(|(visible_text, selected_text, last_command, last_command_output, cwd, exit_code, rows, cols)| {
                TerminalContext {
                    visible_text,
                    selected_text,
                    last_command,
                    last_command_output,
                    cwd,
                    exit_code,
                    rows,
                    cols,
                }
            })
    }

    proptest! {
        // Property 5: ExplainRequest serialization is schema-complete for any
        // TerminalContext. Asserts every required field from ipc-contract/schema.json
        // is present in the JSON with the correct type.
        #[test]
        fn property5_explain_request_serialization_is_schema_complete(
            ctx in arb_terminal_context(),
            request_id in "[a-zA-Z0-9-]{1,40}",
            timestamp in any::<i64>(),
        ) {
            let request = ExplainRequest::from_context(ctx, request_id, timestamp, None);

            let json = serde_json::to_string(&request).expect("serialize ExplainRequest");
            let value: Value = serde_json::from_str(&json).expect("parse JSON");
            let obj = value.as_object().expect("ExplainRequest is a JSON object");

            // ---- ExplainRequest required fields ----
            prop_assert!(
                obj.get("request_id").is_some_and(|v| v.is_string()),
                "request_id missing or not a string",
            );
            prop_assert!(
                obj.get("timestamp").is_some_and(|v| v.is_i64()),
                "timestamp missing or not an integer",
            );
            let term = obj
                .get("terminal")
                .and_then(|v| v.as_object())
                .expect("terminal is a JSON object");

            // ---- TerminalContext required fields ----
            prop_assert!(
                term.get("visible_text").is_some_and(|v| v.is_string()),
                "visible_text missing or not a string",
            );
            // selected_text must be present and either string or null.
            prop_assert!(term.contains_key("selected_text"));
            let selected = &term["selected_text"];
            prop_assert!(
                selected.is_null() || selected.is_string(),
                "selected_text must be string or null",
            );

            prop_assert!(
                term.get("last_command").is_some_and(|v| v.is_string()),
                "last_command missing or not a string",
            );
            prop_assert!(
                term.get("last_command_output").is_some_and(|v| v.is_string()),
                "last_command_output missing or not a string",
            );
            prop_assert!(
                term.get("cwd").is_some_and(|v| v.is_string()),
                "cwd missing or not a string",
            );

            prop_assert!(term.contains_key("exit_code"));
            let exit = &term["exit_code"];
            prop_assert!(
                exit.is_null() || exit.is_i64(),
                "exit_code must be integer or null",
            );

            prop_assert!(
                term.get("rows").is_some_and(|v| v.is_u64()),
                "rows missing or not a non-negative integer",
            );
            prop_assert!(
                term.get("cols").is_some_and(|v| v.is_u64()),
                "cols missing or not a non-negative integer",
            );
        }

        // Round-trip: any TerminalContext serializes and deserializes to itself.
        #[test]
        fn explain_request_round_trips(ctx in arb_terminal_context()) {
            let request = ExplainRequest::from_context(ctx, "req".into(), 0, None);
            let json = serde_json::to_string(&request).unwrap();
            let parsed: ExplainRequest = serde_json::from_str(&json).unwrap();
            prop_assert_eq!(request, parsed);
        }
    }
}
