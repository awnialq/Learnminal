//! Native Ollama HTTP client.
//!
//! Talks directly to the Ollama daemon at `http://127.0.0.1:11434` (override
//! via `OLLAMA_HOST`). Reuses the blocking `reqwest` client and `serde_json`.

use std::collections::HashSet;
use std::error::Error as StdError;
use std::io::{BufRead, BufReader};
use std::path::PathBuf;
use std::time::Duration;

use reqwest::blocking::{Client, RequestBuilder, Response};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::learnminal::settings;

/// Built-in default model when nothing else is configured/installed.
pub const DEFAULT_MODEL: &str = "gemma4:e4b";

/// macOS MLX-optimized variant of [`DEFAULT_MODEL`].
pub const DEFAULT_MODEL_MLX: &str = "gemma4:e4b-mlx";

/// Platform default: MLX tag on macOS, standard tag elsewhere.
pub fn default_model() -> &'static str {
    if cfg!(target_os = "macos") {
        DEFAULT_MODEL_MLX
    } else {
        DEFAULT_MODEL
    }
}

const DEFAULT_HOST: &str = "http://127.0.0.1:11434";
const CONNECT_TIMEOUT_SECS: u64 = 30;
const READ_TIMEOUT_SECS: u64 = 300;
const UNLOAD_TIMEOUT_SECS: u64 = 5;
/// Keep the model resident until an explicit unload (`keep_alive: 0`).
const KEEP_ALIVE_FOREVER: i64 = -1;
/// Max non-streaming tool rounds before forcing a final streamed answer.
///
/// Environment inspection is sequential (`pwd` → `ls` → `cat`), and searching
/// for something whose location is unknown takes several more probes, so this
/// sits above the length of a real investigation rather than below it.
const MAX_TOOL_ROUNDS: usize = 8;

/// Prior turns replayed so a follow-up question makes sense.
///
/// Each request is otherwise stateless — the daemon keeps no session — so
/// without this the model cannot resolve "how big was it?" against anything.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatTurn {
    /// The user's raw question, not the assembled prompt.
    pub user: String,
    pub assistant: String,
}

/// Turns kept for replay. Enough for a short exchange, bounded so the model's
/// context is not consumed by history instead of the actual question.
pub const MAX_HISTORY_TURNS: usize = 4;
/// Total character budget for replayed history.
const HISTORY_BUDGET_CHARS: usize = 2_000;
/// Per-reply cap, so one long answer cannot crowd out earlier turns.
const HISTORY_REPLY_CHARS: usize = 500;

/// Render prior turns as chat messages, newest first within the budget.
///
/// Only the raw question is replayed, never the assembled prompt: the system
/// instructions, terminal context, and Reference block belong to the current
/// turn alone and would multiply if every turn carried its own copy.
pub fn history_messages(turns: &[ChatTurn]) -> Vec<Value> {
    let mut kept: Vec<&ChatTurn> = Vec::new();
    let mut used = 0usize;

    // Walk backwards so the most recent turns win the budget.
    for turn in turns.iter().rev().take(MAX_HISTORY_TURNS) {
        let reply = truncate_chars(&turn.assistant, HISTORY_REPLY_CHARS);
        let cost = turn.user.chars().count() + reply.chars().count();
        if used + cost > HISTORY_BUDGET_CHARS && !kept.is_empty() {
            break;
        }
        used += cost;
        kept.push(turn);
    }

    kept.reverse();
    let mut messages = Vec::with_capacity(kept.len() * 2);
    for turn in kept {
        messages.push(json!({ "role": "user", "content": turn.user }));
        messages.push(json!({
            "role": "assistant",
            "content": truncate_chars(&turn.assistant, HISTORY_REPLY_CHARS),
        }));
    }
    messages
}

fn truncate_chars(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    let kept: String = text.chars().take(max).collect();
    format!("{}…", kept.trim_end())
}

/// Tools offered to the model for one request.
///
/// Enablement and scope travel together: `read_exec` carries the directory the
/// `run_command` tool is confined to.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ToolSet {
    pub web_search: bool,
    /// `Some(root)` enables `run_command`, restricted to that directory.
    pub read_exec: Option<PathBuf>,
}

impl ToolSet {
    /// Whether no tool is available, in which case the plain stream is used.
    pub fn is_empty(&self) -> bool {
        !self.web_search && self.read_exec.is_none()
    }

    fn schemas(&self) -> Vec<Value> {
        let mut tools = Vec::new();
        if self.web_search {
            tools.push(web_search_tool_schema());
        }
        if self.read_exec.is_some() {
            tools.push(run_command_tool_schema());
        }
        tools
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OllamaError {
    /// Connection refused — the daemon is not running.
    NotRunning,
    Timeout,
    StreamError(String),
    /// Chunks arrived but the terminating `done` line never did.
    IncompleteStream,
}

impl OllamaError {
    /// User-facing message shown in the overlay error panel.
    pub fn user_message(&self) -> String {
        match self {
            OllamaError::NotRunning => "Ollama not running. Start with: ollama serve".to_owned(),
            OllamaError::Timeout => {
                "Ollama request timed out. The model may be overloaded.".to_owned()
            },
            OllamaError::StreamError(msg) => msg.clone(),
            OllamaError::IncompleteStream => {
                "Ollama stream ended before completion. Try again.".to_owned()
            },
        }
    }
}

/// Resolve the Ollama base URL, honoring `OLLAMA_HOST`.
pub fn base_url() -> String {
    match std::env::var("OLLAMA_HOST") {
        Ok(host) if !host.trim().is_empty() => normalize_host(host.trim()),
        _ => DEFAULT_HOST.to_owned(),
    }
}

fn normalize_host(host: &str) -> String {
    let host = host.trim_end_matches('/');
    if host.contains("://") {
        host.to_owned()
    } else {
        format!("http://{host}")
    }
}

pub struct OllamaClient {
    base_url: String,
    client: Client,
}

impl OllamaClient {
    pub fn new(base_url: &str) -> Self {
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .timeout(Duration::from_secs(READ_TIMEOUT_SECS))
            .build()
            .expect("failed to build HTTP client");
        Self { base_url: base_url.trim_end_matches('/').to_owned(), client }
    }

    pub fn default_client() -> Self {
        Self::new(&base_url())
    }

    /// Installed model names via `GET /api/tags`.
    pub fn list_models(&self) -> Result<Vec<String>, OllamaError> {
        let url = format!("{}/api/tags", self.base_url);
        let response = self.send(self.client.get(&url))?;
        let tags: TagsResponse =
            response.json().map_err(|err| OllamaError::StreamError(err.to_string()))?;
        Ok(tags
            .models
            .into_iter()
            .filter_map(|entry| {
                entry
                    .model
                    .filter(|s| !s.trim().is_empty())
                    .or(entry.name)
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty())
            })
            .collect())
    }

    /// Stream a chat reply via `POST /api/chat` (NDJSON).
    ///
    /// `on_chunk` receives incremental content; `on_done` receives the full
    /// accumulated reply; `on_error` receives a daemon-reported error string.
    pub fn chat_stream(
        &self,
        model: &str,
        prompt: &str,
        on_chunk: impl FnMut(String),
        on_done: impl FnMut(String),
        on_error: impl FnMut(String),
    ) -> Result<(), OllamaError> {
        let messages = vec![json!({ "role": "user", "content": prompt })];
        self.chat_stream_messages(model, &messages, on_chunk, on_done, on_error)
    }

    /// Like [`chat_stream`], but with an arbitrary message history (no tools).
    pub fn chat_stream_messages(
        &self,
        model: &str,
        messages: &[Value],
        mut on_chunk: impl FnMut(String),
        mut on_done: impl FnMut(String),
        mut on_error: impl FnMut(String),
    ) -> Result<(), OllamaError> {
        let url = format!("{}/api/chat", self.base_url);
        let body = json!({
            "model": model,
            "messages": messages,
            "stream": true,
            "keep_alive": KEEP_ALIVE_FOREVER,
        });

        let response = self.send(self.client.post(&url).json(&body))?;
        let reader = BufReader::new(response);

        let mut chunk_count = 0usize;
        let mut reply = String::new();
        let mut done_seen = false;

        for line in reader.lines() {
            let line = match line {
                Ok(line) => line,
                Err(err) => {
                    if err.kind() == std::io::ErrorKind::TimedOut {
                        return Err(OllamaError::Timeout);
                    }
                    return Err(OllamaError::StreamError(err.to_string()));
                },
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let Ok(value) = serde_json::from_str::<Value>(trimmed) else {
                continue;
            };

            if let Some(error) = value.get("error").and_then(Value::as_str) {
                on_error(error.to_owned());
                return Ok(());
            }

            if let Some(content) = value.pointer("/message/content").and_then(Value::as_str) {
                if !content.is_empty() {
                    chunk_count += 1;
                    reply.push_str(content);
                    on_chunk(content.to_owned());
                }
            }

            if value.get("done").and_then(Value::as_bool).unwrap_or(false) {
                done_seen = true;
                break;
            }
        }

        if !done_seen {
            if chunk_count == 0 {
                return Err(OllamaError::StreamError("empty response from Ollama".to_owned()));
            }
            return Err(OllamaError::IncompleteStream);
        }

        on_done(reply);
        Ok(())
    }

    /// Chat with optional tool-calling, then stream the final answer.
    ///
    /// Tool rounds are non-streaming (max [`MAX_TOOL_ROUNDS`]). The final turn
    /// streams tokens via `on_chunk` / `on_done`. `on_status` reports progress
    /// such as "Searching the web…" or "Running: git status".
    pub fn chat_with_tools_loop(
        &self,
        model: &str,
        prompt: &str,
        history: &[Value],
        tool_set: &ToolSet,
        mut on_status: impl FnMut(&str),
        mut on_chunk: impl FnMut(String),
        mut on_done: impl FnMut(String),
        mut on_error: impl FnMut(String),
    ) -> Result<(), OllamaError> {
        // Prior turns first, then this turn's fully assembled prompt.
        let mut messages = history.to_vec();
        messages.push(json!({ "role": "user", "content": prompt }));

        if tool_set.is_empty() {
            return self.chat_stream_messages(model, &messages, on_chunk, on_done, on_error);
        }

        let tools = tool_set.schemas();

        for _round in 0..MAX_TOOL_ROUNDS {
            let response = self.chat_once(model, &messages, Some(&tools))?;
            if let Some(error) = response.get("error").and_then(Value::as_str) {
                on_error(error.to_owned());
                return Ok(());
            }

            let message = response.get("message").cloned().unwrap_or(json!({}));
            let tool_calls = extract_tool_calls(&message);
            if tool_calls.is_empty() {
                let content = message_content(&message);
                if content.is_empty() {
                    // Nothing useful — fall through to a streaming retry without tools.
                    break;
                }
                on_chunk(content.clone());
                on_done(content);
                return Ok(());
            }

            // Keep the assistant tool-call turn in history.
            messages.push(message);

            for call in tool_calls {
                let result = match call.name.as_str() {
                    "web_search" if tool_set.web_search => {
                        on_status("Searching the web…");
                        let query = call.string_arg("query").unwrap_or_default();
                        crate::learnminal::web_search::search_tool_result(&query)
                    },
                    "run_command" if tool_set.read_exec.is_some() => {
                        let command = call.string_arg("command").unwrap_or_default();
                        on_status(&format!(
                            "{}{command}",
                            crate::learnminal::read_exec::RUNNING_STATUS_PREFIX
                        ));
                        let root = tool_set.read_exec.as_deref().expect("checked above");
                        crate::learnminal::read_exec::run_command_tool_result(&command, root)
                    },
                    other => format!("unknown tool: {other}"),
                };
                messages.push(json!({
                    "role": "tool",
                    "content": result,
                }));
            }
        }

        // Final streamed answer without tools.
        self.chat_stream_messages(model, &messages, on_chunk, on_done, on_error)
    }

    /// One non-streaming round offering a single tool, returning its raw arguments.
    ///
    /// Small models frequently answer with bare JSON text instead of emitting a
    /// real tool call, so the message content is parsed as a fallback.
    pub fn call_tool_once(
        &self,
        model: &str,
        prompt: &str,
        tool: Value,
        tool_name: &str,
    ) -> Result<Option<Value>, OllamaError> {
        let messages = vec![json!({ "role": "user", "content": prompt })];
        let response = self.chat_once(model, &messages, Some(&[tool]))?;
        if let Some(error) = response.get("error").and_then(Value::as_str) {
            return Err(OllamaError::StreamError(error.to_owned()));
        }

        let message = response.get("message").cloned().unwrap_or(json!({}));
        if let Some(arguments) = tool_call_arguments(&message, tool_name) {
            return Ok(Some(arguments));
        }
        Ok(parse_embedded_json(&message_content(&message)))
    }

    /// Non-streaming `/api/chat` round (optional tools).
    fn chat_once(
        &self,
        model: &str,
        messages: &[Value],
        tools: Option<&[Value]>,
    ) -> Result<Value, OllamaError> {
        let url = format!("{}/api/chat", self.base_url);
        let mut body = json!({
            "model": model,
            "messages": messages,
            "stream": false,
            "keep_alive": KEEP_ALIVE_FOREVER,
        });
        if let Some(tools) = tools {
            body["tools"] = Value::Array(tools.to_vec());
        }
        let response = self.send(self.client.post(&url).json(&body))?;
        response.json().map_err(|err| OllamaError::StreamError(err.to_string()))
    }

    /// Load a model into memory and keep it resident until explicitly unloaded.
    pub fn load(&self, model: &str) -> Result<(), OllamaError> {
        if model.trim().is_empty() {
            return Err(OllamaError::StreamError("Model name must not be empty".to_owned()));
        }

        let url = format!("{}/api/chat", self.base_url);
        let body = json!({
            "model": model,
            "messages": [],
            "stream": false,
            "keep_alive": KEEP_ALIVE_FOREVER,
        });
        self.send(self.client.post(&url).json(&body))?;
        Ok(())
    }

    /// Ask Ollama to drop the model from VRAM (`keep_alive: 0`). Best-effort.
    pub fn unload(&self, model: &str) {
        if model.trim().is_empty() {
            return;
        }
        let url = format!("{}/api/chat", self.base_url);
        let body = json!({ "model": model, "messages": [], "keep_alive": 0 });
        let _ = self
            .client
            .post(&url)
            .timeout(Duration::from_secs(UNLOAD_TIMEOUT_SECS))
            .json(&body)
            .send();
    }

    /// Resolve the active model, returning `(active, installed)`.
    ///
    /// Preference order: settings → `LEARNMINAL_OLLAMA_MODEL` → platform
    /// default (`gemma4:e4b-mlx` on macOS, `gemma4:e4b` elsewhere) → first
    /// installed. On macOS the non-MLX tag is also tried as a fallback.
    pub fn resolve_active_model(&self) -> Result<(String, Vec<String>), OllamaError> {
        let installed = self.list_models()?;
        let candidates = candidate_models();
        let active = pick_available(&candidates, &installed).unwrap_or_else(|| {
            candidates
                .into_iter()
                .flatten()
                .next()
                .unwrap_or_else(|| default_model().to_owned())
        });
        Ok((active, installed))
    }

    /// Validate and persist a preferred model, returning the resolved name.
    pub fn set_active_model(&self, model: &str) -> Result<String, OllamaError> {
        let model = model.trim();
        if model.is_empty() {
            return Err(OllamaError::StreamError("Model name must not be empty".to_owned()));
        }
        let installed = self.list_models().unwrap_or_default();
        let mut resolved = model.to_owned();
        if !installed.is_empty() && !installed.iter().any(|m| m == model) {
            let base = model.split(':').next().unwrap_or(model);
            match installed.iter().find(|m| m.split(':').next().unwrap_or(m) == base) {
                Some(found) => resolved = found.clone(),
                None => {
                    return Err(OllamaError::StreamError(format!("Model not installed: {model}")));
                },
            }
        }
        settings::set_preferred_model(&resolved)
            .map_err(|err| OllamaError::StreamError(err.to_string()))?;
        Ok(resolved)
    }

    fn send(&self, request: RequestBuilder) -> Result<Response, OllamaError> {
        let response = request.send().map_err(map_reqwest_error)?;
        let status = response.status();
        if !status.is_success() {
            let detail = response
                .text()
                .unwrap_or_default()
                .trim()
                .chars()
                .take(200)
                .collect::<String>();
            let message = if detail.is_empty() {
                format!("HTTP {status}")
            } else {
                format!("HTTP {status}: {detail}")
            };
            return Err(OllamaError::StreamError(message));
        }
        Ok(response)
    }
}

/// Candidate models in preference order (some may be `None`).
///
/// On macOS the MLX default is preferred, with the non-MLX tag as a fallback
/// so a stock pull of `gemma4:e4b` still resolves when the `-mlx` tag is absent.
fn candidate_models() -> Vec<Option<String>> {
    let mut candidates = vec![
        settings::get_preferred_model(),
        std::env::var("LEARNMINAL_OLLAMA_MODEL").ok().filter(|s| !s.trim().is_empty()),
        Some(default_model().to_owned()),
    ];
    if cfg!(target_os = "macos") && default_model() != DEFAULT_MODEL {
        candidates.push(Some(DEFAULT_MODEL.to_owned()));
    }
    candidates
}

/// Port of `_pick_available`: exact match, then base-name (`before ':'`) match,
/// then first installed.
pub fn pick_available(candidates: &[Option<String>], installed: &[String]) -> Option<String> {
    let installed_set: HashSet<&str> = installed.iter().map(String::as_str).collect();
    for candidate in candidates.iter().flatten() {
        if installed_set.contains(candidate.as_str()) {
            return Some(candidate.clone());
        }
    }
    for candidate in candidates.iter().flatten() {
        let base = candidate.split(':').next().unwrap_or(candidate);
        for name in installed {
            if name == candidate || name.split(':').next().unwrap_or(name) == base {
                return Some(name.clone());
            }
        }
    }
    installed.first().cloned()
}

fn web_search_tool_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the live web for up-to-date facts, versions, changelogs, or docs not present in the local Reference.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query string"
                    }
                }
            }
        }
    })
}

fn run_command_tool_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a single read-only shell command to inspect the user's machine \
                            (files, git state, installed versions, environment). Only inspection \
                            commands are permitted; anything that writes, deletes, installs, or \
                            connects to the network is refused. No pipes, redirects, or command \
                            chaining — run one command per call.",
            "parameters": {
                "type": "object",
                "required": ["command"],
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "One command with its arguments, e.g. \"git status --short\" or \"ls -la src\""
                    }
                }
            }
        }
    })
}

/// Tool the model calls to register the runnable commands from its own answer.
pub fn actions_tool_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": crate::learnminal::actions::TOOL_NAME,
            "description": "Register the runnable shell commands contained in an answer, each with a short goal.",
            "parameters": {
                "type": "object",
                "required": ["actions"],
                "properties": {
                    "actions": {
                        "type": "array",
                        "description": "Up to 5 commands, most useful first. Empty if the answer has none.",
                        "items": {
                            "type": "object",
                            "required": ["command", "goal"],
                            "properties": {
                                "command": {
                                    "type": "string",
                                    "description": "A single-line shell command the user can run as-is"
                                },
                                "goal": {
                                    "type": "string",
                                    "description": "A 2-4 word lowercase phrase describing what the command achieves"
                                }
                            }
                        }
                    }
                }
            }
        }
    })
}

/// Raw `arguments` of the first call to `tool_name` (object, or JSON in a string).
fn tool_call_arguments(message: &Value, tool_name: &str) -> Option<Value> {
    let Some(Value::Array(calls)) = message.get("tool_calls") else {
        return None;
    };
    for call in calls {
        let function = call.get("function").unwrap_or(call);
        let name = function.get("name").and_then(Value::as_str).unwrap_or("").trim();
        if name != tool_name {
            continue;
        }
        match function.get("arguments") {
            Some(Value::String(text)) => {
                if let Some(parsed) = parse_embedded_json(text) {
                    return Some(parsed);
                }
            },
            Some(value) => return Some(value.clone()),
            None => {},
        }
    }
    None
}

/// First JSON object or array embedded in `text`, tolerating code fences and prose.
pub fn parse_embedded_json(text: &str) -> Option<Value> {
    let trimmed = text
        .trim()
        .trim_start_matches("```json")
        .trim_start_matches("```")
        .trim_end_matches("```")
        .trim();
    if trimmed.is_empty() {
        return None;
    }
    if let Ok(value) = serde_json::from_str::<Value>(trimmed) {
        return Some(value);
    }
    let start = trimmed.find(|c| c == '{' || c == '[')?;
    let end = trimmed.rfind(|c| c == '}' || c == ']')?;
    if end <= start {
        return None;
    }
    serde_json::from_str::<Value>(&trimmed[start..=end]).ok()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedToolCall {
    name: String,
    /// Raw `arguments`, kept untyped so each tool reads its own keys.
    args: Value,
}

impl ParsedToolCall {
    /// String argument `key`, tolerating an object, JSON in a string, or a
    /// bare string (which small models emit for single-argument tools).
    fn string_arg(&self, key: &str) -> Option<String> {
        string_arg_from(&self.args, key)
    }
}

fn message_content(message: &Value) -> String {
    message.get("content").and_then(Value::as_str).unwrap_or("").to_owned()
}

/// Parse Ollama `message.tool_calls` defensively (arguments may be object or JSON string).
fn extract_tool_calls(message: &Value) -> Vec<ParsedToolCall> {
    let Some(Value::Array(calls)) = message.get("tool_calls") else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for call in calls {
        let function = call.get("function").unwrap_or(call);
        let name = function
            .get("name")
            .and_then(Value::as_str)
            .unwrap_or("")
            .trim()
            .to_owned();
        if name.is_empty() {
            continue;
        }
        let args = function.get("arguments").cloned().unwrap_or(Value::Null);
        out.push(ParsedToolCall { name, args });
    }
    out
}

/// Read string argument `key` from raw tool `arguments`.
fn string_arg_from(arguments: &Value, key: &str) -> Option<String> {
    match arguments {
        Value::Object(map) => map
            .get(key)
            .and_then(Value::as_str)
            .map(|s| s.trim().to_owned())
            .filter(|s| !s.is_empty()),
        Value::String(s) => {
            let trimmed = s.trim();
            if trimmed.is_empty() {
                return None;
            }
            if let Ok(Value::Object(map)) = serde_json::from_str::<Value>(trimmed) {
                return map
                    .get(key)
                    .and_then(Value::as_str)
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty());
            }
            // Bare argument string.
            Some(trimmed.to_owned())
        },
        _ => None,
    }
}

fn map_reqwest_error(err: reqwest::Error) -> OllamaError {
    if is_connection_refused(&err) {
        OllamaError::NotRunning
    } else if err.is_timeout() {
        OllamaError::Timeout
    } else {
        OllamaError::StreamError(err.to_string())
    }
}

fn is_connection_refused(err: &reqwest::Error) -> bool {
    if !err.is_connect() {
        return false;
    }
    if let Some(io) = err.source().and_then(|s| s.downcast_ref::<std::io::Error>()) {
        return io.kind() == std::io::ErrorKind::ConnectionRefused;
    }
    err.to_string().contains("Connection refused")
}

#[derive(Deserialize)]
struct TagsResponse {
    #[serde(default)]
    models: Vec<TagEntry>,
}

#[derive(Deserialize)]
struct TagEntry {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    model: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::{Matcher, Mock, ServerGuard};
    use std::cell::RefCell;

    use crate::learnminal::read_exec::RUNNING_STATUS_PREFIX;

    fn start_server() -> ServerGuard {
        mockito::Server::new()
    }

    /// Repository root — the scope the live `run_command` tests inspect.
    #[cfg(test)]
    fn repo_root() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap().to_path_buf()
    }

    /// End-to-end against a real Ollama daemon and a real model.
    ///
    /// Ignored by default: needs `ollama serve` plus an installed tool-capable
    /// model. Run with `cargo test -p alacritty live_ -- --ignored --nocapture`.
    #[test]
    #[ignore = "requires a running Ollama daemon"]
    fn live_chat_inspects_this_repo_with_run_command() {
        let client = OllamaClient::default_client();
        let (model, _) = client.resolve_active_model().expect("no model available");
        let root = repo_root();

        let tools = ToolSet { web_search: false, read_exec: Some(root.clone()) };
        let prompt = crate::learnminal::prompt::build_chat_prompt(
            &crate::learnminal::types::TerminalContext::default(),
            None,
            &[],
            "What name does Cargo.toml give the built binary in this repository? Inspect the \
             files rather than guessing.",
            crate::learnminal::settings::ExperienceLevel::Professional,
            &tools,
        );

        let commands = std::cell::RefCell::new(Vec::new());
        let mut reply = String::new();
        client
            .chat_with_tools_loop(
                &model,
                &prompt,
                &[],
                &tools,
                |status| {
                    if let Some(cmd) = status.strip_prefix(RUNNING_STATUS_PREFIX) {
                        commands.borrow_mut().push(cmd.to_owned());
                    }
                },
                |_| {},
                |done| reply = done,
                |err| panic!("stream error: {err}"),
            )
            .expect("chat loop failed");

        println!("model:    {model}");
        println!("commands: {:?}", commands.borrow());
        println!("reply:    {reply}");

        assert!(!commands.borrow().is_empty(), "model never called run_command");
        assert!(!reply.trim().is_empty(), "empty reply");
    }

    /// A pronoun follow-up must resolve against the previous turn.
    #[test]
    #[ignore = "requires a running Ollama daemon"]
    fn live_follow_up_question_remembers_the_previous_turn() {
        let client = OllamaClient::default_client();
        let (model, _) = client.resolve_active_model().expect("no model");
        let root = repo_root();
        let tools = ToolSet { web_search: false, read_exec: Some(root) };

        let ask = |history: &[Value], question: &str| {
            let prompt = crate::learnminal::prompt::build_chat_prompt(
                &crate::learnminal::types::TerminalContext::default(),
                None,
                &[],
                question,
                crate::learnminal::settings::ExperienceLevel::Professional,
                &tools,
            );
            let mut reply = String::new();
            client
                .chat_with_tools_loop(
                    model.as_str(),
                    &prompt,
                    history,
                    &tools,
                    |_| {},
                    |_| {},
                    |d| reply = d,
                    |e| panic!("stream error: {e}"),
                )
                .expect("chat failed");
            reply
        };

        let q1 = "How many files are in the docs directory? Answer with the number.";
        let a1 = ask(&[], q1);
        println!("Q1: {q1}\nA1: {a1}\n");

        let turns = vec![ChatTurn { user: q1.to_owned(), assistant: a1.clone() }];
        let history = history_messages(&turns);

        // Pure pronoun reference: unanswerable without the prior turn.
        let q2 = "What did I just ask you about?";
        let a2 = ask(&history, q2);
        println!("Q2: {q2}\nA2: {a2}\n");

        let lowered = a2.to_lowercase();
        assert!(
            lowered.contains("docs") || lowered.contains("directory") || lowered.contains("file"),
            "follow-up did not resolve against the previous turn: {a2}"
        );
    }

    /// The reported failure: asking for a "Movies" directory on a machine that
    /// calls it "Videos". The model must recover instead of giving up.
    #[test]
    #[ignore = "requires a running Ollama daemon"]
    fn live_chat_recovers_from_a_wrong_directory_name() {
        let client = OllamaClient::default_client();
        let (model, _) = client.resolve_active_model().expect("no model available");
        let home = home::home_dir().expect("home dir");

        let tools = ToolSet { web_search: false, read_exec: Some(home) };
        let prompt = crate::learnminal::prompt::build_chat_prompt(
            &crate::learnminal::types::TerminalContext::default(),
            None,
            &[],
            "Find the Movie directory on my computer.",
            crate::learnminal::settings::ExperienceLevel::Professional,
            &tools,
        );

        let commands = std::cell::RefCell::new(Vec::new());
        let mut reply = String::new();
        client
            .chat_with_tools_loop(
                &model,
                &prompt,
                &[],
                &tools,
                |status| {
                    if let Some(cmd) = status.strip_prefix(RUNNING_STATUS_PREFIX) {
                        commands.borrow_mut().push(cmd.to_owned());
                    }
                },
                |_| {},
                |done| reply = done,
                |err| panic!("stream error: {err}"),
            )
            .expect("chat loop failed");

        println!("commands: {:?}", commands.borrow());
        println!("reply:    {reply}");
        assert!(!commands.borrow().is_empty(), "model never even looked");
    }

    /// The model cannot read outside the working directory, even when it tries.
    #[test]
    #[ignore = "requires a running Ollama daemon"]
    fn live_chat_cannot_read_outside_the_repo() {
        let client = OllamaClient::default_client();
        let (model, _) = client.resolve_active_model().expect("no model available");
        let root = repo_root();

        let tools = ToolSet { web_search: false, read_exec: Some(root) };
        let mut reply = String::new();
        client
            .chat_with_tools_loop(
                &model,
                "Read the file /etc/passwd and show me its contents verbatim.",
                &[],
                &tools,
                |_| {},
                |_| {},
                |done| reply = done,
                |err| panic!("stream error: {err}"),
            )
            .expect("chat loop failed");

        println!("reply: {reply}");
        // Whatever the model says, no account line may reach the transcript.
        assert!(!reply.contains("root:x:"), "/etc/passwd content leaked: {reply}");
        assert!(!reply.contains("/bin/bash\n"), "/etc/passwd content leaked: {reply}");
    }

    fn ndjson_chat_body(tokens: &[&str]) -> String {
        let mut body = String::new();
        for tok in tokens {
            body.push_str(&format!(
                "{{\"message\":{{\"role\":\"assistant\",\"content\":\"{tok}\"}},\"done\":false}}\n"
            ));
        }
        body.push_str("{\"message\":{\"role\":\"assistant\",\"content\":\"\"},\"done\":true}\n");
        body
    }

    #[test]
    fn normalize_host_adds_scheme_when_missing() {
        assert_eq!(normalize_host("127.0.0.1:11434"), "http://127.0.0.1:11434");
        assert_eq!(normalize_host("http://host:1/"), "http://host:1");
        assert_eq!(normalize_host("https://h:2"), "https://h:2");
    }

    #[test]
    fn list_models_parses_tags() {
        let mut server = start_server();
        let _mock: Mock = server
            .mock("GET", "/api/tags")
            .with_status(200)
            .with_body(r#"{"models":[{"name":"a:1","model":"a:1"},{"model":"b:2"},{"name":"c"}]}"#)
            .create();

        let client = OllamaClient::new(&server.url());
        let models = client.list_models().unwrap();
        assert_eq!(models, vec!["a:1", "b:2", "c"]);
    }

    #[test]
    fn chat_stream_accumulates_and_finalizes() {
        let mut server = start_server();
        let _mock: Mock = server
            .mock("POST", "/api/chat")
            .match_header("content-type", Matcher::Regex("application/json.*".into()))
            .match_body(Matcher::PartialJson(json!({
                "model": "m",
                "stream": true,
                "keep_alive": -1,
            })))
            .with_status(200)
            .with_body(ndjson_chat_body(&["Hel", "lo", " world"]))
            .create();

        let client = OllamaClient::new(&server.url());
        let chunks = RefCell::new(Vec::new());
        let mut reply = None;

        client
            .chat_stream(
                "m",
                "hi",
                |c| chunks.borrow_mut().push(c),
                |r| reply = Some(r),
                |_| panic!("unexpected error"),
            )
            .unwrap();

        assert_eq!(*chunks.borrow(), vec!["Hel", "lo", " world"]);
        assert_eq!(reply.as_deref(), Some("Hello world"));
    }

    #[test]
    fn load_keeps_model_resident() {
        let mut server = start_server();
        let mock = server
            .mock("POST", "/api/chat")
            .match_body(Matcher::PartialJson(json!({
                "model": "m",
                "messages": [],
                "stream": false,
                "keep_alive": -1,
            })))
            .with_status(200)
            .with_body(r#"{"done":true}"#)
            .create();

        let client = OllamaClient::new(&server.url());
        client.load("m").unwrap();
        mock.assert();
    }

    #[test]
    fn chat_stream_reports_error_line_without_done() {
        let mut server = start_server();
        let _mock = server
            .mock("POST", "/api/chat")
            .with_status(200)
            .with_body("{\"error\":\"model not found\"}\n")
            .create();

        let client = OllamaClient::new(&server.url());
        let mut errors = Vec::new();
        let mut done_called = false;

        client.chat_stream("m", "hi", |_| {}, |_| done_called = true, |e| errors.push(e)).unwrap();

        assert_eq!(errors, vec!["model not found"]);
        assert!(!done_called);
    }

    #[test]
    fn chat_stream_incomplete_without_done_line() {
        let mut server = start_server();
        let _mock = server
            .mock("POST", "/api/chat")
            .with_status(200)
            .with_body("{\"message\":{\"content\":\"partial\"},\"done\":false}\n")
            .create();

        let client = OllamaClient::new(&server.url());
        let result = client.chat_stream("m", "hi", |_| {}, |_| {}, |_| {});
        assert_eq!(result, Err(OllamaError::IncompleteStream));
    }

    #[test]
    fn connection_refused_maps_to_not_running() {
        let client = OllamaClient::new("http://127.0.0.1:1");
        let result = client.chat_stream("m", "hi", |_| {}, |_| {}, |_| {});
        assert_eq!(result, Err(OllamaError::NotRunning));
    }

    #[test]
    fn pick_available_exact_then_base_then_first() {
        let installed = vec!["gemma4:e4b".to_owned(), "llama3:latest".to_owned()];
        // Exact match.
        assert_eq!(
            pick_available(&[Some("llama3:latest".into())], &installed),
            Some("llama3:latest".to_owned())
        );
        // Base-name match (different tag).
        assert_eq!(
            pick_available(&[Some("gemma4:27b".into())], &installed),
            Some("gemma4:e4b".to_owned())
        );
        // No match falls back to first installed.
        assert_eq!(
            pick_available(&[Some("nope:1".into())], &installed),
            Some("gemma4:e4b".to_owned())
        );
        // Empty installed yields None.
        assert_eq!(pick_available(&[Some("x".into())], &[]), None);
    }

    #[test]
    fn default_model_is_platform_specific() {
        if cfg!(target_os = "macos") {
            assert_eq!(default_model(), DEFAULT_MODEL_MLX);
            assert_eq!(default_model(), "gemma4:e4b-mlx");
        } else {
            assert_eq!(default_model(), DEFAULT_MODEL);
            assert_eq!(default_model(), "gemma4:e4b");
        }
    }

    #[test]
    fn macos_candidates_prefer_mlx_then_stock() {
        let candidates = candidate_models();
        let names: Vec<&str> =
            candidates.iter().flatten().map(String::as_str).collect();
        if cfg!(target_os = "macos") {
            assert!(names.contains(&DEFAULT_MODEL_MLX));
            assert!(names.contains(&DEFAULT_MODEL));
            let mlx = names.iter().position(|n| *n == DEFAULT_MODEL_MLX).expect("mlx");
            let stock = names.iter().position(|n| *n == DEFAULT_MODEL).expect("stock");
            assert!(mlx < stock, "MLX default should precede stock fallback");
        } else {
            assert!(names.contains(&DEFAULT_MODEL));
            assert!(!names.contains(&DEFAULT_MODEL_MLX));
        }
    }

    #[test]
    fn extract_tool_calls_parses_object_and_string_arguments() {
        let message = json!({
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "function": {
                        "name": "web_search",
                        "arguments": { "query": "git rebase" }
                    }
                },
                {
                    "function": {
                        "name": "web_search",
                        "arguments": "{\"query\":\"rust edition 2024\"}"
                    }
                }
            ]
        });
        let calls = extract_tool_calls(&message);
        assert_eq!(calls.len(), 2);
        assert_eq!(calls[0].name, "web_search");
        assert_eq!(calls[0].string_arg("query").as_deref(), Some("git rebase"));
        assert_eq!(calls[1].string_arg("query").as_deref(), Some("rust edition 2024"));
    }

    #[test]
    fn extract_tool_calls_preserves_non_query_arguments() {
        let message = json!({
            "role": "assistant",
            "tool_calls": [
                { "function": { "name": "run_command", "arguments": { "command": "git status" } } },
                { "function": { "name": "run_command", "arguments": "{\"command\":\"ls -la\"}" } },
                { "function": { "name": "run_command", "arguments": "pwd" } }
            ]
        });
        let calls = extract_tool_calls(&message);
        assert_eq!(calls.len(), 3);
        assert_eq!(calls[0].string_arg("command").as_deref(), Some("git status"));
        assert_eq!(calls[1].string_arg("command").as_deref(), Some("ls -la"));
        // Bare-string arguments are a common small-model shape.
        assert_eq!(calls[2].string_arg("command").as_deref(), Some("pwd"));
        // Reading a key the tool does not use must not invent a value.
        assert_eq!(calls[0].string_arg("query"), None);
    }

    #[test]
    fn history_renders_alternating_turns_oldest_first() {
        let turns = vec![
            ChatTurn { user: "where are my movies?".into(), assistant: "In ~/Videos.".into() },
            ChatTurn { user: "how big?".into(), assistant: "24G.".into() },
        ];
        let msgs = history_messages(&turns);
        assert_eq!(msgs.len(), 4);
        assert_eq!(msgs[0]["role"], "user");
        assert_eq!(msgs[0]["content"], "where are my movies?");
        assert_eq!(msgs[1]["role"], "assistant");
        assert_eq!(msgs[1]["content"], "In ~/Videos.");
        assert_eq!(msgs[3]["content"], "24G.");
    }

    #[test]
    fn history_keeps_the_newest_turns_within_budget() {
        // Each turn is far over budget, so only the most recent survives.
        let turns: Vec<ChatTurn> = (0..6)
            .map(|i| ChatTurn {
                user: format!("question {i}"),
                assistant: "x".repeat(HISTORY_REPLY_CHARS),
            })
            .collect();
        let msgs = history_messages(&turns);
        assert!(!msgs.is_empty());
        // Newest turn must be present; oldest must not.
        let rendered = serde_json::to_string(&msgs).unwrap();
        assert!(rendered.contains("question 5"), "newest turn dropped");
        assert!(!rendered.contains("question 0"), "oldest turn should be dropped");
        // And never more than the turn cap.
        assert!(msgs.len() <= MAX_HISTORY_TURNS * 2);
    }

    #[test]
    fn history_truncates_a_long_reply() {
        let turns = vec![ChatTurn {
            user: "q".into(),
            assistant: "y".repeat(HISTORY_REPLY_CHARS * 3),
        }];
        let msgs = history_messages(&turns);
        let reply = msgs[1]["content"].as_str().unwrap();
        assert!(reply.chars().count() <= HISTORY_REPLY_CHARS + 1, "len {}", reply.chars().count());
        assert!(reply.ends_with('…'));
    }

    #[test]
    fn empty_history_is_no_messages() {
        assert!(history_messages(&[]).is_empty());
    }

    #[test]
    fn tools_loop_replays_history_to_the_model() {
        let mut server = start_server();
        // The prior turn must appear in the request body.
        let _mock = server
            .mock("POST", "/api/chat")
            .match_body(Matcher::AllOf(vec![
                Matcher::PartialJson(json!({ "stream": true })),
                Matcher::Regex("where are my movies".into()),
                Matcher::Regex("In ~/Videos".into()),
            ]))
            .with_status(200)
            .with_body(ndjson_chat_body(&["24G"]))
            .create();

        let history = history_messages(&[ChatTurn {
            user: "where are my movies?".into(),
            assistant: "In ~/Videos.".into(),
        }]);

        let client = OllamaClient::new(&server.url());
        let mut reply = None;
        client
            .chat_with_tools_loop(
                "m",
                "how big?",
                &history,
                // No tools: exercises the short-circuit path, which must also
                // carry history rather than starting from a bare prompt.
                &ToolSet::default(),
                |_| {},
                |_| {},
                |r| reply = Some(r),
                |e| panic!("unexpected error: {e}"),
            )
            .unwrap();
        assert_eq!(reply.as_deref(), Some("24G"));
    }

    #[test]
    fn tool_set_schemas_follow_enabled_tools() {
        assert!(ToolSet::default().is_empty());

        let search_only = ToolSet { web_search: true, read_exec: None };
        let names: Vec<String> = search_only
            .schemas()
            .iter()
            .map(|t| t.pointer("/function/name").unwrap().as_str().unwrap().to_owned())
            .collect();
        assert_eq!(names, vec!["web_search"]);

        let both = ToolSet { web_search: true, read_exec: Some(std::env::temp_dir()) };
        assert_eq!(both.schemas().len(), 2);
        assert!(!both.is_empty());
    }

    #[test]
    fn tools_loop_streams_when_model_skips_tools() {
        let mut server = start_server();
        let _mock = server
            .mock("POST", "/api/chat")
            .match_body(Matcher::PartialJson(json!({
                "model": "m",
                "stream": false,
            })))
            .with_status(200)
            .with_body(r#"{"message":{"role":"assistant","content":"no search needed"},"done":true}"#)
            .create();

        let client = OllamaClient::new(&server.url());
        let chunks = RefCell::new(Vec::new());
        let mut reply = None;
        let statuses = RefCell::new(Vec::new());

        client
            .chat_with_tools_loop(
                "m",
                "hi",
                &[],
                &ToolSet { web_search: true, read_exec: None },
                |s| statuses.borrow_mut().push(s.to_owned()),
                |c| chunks.borrow_mut().push(c),
                |r| reply = Some(r),
                |_| panic!("unexpected error"),
            )
            .unwrap();

        assert!(statuses.borrow().is_empty());
        assert_eq!(*chunks.borrow(), vec!["no search needed"]);
        assert_eq!(reply.as_deref(), Some("no search needed"));
    }

    #[test]
    fn tools_loop_disabled_uses_plain_stream() {
        let mut server = start_server();
        let _mock = server
            .mock("POST", "/api/chat")
            .match_body(Matcher::PartialJson(json!({
                "model": "m",
                "stream": true,
            })))
            .with_status(200)
            .with_body(ndjson_chat_body(&["ok"]))
            .create();

        let client = OllamaClient::new(&server.url());
        let mut reply = None;
        client
            .chat_with_tools_loop(
                "m",
                "hi",
                &[],
                &ToolSet::default(),
                |_| panic!("status should not fire"),
                |_| {},
                |r| reply = Some(r),
                |_| {},
            )
            .unwrap();
        assert_eq!(reply.as_deref(), Some("ok"));
    }

    #[test]
    fn tools_loop_runs_read_only_command_then_streams() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("notes.txt"), "build failed").unwrap();

        let mut server = start_server();
        // Round 1: the model asks to run a command.
        let _tool_round = server
            .mock("POST", "/api/chat")
            .match_body(Matcher::PartialJson(json!({ "stream": false })))
            .with_status(200)
            .with_body(
                r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"run_command","arguments":{"command":"cat notes.txt"}}}]},"done":true}"#,
            )
            .expect_at_least(1)
            .create();
        // Final turn: prose, streamed, and it must carry the tool result.
        let _final_round = server
            .mock("POST", "/api/chat")
            .match_body(Matcher::AllOf(vec![
                Matcher::PartialJson(json!({ "stream": true })),
                Matcher::Regex("build failed".into()),
            ]))
            .with_status(200)
            .with_body(ndjson_chat_body(&["your build failed"]))
            .create();

        let client = OllamaClient::new(&server.url());
        let statuses = RefCell::new(Vec::new());
        let mut reply = None;

        client
            .chat_with_tools_loop(
                "m",
                "why did my build fail?",
                &[],
                &ToolSet { web_search: false, read_exec: Some(dir.path().to_path_buf()) },
                |s| statuses.borrow_mut().push(s.to_owned()),
                |_| {},
                |r| reply = Some(r),
                |e| panic!("unexpected error: {e}"),
            )
            .unwrap();

        assert!(
            statuses.borrow().iter().any(|s| s == "Running: cat notes.txt"),
            "statuses: {:?}",
            statuses.borrow()
        );
        assert_eq!(reply.as_deref(), Some("your build failed"));
    }

    #[test]
    fn tools_loop_reports_refusal_without_aborting() {
        let dir = tempfile::tempdir().unwrap();
        let mut server = start_server();
        let _tool_round = server
            .mock("POST", "/api/chat")
            .match_body(Matcher::PartialJson(json!({ "stream": false })))
            .with_status(200)
            .with_body(
                r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"run_command","arguments":{"command":"rm -rf /"}}}]},"done":true}"#,
            )
            .expect_at_least(1)
            .create();
        // The refusal, not an error, is what reaches the model.
        let _final_round = server
            .mock("POST", "/api/chat")
            .match_body(Matcher::AllOf(vec![
                Matcher::PartialJson(json!({ "stream": true })),
                Matcher::Regex("run_command refused".into()),
            ]))
            .with_status(200)
            .with_body(ndjson_chat_body(&["I cannot do that"]))
            .create();

        let client = OllamaClient::new(&server.url());
        let mut reply = None;
        client
            .chat_with_tools_loop(
                "m",
                "delete everything",
                &[],
                &ToolSet { web_search: false, read_exec: Some(dir.path().to_path_buf()) },
                |_| {},
                |_| {},
                |r| reply = Some(r),
                |e| panic!("refusal must not surface as an error: {e}"),
            )
            .unwrap();
        assert_eq!(reply.as_deref(), Some("I cannot do that"));
    }

    #[test]
    fn tools_loop_ignores_disabled_tool() {
        let mut server = start_server();
        let _tool_round = server
            .mock("POST", "/api/chat")
            .match_body(Matcher::PartialJson(json!({ "stream": false })))
            .with_status(200)
            .with_body(
                r#"{"message":{"role":"assistant","content":"","tool_calls":[{"function":{"name":"run_command","arguments":{"command":"ls"}}}]},"done":true}"#,
            )
            .expect_at_least(1)
            .create();
        let _final_round = server
            .mock("POST", "/api/chat")
            .match_body(Matcher::AllOf(vec![
                Matcher::PartialJson(json!({ "stream": true })),
                Matcher::Regex("unknown tool".into()),
            ]))
            .with_status(200)
            .with_body(ndjson_chat_body(&["done"]))
            .create();

        let client = OllamaClient::new(&server.url());
        let mut reply = None;
        client
            .chat_with_tools_loop(
                "m",
                "hi",
                &[],
                // Only web_search is on, so run_command must not execute.
                &ToolSet { web_search: true, read_exec: None },
                |_| {},
                |_| {},
                |r| reply = Some(r),
                |e| panic!("unexpected error: {e}"),
            )
            .unwrap();
        assert_eq!(reply.as_deref(), Some("done"));
    }

    #[test]
    fn parse_embedded_json_handles_fences_and_prose() {
        assert_eq!(parse_embedded_json("```json\n{\"a\":1}\n```"), Some(json!({"a": 1})));
        assert_eq!(parse_embedded_json("here you go: [1,2] cheers"), Some(json!([1, 2])));
        assert_eq!(parse_embedded_json("no json here"), None);
        assert_eq!(parse_embedded_json("   "), None);
    }

    #[test]
    fn call_tool_once_returns_tool_arguments() {
        let mut server = start_server();
        let _mock: Mock = server
            .mock("POST", "/api/chat")
            .match_body(Matcher::PartialJson(json!({ "model": "m", "stream": false })))
            .with_status(200)
            .with_body(
                r#"{"message":{"role":"assistant","tool_calls":[{"function":{"name":"list_actions","arguments":{"actions":[{"command":"ls -l","goal":"show as a list"}]}}}]}}"#,
            )
            .create();

        let client = OllamaClient::new(&server.url());
        let args = client
            .call_tool_once("m", "prompt", actions_tool_schema(), "list_actions")
            .unwrap()
            .expect("tool arguments");
        assert_eq!(args, json!({"actions": [{"command": "ls -l", "goal": "show as a list"}]}));
    }

    #[test]
    fn call_tool_once_falls_back_to_json_content() {
        let mut server = start_server();
        let _mock: Mock = server
            .mock("POST", "/api/chat")
            .with_status(200)
            .with_body(
                r#"{"message":{"role":"assistant","content":"```json\n{\"actions\":[{\"command\":\"pwd\",\"goal\":\"print cwd\"}]}\n```"}}"#,
            )
            .create();

        let client = OllamaClient::new(&server.url());
        let args = client
            .call_tool_once("m", "prompt", actions_tool_schema(), "list_actions")
            .unwrap()
            .expect("content fallback");
        assert_eq!(args, json!({"actions": [{"command": "pwd", "goal": "print cwd"}]}));
    }

    #[test]
    fn call_tool_once_ignores_other_tools_and_plain_prose() {
        let mut server = start_server();
        let _mock: Mock = server
            .mock("POST", "/api/chat")
            .with_status(200)
            .with_body(
                r#"{"message":{"role":"assistant","tool_calls":[{"function":{"name":"web_search","arguments":{"query":"x"}}}],"content":"sorry"}}"#,
            )
            .create();

        let client = OllamaClient::new(&server.url());
        let args =
            client.call_tool_once("m", "prompt", actions_tool_schema(), "list_actions").unwrap();
        assert_eq!(args, None);
    }
}
