//! Native Ollama HTTP client.
//!
//! Talks directly to the Ollama daemon at `http://127.0.0.1:11434` (override
//! via `OLLAMA_HOST`). Reuses the blocking `reqwest` client and `serde_json`.

use std::collections::HashSet;
use std::error::Error as StdError;
use std::io::{BufRead, BufReader};
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
const MAX_TOOL_ROUNDS: usize = 2;

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

    /// Chat with optional `web_search` tool-calling, then stream the final answer.
    ///
    /// Tool rounds are non-streaming (max [`MAX_TOOL_ROUNDS`]). The final turn
    /// streams tokens via `on_chunk` / `on_done`. `on_status` reports progress
    /// such as "Searching the web…".
    pub fn chat_with_tools_loop(
        &self,
        model: &str,
        prompt: &str,
        enable_web_search: bool,
        mut on_status: impl FnMut(&str),
        mut on_chunk: impl FnMut(String),
        mut on_done: impl FnMut(String),
        mut on_error: impl FnMut(String),
    ) -> Result<(), OllamaError> {
        if !enable_web_search {
            return self.chat_stream(model, prompt, on_chunk, on_done, on_error);
        }

        let mut messages = vec![json!({ "role": "user", "content": prompt })];
        let tools = vec![web_search_tool_schema()];

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
                if call.name != "web_search" {
                    messages.push(json!({
                        "role": "tool",
                        "content": format!("unknown tool: {}", call.name),
                    }));
                    continue;
                }
                on_status("Searching the web…");
                let query = call.query.unwrap_or_default();
                let result = crate::learnminal::web_search::search_tool_result(&query);
                messages.push(json!({
                    "role": "tool",
                    "content": result,
                }));
            }
        }

        // Final streamed answer without tools.
        self.chat_stream_messages(model, &messages, on_chunk, on_done, on_error)
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedToolCall {
    name: String,
    query: Option<String>,
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
        let query = tool_query_arg(function.get("arguments"));
        out.push(ParsedToolCall { name, query });
    }
    out
}

fn tool_query_arg(arguments: Option<&Value>) -> Option<String> {
    let Some(arguments) = arguments else {
        return None;
    };
    match arguments {
        Value::Object(map) => map
            .get("query")
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
                    .get("query")
                    .and_then(Value::as_str)
                    .map(|s| s.trim().to_owned())
                    .filter(|s| !s.is_empty());
            }
            // Bare query string.
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

    fn start_server() -> ServerGuard {
        mockito::Server::new()
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
        assert_eq!(calls[0].query.as_deref(), Some("git rebase"));
        assert_eq!(calls[1].query.as_deref(), Some("rust edition 2024"));
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
                true,
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
                false,
                |_| panic!("status should not fire"),
                |_| {},
                |r| reply = Some(r),
                |_| {},
            )
            .unwrap();
        assert_eq!(reply.as_deref(), Some("ok"));
    }
}
