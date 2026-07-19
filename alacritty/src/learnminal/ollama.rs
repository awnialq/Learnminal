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
pub const DEFAULT_MODEL: &str = "qwen3.6:35b-a3b";

const DEFAULT_HOST: &str = "http://127.0.0.1:11434";
const CONNECT_TIMEOUT_SECS: u64 = 30;
const READ_TIMEOUT_SECS: u64 = 300;
const UNLOAD_TIMEOUT_SECS: u64 = 5;
/// Keep the model resident until an explicit unload (`keep_alive: 0`).
const KEEP_ALIVE_FOREVER: i64 = -1;

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
        mut on_chunk: impl FnMut(String),
        mut on_done: impl FnMut(String),
        mut on_error: impl FnMut(String),
    ) -> Result<(), OllamaError> {
        let url = format!("{}/api/chat", self.base_url);
        let body = json!({
            "model": model,
            "messages": [{ "role": "user", "content": prompt }],
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
    /// Preference order: settings → `LEARNMINAL_OLLAMA_MODEL` → built-in
    /// default → first installed.
    pub fn resolve_active_model(&self) -> Result<(String, Vec<String>), OllamaError> {
        let installed = self.list_models()?;
        let candidates = candidate_models();
        let active = pick_available(&candidates, &installed).unwrap_or_else(|| {
            candidates.into_iter().flatten().next().unwrap_or_else(|| DEFAULT_MODEL.to_owned())
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
fn candidate_models() -> Vec<Option<String>> {
    vec![
        settings::get_preferred_model(),
        std::env::var("LEARNMINAL_OLLAMA_MODEL").ok().filter(|s| !s.trim().is_empty()),
        Some(DEFAULT_MODEL.to_owned()),
    ]
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
        let installed = vec!["qwen3:8b".to_owned(), "llama3:latest".to_owned()];
        // Exact match.
        assert_eq!(
            pick_available(&[Some("llama3:latest".into())], &installed),
            Some("llama3:latest".to_owned())
        );
        // Base-name match (different tag).
        assert_eq!(
            pick_available(&[Some("qwen3:14b".into())], &installed),
            Some("qwen3:8b".to_owned())
        );
        // No match falls back to first installed.
        assert_eq!(
            pick_available(&[Some("nope:1".into())], &installed),
            Some("qwen3:8b".to_owned())
        );
        // Empty installed yields None.
        assert_eq!(pick_available(&[Some("x".into())], &[]), None);
    }
}
