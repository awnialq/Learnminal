use std::error::Error as StdError;
use std::io::{BufRead, BufReader};
use std::time::{SystemTime, UNIX_EPOCH};

use reqwest::blocking::Client;
use reqwest::Url;
use serde::Deserialize;
use uuid::Uuid;

use log::warn;

use crate::learnminal::types::{
    parse_chat_done_event, parse_structured_done_event, ChatDoneEvent, ChatRequest,
    CommandReferenceResponse,
    ExplainRequest, ExplainResponse, StreamChunk, SystemInfo, TerminalContext,
};

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:8765";
const CONNECT_TIMEOUT_SECS: u64 = 30;
const READ_TIMEOUT_SECS: u64 = 120;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IpcError {
    BackendNotRunning,
    Timeout,
    StreamError(String),
    /// Tokens were received but no parseable structured `done` event arrived.
    IncompleteStream,
}

pub struct IpcClient {
    base_url: String,
    client: Client,
}

impl IpcClient {
    pub fn new(base_url: &str) -> Self {
        let client = Client::builder()
            .connect_timeout(std::time::Duration::from_secs(CONNECT_TIMEOUT_SECS))
            .timeout(std::time::Duration::from_secs(READ_TIMEOUT_SECS))
            .build()
            .expect("failed to build HTTP client");

        Self { base_url: base_url.to_owned(), client }
    }

    pub fn default_client() -> Self {
        Self::new(DEFAULT_BASE_URL)
    }

    pub fn explain(
        &self,
        ctx: &TerminalContext,
        follow_up_question: Option<&str>,
        mut on_chunk: impl FnMut(StreamChunk),
        mut on_done: impl FnMut(ExplainResponse),
        mut on_error: impl FnMut(String),
    ) -> Result<(), IpcError> {
        let url = format!("{}/explain", self.base_url.trim_end_matches('/'));
        let _ = Url::parse(&url).map_err(|_| IpcError::StreamError("invalid URL".into()))?;

        let request = ExplainRequest::from_context(
            ctx.clone(),
            Uuid::new_v4().to_string(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64,
            follow_up_question.map(str::to_owned),
        );

        let response = match self.client.post(&url).json(&request).send() {
            Ok(response) => response,
            Err(err) => {
                if is_connection_refused(&err) {
                    return Err(IpcError::BackendNotRunning);
                } else if err.is_timeout() {
                    return Err(IpcError::Timeout);
                } else {
                    return Err(IpcError::StreamError(err.to_string()));
                }
            },
        };

        if !response.status().is_success() {
            return Err(IpcError::StreamError(format!("HTTP {}", response.status())));
        }

        let reader = BufReader::new(response);
        let mut chunk_count = 0usize;
        let mut done_called = false;

        for line in reader.lines() {
            let line = match line {
                Ok(line) => line,
                Err(err) => {
                    if err.kind() == std::io::ErrorKind::TimedOut {
                        return Err(IpcError::Timeout);
                    }
                    return Err(IpcError::StreamError(err.to_string()));
                },
            };

            let Some(data) = sse_data_payload(&line) else {
                continue;
            };

            if data == "[DONE]" {
                break;
            }

            if let Ok(error_event) = serde_json::from_str::<SseErrorEvent>(data) {
                on_error(error_event.error);
                return Ok(());
            }

            if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                chunk_count += 1;
                on_chunk(chunk);
                continue;
            }

            if let Some(response) = parse_structured_done_event(data) {
                on_done(response);
                done_called = true;
                continue;
            }

            if data.contains("\"structured\"") && data.contains("\"done\"") {
                warn!("Learnminal: structured done event failed to parse: {data}");
            }
        }

        if chunk_count == 0 && !done_called {
            return Err(IpcError::StreamError("empty SSE stream".into()));
        }

        if !done_called {
            return Err(IpcError::IncompleteStream);
        }

        Ok(())
    }

    /// Fetch cached system environment (`GET /system-info`).
    ///
    /// If the first response is incomplete and `refresh` was false, retries once with
    /// `?refresh=true` so `/info` does not surface a spurious error to the user.
    pub fn system_info(&self, refresh: bool) -> Result<SystemInfo, IpcError> {
        let info = self.system_info_request(refresh)?;
        if refresh || info.is_complete() {
            return Ok(info);
        }
        self.system_info_request(true)
    }

    fn system_info_request(&self, refresh: bool) -> Result<SystemInfo, IpcError> {
        let mut url = format!("{}/system-info", self.base_url.trim_end_matches('/'));
        if refresh {
            url.push_str("?refresh=true");
        }
        let _ = Url::parse(&url).map_err(|_| IpcError::StreamError("invalid URL".into()))?;

        let response = match self.client.get(&url).send() {
            Ok(response) => response,
            Err(err) => {
                if is_connection_refused(&err) {
                    return Err(IpcError::BackendNotRunning);
                } else if err.is_timeout() {
                    return Err(IpcError::Timeout);
                } else {
                    return Err(IpcError::StreamError(err.to_string()));
                }
            },
        };

        if !response.status().is_success() {
            return Err(IpcError::StreamError(format!("HTTP {}", response.status())));
        }

        response.json::<SystemInfo>().map_err(|err| IpcError::StreamError(err.to_string()))
    }

    /// Fetch formatted man/--help for a program (`POST /command-reference`).
    pub fn command_reference(&self, program: &str) -> Result<CommandReferenceResponse, IpcError> {
        let url = format!("{}/command-reference", self.base_url.trim_end_matches('/'));
        let _ = Url::parse(&url).map_err(|_| IpcError::StreamError("invalid URL".into()))?;

        #[derive(serde::Serialize)]
        struct Body<'a> {
            program: &'a str,
        }

        let response = match self.client.post(&url).json(&Body { program }).send() {
            Ok(response) => response,
            Err(err) => {
                if is_connection_refused(&err) {
                    return Err(IpcError::BackendNotRunning);
                } else if err.is_timeout() {
                    return Err(IpcError::Timeout);
                } else {
                    return Err(IpcError::StreamError(err.to_string()));
                }
            },
        };

        if !response.status().is_success() {
            return Err(IpcError::StreamError(format!("HTTP {}", response.status())));
        }

        response
            .json::<CommandReferenceResponse>()
            .map_err(|err| IpcError::StreamError(err.to_string()))
    }

    /// Conversational chat with command context (`POST /chat`, SSE).
    pub fn chat(
        &self,
        ctx: &TerminalContext,
        message: &str,
        mut on_chunk: impl FnMut(StreamChunk),
        mut on_done: impl FnMut(ChatDoneEvent),
        mut on_error: impl FnMut(String),
    ) -> Result<(), IpcError> {
        let url = format!("{}/chat", self.base_url.trim_end_matches('/'));
        let _ = Url::parse(&url).map_err(|_| IpcError::StreamError("invalid URL".into()))?;

        let request = ChatRequest::from_context(
            ctx.clone(),
            Uuid::new_v4().to_string(),
            SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_secs() as i64,
            message.to_owned(),
        );

        let response = match self.client.post(&url).json(&request).send() {
            Ok(response) => response,
            Err(err) => {
                if is_connection_refused(&err) {
                    return Err(IpcError::BackendNotRunning);
                } else if err.is_timeout() {
                    return Err(IpcError::Timeout);
                } else {
                    return Err(IpcError::StreamError(err.to_string()));
                }
            },
        };

        if !response.status().is_success() {
            return Err(IpcError::StreamError(format!("HTTP {}", response.status())));
        }

        let reader = BufReader::new(response);
        let mut chunk_count = 0usize;
        let mut done_called = false;

        for line in reader.lines() {
            let line = match line {
                Ok(line) => line,
                Err(err) => {
                    if err.kind() == std::io::ErrorKind::TimedOut {
                        return Err(IpcError::Timeout);
                    }
                    return Err(IpcError::StreamError(err.to_string()));
                },
            };

            let Some(data) = sse_data_payload(&line) else {
                continue;
            };

            if data == "[DONE]" {
                break;
            }

            if let Ok(error_event) = serde_json::from_str::<SseErrorEvent>(data) {
                if error_event.done {
                    on_error(error_event.error);
                    return Ok(());
                }
            }

            if let Ok(chunk) = serde_json::from_str::<StreamChunk>(data) {
                chunk_count += 1;
                on_chunk(chunk);
                continue;
            }

            if let Some(reply) = parse_chat_done_event(data) {
                on_done(reply);
                done_called = true;
                continue;
            }
        }

        if chunk_count == 0 && !done_called {
            return Err(IpcError::StreamError("empty SSE stream".into()));
        }

        if !done_called {
            return Err(IpcError::IncompleteStream);
        }

        Ok(())
    }
}

/// Extract the payload from an SSE `data:` line (`data: ` or `data:`).
fn sse_data_payload(line: &str) -> Option<&str> {
    let rest = line.strip_prefix("data:")?;
    let payload = rest.strip_prefix(' ').unwrap_or(rest).trim();
    if payload.is_empty() { None } else { Some(payload) }
}

#[derive(Debug, Deserialize)]
struct SseErrorEvent {
    error: String,
    done: bool,
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

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::{Matcher, Mock, ServerGuard};
    use proptest::prelude::*;
    use std::cell::RefCell;

    fn start_server() -> ServerGuard {
        mockito::Server::new()
    }

    /// Build an SSE body with `chunk_count` chunk events followed by a structured-done
    /// event and a `[DONE]` terminator.
    fn build_sse_body(chunk_count: u32) -> String {
        let mut body = String::new();
        for i in 0..chunk_count {
            body.push_str(&format!(
                "data: {{\"text\": \"tok{i}\", \"chunk_index\": {i}}}\n\n",
            ));
        }
        body.push_str(
            "data: {\"structured\": {\"command_name\": \"git\", \"flags_explained\": [], \
             \"general_utility\": \"g\", \"contextual_usage\": \"c\", \"error_fix\": null, \
             \"similar_commands\": [], \"tool_calls_made\": []}, \"done\": true}\n\n",
        );
        body.push_str("data: [DONE]\n\n");
        body
    }

    #[test]
    fn successful_stream_invokes_callbacks_in_order() {
        let mut server = start_server();
        let body = "data: {\"text\": \"hello\", \"chunk_index\": 0}\n\n\
                    data: {\"structured\": {\"command_name\": \"git\", \"flags_explained\": [], \
                    \"general_utility\": \"g\", \"contextual_usage\": \"c\", \"error_fix\": null, \
                    \"similar_commands\": [], \"tool_calls_made\": []}, \"done\": true}\n\n\
                    data: [DONE]\n\n";

        let _mock: Mock = server
            .mock("POST", "/explain")
            .match_header("content-type", Matcher::Regex("application/json.*".into()))
            .with_status(200)
            .with_body(body)
            .create();

        let client = IpcClient::new(&server.url());
        let ctx = TerminalContext::default();
        let mut chunks = Vec::new();
        let mut done = None;

        client
            .explain(
                &ctx,
                None,
                |c| chunks.push(c),
                |r| done = Some(r),
                |_| {},
            )
            .unwrap();

        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello");
        assert!(done.is_some());
    }

    #[test]
    fn connection_refused_returns_backend_not_running() {
        let client = IpcClient::new("http://127.0.0.1:1");
        let ctx = TerminalContext::default();
        let result = client.explain(&ctx, None, |_| {}, |_| {}, |_| {});
        assert_eq!(result, Err(IpcError::BackendNotRunning));
    }

    proptest! {
        // Property 7: on_chunk is called at least once before on_done on every successful
        // stream, regardless of how many chunk events the backend emits (>=1).
        #[test]
        fn property7_on_chunk_called_before_on_done(chunk_count in 1u32..32) {
            let mut server = start_server();
            let body = build_sse_body(chunk_count);
            let _mock: Mock = server
                .mock("POST", "/explain")
                .with_status(200)
                .with_body(body)
                .create();

            let client = IpcClient::new(&server.url());
            let ctx = TerminalContext::default();

            // Track total chunk count seen at the moment on_done fires.
            let chunks_before_done = RefCell::new(None::<usize>);
            let chunks_seen = RefCell::new(0usize);
            let done_count = RefCell::new(0usize);

            let result = client.explain(
                &ctx,
                None,
                |_chunk| {
                    *chunks_seen.borrow_mut() += 1;
                },
                |_response| {
                    *chunks_before_done.borrow_mut() = Some(*chunks_seen.borrow());
                    *done_count.borrow_mut() += 1;
                },
                |_err| {},
            );

            prop_assert!(result.is_ok(), "explain should succeed");
            // on_done called exactly once.
            prop_assert_eq!(*done_count.borrow(), 1);
            let observed = chunks_before_done.borrow().expect("on_done was called");
            // on_chunk fired at least once strictly before on_done.
            prop_assert!(observed >= 1, "on_chunk must fire before on_done (saw {})", observed);
            prop_assert_eq!(observed as u32, chunk_count);
        }
    }

    #[test]
    fn error_event_does_not_call_on_done() {
        let mut server = start_server();
        let body = "data: {\"error\": \"Ollama not running\", \"done\": true}\n\ndata: [DONE]\n\n";

        let _mock = server
            .mock("POST", "/explain")
            .with_status(200)
            .with_body(body)
            .create();

        let client = IpcClient::new(&server.url());
        let ctx = TerminalContext::default();
        let mut errors = Vec::new();
        let mut done_called = false;

        client
            .explain(
                &ctx,
                None,
                |_| {},
                |_| {
                    done_called = true;
                },
                |e| errors.push(e),
            )
            .unwrap();

        assert_eq!(errors, vec!["Ollama not running"]);
        assert!(!done_called);
    }
}
