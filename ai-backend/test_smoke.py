"""
Tasks 8.1 + 8.2 — Integration smoke test with mock Ollama stub.

Starts the full HTTP stack (FastAPI + real Database) but replaces the Ollama
AsyncClient with a stub that returns canned tokens, so no real model is needed.

Validates: Requirements 5.4, 5.6, 5.7, 13.1, 13.2
"""
import json
from unittest.mock import patch

import ollama
import pytest
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

from server import app

client = TestClient(app)


def test_system_info_endpoint():
    response = client.get("/system-info")
    assert response.status_code == 200
    data = response.json()
    assert "os" in data
    assert data["os"]
    assert "shell" in data
    assert isinstance(data.get("package_managers", []), list)
    assert isinstance(data.get("installed_packages", {}), dict)
    assert isinstance(data.get("installed_tools", []), list)
    assert "collected_at_display" in data


def test_system_info_auto_refreshes_incomplete_cache(monkeypatch):
    """GET /system-info collects when the DB cache is empty instead of returning {}."""
    import server as srv

    monkeypatch.setattr(srv._db, "get_all_constants", lambda: {})
    monkeypatch.setattr(srv._db, "needs_system_refresh", lambda: False)
    srv._system_info = {}

    response = client.get("/system-info")
    assert response.status_code == 200
    data = response.json()
    assert data.get("os")
    assert data.get("collected_at") is not None
    assert srv._system_info.get("os")


# ── request fixture ───────────────────────────────────────────────────────────

_REBASE_REQUEST = {
    "request_id": "smoke-test-id-0000",
    "timestamp": 1700000000,
    "terminal": {
        "visible_text": "$ git rebase -i HEAD~3\nerror: could not apply abc1234",
        "selected_text": None,
        "last_command": "git rebase -i HEAD~3",
        "cwd": "/home/user/project",
        "exit_code": 1,
        "rows": 24,
        "cols": 80,
    },
}

_CANNED_JSON = {
    "command_name": "git rebase",
    "flags_explained": [{"flag": "-i", "meaning": "Interactive mode", "example": "git rebase -i HEAD~3"}],
    "general_utility": "Reapplies commits on top of another base tip.",
    "contextual_usage": "User is interactively rebasing the last three commits.",
    "error_fix": "Resolve conflicts, then run `git rebase --continue`.",
    "similar_commands": [],
    "tool_calls_made": [],
}


# ── Ollama stubs ──────────────────────────────────────────────────────────────

def _make_canned_generate():
    """Stub that streams _CANNED_JSON as 20-char token chunks."""
    raw = json.dumps(_CANNED_JSON)
    tokens = [raw[i:i + 20] for i in range(0, len(raw), 20)]

    async def stub(self, **kwargs):
        async def _gen():
            for t in tokens:
                yield {"response": t}
        return _gen()

    return stub


def _make_mid_stream_error_generate():
    """Stub that yields one token then simulates Ollama dying mid-stream."""
    async def stub(self, **kwargs):
        async def _gen():
            yield {"response": '{"partial":'}
            raise ConnectionError("Ollama process died mid-stream")
        return _gen()

    return stub


def _make_connection_refused_generate():
    """Stub that raises ConnectionError immediately (Ollama not running)."""
    async def stub(self, **kwargs):
        raise ConnectionError("Connection refused")

    return stub


# ── helpers ───────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def reset_sse_state():
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None
    yield


def _parse_sse(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        if not line.startswith("data: "):
            continue
        payload = line[6:]
        if payload == "[DONE]":
            events.append({"_type": "sentinel"})
        else:
            try:
                data = json.loads(payload)
                if "text" in data:
                    events.append({"_type": "chunk", **data})
                elif data.get("done") and "structured" in data:
                    events.append({"_type": "structured_done", **data})
                elif data.get("done") and "error" in data:
                    events.append({"_type": "error", **data})
                else:
                    events.append({"_type": "unknown", "raw": payload})
            except json.JSONDecodeError:
                events.append({"_type": "unknown", "raw": payload})
    return events


# ══════════════════════════════════════════════════════════════════════════════
# Task 8.2 — Smoke tests: successful flow end-to-end
# ══════════════════════════════════════════════════════════════════════════════

def test_smoke_status_200():
    """POST /explain returns HTTP 200 for a valid request."""
    with patch.object(ollama.AsyncClient, "generate", _make_canned_generate()):
        resp = client.post("/explain", json=_REBASE_REQUEST)
    assert resp.status_code == 200


def test_smoke_sse_structure():
    """Req 5.6, 3.2: ≥1 chunks → one structured-done → [DONE] last."""
    with patch.object(ollama.AsyncClient, "generate", _make_canned_generate()):
        resp = client.post("/explain", json=_REBASE_REQUEST)

    events = _parse_sse(resp.text)
    assert events, "No SSE events received"
    assert events[-1]["_type"] == "sentinel", f"Last event was not [DONE]: {events[-1]}"

    chunks = [e for e in events if e["_type"] == "chunk"]
    assert len(chunks) >= 1, "Expected at least one chunk event (Req 3.2)"

    done_events = [e for e in events if e["_type"] == "structured_done"]
    assert len(done_events) == 1, f"Expected exactly one structured-done, got {len(done_events)}"


def test_smoke_structured_fields():
    """Req 13.1: structured-done event contains all IPC contract fields."""
    with patch.object(ollama.AsyncClient, "generate", _make_canned_generate()):
        resp = client.post("/explain", json=_REBASE_REQUEST)

    events = _parse_sse(resp.text)
    done_events = [e for e in events if e["_type"] == "structured_done"]
    assert done_events, "No structured-done event"

    structured = done_events[0]["structured"]
    required = {
        "command_name", "flags_explained", "general_utility",
        "contextual_usage", "error_fix", "similar_commands", "tool_calls_made",
    }
    missing = required - set(structured.keys())
    assert not missing, f"IPC contract fields missing from structured response: {missing}"


def test_smoke_tool_calls_populated():
    """Req 6.6: tool_calls_made contains syntax_explanation and error_remediation
    for a request with non-empty last_command and non-zero exit_code."""
    with patch.object(ollama.AsyncClient, "generate", _make_canned_generate()):
        resp = client.post("/explain", json=_REBASE_REQUEST)

    events = _parse_sse(resp.text)
    done_events = [e for e in events if e["_type"] == "structured_done"]
    assert done_events
    tool_calls = done_events[0]["structured"].get("tool_calls_made", [])
    assert "syntax_explanation" in tool_calls, f"syntax_explanation missing: {tool_calls}"
    assert "error_remediation" in tool_calls, f"error_remediation missing: {tool_calls}"


def test_smoke_chunk_index_monotonic():
    """Req 3.1: chunk_index is monotonically increasing across all chunk events."""
    with patch.object(ollama.AsyncClient, "generate", _make_canned_generate()):
        resp = client.post("/explain", json=_REBASE_REQUEST)

    chunks = [e for e in _parse_sse(resp.text) if e["_type"] == "chunk"]
    indices = [c["chunk_index"] for c in chunks]
    assert indices == list(range(len(indices))), f"chunk_index not monotonic: {indices}"


# ══════════════════════════════════════════════════════════════════════════════
# Task 8.1 — Error scenario tests
# ══════════════════════════════════════════════════════════════════════════════

def test_8_1_done_last_on_mid_stream_ollama_death():
    """Req 5.6: [DONE] is still the last event when Ollama dies mid-stream."""
    with patch.object(ollama.AsyncClient, "generate", _make_mid_stream_error_generate()):
        resp = client.post("/explain", json=_REBASE_REQUEST)

    events = _parse_sse(resp.text)
    assert events, "No SSE events received"
    assert events[-1]["_type"] == "sentinel", f"[DONE] not last on mid-stream error: {events[-1]}"

    error_events = [e for e in events if e["_type"] == "error"]
    assert error_events, "Expected an error event when Ollama dies mid-stream"


def test_8_1_done_last_on_ollama_not_running():
    """Req 5.4, 5.6: OllamaConnectionError → human-readable error + [DONE] last."""
    with patch.object(ollama.AsyncClient, "generate", _make_connection_refused_generate()):
        resp = client.post("/explain", json=_REBASE_REQUEST)

    events = _parse_sse(resp.text)
    assert events[-1]["_type"] == "sentinel", f"[DONE] not last on connection refused: {events[-1]}"

    error_events = [e for e in events if e["_type"] == "error"]
    assert error_events, "Expected error event for connection refused"
    assert "ollama" in error_events[0]["error"].lower(), (
        f"Expected Ollama hint in error message: {error_events[0]['error']!r}"
    )


def test_8_1_done_last_on_parse_error():
    """Req 5.6, 11.5: ParseError path emits structured-done fallback then [DONE]."""
    async def garbled_generate(self, **kwargs):
        async def _gen():
            yield {"response": "this is not JSON at all }{{{"}
        return _gen()

    with patch.object(ollama.AsyncClient, "generate", garbled_generate):
        resp = client.post("/explain", json=_REBASE_REQUEST)

    events = _parse_sse(resp.text)
    assert events[-1]["_type"] == "sentinel", f"[DONE] not last on parse error: {events[-1]}"
    # Falls back to structured-done (not an error event) — Req 11.5
    done_events = [e for e in events if e["_type"] == "structured_done"]
    assert done_events, "ParseError should fall back to structured-done, not error event"
