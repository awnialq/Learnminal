"""
Task 6.4 — End-to-end integration test with real Ollama.

Requires both the backend server and Ollama to be running:
  Terminal 1: cd ai-backend && python3 server.py
  Terminal 2: ollama serve   (if not already running)

Run with:  pytest test_e2e.py -v -m integration
Skipped automatically when the backend is unreachable.
"""
import json
import time
import uuid

import pytest
import requests

BACKEND = "http://127.0.0.1:8765"

_REBASE_PAYLOAD = {
    "request_id": str(uuid.uuid4()),
    "timestamp": int(time.time()),
    "terminal": {
        "visible_text": (
            "$ git rebase -i HEAD~3\n"
            "error: could not apply abc1234... add feature\n"
            "Resolve all conflicts manually, mark them as resolved with\n"
            "'git add <conflicted_files>', then run 'git rebase --continue'.\n"
            "Conflict in src/main.py"
        ),
        "selected_text": None,
        "last_command": "git rebase -i HEAD~3",
        "cwd": "/home/user/project",
        "exit_code": 1,
        "rows": 24,
        "cols": 80,
    },
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _backend_alive() -> bool:
    try:
        return requests.get(f"{BACKEND}/health", timeout=2).status_code == 200
    except Exception:
        return False


def _parse_sse(raw: str) -> list[dict]:
    """Return a list of parsed event dicts from a raw SSE response body."""
    events = []
    for line in raw.splitlines():
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


@pytest.fixture(scope="module")
def backend():
    if not _backend_alive():
        pytest.skip("Backend not running at 127.0.0.1:8765 — start with: python3 server.py")


# ── tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_e2e_sse_structure(backend):
    """Req 3.1, 3.2, 5.6: stream has ≥1 chunks, one structured-done, [DONE] last."""
    resp = requests.post(
        f"{BACKEND}/explain",
        json=_REBASE_PAYLOAD,
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=120,
    )
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"

    events = _parse_sse(resp.text)
    assert events, "No SSE events received"

    # Req 5.6: [DONE] must be the absolute last event
    assert events[-1]["_type"] == "sentinel", (
        f"Last event was not [DONE]: {events[-1]}"
    )

    # Skip the structural assertions if Ollama returned an error
    if any(e["_type"] == "error" for e in events):
        pytest.skip("Ollama returned an error — verify qwen3.6:27b-mlx is running via: ollama serve")

    # Req 3.2: at least one chunk before structured-done
    chunks = [e for e in events if e["_type"] == "chunk"]
    assert len(chunks) >= 1, "Expected at least one chunk event"

    # Req 3.1: exactly one structured-done event
    done_events = [e for e in events if e["_type"] == "structured_done"]
    assert len(done_events) == 1, f"Expected exactly one structured-done, got {len(done_events)}"


@pytest.mark.integration
def test_e2e_structured_response_fields(backend):
    """Req 3.3: structured-done contains all IPC contract fields."""
    resp = requests.post(
        f"{BACKEND}/explain",
        json=_REBASE_PAYLOAD,
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=120,
    )
    events = _parse_sse(resp.text)
    done_events = [e for e in events if e["_type"] == "structured_done"]

    if not done_events:
        pytest.skip("No structured-done event — Ollama may not be running")

    structured = done_events[0]["structured"]
    required = {
        "command_name", "flags_explained", "general_utility",
        "contextual_usage", "error_fix", "similar_commands", "tool_calls_made",
    }
    missing = required - set(structured.keys())
    assert not missing, f"Structured response missing fields: {missing}"


@pytest.mark.integration
def test_e2e_tool_calls_populated(backend):
    """Req 6.6: tool_calls_made reflects which tools ran for this request."""
    resp = requests.post(
        f"{BACKEND}/explain",
        json=_REBASE_PAYLOAD,
        headers={"Accept": "text/event-stream"},
        stream=True,
        timeout=120,
    )
    events = _parse_sse(resp.text)
    done_events = [e for e in events if e["_type"] == "structured_done"]

    if not done_events:
        pytest.skip("No structured-done event — Ollama may not be running")

    tool_calls = done_events[0]["structured"].get("tool_calls_made", [])
    # last_command is non-empty → syntax_explanation must have run
    assert "syntax_explanation" in tool_calls, (
        f"syntax_explanation missing from tool_calls_made: {tool_calls}"
    )
    # exit_code=1 → error_remediation must have run
    assert "error_remediation" in tool_calls, (
        f"error_remediation missing from tool_calls_made: {tool_calls}"
    )
