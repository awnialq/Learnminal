"""
Property 15: HTTP_Server rejects malformed ExplainRequest with HTTP 422
Validates: Requirement 13.2

Uses hypothesis to generate every non-empty subset of required fields at both
the top level and inside `terminal`, and asserts each produces a 422 response
without invoking the agent (stream handler).
"""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from fastapi.testclient import TestClient
from sse_starlette.sse import AppStatus

from server import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_sse_state():
    # sse-starlette lazily binds AppStatus.should_exit_event to the current
    # anyio event loop.  TestClient creates a fresh loop per test, so we must
    # reset the singleton to None before each test so it is recreated in the
    # correct loop.
    AppStatus.should_exit = False
    AppStatus.should_exit_event = None
    yield

# ── valid baseline fixtures ────────────────────────────────────────────────────

VALID_TERMINAL = {
    "visible_text": "$ git rebase -i HEAD~3",
    "selected_text": None,
    "last_command": "git rebase -i HEAD~3",
    "cwd": "/home/user",
    "exit_code": 1,
    "rows": 48,
    "cols": 220,
}

VALID_REQUEST = {
    "request_id": "550e8400-e29b-41d4-a716-446655440000",
    "timestamp": 1700000000,
    "terminal": VALID_TERMINAL,
}

REQUEST_REQUIRED = ["request_id", "timestamp", "terminal"]
TERMINAL_REQUIRED = ["visible_text", "selected_text", "last_command", "cwd", "exit_code", "rows", "cols"]


# ── helper ─────────────────────────────────────────────────────────────────────

def assert_validation_error(response, *, missing):
    """Assert 422 and that the response is a Pydantic error, not an SSE stream.

    A 200 SSE stream body starts with 'data:'; a 422 body is JSON with a
    'detail' key.  If the agent had been called, we would see 'data:' here.
    """
    assert response.status_code == 422, (
        f"Expected 422 for missing {missing!r}, got {response.status_code}: {response.text[:200]}"
    )
    body = response.json()
    assert "detail" in body, "422 body should be a Pydantic validation error with 'detail'"
    assert not response.text.startswith("data:"), "Agent must not have been called (SSE stream detected)"


# ── sanity check ───────────────────────────────────────────────────────────────

def test_valid_request_returns_200():
    response = client.post("/explain", json=VALID_REQUEST)
    assert response.status_code == 200


# ── property tests ─────────────────────────────────────────────────────────────

@given(st.frozensets(st.sampled_from(REQUEST_REQUIRED), min_size=1))
@settings(max_examples=20)
def test_missing_top_level_fields_returns_422(missing_fields):
    body = {**VALID_REQUEST, "terminal": {**VALID_TERMINAL}}
    for field in missing_fields:
        del body[field]
    assert_validation_error(client.post("/explain", json=body), missing=missing_fields)


@given(st.frozensets(st.sampled_from(TERMINAL_REQUIRED), min_size=1))
@settings(max_examples=50)
def test_missing_terminal_fields_returns_422(missing_fields):
    terminal = {**VALID_TERMINAL}
    for field in missing_fields:
        del terminal[field]
    body = {**VALID_REQUEST, "terminal": terminal}
    assert_validation_error(client.post("/explain", json=body), missing=missing_fields)


# ── unit tests for specific edge cases ────────────────────────────────────────

def test_empty_body_returns_422():
    assert_validation_error(client.post("/explain", json={}), missing="all fields")


def test_visible_text_over_8000_chars_returns_422():
    body = {**VALID_REQUEST, "terminal": {**VALID_TERMINAL, "visible_text": "x" * 8001}}
    assert_validation_error(client.post("/explain", json=body), missing="visible_text length")


def test_visible_text_exactly_8000_chars_passes():
    body = {**VALID_REQUEST, "terminal": {**VALID_TERMINAL, "visible_text": "x" * 8000}}
    assert client.post("/explain", json=body).status_code == 200


def test_selected_text_null_is_valid():
    """selected_text=null is the normal case when no selection is active."""
    body = {**VALID_REQUEST, "terminal": {**VALID_TERMINAL, "selected_text": None}}
    assert client.post("/explain", json=body).status_code == 200


def test_exit_code_null_is_valid():
    """exit_code=null is valid when exit code is unknown."""
    body = {**VALID_REQUEST, "terminal": {**VALID_TERMINAL, "exit_code": None}}
    assert client.post("/explain", json=body).status_code == 200


def test_health_not_affected():
    """GET /health must never return 422 regardless of what else happens."""
    assert client.get("/health").status_code == 200
