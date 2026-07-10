"""
Task 7.2 — SQLite persistence after stream + ParseError file logging.
Validates: Requirements 5.7, 11.5, 12.3
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import server
from agent import ExplainAgent
from database import CommandRecord, Database
from server import _agent_stream


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_request(last_command="git rebase -i", cwd="/repo", exit_code=0):
    return SimpleNamespace(
        request_id="test-id",
        timestamp=0,
        terminal=SimpleNamespace(
            last_command=last_command,
            visible_text="$ git rebase -i",
            exit_code=exit_code,
            selected_text=None,
            cwd=cwd,
        ),
    )


async def _collect(gen) -> list[str]:
    return [item["data"] async for item in gen]


def _run(gen) -> list[str]:
    return asyncio.run(_collect(gen))


def _valid_payload():
    return {
        "command_name": "git rebase",
        "flags_explained": [],
        "general_utility": "Reapplies commits on top of another base.",
        "contextual_usage": "User is rebasing the last 3 commits.",
        "error_fix": None,
        "similar_commands": [],
        "tool_calls_made": [],
    }


def _mock_agent_json(payload: dict) -> MagicMock:
    text = json.dumps(payload)

    async def fake_stream(_req):
        yield text

    agent = MagicMock(spec=ExplainAgent)
    agent.run_stream = fake_stream
    agent._select_tools.return_value = ([], None, None, [])
    return agent


def _mock_agent_bad_json() -> MagicMock:
    async def fake_stream(_req):
        yield "this is {not valid json"

    agent = MagicMock(spec=ExplainAgent)
    agent.run_stream = fake_stream
    agent._select_tools.return_value = ([], None, None, [])
    return agent


def _mock_agent_connection_error() -> MagicMock:
    async def fail_stream(_req):
        raise ConnectionError("Connection refused")
        yield  # noqa

    agent = MagicMock(spec=ExplainAgent)
    agent.run_stream = fail_stream
    agent._select_tools.return_value = ([], None, None, [])
    return agent


# ── Req 5.7: persistence after successful stream ──────────────────────────────

def test_successful_stream_inserts_record():
    db = Database(":memory:")
    req = _make_request(last_command="git rebase -i", cwd="/repo", exit_code=0)

    _run(_agent_stream(req, _mock_agent_json(_valid_payload()), db=db))

    recent = db.get_recent(limit=1)
    assert len(recent) == 1
    r = recent[0]
    assert r.command == "git rebase -i"
    assert r.cwd == "/repo"
    assert r.exit_code == 0
    assert r.ai_general_utility == _valid_payload()["general_utility"]
    assert r.ai_context_desc == _valid_payload()["contextual_usage"]
    db.close()


def test_record_has_no_forbidden_fields():
    """Req 12.3: persisted record must never contain visible_text or selected_text."""
    import dataclasses
    db = Database(":memory:")
    _run(_agent_stream(_make_request(), _mock_agent_json(_valid_payload()), db=db))
    recent = db.get_recent(limit=1)
    assert recent
    row = dataclasses.asdict(recent[0])
    for bad in ("visible_text", "selected_text"):
        assert bad not in row, f"Forbidden field {bad!r} found in persisted record"
    db.close()


def test_parse_error_does_not_insert():
    db = Database(":memory:")
    _run(_agent_stream(_make_request(), _mock_agent_bad_json(), db=db))
    assert db.get_recent(limit=10) == [], "ParseError must not insert to DB"
    db.close()


def test_connection_error_does_not_insert():
    db = Database(":memory:")
    _run(_agent_stream(_make_request(), _mock_agent_connection_error(), db=db))
    assert db.get_recent(limit=10) == [], "Connection error must not insert to DB"
    db.close()


def test_db_none_does_not_raise():
    """db=None (default) must never crash the stream."""
    events = _run(_agent_stream(_make_request(), _mock_agent_json(_valid_payload()), db=None))
    assert events[-1] == "[DONE]"


# ── Req 11.5: ParseError writes to errors.log ─────────────────────────────────

def test_parse_error_writes_to_error_log(tmp_path):
    error_log = tmp_path / "errors.log"
    original = server._ERROR_LOG_PATH
    try:
        server._ERROR_LOG_PATH = error_log
        _run(_agent_stream(_make_request(), _mock_agent_bad_json(), db=None))
    finally:
        server._ERROR_LOG_PATH = original

    assert error_log.exists(), "errors.log must be created on ParseError"
    content = error_log.read_text()
    assert "ParseError" in content, f"Expected 'ParseError' in log: {content!r}"


def test_successful_stream_does_not_write_error_log(tmp_path):
    error_log = tmp_path / "errors.log"
    original = server._ERROR_LOG_PATH
    try:
        server._ERROR_LOG_PATH = error_log
        _run(_agent_stream(_make_request(), _mock_agent_json(_valid_payload()), db=None))
    finally:
        server._ERROR_LOG_PATH = original

    assert not error_log.exists(), "errors.log must NOT be written on successful stream"


def test_connection_error_does_not_write_error_log(tmp_path):
    error_log = tmp_path / "errors.log"
    original = server._ERROR_LOG_PATH
    try:
        server._ERROR_LOG_PATH = error_log
        _run(_agent_stream(_make_request(), _mock_agent_connection_error(), db=None))
    finally:
        server._ERROR_LOG_PATH = original

    assert not error_log.exists(), "errors.log must NOT be written on connection error"
