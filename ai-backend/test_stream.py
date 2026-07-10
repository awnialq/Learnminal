"""
Property 6: SSE stream always terminates with [DONE] regardless of outcome.
Property 8: Tool selection matches context invariants for any ExplainRequest.

Validates: Requirements 5.6, 6.1, 6.2, 6.3, 6.4, 6.6
"""
import asyncio
import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent import ExplainAgent
from server import _agent_stream
from tools import ActionStep


# ── shared fixtures / helpers ─────────────────────────────────────────────────

def _make_request(
    last_command: str = "ls",
    visible_text: str = "$ ls",
    exit_code=None,
    selected_text=None,
    cwd: str = "/home/user",
):
    return SimpleNamespace(
        request_id="test-id",
        timestamp=0,
        terminal=SimpleNamespace(
            last_command=last_command,
            visible_text=visible_text,
            exit_code=exit_code,
            selected_text=selected_text,
            cwd=cwd,
        ),
    )


async def _collect(gen) -> list[str]:
    """Drain an async generator of SSE dicts and return the data strings."""
    return [item["data"] async for item in gen]


def _run(gen) -> list[str]:
    return asyncio.run(_collect(gen))


def _mock_agent(tokens: list[str]) -> MagicMock:
    """Agent whose run_stream yields the given token strings."""
    captured = tokens

    async def fake_stream(_req):
        for t in captured:
            yield t

    agent = MagicMock(spec=ExplainAgent)
    agent.run_stream = fake_stream
    agent._select_tools.return_value = ([], None, None, [])
    return agent


# ═══════════════════════════════════════════════════════════════════════════════
# Property 6 — SSE always terminates with [DONE]          (Req 5.6)
# ═══════════════════════════════════════════════════════════════════════════════

# 6a: success path — arbitrary token lists
@given(st.lists(st.text(min_size=1, max_size=80), min_size=1, max_size=30))
@settings(max_examples=150)
def test_prop6_done_always_last_on_success(tokens):
    events = _run(_agent_stream(_make_request(), _mock_agent(tokens)))
    assert events[-1] == "[DONE]", f"Last event was not [DONE]: {events[-1]!r}"


# 6b: OllamaConnectionError path
def test_prop6_done_always_last_on_connection_error():
    async def fail_stream(_req):
        raise ConnectionError("Connection refused by Ollama")
        yield  # noqa: makes it an async generator

    agent = MagicMock(spec=ExplainAgent)
    agent.run_stream = fail_stream
    agent._select_tools.return_value = ([], None, None, [])

    events = _run(_agent_stream(_make_request(), agent))
    assert events[-1] == "[DONE]"
    error_event = json.loads(events[-2])
    assert "error" in error_event, f"Expected error event before [DONE]: {events[-2]!r}"


# 6c: generic exception path
@given(st.text(min_size=1, max_size=100))
@settings(max_examples=50)
def test_prop6_done_always_last_on_arbitrary_exception(msg):
    async def raising_stream(_req):
        raise RuntimeError(msg)
        yield  # noqa

    agent = MagicMock(spec=ExplainAgent)
    agent.run_stream = raising_stream
    agent._select_tools.return_value = ([], None, None, [])

    events = _run(_agent_stream(_make_request(), agent))
    assert events[-1] == "[DONE]"


# 6d: ParseError path — agent yields non-JSON tokens
def test_prop6_done_always_last_on_parse_error():
    events = _run(_agent_stream(_make_request(), _mock_agent(["this is {not json"])))
    assert events[-1] == "[DONE]"
    done_event = json.loads(events[-2])
    # Falls back to structured-done with partial data
    assert done_event.get("done") is True, f"Expected structured-done fallback: {events[-2]!r}"


# 6e: empty stream (agent yields nothing)
def test_prop6_done_always_last_on_empty_stream():
    async def empty_stream(_req):
        return
        yield  # noqa

    agent = MagicMock(spec=ExplainAgent)
    agent.run_stream = empty_stream
    agent._select_tools.return_value = ([], None, None, [])

    events = _run(_agent_stream(_make_request(), agent))
    assert events[-1] == "[DONE]"


# ═══════════════════════════════════════════════════════════════════════════════
# Property 8 — tool selection invariants                   (Req 6.1, 6.2, 6.6)
# ═══════════════════════════════════════════════════════════════════════════════

_agent_instance = ExplainAgent()

_text = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "Zs")),
    min_size=0,
    max_size=100,
)


@given(
    _text,                                              # last_command
    st.one_of(st.none(), st.integers(-255, 255)),       # exit_code
    _text,                                              # visible_text
)
@settings(max_examples=400, deadline=None)
def test_prop8_tool_selection_invariants(last_command, exit_code, visible_text):
    req = _make_request(
        last_command=last_command,
        exit_code=exit_code,
        visible_text=visible_text,
    )
    flags, error_remediation, _, tool_calls = _agent_instance._select_tools(req)

    # Invariant 1: syntax_explanation iff last_command is truthy
    if last_command:
        assert "syntax_explanation" in tool_calls, (
            f"syntax_explanation missing for last_command={last_command!r}"
        )
    else:
        assert "syntax_explanation" not in tool_calls, (
            f"syntax_explanation present for empty last_command"
        )

    # Invariant 2: error_remediation iff exit_code is non-null and non-zero
    if exit_code is not None and exit_code != 0:
        assert "error_remediation" in tool_calls, (
            f"error_remediation missing for exit_code={exit_code}"
        )
        assert error_remediation is not None
    else:
        assert "error_remediation" not in tool_calls, (
            f"error_remediation present for exit_code={exit_code}"
        )
        assert error_remediation is None

    # Invariant 3: when error_remediation ran, result has correct action_type
    if error_remediation is not None:
        assert error_remediation.action_type == "error_remediation"
        assert isinstance(error_remediation.root_cause, str)
        assert isinstance(error_remediation.fix_steps, list)

    # Invariant 4: flags is always a list, never raises
    assert isinstance(flags, list)


# ── deterministic spot-checks ─────────────────────────────────────────────────

def test_prop8_empty_command_no_syntax_tool():
    _, _, _, calls = _agent_instance._select_tools(_make_request(last_command=""))
    assert "syntax_explanation" not in calls
    assert "error_remediation" not in calls


def test_prop8_nonzero_exit_triggers_error_remediation():
    _, err, _, calls = _agent_instance._select_tools(
        _make_request(last_command="git rebase -i HEAD~3", exit_code=1)
    )
    assert "syntax_explanation" in calls
    assert "error_remediation" in calls
    assert isinstance(err, ActionStep)
    assert err.action_type == "error_remediation"


def test_prop8_zero_exit_no_error_remediation():
    _, err, _, calls = _agent_instance._select_tools(
        _make_request(last_command="ls -la", exit_code=0)
    )
    assert "syntax_explanation" in calls
    assert "error_remediation" not in calls
    assert err is None


def test_prop8_null_exit_no_error_remediation():
    _, err, _, calls = _agent_instance._select_tools(
        _make_request(last_command="ls", exit_code=None)
    )
    assert "error_remediation" not in calls
    assert err is None
