"""
Property 10: ErrorRemediationTool always returns action_type == "error_remediation"
             and at least one of root_cause/fix_steps is non-empty for non-empty stderr.
Validates: Requirements 8.1, 8.3
"""
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from tools import ActionStep, ErrorRemediationTool

tool = ErrorRemediationTool()


# ── Property 10: action_type invariant ───────────────────────────────────────
# Req 8.1: action_type is ALWAYS "error_remediation" regardless of input.

@given(
    st.text(min_size=0, max_size=500),
    st.one_of(st.none(), st.integers(-255, 255)),
)
@settings(max_examples=300)
def test_prop10_action_type_always_error_remediation(stderr, exit_code):
    result = tool.run(stderr, exit_code)
    assert result.action_type == "error_remediation"


# ── Property 10: partial-result guarantee ────────────────────────────────────
# Req 8.3: for any non-empty stderr at least one of root_cause / fix_steps is set.

@given(
    st.text(min_size=1, max_size=500).filter(lambda s: s.strip()),
    st.integers(1, 255),
)
@settings(max_examples=200)
def test_prop10_non_empty_stderr_yields_partial_result(stderr, exit_code):
    result = tool.run(stderr, exit_code)
    assert result.root_cause or result.fix_steps, (
        f"At least one of root_cause/fix_steps must be non-empty for non-empty stderr. "
        f"Got root_cause={result.root_cause!r}, fix_steps={result.fix_steps!r}"
    )


# Req 8.1, 8.3: result fields always have correct types, never raises.
@given(
    st.one_of(st.none(), st.text(max_size=1000)),
    st.one_of(st.none(), st.integers(-255, 255)),
)
@settings(max_examples=300)
def test_prop10_never_raises_and_types_correct(stderr, exit_code):
    result = tool.run(stderr or "", exit_code)
    assert result.action_type == "error_remediation"
    assert isinstance(result.root_cause, str)
    assert isinstance(result.fix_steps, list)
    for step in result.fix_steps:
        assert isinstance(step, str)


# ── Known-pattern tests (Req 8.2) ────────────────────────────────────────────

@pytest.mark.parametrize("text", [
    "CONFLICT (content): Merge conflict in README.md",
    "Automatic merge failed; fix conflicts and then commit the result.",
    "merge conflict detected in src/main.py",
])
def test_git_merge_conflict_recognized(text):
    result = tool.run(text, exit_code=1)
    assert result.root_cause
    assert result.fix_steps
    assert "conflict" in result.root_cause.lower() or "merge" in result.root_cause.lower()


@pytest.mark.parametrize("text", [
    "bash: ./script.sh: Permission denied",
    "open /etc/shadow: permission denied",
    "Error: EACCES: permission denied, open '/etc/hosts'",
])
def test_permission_denied_recognized(text):
    result = tool.run(text, exit_code=1)
    assert result.root_cause
    assert result.fix_steps
    assert "permission" in result.root_cause.lower()


# ── Edge cases ────────────────────────────────────────────────────────────────

def test_empty_stderr_returns_valid_action_step():
    result = tool.run("", exit_code=1)
    assert result.action_type == "error_remediation"
    # root_cause and fix_steps may both be empty for truly empty input — partial result

def test_none_stderr_does_not_raise():
    result = tool.run(None, exit_code=2)  # type: ignore[arg-type]
    assert result.action_type == "error_remediation"


def test_unknown_error_returns_first_line_as_root_cause():
    result = tool.run("some obscure runtime error on line 42\nmore details here")
    assert result.root_cause == "some obscure runtime error on line 42"
    # fix_steps may be empty — that's a valid partial result per Req 8.3


def test_whitespace_only_stderr_empty_root_cause():
    result = tool.run("   \n\t\n  ", exit_code=1)
    assert result.action_type == "error_remediation"
    assert result.root_cause == ""
    assert result.fix_steps == []
