"""
Property 9:  SyntaxExplanationTool produces a FlagExplanation for every parsed flag.
Property 14: Prompt always contains all required context fields and wraps
             user-controlled content in code blocks.
Validates: Requirements 7.1, 7.2, 7.3, 10.1, 10.2, 10.3, 10.6
"""
from types import SimpleNamespace

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agent import ExplainAgent, FlagExplanation, SyntaxExplanationTool

tool = SyntaxExplanationTool()
agent = ExplainAgent()


# ── helpers ───────────────────────────────────────────────────────────────────

def make_request(
    *,
    last_command="ls",
    visible_text="$ ls",
    exit_code=None,
    selected_text=None,
    cwd="/home/user",
):
    return SimpleNamespace(
        terminal=SimpleNamespace(
            last_command=last_command,
            visible_text=visible_text,
            exit_code=exit_code,
            selected_text=selected_text,
            cwd=cwd,
        )
    )


# ── Property 9: SyntaxExplanationTool ────────────────────────────────────────
# Req 7.3: empty or whitespace-only → empty list, no exception

@given(st.just(""))
def test_prop9_empty_string_returns_empty_list(cmd):
    assert tool.run(cmd) == []


@given(st.text(alphabet=" \t\n\r", min_size=1, max_size=40))
@settings(max_examples=50)
def test_prop9_whitespace_only_returns_empty_list(cmd):
    assert tool.run(cmd) == []


# Req 7.2: no exception for ANY input string
@given(st.text(min_size=0, max_size=200))
@settings(max_examples=200)
def test_prop9_never_raises(cmd):
    result = tool.run(cmd)
    assert isinstance(result, list)
    for entry in result:
        assert isinstance(entry, FlagExplanation)


# Req 7.1: every token after the base command produces exactly one FlagExplanation
# with a non-empty `flag` field.  meaning/example are filled by the LLM later.
@given(
    st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz-_0123456789", min_size=1, max_size=12),
        min_size=2,  # base command + at least one token
        max_size=8,
    )
)
@settings(max_examples=150)
def test_prop9_multi_token_command_yields_one_entry_per_token(tokens):
    cmd = " ".join(tokens)
    result = tool.run(cmd)
    # Exactly len(tokens)-1 FlagExplanation entries (first token is base command)
    assert len(result) == len(tokens) - 1
    for entry in result:
        assert entry.flag, "flag field must be non-empty"


# Single-word command → no flags/args → empty list
@given(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=20))
@settings(max_examples=50)
def test_prop9_single_word_command_returns_empty_list(word):
    assert tool.run(word) == []


# ── Property 14: _build_context ──────────────────────────────────────────────
# Use printable ASCII to avoid Jinja2 / template edge cases with exotic unicode.
_printable = st.text(
    alphabet=st.characters(whitelist_categories=("L", "N", "P", "S", "Zs")),
    min_size=1,
    max_size=120,
)


# Req 10.1: prompt always includes visible_text, last_command, cwd, exit_code
@given(_printable, _printable, _printable,
       st.one_of(st.none(), st.integers(-255, 255)))
@settings(max_examples=100)
def test_prop14_required_fields_always_in_prompt(visible_text, last_command, cwd, exit_code):
    req = make_request(visible_text=visible_text, last_command=last_command,
                       cwd=cwd, exit_code=exit_code)
    prompt = agent._build_context(req)
    assert visible_text in prompt
    assert last_command in prompt
    assert cwd in prompt
    exit_code_str = str(exit_code) if exit_code is not None else "unknown"
    assert exit_code_str in prompt


# Req 10.6: visible_text and last_command always inside triple-backtick fences
@given(_printable, _printable)
@settings(max_examples=100)
def test_prop14_visible_text_and_last_command_in_code_blocks(visible_text, last_command):
    req = make_request(visible_text=visible_text, last_command=last_command)
    prompt = agent._build_context(req)
    assert f"```\n{visible_text}\n```" in prompt
    assert f"```\n{last_command}\n```" in prompt


# Req 10.6: selected_text inside triple-backtick fences when non-null
@given(_printable)
@settings(max_examples=100)
def test_prop14_selected_text_in_code_block_when_present(selected_text):
    req = make_request(selected_text=selected_text)
    prompt = agent._build_context(req)
    assert f"```\n{selected_text}\n```" in prompt


# Req 10.4: no selection section when selected_text is null
def test_prop14_no_selection_section_when_null():
    prompt = agent._build_context(make_request(selected_text=None))
    assert "highlighted" not in prompt


# Req 10.2: failure notice is the VERY FIRST LINE when exit_code is non-null and non-zero
@given(st.integers(-255, 255).filter(lambda x: x != 0))
@settings(max_examples=50)
def test_prop14_failure_notice_is_first_on_nonzero_exit(exit_code):
    prompt = agent._build_context(make_request(exit_code=exit_code))
    first_line = prompt.strip().splitlines()[0]
    assert "IMPORTANT" in first_line, (
        f"Failure notice must be first line for exit_code={exit_code}, got: {first_line!r}"
    )


# Req 10.2: exit_code=0 → no IMPORTANT failure notice
def test_prop14_no_failure_notice_on_zero_exit_code():
    prompt = agent._build_context(make_request(exit_code=0))
    assert "IMPORTANT" not in prompt


# Req 10.3: failure notice present when exit_code is null + failure keyword in visible_text
@pytest.mark.parametrize("keyword", [
    "error", "Error", "ERROR",
    "exception", "fatal", "Fatal",
    "traceback", "Traceback",
    "failed", "Failed",
    "permission denied",
    "command not found",
    "panic",
])
def test_prop14_failure_notice_on_keyword_with_null_exit(keyword):
    req = make_request(exit_code=None, visible_text=f"output: {keyword} in process")
    prompt = agent._build_context(req)
    assert "NOTE:" in prompt


# Req 10.3: no failure notice when exit_code is null and no keywords present
def test_prop14_no_failure_notice_without_keywords_and_null_exit():
    req = make_request(exit_code=None, visible_text="total 8\ndrwxr-xr-x 2 user group 64 Jan 1 README.md")
    prompt = agent._build_context(req)
    assert "IMPORTANT" not in prompt
    assert "NOTE:" not in prompt


# ── Snapshot tests ────────────────────────────────────────────────────────────

def test_snapshot_git_rebase_failure():
    req = make_request(
        last_command="git rebase -i HEAD~3",
        visible_text="$ git rebase -i HEAD~3\nerror: could not apply abc1234",
        exit_code=1,
        cwd="/home/user/myproject",
    )
    prompt = agent._build_context(req)
    assert prompt.strip().startswith("IMPORTANT")           # failure notice first
    assert "git rebase -i HEAD~3" in prompt                 # last_command present
    assert "/home/user/myproject" in prompt                 # cwd present
    assert "error: could not apply abc1234" in prompt       # visible_text present
    assert "```\ngit rebase -i HEAD~3\n```" in prompt       # last_command in code block
    assert "1" in prompt                                    # exit_code present


def test_snapshot_successful_ls():
    req = make_request(
        last_command="ls -la",
        visible_text="total 8\ndrwxr-xr-x 2 user group 64",
        exit_code=0,
        selected_text=None,
        cwd="/tmp",
    )
    prompt = agent._build_context(req)
    assert "IMPORTANT" not in prompt
    assert "NOTE:" not in prompt
    assert "highlighted" not in prompt
    assert "```\nls -la\n```" in prompt
    assert "/tmp" in prompt


def test_snapshot_selected_text_focus():
    req = make_request(
        last_command="git rebase",
        visible_text="$ git rebase\nerror: could not apply abc1234",
        selected_text="error: could not apply abc1234",
        exit_code=None,
    )
    prompt = agent._build_context(req)
    assert "highlighted" in prompt
    assert "```\nerror: could not apply abc1234\n```" in prompt
    # failure keywords in visible_text with null exit_code → NOTE: present
    assert "NOTE:" in prompt
