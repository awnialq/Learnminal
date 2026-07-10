"""Tests for actionable_items derivation."""

from actionable import derive_actionable_items, extract_from_prose


def test_extract_from_prose_code_fence():
    text = "Run:\n```\ngit status\n```"
    items = extract_from_prose(text)
    assert items == ["git status"]


def test_derive_from_error_fix_and_flags():
    structured = {
        "command_name": "git",
        "flags_explained": [{"flag": "-a", "meaning": "all", "example": "git add -A"}],
        "error_fix": "1. git status\n2. git diff",
        "actionable_items": [],
    }
    items = derive_actionable_items(structured)
    assert "git status" in items
    assert "git add -A" in items


def test_derive_preserves_llm_items():
    structured = {"actionable_items": ["cargo build", "cargo test"]}
    assert derive_actionable_items(structured) == ["cargo build", "cargo test"]
