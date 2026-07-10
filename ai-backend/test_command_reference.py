"""Tests for command reference formatting (Command mode)."""
import pytest

from command_reference import extract_program_name, format_command_reference
from tools import ManPageResult


def test_extract_program_name_first_token():
    assert extract_program_name("git rebase -i HEAD~3") == "git"
    assert extract_program_name("  ls -la  ") == "ls"
    assert extract_program_name("") == ""


def test_format_command_reference_not_found():
    man = ManPageResult(source="none")
    out = format_command_reference(man, "nonexistent-cmd-xyz")
    assert out["source"] == "none"
    assert out["sections"][0]["name"] == "Not found"


def test_format_command_reference_sections():
    body = "NAME\n    git - the stupid content tracker\n\nSYNOPSIS\n    git [--version]\n\nDESCRIPTION\n    Git is a VCS.\n"
    man = ManPageResult(source="man", lookup="git", body=body)
    out = format_command_reference(man, "git")
    assert out["program"] == "git"
    names = [s["name"] for s in out["sections"]]
    assert "NAME" in names
    assert "SYNOPSIS" in names
