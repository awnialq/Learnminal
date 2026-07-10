"""Tests for per-manager installed package enumeration."""
from unittest.mock import patch

from packages import (
    _cap,
    _probe_pacman,
    collect_installed_packages,
    summarize_for_prompt,
    total_package_count,
)


def test_cap_dedupes_and_limits():
    names = ["vim", "VIM", "git", "nano", "curl"] * 200
    capped = _cap(names, limit=3)
    assert len(capped) == 3
    assert capped == sorted(capped, key=str.lower)
    assert len({n.lower() for n in capped}) == 3


def test_summarize_for_prompt_truncates():
    data = {"pacman": [f"pkg{i}" for i in range(100)]}
    text = summarize_for_prompt(data, max_per_manager=5)
    assert "pacman (100)" in text
    assert "+95 more" in text


def test_collect_installed_packages_invokes_probes():
    with patch.dict(
        "packages._MANAGER_PROBES",
        {"pacman": lambda: ["vim", "git"]},
        clear=False,
    ):
        result = collect_installed_packages(["pacman", "unknown-tool"])
    assert result["pacman"] == ["vim", "git"]


def test_probe_pacman_parses_output():
    with patch("packages._run_command", return_value=(0, "vim\ngit\n")):
        assert _probe_pacman() == ["git", "vim"]


def test_total_package_count():
    assert total_package_count({"a": ["x"], "b": ["y", "z"]}) == 3
