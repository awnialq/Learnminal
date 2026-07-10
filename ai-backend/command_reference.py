"""
Format man/--help output into human-readable sections for Command mode.
"""
import re
import shlex
from typing import Any

from tools import ManPageResult, ManPageTool

_SECTION_RE = re.compile(r"^([A-Z][A-Z0-9][A-Z0-9 -]{0,40})$")
_PRIORITY_SECTIONS = ("NAME", "SYNOPSIS", "DESCRIPTION", "OPTIONS", "FLAGS", "COMMANDS", "EXAMPLES")


def extract_program_name(last_command: str) -> str:
    """First token of the command line (the program name)."""
    if not last_command or not last_command.strip():
        return ""
    try:
        tokens = shlex.split(last_command.strip())
    except ValueError:
        tokens = last_command.split()
    return tokens[0] if tokens else ""


def _split_man_sections(body: str) -> list[tuple[str, list[str]]]:
    """Split classic man page text into (section_name, lines) pairs."""
    sections: list[tuple[str, list[str]]] = []
    current_name = "Reference"
    current_lines: list[str] = []

    for raw in body.splitlines():
        line = raw.rstrip()
        if _SECTION_RE.match(line.strip()):
            if current_lines:
                sections.append((current_name, current_lines))
            current_name = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_name, current_lines))

    return sections


def _wrap_lines(lines: list[str], width: int = 78) -> list[str]:
    """Soft-wrap long lines for overlay display."""
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            out.append("")
            continue
        if line.startswith((" ", "\t")) or len(stripped) <= width:
            out.append(line.rstrip())
            continue
        words = stripped.split()
        current = ""
        for word in words:
            candidate = f"{current} {word}".strip()
            if len(candidate) <= width:
                current = candidate
            else:
                if current:
                    out.append(current)
                current = word
        if current:
            out.append(current)
    return out


def _one_line_summary(body: str, program: str) -> str:
    for line in body.splitlines():
        text = line.strip()
        if text and not text.startswith("-"):
            if " - " in text:
                return text
            if text.lower().startswith(program.lower()):
                return text
    first = next((ln.strip() for ln in body.splitlines() if ln.strip()), "")
    return first or program


def format_command_reference(man: ManPageResult, program: str) -> dict[str, Any]:
    """Build overlay-friendly JSON for Command mode."""
    if man.source == "none" or not man.body.strip():
        return {
            "program": program,
            "source": "none",
            "title": program,
            "sections": [
                {
                    "name": "Not found",
                    "lines": [
                        f"No manual found for `{program}`.",
                        f"Try running `{program} --help` in your terminal.",
                    ],
                }
            ],
        }

    title_summary = _one_line_summary(man.body, program)
    title = f"{program} — {title_summary}" if title_summary != program else program

    raw_sections = _split_man_sections(man.body)
    by_name = {name: _wrap_lines(lines) for name, lines in raw_sections}

    ordered: list[dict[str, Any]] = []
    seen: set[str] = set()
    for name in _PRIORITY_SECTIONS:
        if name in by_name:
            ordered.append({"name": name, "lines": by_name[name]})
            seen.add(name)

    for name, lines in raw_sections:
        if name not in seen:
            ordered.append({"name": name, "lines": _wrap_lines(lines)})

    if not ordered:
        ordered.append({"name": "Reference", "lines": _wrap_lines(man.body.splitlines())})

    return {
        "program": program,
        "source": man.source,
        "title": title,
        "sections": ordered,
    }


def lookup_command_reference(program: str, man_tool: ManPageTool | None = None) -> dict[str, Any]:
    """Fetch and format reference for a single program name."""
    tool = man_tool or ManPageTool()
    man = tool.run_program(program)
    return format_command_reference(man, program)
