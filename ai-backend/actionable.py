"""Derive shell-ready actionable_items from structured explain responses."""

from __future__ import annotations

import re

_MAX_ITEMS = 5


def _looks_like_command(s: str) -> bool:
    s = s.strip()
    if not s or len(s) > 200:
        return False
    if " should " in s or " because " in s:
        return False
    return (
        s.startswith("$")
        or "git " in s
        or "sudo " in s
        or (s[0].isalnum() if s else False)
    )


def extract_from_prose(prose: str | None) -> list[str]:
    if not prose or not prose.strip():
        return []

    items: list[str] = []
    in_fence = False
    for line in prose.splitlines():
        trimmed = line.strip()
        if trimmed.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence and trimmed:
            items.append(trimmed)
            continue
        m = re.match(r"^\d+[\.\)]\s*(.+)$", trimmed)
        if m and _looks_like_command(m.group(1)):
            items.append(m.group(1).strip())
        elif trimmed.startswith("- ") and _looks_like_command(trimmed[2:]):
            items.append(trimmed[2:].strip())

    return _dedupe(items)


def _dedupe(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(key)
        if len(out) >= _MAX_ITEMS:
            break
    return out


def derive_actionable_items(structured: dict) -> list[str]:
    """Fill actionable_items when the LLM omits them."""
    existing = structured.get("actionable_items")
    if isinstance(existing, list) and existing:
        return _dedupe([str(x) for x in existing if str(x).strip()])

    items: list[str] = []
    error_fix = structured.get("error_fix")
    if isinstance(error_fix, str):
        items.extend(extract_from_prose(error_fix))

    flags = structured.get("flags_explained") or []
    if isinstance(flags, list):
        for entry in flags:
            if isinstance(entry, dict):
                example = entry.get("example", "")
                if isinstance(example, str) and example.strip():
                    items.append(example.strip())

    return _dedupe(items)
