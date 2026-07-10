"""
ErrorRemediationTool: parse terminal output for known error patterns and
return actionable fix steps.

Req 8.1: always returns action_type == "error_remediation".
Req 8.2: recognises git merge conflicts and permission denied.
Req 8.3: returns partial result (root_cause only) when fix steps cannot be
         determined, rather than raising.
"""
import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field
from typing import Optional


# ── Known-pattern regexes ─────────────────────────────────────────────────────

_GIT_CONFLICT_RE = re.compile(
    r"(CONFLICT\s*[\(\[]|Automatic merge failed|merge conflict|Cannot merge)",
    re.IGNORECASE,
)
_PERMISSION_DENIED_RE = re.compile(r"permission denied", re.IGNORECASE)
_DISK_FULL_RE = re.compile(
    r"(no space left on device|disk quota exceeded)",
    re.IGNORECASE,
)
_PORT_IN_USE_RE = re.compile(
    r"(address already in use|port\b.*\bin use|bind.*failed)",
    re.IGNORECASE,
)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ActionStep:
    action_type: str
    root_cause: str = ""
    fix_steps: list = field(default_factory=list)


# ── Tool ──────────────────────────────────────────────────────────────────────

class ErrorRemediationTool:
    """Parse visible_text/stderr for known error patterns.

    Never raises.  When no known pattern is detected, the first non-empty line
    of the text becomes root_cause and fix_steps is left empty (partial result,
    Req 8.3).
    """

    def run(self, stderr: str, exit_code: Optional[int] = None) -> ActionStep:
        text = (stderr or "").strip()
        root_cause = ""
        fix_steps: list[str] = []

        if _GIT_CONFLICT_RE.search(text):
            root_cause = (
                "Git merge conflict: conflicting changes cannot be merged automatically."
            )
            fix_steps = [
                "Identify conflicted files: git status",
                "Open each conflicted file and resolve the <<<< / ==== / >>>> markers.",
                "Stage resolved files: git add <file>",
                "Complete the merge: git merge --continue",
                "Or abort the merge entirely: git merge --abort",
            ]
        elif _PERMISSION_DENIED_RE.search(text):
            root_cause = (
                "Permission denied — the current user lacks access to the resource."
            )
            fix_steps = [
                "Check ownership and permissions: ls -la <path>",
                "Run with elevated privileges: sudo <command>",
                "Or fix permissions: chmod u+rw <path>",
            ]
        elif _DISK_FULL_RE.search(text):
            root_cause = "Disk is full or quota exceeded."
            fix_steps = [
                "Check disk usage: df -h",
                "Free space by removing large or unnecessary files.",
                "Check per-user quota: quota -s",
            ]
        elif _PORT_IN_USE_RE.search(text):
            root_cause = (
                "The requested network port is already bound by another process."
            )
            fix_steps = [
                "Find the process using the port: lsof -i :<port>",
                "Kill it: kill -9 <PID>",
                "Or change your application to use a different port.",
            ]
        else:
            # Generic fallback: first non-empty line as root_cause (partial result)
            for line in text.splitlines():
                if line.strip():
                    root_cause = line.strip()
                    break

        return ActionStep(
            action_type="error_remediation",
            root_cause=root_cause,
            fix_steps=fix_steps,
        )


# ── ManPageTool ───────────────────────────────────────────────────────────────

@dataclass
class ManPageResult:
    source: str   # "man", "help", or "none"
    lookup: str = ""   # command name passed to man/--help
    body: str = ""     # full manual text (may be truncated)


_ANSI_RE = re.compile(r'\x1b\[[0-9;]*[mK]')
_BACKSPACE_RE = re.compile(r'.\x08')


class ManPageTool:
    """Fetch the full man page or --help output before the LLM runs.

    Never raises — returns source="none" on any failure.
    """

    _TIMEOUT = 5  # seconds per subprocess call
    _MAX_CHARS = 12_000  # keep prompt within context budget

    def run_program(self, program: str) -> ManPageResult:
        """Fetch man/--help for a single program name (first token only)."""
        empty = ManPageResult(source="none")
        if not program or not program.strip():
            return empty

        base_cmd = program.strip()
        text, source = self._fetch_man(base_cmd)
        lookup = base_cmd
        if not text:
            text, source = self._fetch_help(base_cmd)
        if not text:
            return empty

        return ManPageResult(
            source=source,
            lookup=lookup,
            body=self._truncate(text),
        )

    def run(self, command: str) -> ManPageResult:
        empty = ManPageResult(source="none")
        if not command or not command.strip():
            return empty

        try:
            tokens = shlex.split(command.strip())
        except ValueError:
            return empty

        if not tokens:
            return empty

        base_cmd = tokens[0]

        # For compound commands like "git commit", try "man git-commit" first.
        lookup_cmds = [base_cmd]
        if len(tokens) > 1 and not tokens[1].startswith("-"):
            lookup_cmds.insert(0, f"{base_cmd}-{tokens[1]}")

        text, source, lookup = "", "", ""
        for candidate in lookup_cmds:
            text, source = self._fetch_man(candidate)
            if text:
                lookup = candidate
                break

        if not text:
            text, source = self._fetch_help(base_cmd)
            lookup = base_cmd

        if not text:
            return empty

        return ManPageResult(
            source=source,
            lookup=lookup,
            body=self._truncate(text),
        )

    # ── private ──────────────────────────────────────────────────────────────

    def _env(self) -> dict:
        env = os.environ.copy()
        env.update({"MANPAGER": "cat", "PAGER": "cat", "MAN_POSIXLY_CORRECT": "1"})
        return env

    def _clean(self, text: str) -> str:
        text = _ANSI_RE.sub("", text)
        text = _BACKSPACE_RE.sub("", text)
        return text

    def _fetch_man(self, cmd: str) -> tuple:
        try:
            result = subprocess.run(
                ["man", "-P", "cat", cmd],
                capture_output=True,
                text=True,
                timeout=self._TIMEOUT,
                env=self._env(),
            )
            text = self._clean(result.stdout).strip()
            if len(text) > 100:
                return text, "man"
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            pass
        return "", ""

    def _fetch_help(self, cmd: str) -> tuple:
        for flag in ("--help", "-h"):
            try:
                result = subprocess.run(
                    [cmd, flag],
                    capture_output=True,
                    text=True,
                    timeout=self._TIMEOUT,
                )
                text = self._clean((result.stdout + result.stderr)).strip()
                if len(text) > 50:
                    return text, "help"
            except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
                continue
        return "", ""

    def _truncate(self, text: str) -> str:
        if len(text) <= self._MAX_CHARS:
            return text
        return text[: self._MAX_CHARS] + "\n\n... [manual truncated] ..."
