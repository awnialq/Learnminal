"""
ExplainAgent: orchestrates SyntaxExplanationTool calls and Ollama streaming
to produce structured terminal command explanations.
"""
import os
import re
import shlex
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional

import ollama
from jinja2 import Template

from adalflow.core import Component
from adalflow import DataClass

from tools import ActionStep, ErrorRemediationTool, ManPageResult, ManPageTool

MODEL = os.environ.get("LEARNMINAL_OLLAMA_MODEL", "qwen3.6:27b-mlx")


def _chunk_text(chunk) -> str:
    """Extract streamed text from an Ollama generate chunk (dict or response object)."""
    if isinstance(chunk, dict):
        data = chunk
    elif hasattr(chunk, "model_dump"):
        data = chunk.model_dump()
    else:
        data = {}
    # Reasoning models may stream internal tokens in `thinking`; user-facing text is in `response`.
    return data.get("response") or ""

# Keywords that suggest a failure even when exit_code is null (Req 10.3)
_FAILURE_RE = re.compile(
    r'\b(error|exception|traceback|failed|fatal|panic|'
    r'permission\s+denied|no\s+such\s+file|command\s+not\s+found|'
    r'segfault|core\s+dumped|syntax\s+error|undefined\s+reference)\b',
    re.IGNORECASE,
)

# Req 10.2: failure notice must be the VERY FIRST content of the prompt.
# Req 10.6: user-controlled fields wrapped in code blocks.
_PROMPT_TEMPLATE = Template("""\
{% if failure_notice %}\
{{ failure_notice }}

{% endif %}\
You are an expert command-line educator. Explain terminal commands clearly and
concisely to developers who want to learn.

{% if system_info %}\
User's environment — tailor all examples and suggestions to this:
  OS: {{ system_info.os }}{% if system_info.arch %} ({{ system_info.arch }}){% endif %}

  Shell: {{ system_info.shell }}
  Package managers: {{ system_info.package_managers | join(", ") if system_info.package_managers else "none detected" }}
  Installed tools: {{ system_info.installed_tools | join(", ") if system_info.installed_tools else "none detected" }}
{% if system_info.installed_packages_summary %}
  Installed packages (use these managers for install suggestions — do not guess):
{{ system_info.installed_packages_summary }}
{% elif system_info.installed_packages %}
{% for mgr, pkgs in system_info.installed_packages.items() %}
  {{ mgr }} ({{ pkgs|length }} packages){% if pkgs[:40] %}: {{ pkgs[:40]|join(", ") }}{% if pkgs|length > 40 %} …{% endif %}{% endif %}

{% endfor %}
{% endif %}

{% endif %}\

{% if selected_text %}\
The user has highlighted this specific text — focus your explanation on it:
```
{{ selected_text }}
```

{% endif %}\
{% if follow_up_question %}\
The user is asking a follow-up question about the command/context above. Answer it
directly in contextual_usage (and general_utility if needed); keep the same command focus:
```
{{ follow_up_question }}
```

{% endif %}\
Terminal context:
```
{{ visible_text }}
```

Last command:
```
{{ last_command }}
```

{% if last_command_output %}\
Command output:
```
{{ last_command_output }}
```

{% endif %}\
{% if man_page %}\
IMPORTANT — Official manual for `{{ man_lookup }}` ({{ man_source }}):
```
{{ man_page }}
```
Base your flag explanations and examples DIRECTLY on this documentation.
Do not guess flag meanings; use only what the manual states above.

{% endif %}\
Working directory: {{ cwd }}
Exit code: {{ exit_code if exit_code is not none else "unknown" }}

{% if flags %}\
Parsed tokens from the command (base command + flags/args):
{% for f in flags %}\
  {{ f.flag }}
{% endfor %}

{% endif %}\
{% if history_results %}\
Similar past commands from this user's history:
{% for r in history_results %}\
  - {{ r.command }} ({{ r.ai_general_utility }})
{% endfor %}

{% endif %}\
{% if error_remediation and (error_remediation.root_cause or error_remediation.fix_steps) %}\
Pre-computed error analysis:
Root cause: {{ error_remediation.root_cause }}
{% if error_remediation.fix_steps %}\
Suggested fix steps:
{% for step in error_remediation.fix_steps %}\
  {{ loop.index }}. {{ step }}
{% endfor %}
{% endif %}

{% endif %}\
Respond with a single JSON object — no markdown fences, no extra text — matching
this schema exactly:
{
  "command_name": "<base command, e.g. 'git rebase'>",
  "flags_explained": [
    {"flag": "<token>", "meaning": "<what it does>", "example": "<short example>"}
  ],
  "general_utility": "<one sentence: what the command does>",
  "contextual_usage": "<explanation tailored to this specific terminal context>",
  "error_fix": "<actionable fix steps if the command failed, else null>",
  "similar_commands": [],
  "tool_calls_made": [],
  "actionable_items": ["<up to 5 complete shell commands or copy-paste one-liners, no prose>"]
}\
""")

_CHAT_TEMPLATE = Template("""\
You are an expert command-line educator helping a developer understand their shell.

The user is asking a follow-up question about a command they ran. Answer in clear,
conversational plain prose. Be specific to their terminal context. Do not output JSON.

Last command (focus your answer on this program and how it applies here):
```
{{ last_command }}
```

{% if last_command_output %}\
Recent command output:
```
{{ last_command_output }}
```
{% endif %}\
{% if selected_text %}\
Highlighted selection:
```
{{ selected_text }}
```
{% endif %}\
Working directory: {{ cwd }}
Exit code: {{ exit_code if exit_code is not none else "unknown" }}

Terminal context (excerpt):
```
{{ visible_text }}
```

User question:
```
{{ message }}
```
""")


# ── Data models ───────────────────────────────────────────────────────────────

@dataclass
class FlagExplanation(DataClass):
    flag: str = field(metadata={"desc": "The flag or argument, e.g. '-i'"})
    meaning: str = field(metadata={"desc": "What this flag does"})
    example: str = field(metadata={"desc": "Short usage example"})


@dataclass
class ExplainResponse(DataClass):
    command_name: str = field(metadata={"desc": "Primary command, e.g. 'git rebase'"})
    flags_explained: list = field(metadata={"desc": "Per-flag breakdown"})
    general_utility: str = field(metadata={"desc": "One-sentence description"})
    contextual_usage: str = field(metadata={"desc": "Explanation tailored to the context"})
    error_fix: Optional[str] = field(default=None, metadata={"desc": "Fix steps if exit_code != 0"})
    similar_commands: list = field(default_factory=list, metadata={"desc": "Related commands from history"})
    tool_calls_made: list = field(default_factory=list, metadata={"desc": "Tools invoked during request"})
    actionable_items: list = field(
        default_factory=list,
        metadata={"desc": "Up to 5 shell-ready commands the user can run verbatim"},
    )


# ── SyntaxExplanationTool ─────────────────────────────────────────────────────

class SyntaxExplanationTool:
    """Parses a command string into its base name and individual flags/args.

    Returns a FlagExplanation (with empty meaning/example — the LLM fills those
    in during synthesis) for every token after the base command.  Returns an
    empty list for empty, whitespace-only, or unparseable input without raising.
    """

    def run(self, command: str) -> list[FlagExplanation]:
        # Req 7.3: empty or whitespace → empty list
        if not command or not command.strip():
            return []

        try:
            tokens = shlex.split(command.strip())
        except ValueError:
            # Unterminated quotes or other shlex errors → Req 7.2 partial-failure path
            return []

        if not tokens:
            return []

        # Req 7.1: first token is the base command; remaining are flags/args
        return [
            FlagExplanation(flag=token, meaning="", example="")
            for token in tokens[1:]
        ]


# ── ExplainAgent ──────────────────────────────────────────────────────────────

class ExplainAgent(Component):

    def __init__(self, db=None, system_info: dict | None = None) -> None:
        super().__init__()
        self._syntax_tool = SyntaxExplanationTool()
        self._error_tool = ErrorRemediationTool()
        self._man_tool = ManPageTool()
        self._async_client = ollama.AsyncClient()
        self._system_info = system_info or {}
        # HistoricalQueryTool is optional; wired in when a Database is provided
        if db is not None:
            from database import HistoricalQueryTool
            self._history_tool = HistoricalQueryTool(db)
        else:
            self._history_tool = None

    def _build_context(
        self,
        request,
        *,
        flags: list[FlagExplanation] | None = None,
        history_results: list | None = None,
        error_remediation: ActionStep | None = None,
        man_result: ManPageResult | None = None,
    ) -> str:
        """Build the Jinja2 prompt from the ExplainRequest and tool results.

        Req 10.1: includes visible_text, last_command, cwd, exit_code.
        Req 10.2: failure notice is the very first content when exit_code != 0.
        Req 10.3: failure notice also when exit_code is null but failure keywords
                  are present in visible_text.
        Req 10.4: selected_text included with focus instruction when non-null.
        Req 10.6: visible_text, selected_text, last_command wrapped in code blocks.
        """
        t = request.terminal
        failure_notice: str | None = None

        if t.exit_code is not None and t.exit_code != 0:
            # Req 10.2 — explicit non-zero exit code
            failure_notice = (
                f"IMPORTANT: The last command FAILED with exit code {t.exit_code}. "
                "Focus on diagnosing the error and providing actionable fix steps."
            )
        elif t.exit_code is None and _FAILURE_RE.search(t.visible_text):
            # Req 10.3 — null exit code but failure indicators present
            failure_notice = (
                "NOTE: The terminal output contains potential failure indicators. "
                "Focus on diagnosing the potential error."
            )

        follow_up = getattr(request, "follow_up_question", None)

        man_page = man_result.body if man_result and man_result.source != "none" else ""
        man_lookup = man_result.lookup if man_result and man_result.source != "none" else ""
        man_source = man_result.source if man_result else "none"

        return _PROMPT_TEMPLATE.render(
            failure_notice=failure_notice,
            system_info=self._system_info or None,
            visible_text=t.visible_text,
            selected_text=t.selected_text,
            follow_up_question=follow_up,
            last_command=t.last_command,
            last_command_output=getattr(t, "last_command_output", ""),
            cwd=t.cwd,
            exit_code=t.exit_code,
            flags=flags or [],
            man_page=man_page,
            man_lookup=man_lookup,
            man_source=man_source,
            history_results=history_results or [],
            error_remediation=error_remediation,
        )

    def _select_tools(
        self, request
    ) -> tuple[list[FlagExplanation], "ActionStep | None", "ManPageResult | None", list[str]]:
        """Invoke pre-LLM tools; return (flags, error_remediation, man_result, tool_calls).

        Sync, no Ollama call. Extracted for testability (Property 8).
        """
        flags: list[FlagExplanation] = []
        error_remediation: ActionStep | None = None
        man_result: ManPageResult | None = None
        tool_calls: list[str] = []

        # Req 6.1: SyntaxExplanationTool when last_command is non-empty
        if request.terminal.last_command:
            flags = self._syntax_tool.run(request.terminal.last_command)
            tool_calls.append("syntax_explanation")

        # ManPageTool: always run when there is a command, before the LLM.
        if request.terminal.last_command:
            man_result = self._man_tool.run(request.terminal.last_command)
            if man_result.source != "none" and man_result.body:
                tool_calls.append("man_page")

        # Req 6.2: ErrorRemediationTool when exit_code is non-null and non-zero
        if request.terminal.exit_code is not None and request.terminal.exit_code != 0:
            error_remediation = self._error_tool.run(
                request.terminal.visible_text,
                exit_code=request.terminal.exit_code,
            )
            tool_calls.append("error_remediation")

        return flags, error_remediation, man_result, tool_calls

    async def run_stream(self, request) -> AsyncIterator[str]:
        """Yield raw token strings from Ollama.

        The server collects these tokens, then parses the final JSON into an
        ExplainResponse and emits the structured-done SSE event.
        """
        flags, error_remediation, man_result, tool_calls = self._select_tools(request)
        history_results: list = []

        # Req 6.4: invoke HistoricalQueryTool when last_command has more than one word
        # Req 6.5: only add to tool_calls_made when results are non-empty
        if (
            self._history_tool is not None
            and len(request.terminal.last_command.split()) > 1
        ):
            results = self._history_tool.run(request.terminal.last_command, limit=5)
            if results:
                history_results = results
                tool_calls.append("historical_query")

        prompt = self._build_context(
            request,
            flags=flags,
            history_results=history_results,
            error_remediation=error_remediation,
            man_result=man_result,
        )

        async for chunk in await self._async_client.generate(
            model=MODEL,
            prompt=prompt,
            format="json",
            stream=True,
        ):
            token = _chunk_text(chunk)
            if token:
                yield token

    def _build_chat_prompt(self, request, message: str) -> str:
        t = request.terminal
        excerpt = t.visible_text
        if len(excerpt) > 2000:
            excerpt = excerpt[:2000] + "\n... [truncated]"
        return _CHAT_TEMPLATE.render(
            last_command=t.last_command,
            last_command_output=getattr(t, "last_command_output", ""),
            selected_text=t.selected_text,
            cwd=t.cwd,
            exit_code=t.exit_code,
            visible_text=excerpt,
            message=message,
        )

    async def run_chat_stream(self, request, message: str) -> AsyncIterator[str]:
        """Yield plain-text tokens for Chat mode (no JSON)."""
        prompt = self._build_chat_prompt(request, message)
        async for chunk in await self._async_client.generate(
            model=MODEL,
            prompt=prompt,
            stream=True,
        ):
            token = _chunk_text(chunk)
            if token:
                yield token
