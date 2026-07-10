import json
import logging
import time
from pathlib import Path
from typing import Optional

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

import sysinfo
from packages import summarize_for_prompt, total_package_count
from agent import ExplainAgent, MODEL
from command_reference import extract_program_name, lookup_command_reference
from actionable import derive_actionable_items, extract_from_prose
from database import CommandRecord, Database

log = logging.getLogger(__name__)

app = FastAPI()
_db = Database()

# Collect system environment on first boot or when stale (>7 days).
def _refresh_system_info(db: Database, *, force: bool = False) -> dict:
    if force or db.needs_system_refresh():
        info = sysinfo.collect()
        for key, value in info.items():
            db.upsert_constant(key, value)
        log.info("System info collected: %s", info.get("os"))
        return info
    return db.get_all_constants()


def _system_info_is_complete(info: dict) -> bool:
    """True when cached/collected data has the fields /info and the agent need."""
    if not info:
        return False
    if not str(info.get("os", "")).strip():
        return False
    if info.get("collected_at") is None:
        return False
    # Package inventory added after initial releases — refresh stale caches.
    if info.get("package_managers") and "installed_packages" not in info:
        return False
    return True


def _enrich_system_info(info: dict) -> dict:
    """Add display-friendly fields for clients (e.g. overlay /info)."""
    out = dict(info)
    if isinstance(out.get("installed_packages"), dict):
        if not out.get("installed_packages_summary"):
            out["installed_packages_summary"] = summarize_for_prompt(out["installed_packages"])
        if out.get("installed_packages_total") is None:
            out["installed_packages_total"] = total_package_count(out["installed_packages"])
    ts = out.get("collected_at")
    if isinstance(ts, str) and ts.isdigit():
        ts = int(ts)
    if isinstance(ts, (int, float)):
        out["collected_at_display"] = time.strftime(
            "%Y-%m-%d %H:%M:%S %Z", time.localtime(int(ts))
        )
    return out


def _ensure_system_info(db: Database, cached: dict, *, force: bool = False) -> dict:
    """Return complete system info, collecting on disk when missing or stale."""
    if force or not _system_info_is_complete(cached):
        if not force:
            log.info("System info incomplete or missing; collecting now")
        return _refresh_system_info(db, force=True)
    return cached

_system_info = _refresh_system_info(_db)
_agent = ExplainAgent(db=_db, system_info=_system_info)

# Req 11.5: parse errors are appended here; module-level so tests can patch it
_ERROR_LOG_PATH = Path.home() / ".ai-cli-learning" / "errors.log"


def _log_parse_error_to_file(exc: Exception, raw: str) -> None:
    """Append parse error to _ERROR_LOG_PATH (Req 11.5). Best-effort; never raises."""
    try:
        _ERROR_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with _ERROR_LOG_PATH.open("a") as fh:
            ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
            fh.write(f"{ts} ParseError: {exc} — raw: {raw[:300]}\n")
    except OSError:
        pass

def _parse_llm_json(accumulated: str) -> dict:
    """Parse JSON from model output, tolerating optional ``` fences."""
    text = accumulated.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return json.loads(text)


_REQUIRED_STRUCTURED_KEYS = [
    ("command_name", "unknown"),
    ("flags_explained", []),
    ("general_utility", ""),
    ("contextual_usage", ""),
    ("error_fix", None),
    ("similar_commands", []),
    ("tool_calls_made", []),
    ("actionable_items", []),
]


class TerminalContext(BaseModel):
    visible_text: str = Field(max_length=8000)
    selected_text: Optional[str]
    last_command: str
    last_command_output: str = ""
    cwd: str
    exit_code: Optional[int]
    rows: int
    cols: int


class ExplainRequest(BaseModel):
    request_id: str
    timestamp: int
    terminal: TerminalContext
    follow_up_question: Optional[str] = None


class CommandReferenceRequest(BaseModel):
    program: str = Field(min_length=1)


class ReferenceSection(BaseModel):
    name: str
    lines: list[str]


class CommandReferenceResponse(BaseModel):
    program: str
    source: str
    title: str
    sections: list[ReferenceSection]


class ChatRequest(BaseModel):
    request_id: str
    timestamp: int
    terminal: TerminalContext
    message: str = Field(min_length=1)


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.get("/system-info")
def system_info(refresh: bool = False):
    """Return cached system environment (OS, package managers, installed tools)."""
    global _system_info, _agent
    _system_info = _ensure_system_info(_db, _system_info, force=refresh)
    _agent._system_info = _system_info
    return _enrich_system_info(_system_info)


async def _agent_stream(request, agent: ExplainAgent, db=None):
    """Core SSE generator. Always emits [DONE] as the final event (Req 5.6).

    Yields dicts with a "data" key consumed by EventSourceResponse.
    Extracted from the route handler so it can be unit-tested directly.
    db is optional; when provided, a CommandRecord is inserted after a
    successful parse (Req 5.7). Never inserts on parse failure or errors.
    """
    accumulated = ""
    chunk_index = 0
    parse_ok = False
    try:
        async for token in agent.run_stream(request):
            accumulated += token
            yield {"data": json.dumps({"text": token, "chunk_index": chunk_index})}
            chunk_index += 1

        # Parse accumulated LLM output into the structured-done event
        try:
            structured = _parse_llm_json(accumulated)
            parse_ok = True
        except (json.JSONDecodeError, ValueError) as exc:
            # Req 5.5: fall back to raw text; log parse error to stderr and file
            log.error("ParseError: %s — raw: %.300s", exc, accumulated)
            _log_parse_error_to_file(exc, accumulated)  # Req 11.5
            structured = {
                "command_name": request.terminal.last_command or "unknown",
                "flags_explained": [],
                "general_utility": accumulated[:500] if accumulated else "Could not parse response.",
                "contextual_usage": "",
                "error_fix": None,
                "similar_commands": [],
                "tool_calls_made": [],
            }

        for key, default in _REQUIRED_STRUCTURED_KEYS:
            structured.setdefault(key, default)

        # Override tool_calls_made with the authoritative list from _select_tools;
        # the LLM output for this field is unreliable (Req 6.6).
        _, _, _, pre_llm_tools = agent._select_tools(request)
        structured["tool_calls_made"] = pre_llm_tools
        structured["actionable_items"] = derive_actionable_items(structured)

        yield {"data": json.dumps({"structured": structured, "done": True})}

        # Req 5.7: persist only after a successful JSON parse; never on fallback
        if parse_ok and db is not None:
            try:
                db.insert_command(CommandRecord(
                    command=request.terminal.last_command or "",
                    cwd=request.terminal.cwd or "",
                    exit_code=request.terminal.exit_code,
                    ai_general_utility=structured.get("general_utility", ""),
                    ai_context_desc=structured.get("contextual_usage", ""),
                ))
            except Exception as db_exc:
                log.error("DB insert failed: %s", db_exc)

    except Exception as exc:
        msg = str(exc)
        if any(kw in msg.lower() for kw in ("connection", "refused", "connect")):
            msg = "Ollama not running. Start with: ollama serve"
        log.error("Stream error: %s", exc)
        yield {"data": json.dumps({"error": msg, "done": True})}

    finally:
        # Req 5.6: [DONE] must be the very last event on every code path
        yield {"data": "[DONE]"}


@app.post("/explain")
async def explain(request: ExplainRequest):
    return EventSourceResponse(_agent_stream(request, _agent, _db))


@app.post("/command-reference", response_model=CommandReferenceResponse)
def command_reference(request: CommandReferenceRequest):
    """Return formatted man/--help for a single program (Command mode)."""
    program = request.program.strip()
    data = lookup_command_reference(program)
    return CommandReferenceResponse(**data)


@app.post("/command-reference/from-terminal", response_model=CommandReferenceResponse)
def command_reference_from_terminal(terminal: TerminalContext):
    """Extract first token from last_command and return its reference."""
    program = extract_program_name(terminal.last_command)
    if not program:
        return CommandReferenceResponse(
            program="",
            source="none",
            title="No command detected",
            sections=[
                ReferenceSection(
                    name="Hint",
                    lines=[
                        "Could not detect a command in the terminal.",
                        "Run a command, then press Ctrl+Shift+E again.",
                    ],
                )
            ],
        )
    data = lookup_command_reference(program)
    return CommandReferenceResponse(**data)


async def _chat_stream(request: ChatRequest, agent: ExplainAgent):
    """SSE stream for Chat mode: plain text chunks + reply done event."""
    accumulated = ""
    chunk_index = 0
    try:
        async for token in agent.run_chat_stream(request, request.message):
            accumulated += token
            yield {"data": json.dumps({"text": token, "chunk_index": chunk_index})}
            chunk_index += 1

        yield {
            "data": json.dumps({
                "reply": accumulated,
                "actionable_items": extract_from_prose(accumulated),
                "done": True,
            })
        }
    except Exception as exc:
        msg = str(exc)
        if any(kw in msg.lower() for kw in ("connection", "refused", "connect")):
            msg = "Ollama not running. Start with: ollama serve"
        log.error("Chat stream error: %s", exc)
        yield {"data": json.dumps({"error": msg, "done": True})}
    finally:
        yield {"data": "[DONE]"}


@app.post("/chat")
async def chat(request: ChatRequest):
    return EventSourceResponse(_chat_stream(request, _agent))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
