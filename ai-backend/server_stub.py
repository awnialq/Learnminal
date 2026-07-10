#!/usr/bin/env python3
"""Minimal stub server for Rust-side Learnminal integration testing."""

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import uvicorn

from agent import MODEL

app = FastAPI()


class TerminalContext(BaseModel):
    visible_text: str
    selected_text: Optional[str] = None
    last_command: str
    cwd: str
    exit_code: Optional[int] = None
    rows: int
    cols: int


class ExplainRequest(BaseModel):
    request_id: str
    timestamp: int
    terminal: TerminalContext


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL}


@app.post("/explain")
def explain(_request: ExplainRequest):
    async def stream():
        yield 'data: {"text": "Learnminal stub response", "chunk_index": 0}\n\n'
        yield (
            'data: {"structured": {"command_name": "stub", "flags_explained": [], '
            '"general_utility": "Stub utility.", "contextual_usage": "Stub context.", '
            '"error_fix": null, "similar_commands": [], "tool_calls_made": [], '
            '"actionable_items": []}, "done": true}\n\n'
        )
        yield "data: [DONE]\n\n"

    return StreamingResponse(stream(), media_type="text/event-stream")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8765, log_level="info")
