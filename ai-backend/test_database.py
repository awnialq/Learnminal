"""
Property 11: search_similar returns ≤ limit records, all with non-null command.
Property 12: FTS5 index is immediately searchable after insert.
Property 13: CommandRecord stores only permitted fields (no raw visible_text).
Validates: Requirements 9.1, 9.2, 9.3, 9.5, 12.3
"""
import dataclasses

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from database import CommandRecord, Database, HistoricalQueryTool


# ── helpers ───────────────────────────────────────────────────────────────────

def fresh_db() -> Database:
    """In-memory database — WAL not required, schema still applied."""
    return Database(":memory:")


def make_record(**kwargs) -> CommandRecord:
    defaults = {"command": "git status", "cwd": "/home/user"}
    defaults.update(kwargs)
    return CommandRecord(**defaults)


# Safe alphabet: no FTS5 special characters (*, ", -, ^, (), OR, AND, NOT)
_safe_word = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789",
    min_size=3,
    max_size=12,
)
_safe_command = st.lists(_safe_word, min_size=1, max_size=6).map(" ".join)


# ── Property 11 ───────────────────────────────────────────────────────────────
# Req 9.1, 9.2, 9.5: at most `limit` results, all with non-null command

@given(
    st.lists(_safe_command, min_size=0, max_size=15),
    _safe_word,
    st.integers(min_value=1, max_value=10),
)
@settings(max_examples=100)
def test_prop11_returns_at_most_limit_and_non_null_command(commands, query, limit):
    db = fresh_db()
    try:
        for cmd in commands:
            db.insert_command(make_record(command=cmd))

        results = db.search_similar(query, limit)

        assert len(results) <= limit, (
            f"Got {len(results)} results but limit was {limit}"
        )
        for r in results:
            assert r.command is not None, "search_similar must never return null command"
    finally:
        db.close()


# ── Property 12 ───────────────────────────────────────────────────────────────
# Req 9.3: FTS5 trigger fires on insert; record immediately findable

@given(_safe_word, _safe_word)
@settings(max_examples=100)
def test_prop12_record_findable_immediately_after_insert(unique_word, other_word):
    # Build command with a guaranteed unique prefix word
    command = f"{unique_word} {other_word}"
    db = fresh_db()
    try:
        db.insert_command(make_record(command=command))

        results = db.search_similar(unique_word, limit=10)
        found_commands = [r.command for r in results]
        assert command in found_commands, (
            f"Inserted {command!r} but searching {unique_word!r} returned {found_commands!r}"
        )
    finally:
        db.close()


# ── Property 13 ───────────────────────────────────────────────────────────────
# Req 12.3: CommandRecord schema never includes visible_text or selected_text

_PERMITTED_FIELDS = {"id", "timestamp", "command", "cwd", "exit_code",
                     "ai_general_utility", "ai_context_desc"}
_FORBIDDEN_FIELDS = {"visible_text", "selected_text"}


def test_prop13_command_record_schema_has_only_permitted_fields():
    actual = {f.name for f in dataclasses.fields(CommandRecord)}
    assert actual == _PERMITTED_FIELDS, (
        f"Unexpected fields in CommandRecord: {actual - _PERMITTED_FIELDS}"
    )
    assert not (actual & _FORBIDDEN_FIELDS), (
        f"Forbidden fields present: {actual & _FORBIDDEN_FIELDS}"
    )


@given(_safe_command)
@settings(max_examples=50)
def test_prop13_retrieved_records_contain_no_forbidden_fields(command):
    db = fresh_db()
    try:
        db.insert_command(make_record(command=command))
        recent = db.get_recent(limit=100)
        for r in recent:
            row = dataclasses.asdict(r)
            for bad in _FORBIDDEN_FIELDS:
                assert bad not in row, f"Field {bad!r} must never appear in a retrieved record"
    finally:
        db.close()


# ── Unit tests ────────────────────────────────────────────────────────────────

def test_memory_db_initialises_without_wal():
    db = fresh_db()
    db.close()  # must not raise


def test_insert_and_get_recent_roundtrip():
    db = fresh_db()
    record = make_record(
        command="git rebase -i HEAD~3",
        cwd="/home/user/project",
        exit_code=1,
        ai_general_utility="Rewrites commit history interactively.",
        ai_context_desc="User ran an interactive rebase that failed.",
    )
    returned_id = db.insert_command(record)
    assert returned_id == record.id

    recent = db.get_recent(limit=1)
    assert len(recent) == 1
    r = recent[0]
    assert r.command == "git rebase -i HEAD~3"
    assert r.exit_code == 1
    assert r.ai_general_utility == "Rewrites commit history interactively."
    db.close()


def test_search_empty_query_returns_empty():
    db = fresh_db()
    db.insert_command(make_record(command="git status"))
    assert db.search_similar("") == []
    assert db.search_similar("   ") == []
    db.close()


def test_search_limit_zero_returns_empty():
    db = fresh_db()
    db.insert_command(make_record(command="git status"))
    assert db.search_similar("git", limit=0) == []
    db.close()


def test_search_no_match_returns_empty():
    db = fresh_db()
    db.insert_command(make_record(command="ls -la"))
    results = db.search_similar("xyzzy123nonexistent", limit=5)
    assert results == []
    db.close()


def test_fts5_trigger_keeps_index_in_sync():
    db = fresh_db()
    db.insert_command(make_record(command="docker run ubuntu"))
    results = db.search_similar("docker", limit=5)
    assert any(r.command == "docker run ubuntu" for r in results)
    db.close()


def test_multiple_inserts_respects_limit():
    db = fresh_db()
    for i in range(10):
        db.insert_command(make_record(command=f"git commit {i}"))
    results = db.search_similar("git", limit=3)
    assert len(results) <= 3
    db.close()


def test_historical_query_tool_returns_empty_on_no_match():
    db = fresh_db()
    tool = HistoricalQueryTool(db)
    assert tool.run("xyzzy123nonexistent") == []
    db.close()


def test_historical_query_tool_finds_inserted_record():
    db = fresh_db()
    db.insert_command(make_record(command="kubectl apply -f manifest.yaml"))
    tool = HistoricalQueryTool(db)
    results = tool.run("kubectl", limit=5)
    assert any(r.command == "kubectl apply -f manifest.yaml" for r in results)
    db.close()
