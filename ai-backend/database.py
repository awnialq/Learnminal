"""
SQLite persistence layer with WAL mode and FTS5 full-text search.
Stores only permitted fields — never raw visible_text or selected_text (Req 12.3).
"""
import json
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

DEFAULT_DB_PATH = str(Path.home() / ".ai-cli-learning" / "history.db")

# WAL and foreign-key PRAGMAs are applied separately before this script
# (WAL is filesystem-only and not valid for :memory: databases).
_SYSTEM_INFO_MAX_AGE = 7 * 24 * 3600  # refresh after 7 days

_SCHEMA_SQL = """\
PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS system_constants (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL,
    updated_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS command_history (
    id                TEXT PRIMARY KEY,
    timestamp         INTEGER NOT NULL,
    command           TEXT NOT NULL,
    cwd               TEXT NOT NULL DEFAULT '',
    exit_code         INTEGER,
    ai_general_utility TEXT,
    ai_context_desc   TEXT
);

CREATE VIRTUAL TABLE IF NOT EXISTS command_history_fts
USING fts5(
    command,
    ai_general_utility,
    ai_context_desc,
    content='command_history',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS command_history_ai
AFTER INSERT ON command_history BEGIN
    INSERT INTO command_history_fts(rowid, command, ai_general_utility, ai_context_desc)
    VALUES (new.rowid, new.command, new.ai_general_utility, new.ai_context_desc);
END;
"""


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class CommandRecord:
    """Persisted command record.

    Permitted fields only — visible_text and selected_text are intentionally
    absent to protect user privacy (Req 12.3).
    """
    command: str
    cwd: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: int = field(default_factory=lambda: int(time.time()))
    exit_code: Optional[int] = None
    ai_general_utility: Optional[str] = None
    ai_context_desc: Optional[str] = None


def _row_to_record(row: sqlite3.Row) -> CommandRecord:
    return CommandRecord(
        id=row["id"],
        timestamp=row["timestamp"],
        command=row["command"],
        cwd=row["cwd"],
        exit_code=row["exit_code"],
        ai_general_utility=row["ai_general_utility"],
        ai_context_desc=row["ai_context_desc"],
    )


# ── Database ──────────────────────────────────────────────────────────────────

class Database:

    def __init__(self, path: str = DEFAULT_DB_PATH) -> None:
        self._path = path
        is_memory = (path == ":memory:")

        if not is_memory:
            db_file = Path(path).expanduser()
            db_file.parent.mkdir(parents=True, exist_ok=True)

        conn_path = ":memory:" if is_memory else str(Path(path).expanduser())
        self._conn = sqlite3.connect(conn_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row

        if not is_memory:
            # Req 9.4: WAL required; no fallback — fail hard if unavailable.
            row = self._conn.execute("PRAGMA journal_mode = WAL").fetchone()
            if row[0].upper() != "WAL":
                self._conn.close()
                raise RuntimeError(
                    f"WAL journal mode required but got '{row[0]}'. "
                    "Database cannot start without WAL (Req 9.4)."
                )

        self._conn.executescript(_SCHEMA_SQL)

        if not is_memory:
            # Req 9.6, 12.4: restrict file to owning user only.
            db_file = Path(path).expanduser()
            if db_file.exists():
                os.chmod(db_file, 0o600)

    def upsert_constant(self, key: str, value) -> None:
        """Insert or replace a system constant. Lists/dicts are JSON-encoded."""
        if not isinstance(value, str):
            value = json.dumps(value)
        self._conn.execute(
            "INSERT OR REPLACE INTO system_constants (key, value, updated_at) VALUES (?, ?, ?)",
            (key, value, int(time.time())),
        )
        self._conn.commit()

    def get_constant(self, key: str) -> Optional[str]:
        row = self._conn.execute(
            "SELECT value FROM system_constants WHERE key = ?", (key,)
        ).fetchone()
        return row[0] if row else None

    def get_all_constants(self) -> dict:
        """Return all system constants as a dict, JSON-decoding list/dict values."""
        rows = self._conn.execute(
            "SELECT key, value FROM system_constants"
        ).fetchall()
        result = {}
        for row in rows:
            v = row["value"]
            try:
                parsed = json.loads(v)
                result[row["key"]] = parsed if isinstance(parsed, (list, dict)) else v
            except (json.JSONDecodeError, ValueError):
                result[row["key"]] = v
        return result

    def needs_system_refresh(self) -> bool:
        """True if system info has never been collected or is older than max age."""
        row = self._conn.execute(
            "SELECT updated_at FROM system_constants WHERE key = 'os'"
        ).fetchone()
        if row is None:
            return True
        return (int(time.time()) - row[0]) > _SYSTEM_INFO_MAX_AGE

    def insert_command(self, record: CommandRecord) -> str:
        """Persist a CommandRecord. Returns the record id."""
        self._conn.execute(
            """
            INSERT INTO command_history
                (id, timestamp, command, cwd, exit_code, ai_general_utility, ai_context_desc)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.id, record.timestamp, record.command, record.cwd,
                record.exit_code, record.ai_general_utility, record.ai_context_desc,
            ),
        )
        self._conn.commit()
        return record.id

    def search_similar(self, query: str, limit: int = 5) -> list[CommandRecord]:
        """BM25-ranked FTS5 search; returns at most `limit` records with non-null command.

        Returns empty list for blank query, limit < 1, or malformed FTS5 syntax.
        """
        if not query or not query.strip() or limit < 1:
            return []

        try:
            rows = self._conn.execute(
                """
                SELECT c.id, c.timestamp, c.command, c.cwd, c.exit_code,
                       c.ai_general_utility, c.ai_context_desc
                FROM command_history_fts f
                JOIN command_history c ON c.rowid = f.rowid
                WHERE command_history_fts MATCH ?
                  AND c.command IS NOT NULL
                ORDER BY f.rank
                LIMIT ?
                """,
                (query, limit),
            ).fetchall()
        except sqlite3.OperationalError:
            # Malformed FTS5 query syntax (special chars etc.) — return empty
            return []

        return [_row_to_record(r) for r in rows]

    def get_recent(self, limit: int = 10) -> list[CommandRecord]:
        rows = self._conn.execute(
            "SELECT * FROM command_history ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [_row_to_record(r) for r in rows]

    def close(self) -> None:
        self._conn.close()


# ── HistoricalQueryTool ───────────────────────────────────────────────────────

class HistoricalQueryTool:
    """Queries command history via FTS5 BM25 search.

    Returns an empty list when no records match. ExplainAgent checks for
    non-empty results before adding 'historical_query' to tool_calls_made
    (Req 6.5).
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    def run(self, query: str, limit: int = 5) -> list[CommandRecord]:
        return self._db.search_similar(query, limit)
