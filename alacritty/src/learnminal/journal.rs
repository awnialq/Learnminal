//! Local SQLite journal of past chat interactions keyed by program.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use log::warn;
use rusqlite::{params, Connection, OptionalExtension};

use crate::learnminal::settings::SETTINGS_DIR_NAME;

const DB_FILE_NAME: &str = "journal.db";
pub const GENERAL_PROGRAM: &str = "_general";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JournalNote {
    pub id: i64,
    pub program: String,
    pub question: String,
    pub reply: String,
    pub last_command: String,
    pub reference_source: String,
    /// `Some(true)` all flags matched, `Some(false)` had unverified flags, `None` skipped.
    pub verified: Option<bool>,
    pub created_at: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProgramSummary {
    pub program: String,
    pub note_count: i64,
}

fn default_db_path() -> Option<PathBuf> {
    home::home_dir().map(|home| home.join(SETTINGS_DIR_NAME).join(DB_FILE_NAME))
}

/// Open (or create) the journal database at the default path.
pub fn open() -> Option<Connection> {
    open_at(&default_db_path()?)
}

/// Open (or create) the journal database at an explicit path.
pub fn open_at(path: &Path) -> Option<Connection> {
    if let Some(parent) = path.parent() {
        if let Err(err) = std::fs::create_dir_all(parent) {
            warn!("learnminal journal: create_dir_all failed: {err}");
            return None;
        }
    }
    match Connection::open(path) {
        Ok(conn) => {
            if let Err(err) = migrate(&conn) {
                warn!("learnminal journal: migrate failed: {err}");
                return None;
            }
            Some(conn)
        },
        Err(err) => {
            warn!("learnminal journal: open failed: {err}");
            None
        },
    }
}

fn migrate(conn: &Connection) -> rusqlite::Result<()> {
    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS command_notes (
            id INTEGER PRIMARY KEY,
            program TEXT NOT NULL,
            question TEXT NOT NULL,
            reply TEXT NOT NULL,
            last_command TEXT,
            reference_source TEXT,
            verified INTEGER,
            created_at INTEGER NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_command_notes_program_created
            ON command_notes(program, created_at DESC);
        ",
    )
}

fn now_secs() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs() as i64).unwrap_or(0)
}

/// Normalize a program key for storage/lookup.
pub fn normalize_program(program: &str) -> String {
    let trimmed = program.trim();
    if trimmed.is_empty() {
        GENERAL_PROGRAM.to_owned()
    } else {
        // Strip path components: `/usr/bin/git` → `git`
        Path::new(trimmed)
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_else(|| trimmed.to_owned())
    }
}

/// Insert a note. Best-effort — returns `false` on failure.
pub fn insert_note(
    conn: &Connection,
    program: &str,
    question: &str,
    reply: &str,
    last_command: &str,
    reference_source: &str,
    verified: Option<bool>,
) -> bool {
    let program = normalize_program(program);
    let question = question.trim();
    let reply = reply.trim();
    if question.is_empty() || reply.is_empty() {
        return false;
    }
    let verified_i = verified.map(|v| if v { 1i64 } else { 0 });
    match conn.execute(
        "INSERT INTO command_notes
            (program, question, reply, last_command, reference_source, verified, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        params![
            program,
            question,
            reply,
            last_command,
            reference_source,
            verified_i,
            now_secs()
        ],
    ) {
        Ok(_) => true,
        Err(err) => {
            warn!("learnminal journal: insert failed: {err}");
            false
        },
    }
}

/// Best-effort insert using the default database path.
pub fn insert_note_default(
    program: &str,
    question: &str,
    reply: &str,
    last_command: &str,
    reference_source: &str,
    verified: Option<bool>,
) -> bool {
    let Some(conn) = open() else {
        return false;
    };
    insert_note(&conn, program, question, reply, last_command, reference_source, verified)
}

fn row_to_note(row: &rusqlite::Row<'_>) -> rusqlite::Result<JournalNote> {
    let verified_i: Option<i64> = row.get(6)?;
    Ok(JournalNote {
        id: row.get(0)?,
        program: row.get(1)?,
        question: row.get(2)?,
        reply: row.get(3)?,
        last_command: row.get::<_, Option<String>>(4)?.unwrap_or_default(),
        reference_source: row.get::<_, Option<String>>(5)?.unwrap_or_else(|| "none".into()),
        verified: verified_i.map(|v| v != 0),
        created_at: row.get(7)?,
    })
}

/// Recent notes for a program, newest first.
pub fn recent_for_program(conn: &Connection, program: &str, limit: usize) -> Vec<JournalNote> {
    let program = normalize_program(program);
    let limit = limit.max(1) as i64;
    let mut stmt = match conn.prepare(
        "SELECT id, program, question, reply, last_command, reference_source, verified, created_at
         FROM command_notes
         WHERE program = ?1
         ORDER BY created_at DESC, id DESC
         LIMIT ?2",
    ) {
        Ok(stmt) => stmt,
        Err(err) => {
            warn!("learnminal journal: prepare recent failed: {err}");
            return Vec::new();
        },
    };
    match stmt.query_map(params![program, limit], row_to_note) {
        Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
        Err(err) => {
            warn!("learnminal journal: query recent failed: {err}");
            Vec::new()
        },
    }
}

pub fn recent_for_program_default(program: &str, limit: usize) -> Vec<JournalNote> {
    open().map(|c| recent_for_program(&c, program, limit)).unwrap_or_default()
}

/// Substring search within a program's notes.
pub fn search(conn: &Connection, program: &str, query: &str, limit: usize) -> Vec<JournalNote> {
    let program = normalize_program(program);
    let pattern = format!("%{}%", query.trim());
    let limit = limit.max(1) as i64;
    let mut stmt = match conn.prepare(
        "SELECT id, program, question, reply, last_command, reference_source, verified, created_at
         FROM command_notes
         WHERE program = ?1 AND (question LIKE ?2 OR reply LIKE ?2)
         ORDER BY created_at DESC, id DESC
         LIMIT ?3",
    ) {
        Ok(stmt) => stmt,
        Err(err) => {
            warn!("learnminal journal: prepare search failed: {err}");
            return Vec::new();
        },
    };
    match stmt.query_map(params![program, pattern, limit], row_to_note) {
        Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
        Err(err) => {
            warn!("learnminal journal: search failed: {err}");
            Vec::new()
        },
    }
}

/// Programs with note counts, most recently used first.
pub fn list_programs(conn: &Connection, limit: usize) -> Vec<ProgramSummary> {
    let limit = limit.max(1) as i64;
    let mut stmt = match conn.prepare(
        "SELECT program, COUNT(*) AS cnt, MAX(created_at) AS last_at
         FROM command_notes
         GROUP BY program
         ORDER BY last_at DESC
         LIMIT ?1",
    ) {
        Ok(stmt) => stmt,
        Err(err) => {
            warn!("learnminal journal: prepare list_programs failed: {err}");
            return Vec::new();
        },
    };
    match stmt.query_map(params![limit], |row| {
        Ok(ProgramSummary { program: row.get(0)?, note_count: row.get(1)? })
    }) {
        Ok(rows) => rows.filter_map(|r| r.ok()).collect(),
        Err(err) => {
            warn!("learnminal journal: list_programs failed: {err}");
            Vec::new()
        },
    }
}

pub fn list_programs_default(limit: usize) -> Vec<ProgramSummary> {
    open().map(|c| list_programs(&c, limit)).unwrap_or_default()
}

/// Delete all notes for a program. Returns rows deleted.
pub fn clear_program(conn: &Connection, program: &str) -> usize {
    let program = normalize_program(program);
    match conn.execute("DELETE FROM command_notes WHERE program = ?1", params![program]) {
        Ok(n) => n,
        Err(err) => {
            warn!("learnminal journal: clear failed: {err}");
            0
        },
    }
}

pub fn clear_program_default(program: &str) -> usize {
    open().map(|c| clear_program(&c, program)).unwrap_or(0)
}

/// Whether any notes exist for a program (tests / diagnostics).
#[cfg(test)]
pub fn count_for_program(conn: &Connection, program: &str) -> i64 {
    let program = normalize_program(program);
    conn.query_row(
        "SELECT COUNT(*) FROM command_notes WHERE program = ?1",
        params![program],
        |row| row.get(0),
    )
    .optional()
    .ok()
    .flatten()
    .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_conn() -> Connection {
        let conn = Connection::open_in_memory().unwrap();
        migrate(&conn).unwrap();
        conn
    }

    #[test]
    fn normalize_strips_path_and_empty_to_general() {
        assert_eq!(normalize_program("/usr/bin/git"), "git");
        assert_eq!(normalize_program("  kubectl  "), "kubectl");
        assert_eq!(normalize_program(""), GENERAL_PROGRAM);
    }

    #[test]
    fn insert_and_recent_round_trip() {
        let conn = temp_conn();
        assert!(insert_note(
            &conn,
            "git",
            "how do I rebase?",
            "Use git rebase -i",
            "git status",
            "man",
            Some(true),
        ));
        assert!(insert_note(
            &conn,
            "git",
            "how do I stash?",
            "Use git stash",
            "",
            "help",
            None,
        ));
        let notes = recent_for_program(&conn, "git", 3);
        assert_eq!(notes.len(), 2);
        assert_eq!(notes[0].question, "how do I stash?");
        assert_eq!(notes[1].verified, Some(true));
    }

    #[test]
    fn search_and_clear() {
        let conn = temp_conn();
        insert_note(&conn, "git", "rebase interactive", "git rebase -i", "", "man", Some(true));
        insert_note(&conn, "git", "push force", "git push --force-with-lease", "", "man", Some(false));
        let hits = search(&conn, "git", "rebase", 10);
        assert_eq!(hits.len(), 1);
        assert_eq!(clear_program(&conn, "git"), 2);
        assert_eq!(count_for_program(&conn, "git"), 0);
    }

    #[test]
    fn list_programs_orders_by_recency() {
        let conn = temp_conn();
        insert_note(&conn, "git", "q1", "a1", "", "man", None);
        std::thread::sleep(std::time::Duration::from_millis(5));
        // Force distinct timestamps.
        conn.execute(
            "UPDATE command_notes SET created_at = created_at - 10 WHERE program = 'git'",
            [],
        )
        .unwrap();
        insert_note(&conn, "cargo", "q2", "a2", "", "none", None);
        let programs = list_programs(&conn, 10);
        assert_eq!(programs[0].program, "cargo");
        assert_eq!(programs.iter().find(|p| p.program == "git").unwrap().note_count, 1);
    }

    #[test]
    fn rejects_empty_question_or_reply() {
        let conn = temp_conn();
        assert!(!insert_note(&conn, "git", "", "reply", "", "none", None));
        assert!(!insert_note(&conn, "git", "q", "  ", "", "none", None));
    }
}
