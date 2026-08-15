"""
db.py
======
Local SQLite persistence for the proctoring pipeline — keeps it consistent
with the project's offline-first design (same reasoning as using ChromaDB
and Ollama locally: no external DB server needed).

Two things get stored, matching what was asked for:
    1. Extracted event logs (per session) — for future reference / re-review.
    2. Generated reports (risks, events, recommendation) — the final AI
       output, tied to the session it was generated for.

Place this file inside rag_system/, next to rag_reporter.py.

USAGE
-----
    import db
    db.init_db()
    db.save_session_events(session_data)          # after extraction
    db.save_report(session_id, report_md, model)   # after report generation

    db.list_sessions()                # for a "past sessions" view
    db.get_session(session_id)        # events for one session
    db.get_latest_report(session_id)  # most recent report for one session
"""

import os
import sqlite3
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "db", "proctoring.db")

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    candidate_name TEXT,
    exam_name TEXT,
    date TEXT,
    duration_minutes REAL,
    created_at TEXT
);

CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    timestamp TEXT,
    event_type TEXT,
    confidence REAL,
    duration_ms INTEGER,
    frame_ref TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);

CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    generated_at TEXT,
    model_used TEXT,
    report_markdown TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
);
"""


def get_connection():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db():
    conn = get_connection()
    conn.executescript(SCHEMA)
    conn.commit()
    conn.close()


def save_session_events(session_data):
    """Upserts the session row, then replaces its event rows (so re-running
    extraction on the same session_id doesn't duplicate events)."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO sessions (session_id, candidate_name, exam_name, date, duration_minutes, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            candidate_name=excluded.candidate_name,
            exam_name=excluded.exam_name,
            date=excluded.date,
            duration_minutes=excluded.duration_minutes
        """,
        (
            session_data["session_id"],
            session_data.get("candidate_name"),
            session_data.get("exam_name"),
            session_data.get("date"),
            session_data.get("duration_minutes", 0),
            datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
        ),
    )
    cur.execute("DELETE FROM events WHERE session_id = ?", (session_data["session_id"],))
    for ev in session_data.get("events", []):
        cur.execute(
            """
            INSERT INTO events (session_id, timestamp, event_type, confidence, duration_ms, frame_ref)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_data["session_id"],
                ev.get("timestamp"),
                ev.get("event_type"),
                ev.get("confidence"),
                ev.get("duration_ms"),
                ev.get("frame_ref"),
            ),
        )
    conn.commit()
    conn.close()


def save_report(session_id, report_markdown, model_used="unknown"):
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO reports (session_id, generated_at, model_used, report_markdown)
        VALUES (?, ?, ?, ?)
        """,
        (session_id, datetime.now().strftime("%Y-%m-%dT%H:%M:%S"), model_used, report_markdown),
    )
    conn.commit()
    conn.close()


def get_session(session_id):
    """Returns the session + its events in the same schema as the extractor
    output, or None if the session_id isn't in the DB."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    session = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not session:
        conn.close()
        return None
    events = conn.execute(
        "SELECT timestamp, event_type, confidence, duration_ms, frame_ref "
        "FROM events WHERE session_id = ? ORDER BY timestamp",
        (session_id,),
    ).fetchall()
    conn.close()
    return {
        "session_id": session["session_id"],
        "candidate_name": session["candidate_name"],
        "exam_name": session["exam_name"],
        "date": session["date"],
        "duration_minutes": session["duration_minutes"],
        "events": [dict(e) for e in events],
    }


def list_sessions():
    """For a 'past sessions' view — newest first."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT session_id, candidate_name, exam_name, date, created_at "
        "FROM sessions ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_latest_report(session_id):
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM reports WHERE session_id = ? ORDER BY generated_at DESC LIMIT 1",
        (session_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None
