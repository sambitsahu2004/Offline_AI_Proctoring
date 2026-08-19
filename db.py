"""
db.py
======
Local SQLite persistence for the proctoring pipeline — keeps it consistent
with the project's offline-first design (same reasoning as using ChromaDB
and Ollama locally: no external DB server needed).

Three things get stored:
    1. Extracted event logs (per session) — for future reference / re-review.
    2. Flagged frame IMAGES themselves — stored as BLOBs in the `frames`
       table, not as loose .jpg files on disk. A frame is only ever
       reachable through the database now: no data/frames/ folder, no file
       path that can go stale if something gets moved/renamed. Candidate
       info is one join away (frames -> sessions on session_id).
    3. Generated reports (risks, events, recommendation) — the final AI
       output, tied to the session it was generated for. This already
       happens automatically after every report generation in
       run_full_demo.py and streamlit_app.py — no manual download needed
       to have it persisted; download is just a convenience export.

Place this file inside rag_system/, next to rag_reporter.py.

USAGE
-----
    import db
    db.init_db()
    db.save_session_events(session_data)          # after extraction — also
                                                    # persists any frame
                                                    # bytes attached to events
    db.save_report(session_id, report_md, model)   # after report generation

    db.list_sessions()                  # for a "past sessions" view
    db.get_session(session_id)          # events for one session (frame_id only, not bytes)
    db.get_frame(frame_id)              # fetch one frame's raw bytes on demand
    db.get_frames_for_session(session_id)  # all frames for a session, WITH candidate info joined in
    db.get_latest_report(session_id)    # most recent report for one session

MIGRATING AN EXISTING db/proctoring.db
---------------------------------------
If you already have a database from before this change, init_db() detects
the missing `frame_id` column on `events` and adds it automatically
(ALTER TABLE) the next time it runs — no manual migration step needed.
Old rows simply have frame_id = NULL (their frame_ref path, if any, is
left alone in the old column so nothing is silently lost, but the pipeline
no longer writes to it going forward).
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

CREATE TABLE IF NOT EXISTS frames (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    event_type TEXT,
    timestamp TEXT,
    mime_type TEXT DEFAULT 'image/jpeg',
    image_bytes BLOB NOT NULL,
    created_at TEXT,
    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
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


def _migrate_schema(conn):
    """Adds the frame_id column to a pre-existing events table that
    predates frame-in-database storage. Safe to call every time init_db()
    runs — checks PRAGMA table_info first, only ALTERs if the column is
    genuinely missing."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(events)").fetchall()}
    if "frame_id" not in cols:
        conn.execute("ALTER TABLE events ADD COLUMN frame_id INTEGER REFERENCES frames(id)")


def init_db():
    conn = get_connection()
    conn.executescript(SCHEMA)
    _migrate_schema(conn)
    conn.commit()
    conn.close()


def save_frame(session_id, event_type, timestamp, image_bytes, mime_type="image/jpeg"):
    """Stores one frame's raw bytes and returns its new frame_id. Called by
    save_session_events() for each event that has frame bytes attached —
    you generally won't call this directly."""
    conn = get_connection()
    cur = conn.execute(
        """
        INSERT INTO frames (session_id, event_type, timestamp, mime_type, image_bytes, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (session_id, event_type, timestamp, mime_type, image_bytes, datetime.now().strftime("%Y-%m-%dT%H:%M:%S")),
    )
    frame_id = cur.lastrowid
    conn.commit()
    conn.close()
    return frame_id


def get_frame(frame_id):
    """Fetches one frame's raw bytes + metadata. Returns None if not found.
    Use this on demand (e.g. right before an st.image() call) — don't
    pre-load every frame's bytes when just listing/browsing sessions."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    row = conn.execute("SELECT * FROM frames WHERE id = ?", (frame_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_frames_for_session(session_id):
    """All frames for a session, joined with the session's candidate info.
    Returns a list of dicts with image_bytes included — fine for a single
    session's handful of flagged frames, but don't call this in a loop
    over many sessions (use get_session() for lightweight browsing, then
    get_frames_for_session() only for the session the user actually opens)."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """
        SELECT f.id AS frame_id, f.event_type, f.timestamp, f.mime_type, f.image_bytes,
               s.session_id, s.candidate_name, s.exam_name, s.date
        FROM frames f
        JOIN sessions s ON f.session_id = s.session_id
        WHERE f.session_id = ?
        ORDER BY f.timestamp
        """,
        (session_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def save_session_events(session_data):
    """Upserts the session row, then replaces its event rows (so re-running
    extraction on the same session_id doesn't duplicate events) — and
    persists any frame bytes attached to those events into the frames
    table, linking each event to its frame via frame_id.

    Expects each event dict to optionally carry a "frame_jpeg_b64" key
    (base64-encoded JPEG bytes, as produced by extractor_agent.py's
    _encode_frame()) instead of the old file-path "frame_ref". Events
    without frame data (e.g. a report row with no associated image) are
    stored with frame_id = NULL.
    """
    import base64

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
    # Re-running extraction on the same session_id shouldn't duplicate rows —
    # clear old events AND old frames for this session first.
    cur.execute("DELETE FROM events WHERE session_id = ?", (session_data["session_id"],))
    cur.execute("DELETE FROM frames WHERE session_id = ?", (session_data["session_id"],))

    for ev in session_data.get("events", []):
        frame_id = None
        b64 = ev.get("frame_jpeg_b64")
        if b64:
            image_bytes = base64.b64decode(b64)
            frame_id_cur = cur.execute(
                """
                INSERT INTO frames (session_id, event_type, timestamp, mime_type, image_bytes, created_at)
                VALUES (?, ?, ?, 'image/jpeg', ?, ?)
                """,
                (
                    session_data["session_id"], ev.get("event_type"), ev.get("timestamp"),
                    image_bytes, datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
                ),
            )
            frame_id = frame_id_cur.lastrowid

        cur.execute(
            """
            INSERT INTO events (session_id, timestamp, event_type, confidence, duration_ms, frame_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                session_data["session_id"],
                ev.get("timestamp"),
                ev.get("event_type"),
                ev.get("confidence"),
                ev.get("duration_ms"),
                frame_id,
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
    """Returns the session + its events (frame_id only, NOT frame bytes —
    call get_frame(frame_id) or get_frames_for_session() to actually view
    an image). Returns None if the session_id isn't in the DB."""
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    session = conn.execute(
        "SELECT * FROM sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    if not session:
        conn.close()
        return None
    events = conn.execute(
        "SELECT timestamp, event_type, confidence, duration_ms, frame_id "
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
