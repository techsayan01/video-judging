"""
db.py — SQLite central store for film analysis and reviews.

Schema
------
films   — one row per unique film; stores the Gemini analysis JSON so video
           is never re-uploaded for a film that already exists in the library.
reviews — one row per (film, festival) pair; lets the same film generate
           reviews for multiple festivals without touching Gemini video again.

Deduplication strategy (in priority order)
  1. screener_url match  — exact URL from FilmFreeway or manual entry
  2. title + director    — normalised lowercase match
"""

import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock

DB_PATH = Path("film_judging.db")
_lock = Lock()


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")   # allow concurrent reads
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_db():
    """Create tables if they don't exist. Call once at app startup."""
    with _lock, _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS films (
                film_id            TEXT PRIMARY KEY,
                title              TEXT NOT NULL,
                director           TEXT NOT NULL,
                logline            TEXT DEFAULT '',
                director_statement TEXT DEFAULT '',
                genre              TEXT DEFAULT '',
                runtime            TEXT DEFAULT '',
                country            TEXT DEFAULT '',
                screener_url       TEXT DEFAULT '',
                analysis           TEXT,
                analysed_at        TEXT,
                created_at         TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_films_screener
                ON films(screener_url)
                WHERE screener_url != '';

            CREATE INDEX IF NOT EXISTS idx_films_identity
                ON films(LOWER(title), LOWER(director));

            CREATE TABLE IF NOT EXISTS reviews (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                film_id      TEXT NOT NULL REFERENCES films(film_id) ON DELETE CASCADE,
                festival_key TEXT NOT NULL,
                review_text  TEXT DEFAULT '',
                created_at   TEXT NOT NULL,
                wp_post_id   TEXT DEFAULT '',
                wp_url       TEXT DEFAULT '',
                UNIQUE(film_id, festival_key)
            );

            CREATE INDEX IF NOT EXISTS idx_reviews_film
                ON reviews(film_id);
        """)


# ── Film helpers ──────────────────────────────────────────────────────────────

def _row_to_film(row) -> dict:
    d = dict(row)
    if d.get("analysis"):
        try:
            d["analysis"] = json.loads(d["analysis"])
        except Exception:
            d["analysis"] = {}
    return d


def film_list() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM films ORDER BY created_at DESC"
        ).fetchall()
    return [_row_to_film(r) for r in rows]


def film_get(film_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM films WHERE film_id = ?", (film_id,)
        ).fetchone()
    return _row_to_film(row) if row else None


def film_find_by_screener(screener_url: str) -> dict | None:
    """Return existing film if the screener URL matches exactly."""
    if not screener_url:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM films WHERE screener_url = ? LIMIT 1",
            (screener_url.strip(),),
        ).fetchone()
    return _row_to_film(row) if row else None


def film_find_by_identity(title: str, director: str) -> dict | None:
    """Return existing film by normalised title + director match."""
    if not title or not director:
        return None
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM films WHERE LOWER(title) = LOWER(?) AND LOWER(director) = LOWER(?) LIMIT 1",
            (title.strip(), director.strip()),
        ).fetchone()
    return _row_to_film(row) if row else None


def film_create(data: dict) -> str:
    """Insert a new film row. Returns the film_id."""
    film_id = data.get("film_id") or uuid.uuid4().hex
    now = datetime.now().isoformat()
    with _lock, _connect() as conn:
        conn.execute(
            """INSERT INTO films
               (film_id, title, director, logline, director_statement,
                genre, runtime, country, screener_url, analysis, analysed_at, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                film_id,
                data.get("title", ""),
                data.get("director", ""),
                data.get("logline", ""),
                data.get("director_statement", ""),
                data.get("genre", ""),
                data.get("runtime", ""),
                data.get("country", ""),
                data.get("screener_url", ""),
                json.dumps(data.get("analysis") or {}),
                data.get("analysed_at", now),
                now,
            ),
        )
    return film_id


def film_save_analysis(film_id: str, analysis: dict):
    """Write the Gemini analysis JSON into an existing film row."""
    with _lock, _connect() as conn:
        conn.execute(
            "UPDATE films SET analysis = ?, analysed_at = ? WHERE film_id = ?",
            (json.dumps(analysis), datetime.now().isoformat(), film_id),
        )


def film_update_meta(film_id: str, data: dict):
    """Update editable metadata fields (logline, director_statement, etc.)."""
    fields = ["logline", "director_statement", "genre", "runtime", "country", "screener_url"]
    updates = {k: data[k] for k in fields if k in data}
    if not updates:
        return
    cols = ", ".join(f"{k} = ?" for k in updates)
    with _lock, _connect() as conn:
        conn.execute(
            f"UPDATE films SET {cols} WHERE film_id = ?",
            (*updates.values(), film_id),
        )


# ── Review helpers ────────────────────────────────────────────────────────────

def _row_to_review(row) -> dict:
    return dict(row)


def review_list_for_film(film_id: str) -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM reviews WHERE film_id = ? ORDER BY created_at DESC",
            (film_id,),
        ).fetchall()
    return [_row_to_review(r) for r in rows]


def review_get(film_id: str, festival_key: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM reviews WHERE film_id = ? AND festival_key = ?",
            (film_id, festival_key),
        ).fetchone()
    return _row_to_review(row) if row else None


def review_upsert(film_id: str, festival_key: str, review_text: str) -> int:
    """Insert or replace a review. Returns the row id."""
    now = datetime.now().isoformat()
    with _lock, _connect() as conn:
        cur = conn.execute(
            """INSERT INTO reviews (film_id, festival_key, review_text, created_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(film_id, festival_key)
               DO UPDATE SET review_text = excluded.review_text,
                             created_at  = excluded.created_at""",
            (film_id, festival_key, review_text, now),
        )
    return cur.lastrowid


def review_set_wp(film_id: str, festival_key: str, post_id: str, url: str):
    """Record WordPress publish details on an existing review row."""
    with _lock, _connect() as conn:
        conn.execute(
            """UPDATE reviews SET wp_post_id = ?, wp_url = ?
               WHERE film_id = ? AND festival_key = ?""",
            (post_id, url, film_id, festival_key),
        )
