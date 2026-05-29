"""
db.py — MongoDB central store for films, reviews, and jobs.

Collections
-----------
films   — one row per unique film; Gemini analysis stored once so video is
           never re-uploaded for repeat festival submissions.
reviews — one document per (film_id, festival_key) pair.
jobs    — transient processing state; replaces the in-memory JOBS dict so
           Cloud Run instances share state across restarts and scale-outs.

TTL indexes (auto-purge — no manual cleanup needed)
  films   → deleted after 365 days  (keeps DB under 512 MB M0 free tier)
  reviews → deleted after 365 days
  jobs    → deleted after 2 days    (transient, only needed during polling)

Deduplication strategy (priority order)
  1. screener_url exact match
  2. title + director  (case-insensitive)

Connection
  Set MONGO_URI in .env — e.g.
  MONGO_URI=mongodb+srv://user:pass@cluster.mongodb.net/film_judging?retryWrites=true&w=majority
"""

import os
import uuid
from datetime import datetime, timezone
from pymongo import MongoClient, ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

_client: MongoClient | None = None
_db = None

TTL_FILMS_DAYS = int(os.getenv("TTL_FILMS_DAYS", "365"))
TTL_JOBS_DAYS  = int(os.getenv("TTL_JOBS_DAYS",  "2"))


def _get_db():
    global _client, _db
    if _db is None:
        uri    = os.getenv("MONGO_URI", "mongodb://localhost:27017/film_judging")
        _client = MongoClient(uri)
        _db     = _client.get_default_database() if "/" in uri.rsplit("@", 1)[-1] else _client["film_judging"]
    return _db


def init_db():
    """Create indexes. Safe to call multiple times (idempotent)."""
    d = _get_db()

    # ── films ──────────────────────────────────────────────────────────────
    films = d["films"]
    films.create_index("screener_url", sparse=True)
    films.create_index([("title_lc", ASCENDING), ("director_lc", ASCENDING)])
    films.create_index(
        "created_at",
        expireAfterSeconds=TTL_FILMS_DAYS * 86400,
        name="ttl_films",
    )

    # ── reviews ────────────────────────────────────────────────────────────
    reviews = d["reviews"]
    reviews.create_index(
        [("film_id", ASCENDING), ("festival_key", ASCENDING)],
        unique=True,
        name="uniq_film_festival",
    )
    reviews.create_index("film_id")
    reviews.create_index(
        "created_at",
        expireAfterSeconds=TTL_FILMS_DAYS * 86400,
        name="ttl_reviews",
    )

    # ── jobs ───────────────────────────────────────────────────────────────
    jobs = d["jobs"]
    jobs.create_index(
        "created_at",
        expireAfterSeconds=TTL_JOBS_DAYS * 86400,
        name="ttl_jobs",
    )


def _now() -> datetime:
    return datetime.now(timezone.utc)


# ── Film helpers ──────────────────────────────────────────────────────────────

def _clean_film(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    doc = dict(doc)
    doc.pop("_id", None)
    doc.pop("title_lc", None)
    doc.pop("director_lc", None)
    # Convert datetime → ISO string for JSON serialisation
    for k in ("created_at", "analysed_at"):
        if isinstance(doc.get(k), datetime):
            doc[k] = doc[k].isoformat()
    return doc


def film_list() -> list[dict]:
    docs = _get_db()["films"].find({}, sort=[("created_at", DESCENDING)])
    return [_clean_film(d) for d in docs]


def film_get(film_id: str) -> dict | None:
    return _clean_film(_get_db()["films"].find_one({"film_id": film_id}))


def film_find_by_screener(screener_url: str) -> dict | None:
    if not screener_url or not screener_url.strip():
        return None
    return _clean_film(
        _get_db()["films"].find_one({"screener_url": screener_url.strip()})
    )


def film_find_by_identity(title: str, director: str) -> dict | None:
    if not title or not director:
        return None
    return _clean_film(
        _get_db()["films"].find_one({
            "title_lc":    title.strip().lower(),
            "director_lc": director.strip().lower(),
        })
    )


def film_create(data: dict) -> str:
    """Insert a new film document. Returns film_id."""
    film_id = data.get("film_id") or uuid.uuid4().hex
    now     = _now()
    doc = {
        "film_id":            film_id,
        "title":              data.get("title", ""),
        "director":           data.get("director", ""),
        "title_lc":           data.get("title", "").strip().lower(),
        "director_lc":        data.get("director", "").strip().lower(),
        "logline":            data.get("logline", ""),
        "director_statement": data.get("director_statement", ""),
        "genre":              data.get("genre", ""),
        "runtime":            data.get("runtime", ""),
        "country":            data.get("country", ""),
        "screener_url":       data.get("screener_url", ""),
        "analysis":           data.get("analysis") or {},
        "analysed_at":        None,
        "created_at":         now,
    }
    _get_db()["films"].insert_one(doc)
    return film_id


def film_save_analysis(film_id: str, analysis: dict):
    """Write Gemini analysis into an existing film document."""
    _get_db()["films"].update_one(
        {"film_id": film_id},
        {"$set": {"analysis": analysis, "analysed_at": _now()}},
    )


def film_update_meta(film_id: str, data: dict):
    """Update editable metadata fields."""
    fields = ["logline", "director_statement", "genre", "runtime", "country", "screener_url"]
    updates = {k: data[k] for k in fields if k in data}
    if updates:
        _get_db()["films"].update_one({"film_id": film_id}, {"$set": updates})


# ── Review helpers ────────────────────────────────────────────────────────────

def _clean_review(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    doc = dict(doc)
    doc.pop("_id", None)
    if isinstance(doc.get("created_at"), datetime):
        doc["created_at"] = doc["created_at"].isoformat()
    return doc


def review_get(film_id: str, festival_key: str) -> dict | None:
    return _clean_review(
        _get_db()["reviews"].find_one(
            {"film_id": film_id, "festival_key": festival_key}
        )
    )


def review_list_for_film(film_id: str) -> list[dict]:
    docs = _get_db()["reviews"].find(
        {"film_id": film_id}, sort=[("created_at", DESCENDING)]
    )
    return [_clean_review(d) for d in docs]


def review_upsert(film_id: str, festival_key: str, review_text: str):
    _get_db()["reviews"].update_one(
        {"film_id": film_id, "festival_key": festival_key},
        {"$set": {
            "review_text":  review_text,
            "created_at":   _now(),
        }, "$setOnInsert": {
            "wp_post_id": "",
            "wp_url":     "",
        }},
        upsert=True,
    )


def review_set_wp(film_id: str, festival_key: str, post_id: str, url: str):
    _get_db()["reviews"].update_one(
        {"film_id": film_id, "festival_key": festival_key},
        {"$set": {"wp_post_id": post_id, "wp_url": url}},
    )


# ── Job helpers (replaces in-memory JOBS dict) ────────────────────────────────

def _clean_job(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    doc = dict(doc)
    doc.pop("_id", None)
    for k in ("created_at", "completed_at"):
        if isinstance(doc.get(k), datetime):
            doc[k] = doc[k].isoformat()
    return doc


def job_create(job_id: str, data: dict):
    """Create a new job document."""
    doc = {
        "job_id":     job_id,
        "created_at": _now(),
        **data,
    }
    _get_db()["jobs"].insert_one(doc)


def job_get(job_id: str) -> dict | None:
    return _clean_job(_get_db()["jobs"].find_one({"job_id": job_id}))


def job_update(job_id: str, updates: dict):
    """Merge updates into an existing job document."""
    if "completed_at" in updates and isinstance(updates["completed_at"], str):
        try:
            updates["completed_at"] = datetime.fromisoformat(updates["completed_at"])
        except Exception:
            pass
    _get_db()["jobs"].update_one(
        {"job_id": job_id},
        {"$set": updates},
    )
