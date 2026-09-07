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


_indexes_created = False


def _get_db():
    global _client, _db, _indexes_created
    if _db is None:
        uri     = os.getenv("MONGODB_URI") or os.getenv("MONGO_URI", "mongodb://localhost:27017/festival_reviewer")
        db_name = os.getenv("MONGO_DB_NAME", "")
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000, maxPoolSize=20)
        if db_name:
            _db = _client[db_name]
        elif "/" in uri.rsplit("@", 1)[-1]:
            _db = _client.get_default_database()
        else:
            _db = _client["festival_reviewer"]
    if not _indexes_created:
        _indexes_created = True
        _create_indexes(_db)
    return _db


def _create_indexes(d):
    """Create all indexes. Called once on first DB connection."""

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

    # ── users ──────────────────────────────────────────────────────────────
    d["users"].create_index("email", unique=True, name="uniq_email")

    # ── festivals ──────────────────────────────────────────────────────────
    d["festivals"].create_index("key", unique=True, name="uniq_festival_key")

    # ── seasons ────────────────────────────────────────────────────────────
    seasons = d["seasons"]
    seasons.create_index(
        [("festival_key", ASCENDING), ("name", ASCENDING)],
        unique=True,
        name="uniq_festival_season",
    )
    seasons.create_index("festival_key")

    # sessions collection — Flask-Session creates its own TTL index on "expiration"
    # automatically; we must not create a duplicate here.


def _now() -> datetime:
    return datetime.now(timezone.utc)


def init_db():
    """No-op kept for compatibility — indexes are created lazily on first _get_db() call."""
    pass


def get_mongo_client() -> MongoClient:
    """Return the shared MongoClient (initialised lazily). Used by Flask-Session."""
    _get_db()          # ensure _client is initialised
    return _client     # type: ignore[return-value]


def get_db_name() -> str:
    """Return the database name in use."""
    return _get_db().name


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


def film_list(festival_keys: list[str] | None = None) -> list[dict]:
    """List films. If festival_keys is given, restrict to those festivals
    (tenant scoping). Passing None returns every film — callers that expose
    results to end users MUST pass an explicit key list."""
    query: dict = {}
    if festival_keys is not None:
        query = {"festival_key": {"$in": list(festival_keys)}}
    docs = _get_db()["films"].find(query, sort=[("created_at", DESCENDING)])
    return [_clean_film(d) for d in docs]


def film_get(film_id: str) -> dict | None:
    return _clean_film(_get_db()["films"].find_one({"film_id": film_id}))


def film_find_by_screener(screener_url: str) -> dict | None:
    if not screener_url or not screener_url.strip():
        return None
    return _clean_film(
        _get_db()["films"].find_one({"screener_url": screener_url.strip()})
    )


def film_find_by_identity(title: str, director: str, festival_key: str | None = None,
                           duration_min: float | None = None) -> dict | None:
    """Match by title+director, confined to one festival. If duration_min given,
    also require runtime within ±2 min. When several records share the same
    identity (e.g. earlier failed attempts that never produced an analysis),
    prefer the one that actually has a cached analysis.
    Dedup is intentionally scoped per-festival — the same film submitted to two
    different festivals should be judged independently, not share cached analysis.
    """
    if not title or not director:
        return None
    query: dict = {
        "title_lc":    title.strip().lower(),
        "director_lc": director.strip().lower(),
    }
    if festival_key:
        query["festival_key"] = festival_key
    if duration_min is not None:
        query["runtime_min"] = {"$gte": duration_min - 2, "$lte": duration_min + 2}
    # Prefer a record with a non-empty analysis; fall back to the most recent match
    with_analysis = _get_db()["films"].find_one(
        {**query, "analysis": {"$exists": True, "$nin": [None, {}]}},
        sort=[("created_at", DESCENDING)],
    )
    if with_analysis:
        return _clean_film(with_analysis)
    return _clean_film(_get_db()["films"].find_one(query, sort=[("created_at", DESCENDING)]))


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
        "runtime_min":        data.get("runtime_min"),
        "country":            data.get("country", ""),
        "screener_url":       data.get("screener_url", ""),
        "festival_key":       data.get("festival_key", ""),
        "analysis":           data.get("analysis") or {},
        "analysed_at":        None,
        "created_at":         now,
    }
    _get_db()["films"].insert_one(doc)
    return film_id


def film_backfill_festival_keys() -> int:
    """One-time migration: films created before festival_key was recorded on
    the film document have festival_key missing/empty, which breaks tenant
    scoping (review_detail, api_publish_review, api_films all filter on it).
    Derive festival_key from that film's earliest review and backfill it.
    Returns the number of films updated."""
    films_coll   = _get_db()["films"]
    reviews_coll = _get_db()["reviews"]
    updated = 0
    for film in films_coll.find({"$or": [{"festival_key": {"$exists": False}}, {"festival_key": ""}]},
                                 {"film_id": 1}):
        review = reviews_coll.find_one({"film_id": film["film_id"]}, sort=[("created_at", ASCENDING)])
        if review and review.get("festival_key"):
            films_coll.update_one({"film_id": film["film_id"]},
                                   {"$set": {"festival_key": review["festival_key"]}})
            updated += 1
    return updated


def film_save_analysis(film_id: str, analysis: dict):
    """Write Gemini analysis into an existing film document."""
    _get_db()["films"].update_one(
        {"film_id": film_id},
        {"$set": {"analysis": analysis, "analysed_at": _now()}},
    )


def film_update_meta(film_id: str, data: dict):
    """Update editable metadata fields."""
    fields = ["logline", "director_statement", "genre", "runtime", "runtime_min", "country", "screener_url"]
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


def review_list_for_festival(festival_key: str, season: str = "", limit: int = 200) -> list[dict]:
    """Return all reviews for a festival (optionally filtered by season), newest first, joined with film metadata."""
    query: dict = {"festival_key": festival_key}
    if season:
        query["season"] = season
    reviews = list(_get_db()["reviews"].find(
        query,
        sort=[("created_at", DESCENDING)],
        limit=limit,
    ))
    if not reviews:
        return []
    film_ids = [r["film_id"] for r in reviews]
    films = {
        f["film_id"]: f
        for f in _get_db()["films"].find({"film_id": {"$in": film_ids}})
    }
    result = []
    for r in reviews:
        r = _clean_review(r)
        film = films.get(r["film_id"], {})
        r["title"]     = film.get("title", "—")
        r["director"]  = film.get("director", "—")
        r["genre"]     = film.get("genre", "")
        r["runtime"]   = film.get("runtime", "")
        r["country"]   = film.get("country", "")
        result.append(r)
    return result


def review_upsert(film_id: str, festival_key: str, review_text: str, season: str = "", overall_rating: float | None = None):
    updates: dict = {
        "review_text": review_text,
        "season":      season,
        "created_at":  _now(),
    }
    if overall_rating is not None:
        updates["overall_rating"] = overall_rating
    _get_db()["reviews"].update_one(
        {"film_id": film_id, "festival_key": festival_key},
        {"$set": updates, "$setOnInsert": {"wp_post_id": "", "wp_url": ""}},
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
    updates.setdefault("updated_at", _now())   # always stamp last-modified time
    _get_db()["jobs"].update_one(
        {"job_id": job_id},
        {"$set": updates},
    )


# ── User helpers ──────────────────────────────────────────────────────────────

def _clean_user(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    doc = dict(doc)
    doc.pop("_id", None)
    doc.pop("password_hash", None)          # never leak the hash
    if isinstance(doc.get("created_at"), datetime):
        doc["created_at"] = doc["created_at"].isoformat()
    return doc


def user_list() -> list[dict]:
    docs = _get_db()["users"].find({}, sort=[("created_at", ASCENDING)])
    return [_clean_user(d) for d in docs]


def user_get(email: str) -> dict | None:
    """Return full user doc including password_hash (for auth only)."""
    doc = _get_db()["users"].find_one({"email": email.strip().lower()})
    if doc is None:
        return None
    doc = dict(doc)
    doc.pop("_id", None)
    if isinstance(doc.get("created_at"), datetime):
        doc["created_at"] = doc["created_at"].isoformat()
    return doc


def user_create(email: str, password_hash: str, role: str = "user", festival_key: str = "") -> bool:
    """Insert a new user. Returns False if email already exists."""
    try:
        _get_db()["users"].insert_one({
            "email":         email.strip().lower(),
            "password_hash": password_hash,
            "role":          role,
            "festival_key":  festival_key if role != "admin" else "",
            "created_at":    _now(),
        })
        return True
    except Exception:
        return False


def user_update_password(email: str, password_hash: str) -> bool:
    """Update a user's password hash. Returns False if not found."""
    result = _get_db()["users"].update_one(
        {"email": email.strip().lower()},
        {"$set": {"password_hash": password_hash}}
    )
    return result.matched_count > 0


def user_delete(email: str) -> bool:
    """Delete a user by email. Returns False if not found."""
    result = _get_db()["users"].delete_one({"email": email.strip().lower()})
    return result.deleted_count > 0


def user_count() -> int:
    return _get_db()["users"].count_documents({})


# ── Festival helpers ──────────────────────────────────────────────────────────

def _clean_festival(doc: dict | None) -> dict | None:
    if doc is None:
        return None
    doc = dict(doc)
    doc.pop("_id", None)
    if isinstance(doc.get("created_at"), datetime):
        doc["created_at"] = doc["created_at"].isoformat()
    return doc


def festival_list() -> list[dict]:
    return [_clean_festival(d) for d in _get_db()["festivals"].find({}, sort=[("name", ASCENDING)])]


def festival_get(key: str) -> dict | None:
    return _clean_festival(_get_db()["festivals"].find_one({"key": key}))


def festival_upsert(key: str, data: dict):
    """Insert or update a festival document."""
    doc = {k: v for k, v in data.items() if k != "_id"}
    doc["key"] = key
    _get_db()["festivals"].update_one(
        {"key": key},
        {"$set": doc, "$setOnInsert": {"created_at": _now()}},
        upsert=True,
    )


def festival_delete(key: str) -> bool:
    result = _get_db()["festivals"].delete_one({"key": key})
    return result.deleted_count > 0


def festival_count() -> int:
    return _get_db()["festivals"].count_documents({})


# ── Season helpers ────────────────────────────────────────────────────────────

def season_list(festival_key: str) -> list[dict]:
    docs = _get_db()["seasons"].find(
        {"festival_key": festival_key},
        sort=[("name", ASCENDING)],
    )
    return [{"name": d["name"], "festival_key": d["festival_key"]} for d in docs]


def season_create(festival_key: str, name: str) -> bool:
    """Add a season. Returns False if it already exists."""
    try:
        _get_db()["seasons"].insert_one({
            "festival_key": festival_key,
            "name":         name.strip(),
            "created_at":   _now(),
        })
        return True
    except Exception:
        return False


def season_delete(festival_key: str, name: str) -> bool:
    result = _get_db()["seasons"].delete_one({"festival_key": festival_key, "name": name.strip()})
    return result.deleted_count > 0


def season_rename(festival_key: str, old_name: str, new_name: str) -> bool:
    result = _get_db()["seasons"].update_one(
        {"festival_key": festival_key, "name": old_name.strip()},
        {"$set": {"name": new_name.strip()}},
    )
    return result.matched_count > 0


# ── Category helpers (stored in festival document) ────────────────────────────

def category_list(festival_key: str) -> list[str]:
    doc = festival_get(festival_key)
    return doc.get("categories", []) if doc else []


def category_add(festival_key: str, name: str) -> bool:
    name = name.strip()
    if not name:
        return False
    result = _get_db()["festivals"].update_one(
        {"key": festival_key, "categories": {"$ne": name}},
        {"$push": {"categories": name}},
    )
    return result.modified_count > 0


def category_delete(festival_key: str, name: str) -> bool:
    name = name.strip()
    result = _get_db()["festivals"].update_one(
        {"key": festival_key},
        # also drop any per-category prompt stored for this category
        {"$pull": {"categories": name}, "$unset": {f"category_prompts.{name}": ""}},
    )
    return result.modified_count > 0


def category_rename(festival_key: str, old_name: str, new_name: str) -> bool:
    old_name = old_name.strip()
    new_name = new_name.strip()
    doc = festival_get(festival_key)
    if not doc:
        return False
    cats = doc.get("categories", [])
    if old_name not in cats or new_name in cats:
        return False
    cats = [new_name if c == old_name else c for c in cats]
    # Carry the per-category prompt across the rename
    prompts = doc.get("category_prompts", {}) or {}
    if old_name in prompts:
        prompts[new_name] = prompts.pop(old_name)
    result = _get_db()["festivals"].update_one(
        {"key": festival_key},
        {"$set": {"categories": cats, "category_prompts": prompts}},
    )
    return result.modified_count > 0


def category_get_prompt(festival_key: str, name: str) -> str:
    """Return the per-category judging emphasis for a category, or '' if none set."""
    doc = festival_get(festival_key)
    if not doc:
        return ""
    return (doc.get("category_prompts", {}) or {}).get(name.strip(), "")


def category_set_prompt(festival_key: str, name: str, prompt: str) -> bool:
    """Set (or clear) the per-category judging emphasis for a category."""
    name = name.strip()
    result = _get_db()["festivals"].update_one(
        {"key": festival_key, "categories": name},   # only if the category exists
        {"$set": {f"category_prompts.{name}": prompt}},
    )
    return result.matched_count > 0
