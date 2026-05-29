"""
Festival Festival Reviewer
Run locally: python review_app.py
Deploy:      docker build + gcloud run deploy

Cost optimisations applied:
  1. SQLite film library — analysis JSON stored in film_judging.db per film.
     Repeat reviews (same film, different festival) skip Gemini video upload
     entirely and only call the cheap text review-writing step (~0.01¢ vs ~1.6¢).
     Dedup by screener URL or title+director — works across festivals and
     across the batch pipeline (pipeline.py shares the same DB).
  2. Lite model for review writing — gemini-2.5-flash-lite used for the
     text-only review step (prose quality identical, ~70% cheaper output tokens).
  3. Thinking disabled on review call — gemini-2.5-flash thinking tokens
     cost $3.50/1M; disabled with thinking_budget=0 on prose generation.
"""

import os, json, uuid, threading, tempfile, time, re, secrets, logging
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from urllib.parse import urlparse
import bleach
from flask import Flask, request, jsonify, session, redirect, render_template_string, g
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from werkzeug.security import generate_password_hash, check_password_hash
from google import genai
from google.genai import types
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

from festivals import FESTIVALS as SEED_FESTIVALS, DEFAULT_FESTIVAL, get_festival, DOMAIN_FESTIVAL_MAP
from prompts import build_analysis_prompt, build_review_prompt
from wordpress import publish_review as wp_publish_review, publish_post as wp_publish_post
import db

load_dotenv()

app = Flask(__name__)

flask_secret = os.getenv("FLASK_SECRET", "")
if not flask_secret:
    flask_secret = secrets.token_hex(32)
    logging.warning("[security] FLASK_SECRET not set — using ephemeral key; sessions will reset on restart")
app.secret_key = flask_secret

# ── Session hardening ─────────────────────────────────────
app.config.update(
    SESSION_COOKIE_HTTPONLY  = True,
    SESSION_COOKIE_SAMESITE  = "Lax",
    SESSION_COOKIE_SECURE    = os.getenv("HTTPS", "false").lower() == "true",
    PERMANENT_SESSION_LIFETIME = 3600 * 8,  # 8 hours
    MAX_CONTENT_LENGTH       = 1900 * 1024 * 1024,  # 1.9 GB hard Flask limit
)

# ── Rate limiter ──────────────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per minute"],
    storage_uri="memory://",
)

# ── Config ────────────────────────────────────────────────
MAX_UPLOAD_MB  = 1800
ALLOWED_EXT    = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".mpeg"}

# Field length caps (prevent storage/API abuse)
MAX_FIELD_LEN  = 500
MAX_TEXTAREA_LEN = 5000

# Lite model used for text-only review writing (cheaper, same prose quality)
REVIEW_MODEL = "gemini-2.5-flash-lite"

# ── Input sanitisation helpers ────────────────────────────
_ALLOWED_TAGS: list[str] = []  # strip ALL HTML from text fields

def _sanitise(value: str, max_len: int = MAX_FIELD_LEN) -> str:
    """Strip HTML tags and truncate to max_len."""
    return bleach.clean(value, tags=_ALLOWED_TAGS, strip=True)[:max_len]

def _safe_url(url: str) -> str:
    """Allow only http/https schemes to block SSRF via file://, ftp://, etc."""
    url = url.strip()[:2048]
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"URL scheme '{parsed.scheme}' not allowed")
    return url

def _safe_slug(value: str) -> str:
    """Only allow safe slug characters."""
    slug = re.sub(r"[^a-z0-9_]", "", value.strip().lower())[:64]
    if not slug:
        raise ValueError("Slug must contain alphanumeric characters")
    return slug

def get_festivals() -> dict:
    """Load festivals from MongoDB, merging env-var API keys by slug."""
    rows = db.festival_list()
    result = {}
    for f in rows:
        key = f["key"]
        env = key.upper()
        f["gemini_api_key"] = (
            f.get("gemini_api_key") or
            os.getenv(f"{env}_GEMINI_KEY") or
            os.getenv("GEMINI_API_KEY", "")
        )
        f.setdefault("gemini_model", "gemini-2.5-flash")
        f.setdefault("word_count", 500)
        result[key] = f
    return result


def _seed_admin():
    """On first run, create admin accounts from env vars if DB is empty."""
    try:
        if db.user_count() > 0:
            return
        # Support ADMIN_1_EMAIL/ADMIN_1_PASS ... ADMIN_9_EMAIL/ADMIN_9_PASS
        # plus legacy ADMIN_EMAIL/ADMIN_PASS
        seeded = False
        for i in range(1, 10):
            email = os.getenv(f"ADMIN_{i}_EMAIL", "")
            pw    = os.getenv(f"ADMIN_{i}_PASS",  "")
            if email and pw:
                db.user_create(email, generate_password_hash(pw), role="admin")
                print(f"[init] Admin created: {email}")
                seeded = True
        if not seeded:
            email = os.getenv("ADMIN_EMAIL", "admin@reviewportal.com")
            pw    = os.getenv("ADMIN_PASS",  "change_me_admin")
            db.user_create(email, generate_password_hash(pw), role="admin")
            print(f"[init] Admin created: {email}")
    except Exception as e:
        print(f"[init] Admin seed deferred: {e}")


def _seed_festivals():
    """On first run, seed festivals from festivals.py into MongoDB."""
    try:
        if db.festival_count() > 0:
            return
        for key, cfg in SEED_FESTIVALS.items():
            db.festival_upsert(key, cfg)
            print(f"[init] Festival seeded: {key}")
    except Exception as e:
        print(f"[init] Festival seed deferred: {e}")


# ── Security headers on every response ───────────────────
@app.after_request
def set_security_headers(resp):
    resp.headers["X-Content-Type-Options"]    = "nosniff"
    resp.headers["X-Frame-Options"]           = "DENY"
    resp.headers["Referrer-Policy"]           = "strict-origin-when-cross-origin"
    resp.headers["Permissions-Policy"]        = "camera=(), microphone=(), geolocation=()"
    resp.headers["Content-Security-Policy"]   = (
        "default-src 'self' https://fonts.googleapis.com https://fonts.gstatic.com; "
        "script-src 'self' 'unsafe-inline'; "   # inline JS in templates — tighten if moving to external .js
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "img-src 'self' data:; "
        "connect-src 'self';"
    )
    if os.getenv("HTTPS", "false").lower() == "true":
        resp.headers["Strict-Transport-Security"] = "max-age=63072000; includeSubDomains; preload"
    return resp


# ── Auth ──────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper


def admin_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect("/login")
        if session.get("role") != "admin":
            return redirect("/")
        return f(*args, **kwargs)
    return wrapper


# ── Helpers ───────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    # Cap input to prevent ReDoS on huge / malformed LLM responses
    raw = raw[:65536]
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Find the first complete {...} block non-greedily to avoid catastrophic backtracking
        start = clean.find("{")
        end   = clean.rfind("}")
        if start != -1 and end > start:
            return json.loads(clean[start:end + 1])
        raise


def _write_review(client: genai.Client, meta: dict,
                  analysis: dict, festival: dict) -> str:
    """Text-only review generation using the lite model — cheapest step."""
    resp = client.models.generate_content(
        model=REVIEW_MODEL,
        contents=build_review_prompt(meta, analysis, festival),
        config=types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=2048,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    return resp.text.strip()


# ── Processing thread (new video) ─────────────────────────
def process_video(job_id: str, video_path: str, meta: dict):
    """
    Two modes controlled by GCS_BUCKET env var:

    GCS mode (GCS_BUCKET set):
      yt-dlp → stream to GCS → Gemini reads gs:// URI → GCS blob deleted in finally.
      No local disk used beyond the already-saved upload temp file.
      The local temp file is also deleted in finally.

    Local mode (GCS_BUCKET not set):
      Local file uploaded to Gemini Files API → Gemini deletes the file after analysis.
    """
    from downloader import get_video_duration, stream_to_gcs, delete_from_gcs

    GCS_BUCKET = os.getenv("GCS_BUCKET", "")

    festival  = get_festival(meta.get("festival_key", DEFAULT_FESTIVAL))
    client    = genai.Client(api_key=festival["gemini_api_key"])
    model_id  = festival.get("gemini_model", "gemini-2.5-flash")
    uploaded  = None   # Gemini Files API handle (local mode only)
    gcs_blob  = None   # GCS blob name (GCS mode only)

    try:
        film_id = meta.get("film_id") or uuid.uuid4().hex
        screener_url_meta = meta.get("screener_url", "")
        password          = meta.get("screener_password", "")

        # ── 0. Acquire video & extract runtime ────────────────────────────
        if not video_path:
            # Link mode — download from screener_url first
            db.job_update(job_id, {"status": "uploading", "progress": 10,
                                    "message": "Fetching video from link..."})
            from downloader import download_screener
            dl = download_screener(screener_url_meta, password, film_id)
            if not dl["success"]:
                raise RuntimeError(f"Could not download video from link: {screener_url_meta}")
            video_path   = dl["path"]
            duration_min = dl["duration_min"]
        else:
            duration_min = get_video_duration(video_path)

        runtime_str = f"{round(duration_min)} min"
        meta = {**meta, "runtime": runtime_str, "runtime_min": duration_min}
        db.film_update_meta(film_id, {"runtime": runtime_str, "runtime_min": duration_min})

        # ── 1. Get video into Gemini ───────────────────────────────────────
        db.job_update(job_id, {"status": "uploading", "progress": 20,
                                "message": f"Uploading to Gemini [{festival['name']}]..."})

        if GCS_BUCKET:
            gcs_blob   = f"tmp/{film_id}.mp4"
            gs_uri, _  = stream_to_gcs(screener_url_meta, password, GCS_BUCKET, gcs_blob)
            video_part = types.Part.from_uri(file_uri=gs_uri, mime_type="video/mp4")
        else:
            uploaded   = client.files.upload(file=video_path)
            db.job_update(job_id, {"status": "processing", "progress": 35,
                                    "message": "Gemini is watching the film..."})
            while uploaded.state.name == "PROCESSING":
                time.sleep(6)
                uploaded = client.files.get(name=uploaded.name)
            if uploaded.state.name != "ACTIVE":
                raise RuntimeError(f"Gemini processing failed: {uploaded.state.name}")
            video_part = uploaded

        # ── 2. Analyse ─────────────────────────────────────────────────────
        db.job_update(job_id, {"status": "analysing", "progress": 55,
                                "message": "Analysing story, direction, technical..."})
        analysis_resp = client.models.generate_content(
            model=model_id,
            contents=[video_part, build_analysis_prompt(festival)],
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024),
        )
        analysis = _parse_json(analysis_resp.text)

        # ── 3. Persist to DB ───────────────────────────────────────────────
        film_id = meta.get("film_id") or uuid.uuid4().hex
        db.film_save_analysis(film_id, analysis)
        db.film_update_meta(film_id, meta)

        # ── 4. Write review ────────────────────────────────────────────────
        db.job_update(job_id, {"status": "writing", "progress": 78,
                                "message": "Writing Expert Review..."})
        review = _write_review(client, meta, analysis, festival)
        db.review_upsert(film_id, meta.get("festival_key", DEFAULT_FESTIVAL), review)

        db.job_update(job_id, {
            "status": "done", "progress": 100, "message": "Review ready",
            "analysis": analysis, "review": review,
            "film_id": film_id,
            "completed_at": datetime.now(timezone.utc),
        })

    except Exception as e:
        db.job_update(job_id, {"status": "error", "progress": 0, "message": str(e)[:300]})
    finally:
        # Always clean up — GCS blob and local temp file
        if gcs_blob and GCS_BUCKET:
            delete_from_gcs(GCS_BUCKET, gcs_blob)
        if uploaded:
            try: client.files.delete(name=uploaded.name)
            except Exception: pass
        Path(video_path).unlink(missing_ok=True)


# ── Rewrite thread (cached analysis) ──────────────────────
def process_rewrite(job_id: str, film: dict, meta: dict):
    """
    Skip video upload entirely — use stored analysis, only call review writer.
    Cost: ~0.01¢ instead of ~1.6¢.
    """
    festival = get_festival(meta.get("festival_key", DEFAULT_FESTIVAL))
    client   = genai.Client(api_key=festival["gemini_api_key"])

    try:
        db.job_update(job_id, {"status": "writing", "progress": 60,
                                "message": f"Rewriting review for {festival['name']} (no video re-upload)..."})

        merged_meta = {**film, **meta}
        review = _write_review(client, merged_meta, film["analysis"], festival)
        db.review_upsert(film["film_id"], meta.get("festival_key", DEFAULT_FESTIVAL), review)

        db.job_update(job_id, {
            "status": "done", "progress": 100, "message": "Review ready",
            "analysis": film["analysis"], "review": review,
            "film_id": film["film_id"],
            "from_cache": True,
            "completed_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        db.job_update(job_id, {"status": "error", "progress": 0, "message": str(e)[:300]})


# ── Routes ────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
@limiter.limit("10 per minute")
def login():
    error = ""
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()[:254]
        password = request.form.get("password", "")
        try:
            _seed_admin()
            _seed_festivals()
            user = db.user_get(email)
            if user and check_password_hash(user["password_hash"], password):
                # Regenerate session ID on login to prevent session fixation
                session.clear()
                role = user.get("role", "user")
                session["logged_in"] = True
                session["user"]      = email
                session["role"]      = role
                session.permanent    = True
                logging.info("[auth] Login success: %s role=%s ip=%s", email, role, request.remote_addr)
                return redirect("/admin" if role == "admin" else "/")
            logging.warning("[auth] Login fail: %s ip=%s", email, request.remote_addr)
        except Exception as exc:
            logging.error("[auth] Login error: %s", exc)
        error = "Invalid credentials"
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
@login_required
def index():
    # Admins belong on /admin, not the review form
    if session.get("role") == "admin":
        return redirect("/admin")
    user_doc = db.user_get(session.get("user", "")) or {}
    user_role        = user_doc.get("role", "user")
    festivals        = get_festivals()
    user_festival    = user_doc.get("festival_key", "") or DEFAULT_FESTIVAL
    if user_festival not in festivals:
        user_festival = next(iter(festivals), DEFAULT_FESTIVAL)
    return render_template_string(APP_HTML,
                                  festival=festivals.get(DEFAULT_FESTIVAL, {}).get("name", "Festival Reviewer"),
                                  festivals=festivals,
                                  default_festival=DEFAULT_FESTIVAL,
                                  user=session.get("user", ""),
                                  user_role=user_role,
                                  user_festival=user_festival)


@app.route("/api/films")
@login_required
def api_films():
    """Return all films in the DB (for the frontend picker)."""
    return jsonify(db.film_list())


@app.route("/upload", methods=["POST"])
@login_required
@limiter.limit("5 per minute; 30 per hour")
def upload():
    """New film — full video analysis + review.
    Accepts either a screener_url (YouTube/Vimeo link) or a direct video file upload.
    """
    # ── Sanitise text fields ──────────────────────────────
    title    = _sanitise(request.form.get("title",    ""))
    director = _sanitise(request.form.get("director", ""))
    genre    = _sanitise(request.form.get("genre",    ""))
    logline  = _sanitise(request.form.get("logline",  ""), MAX_TEXTAREA_LEN)
    dir_stmt = _sanitise(request.form.get("director_statement", ""), MAX_TEXTAREA_LEN)
    synopsis = _sanitise(request.form.get("synopsis", ""), MAX_TEXTAREA_LEN)

    if not title or not director:
        return jsonify({"error": "Title and director are required"}), 400
    if not genre:
        return jsonify({"error": "Genre is required"}), 400

    screener_url      = request.form.get("screener_url", "").strip()
    screener_password = _sanitise(request.form.get("screener_password", ""), 256)
    has_file          = "video" in request.files and request.files["video"].filename

    if not screener_url and not has_file:
        return jsonify({"error": "Provide a video link or upload a file"}), 400

    # ── Validate URL scheme (block SSRF via file://, gopher://, etc.) ──
    if screener_url:
        try:
            screener_url = _safe_url(screener_url)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

    # ── Festival access control ───────────────────────────
    festivals_db  = get_festivals()
    user_role     = session.get("role", "user")
    user_festival = db.user_get(session.get("user", "")) or {}
    user_fk       = user_festival.get("festival_key", "")

    festival_key = request.form.get("festival_key", "").strip()
    if user_role != "admin":
        # Non-admins are locked to their assigned festival — ignore client input
        festival_key = user_fk or DEFAULT_FESTIVAL
    if festival_key not in festivals_db:
        festival_key = next(iter(festivals_db), DEFAULT_FESTIVAL)
    festival = festivals_db[festival_key]

    if not festival.get("gemini_api_key"):
        return jsonify({"error": f"No Gemini API key configured for {festival['name']}"}), 400

    # Save uploaded file to a temp path (link mode: empty string — process_video will download)
    if has_file:
        f   = request.files["video"]
        # Validate extension from original filename only (ignore path components)
        ext = Path(Path(f.filename).name).suffix.lower()
        if ext not in ALLOWED_EXT:
            return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXT)}"}), 400
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        f.save(tmp.name)
        size_mb = Path(tmp.name).stat().st_size / (1024 * 1024)
        if size_mb > MAX_UPLOAD_MB:
            Path(tmp.name).unlink(missing_ok=True)
            return jsonify({"error": f"File too large ({size_mb:.0f}MB). Max {MAX_UPLOAD_MB}MB"}), 400
        video_path = tmp.name
    else:
        video_path = ""   # process_video will download from screener_url

    film_id = uuid.uuid4().hex
    db.film_create({
        "film_id":            film_id,
        "title":              title,
        "director":           director,
        "logline":            logline,
        "director_statement": dir_stmt,
        "genre":              genre,
        "runtime":            "",
        "screener_url":       screener_url,
    })

    meta = {
        "film_id":            film_id,
        "title":              title,
        "director":           director,
        "logline":            logline,
        "director_statement": dir_stmt,
        "genre":              genre,
        "synopsis":           synopsis,
        "screener_url":       screener_url,
        "screener_password":  screener_password,
        "festival_key":       festival_key,
        "festival_name":      festival["name"],
    }
    logging.info("[upload] job queued film_id=%s festival=%s user=%s", film_id, festival_key, session.get("user"))

    job_id = uuid.uuid4().hex
    db.job_create(job_id, {
        "status": "queued", "progress": 5,
        "message": "Queued for processing...",
        "meta": meta, "analysis": None, "review": None,
    })
    threading.Thread(target=process_video,
                     args=(job_id, video_path, meta), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.route("/rewrite", methods=["POST"])
@login_required
def rewrite():
    """
    Rewrite review for an existing film using cached analysis.
    No video upload — only the cheap text review step runs.
    """
    film_id = request.form.get("film_id", "").strip()
    film    = db.film_get(film_id)
    if not film:
        return jsonify({"error": "Film not found in library"}), 404
    if not film.get("analysis"):
        return jsonify({"error": "No cached analysis for this film"}), 400

    festivals_db = get_festivals()
    festival_key = request.form.get("festival_key", DEFAULT_FESTIVAL)
    if festival_key not in festivals_db:
        festival_key = next(iter(festivals_db), DEFAULT_FESTIVAL)
    festival = festivals_db[festival_key]

    if not festival.get("gemini_api_key"):
        return jsonify({"error": f"No Gemini API key configured for {festival['name']}"}), 400

    # Allow overriding any film field from the form
    meta = {
        "film_id":            film_id,
        "title":              request.form.get("title", film.get("title", "")).strip(),
        "director":           request.form.get("director", film.get("director", "")).strip(),
        "logline":            request.form.get("logline", film.get("logline", "")).strip(),
        "director_statement": request.form.get("director_statement", film.get("director_statement", "")).strip(),
        "genre":              request.form.get("genre", film.get("genre", "")).strip(),
        "runtime":            film.get("runtime", ""),
        "festival_key":       festival_key,
        "festival_name":      festival["name"],
    }

    job_id = uuid.uuid4().hex
    db.job_create(job_id, {
        "status": "queued", "progress": 5,
        "message": "Loading cached analysis...",
        "meta": meta, "analysis": None, "review": None,
    })
    threading.Thread(target=process_rewrite,
                     args=(job_id, film, meta), daemon=True).start()
    return jsonify({"job_id": job_id, "from_cache": True})


@app.route("/status/<job_id>")
@login_required
def status(job_id):
    job = db.job_get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    # Strip internal file paths from error messages before sending to browser
    if job.get("status") == "error" and job.get("message"):
        msg = re.sub(r"(/[^\s]+)", "[path]", job["message"])
        job = {**job, "message": msg}
    return jsonify(job)


@app.route("/publish/<job_id>", methods=["POST"])
@login_required
def publish(job_id):
    job = db.job_get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Review not ready"}), 400

    festival  = get_festival(job["meta"].get("festival_key", DEFAULT_FESTIVAL))
    wp_status = (request.json or {}).get("status", "draft")
    result    = wp_publish_review(meta=job["meta"], analysis=job["analysis"],
                                  review=job["review"], festival=festival, status=wp_status)

    if result["success"]:
        db.job_update(job_id, {
            "wp_post_id": str(result["post_id"]),
            "wp_url":     result["url"],
            "wp_status":  wp_status,
        })
        film_id = job["meta"].get("film_id")
        if film_id:
            db.review_set_wp(film_id, job["meta"].get("festival_key", DEFAULT_FESTIVAL),
                             str(result["post_id"]), result["url"])
    return jsonify(result)


@app.route("/publish_live/<job_id>", methods=["POST"])
@login_required
def publish_live(job_id):
    job     = db.job_get(job_id)
    post_id = job.get("wp_post_id") if job else None
    if not post_id:
        return jsonify({"error": "Not published to WordPress yet"}), 400
    festival = get_festival(job["meta"].get("festival_key", DEFAULT_FESTIVAL))
    return jsonify(wp_publish_post(post_id, festival))


# ── Admin routes ─────────────────────────────────────────
@app.route("/admin")
@admin_required
def admin():
    users = db.user_list()
    return render_template_string(ADMIN_HTML,
                                  users=users,
                                  festivals=get_festivals(),
                                  domain_festival_map=DOMAIN_FESTIVAL_MAP,
                                  current_user=session.get("user", ""))


@app.route("/admin/users", methods=["POST"])
@admin_required
@limiter.limit("20 per minute")
def admin_add_user():
    email       = request.form.get("email", "").strip().lower()[:254]
    password    = request.form.get("password", "").strip()
    role        = request.form.get("role", "user")
    festival_key = request.form.get("festival_key", "").strip()
    if not email or not password:
        return redirect("/admin?error=Email+and+password+required")
    if len(password) < 8:
        return redirect("/admin?error=Password+must+be+at+least+8+characters")
    if role not in ("admin", "user"):
        role = "user"
    if festival_key not in get_festivals():
        festival_key = ""
    ok = db.user_create(email, generate_password_hash(password), role=role, festival_key=festival_key)
    if not ok:
        return redirect("/admin?error=User+already+exists")
    logging.info("[admin] User created: %s role=%s by %s", email, role, session.get("user"))
    return redirect("/admin?success=User+added")


@app.route("/admin/users/delete", methods=["POST"])
@admin_required
@limiter.limit("20 per minute")
def admin_delete_user():
    email = request.form.get("email", "").strip().lower()
    if email == session.get("user"):
        return redirect("/admin?error=Cannot+delete+your+own+account")
    db.user_delete(email)
    logging.info("[admin] User deleted: %s by %s", email, session.get("user"))
    return redirect("/admin?success=User+deleted")


def _festival_fields_from_form() -> dict:
    """Extract all editable festival fields from request.form."""
    raw_domains = request.form.get("email_domains", "")
    email_domains = [
        re.sub(r"[^a-z0-9.\-]", "", d.strip().lower())
        for d in raw_domains.split(",")
        if d.strip()
    ][:20]  # cap at 20 domains

    wp_url = request.form.get("wp_url", "").strip()
    if wp_url:
        try:
            wp_url = _safe_url(wp_url)
        except ValueError:
            wp_url = ""

    return {
        "name":           _sanitise(request.form.get("name", "")),
        "full_name":      _sanitise(request.form.get("full_name", "")),
        "focus":          _sanitise(request.form.get("focus", ""), MAX_TEXTAREA_LEN),
        "tone":           _sanitise(request.form.get("tone", "")) or "professional, honest, and encouraging",
        "analysis_focus": _sanitise(request.form.get("analysis_focus", ""), MAX_TEXTAREA_LEN),
        "review_prompt":  _sanitise(request.form.get("review_prompt", ""), MAX_TEXTAREA_LEN),
        "email_domains":  email_domains,
        "gemini_api_key": request.form.get("gemini_api_key", "").strip()[:256],
        "gemini_model":   "gemini-2.5-flash",
        "word_count":     500,
        "wp_url":         wp_url,
        "wp_user":        _sanitise(request.form.get("wp_user", "")) or "admin",
        "wp_app_pass":    request.form.get("wp_app_pass", "").strip()[:256],
    }


@app.route("/admin/festivals", methods=["POST"])
@admin_required
@limiter.limit("20 per minute")
def admin_create_festival():
    try:
        key = _safe_slug(request.form.get("key", ""))
    except ValueError:
        return redirect("/admin?error=Invalid+festival+key")
    fields = _festival_fields_from_form()
    if not fields["name"] or not fields["focus"]:
        return redirect("/admin?error=Festival+name+and+focus+are+required")
    fields["full_name"] = fields["full_name"] or fields["name"]
    db.festival_upsert(key, fields)
    logging.info("[admin] Festival created: %s by %s", key, session.get("user"))
    return redirect("/admin?success=Festival+created")


@app.route("/admin/festivals/edit", methods=["POST"])
@admin_required
@limiter.limit("30 per minute")
def admin_edit_festival():
    try:
        key = _safe_slug(request.form.get("key", ""))
    except ValueError:
        return redirect("/admin?error=Invalid+festival+key")
    existing = db.festival_get(key)
    if not existing:
        return redirect("/admin?error=Festival+not+found")
    fields = _festival_fields_from_form()
    fields["name"]      = fields["name"]      or existing.get("name", key)
    fields["full_name"] = fields["full_name"] or existing.get("full_name", key)
    fields["focus"]     = fields["focus"]     or existing.get("focus", "")
    db.festival_upsert(key, {**existing, **fields})
    logging.info("[admin] Festival updated: %s by %s", key, session.get("user"))
    return redirect("/admin?success=Festival+updated")


@app.route("/admin/festivals/delete", methods=["POST"])
@admin_required
@limiter.limit("10 per minute")
def admin_delete_festival():
    try:
        key = _safe_slug(request.form.get("key", ""))
    except ValueError:
        return redirect("/admin?error=Invalid+festival+key")
    db.festival_delete(key)
    logging.info("[admin] Festival deleted: %s by %s", key, session.get("user"))
    return redirect("/admin?success=Festival+deleted")


# ── HTML ──────────────────────────────────────────────────
LOGIN_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Festival Reviewer — Sign In</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#e0dbd0;font-family:'DM Sans',sans-serif;
     min-height:100vh;display:flex;align-items:center;justify-content:center;
     background-image:radial-gradient(ellipse 60% 50% at 50% 0%,rgba(201,168,76,.07),transparent)}
.box{width:380px;background:#13131a;border:1px solid rgba(201,168,76,.2);border-radius:16px;padding:40px;text-align:center}
.logo{font-family:'Bebas Neue',sans-serif;font-size:36px;color:#C9A84C;letter-spacing:2px;margin-bottom:4px}
.sub{font-size:12px;color:#6a6560;font-family:'DM Mono',monospace;letter-spacing:2px;text-transform:uppercase;margin-bottom:32px}
label{display:block;text-align:left;font-size:11px;color:#6a6560;letter-spacing:1.5px;text-transform:uppercase;font-family:'DM Mono',monospace;margin-bottom:6px}
input{width:100%;background:#1a1a24;border:1px solid rgba(255,255,255,.08);border-radius:8px;padding:12px 14px;color:#e0dbd0;font-family:'DM Sans',sans-serif;font-size:14px;outline:none;margin-bottom:16px;transition:border-color .2s}
input:focus{border-color:#C9A84C}
button{width:100%;background:#C9A84C;color:#000;border:none;border-radius:8px;padding:13px;font-family:'DM Sans',sans-serif;font-weight:600;font-size:14px;cursor:pointer;margin-top:8px;transition:background .2s}
button:hover{background:#e8c97a}
.error{background:rgba(224,90,90,.1);border:1px solid rgba(224,90,90,.3);border-radius:8px;padding:10px;font-size:13px;color:#e08080;margin-bottom:16px}
</style></head><body>
<div class="box">
  <div class="logo">Festival Reviewer</div>
  <div class="sub">AI-Powered Film Review</div>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <label>Email</label>
    <input type="email" name="email" placeholder="your@email.com" required autofocus>
    <label>Password</label>
    <input type="password" name="password" placeholder="••••••••" required>
    <button type="submit">Sign In</button>
  </form>
</div></body></html>"""


ADMIN_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Admin — User Management</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#e0dbd0;font-family:'DM Sans',sans-serif;min-height:100vh;
     background-image:radial-gradient(ellipse 80% 40% at 50% 0%,rgba(201,168,76,.06),transparent)}
:root{--gold:#C9A84C;--gold-l:#E8C97A;--gold-d:#8A6E2F;
      --bg:#0a0a0f;--bg2:#13131a;--bg3:#1a1a24;
      --border:rgba(201,168,76,.15);--text:#e0dbd0;--muted:#6a6560;
      --green:#4caf7a;--red:#e05a5a}
.header{background:var(--bg2);border-bottom:1px solid var(--border);
        padding:14px 28px;display:flex;align-items:center;justify-content:space-between}
.header-logo{font-family:'Bebas Neue',sans-serif;font-size:22px;color:var(--gold);letter-spacing:2px}
.header-right{display:flex;align-items:center;gap:16px}
.nav-link{font-size:11px;color:var(--muted);text-decoration:none;font-family:'DM Mono',monospace;
          letter-spacing:1px;transition:color .2s}
.nav-link:hover,.nav-link.active{color:var(--gold)}
.main{max-width:780px;margin:0 auto;padding:32px 20px}
.page-title{font-family:'Bebas Neue',sans-serif;font-size:40px;color:var(--gold);letter-spacing:1px;margin-bottom:4px}
.page-sub{font-size:13px;color:var(--muted);margin-bottom:28px}
.card{background:var(--bg2);border:1px solid var(--border);border-radius:14px;overflow:hidden;margin-bottom:20px}
.card-head{padding:16px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
.card-head-icon{width:28px;height:28px;border-radius:6px;background:rgba(201,168,76,.1);
                display:flex;align-items:center;justify-content:center;font-size:14px}
.card-head-title{font-size:12px;letter-spacing:2px;color:var(--gold-d);text-transform:uppercase;
                 font-family:'DM Mono',monospace;font-weight:500}
.card-body{padding:20px 24px}
.form-row{display:grid;grid-template-columns:1fr 1fr auto auto;gap:10px;align-items:end}
label{font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;
      font-family:'DM Mono',monospace;display:block;margin-bottom:5px}
input,select{background:var(--bg3);border:1px solid rgba(255,255,255,.07);border-radius:8px;
             padding:10px 12px;color:var(--text);font-family:'DM Sans',sans-serif;
             font-size:13px;outline:none;width:100%;transition:border-color .2s}
input:focus,select:focus{border-color:var(--gold)}
select option{background:var(--bg3)}
.btn{border:none;border-radius:8px;padding:10px 18px;font-family:'DM Sans',sans-serif;
     font-weight:600;font-size:13px;cursor:pointer;transition:all .2s;white-space:nowrap}
.btn-gold{background:var(--gold);color:#000}.btn-gold:hover{background:var(--gold-l)}
.btn-red{background:rgba(224,90,90,.15);color:var(--red);border:1px solid rgba(224,90,90,.25)}
.btn-red:hover{background:rgba(224,90,90,.25)}
.user-table{width:100%;border-collapse:collapse}
.user-table th{text-align:left;font-size:10px;color:var(--muted);letter-spacing:1.5px;
               text-transform:uppercase;font-family:'DM Mono',monospace;padding:0 0 12px;
               border-bottom:1px solid var(--border)}
.user-table td{padding:13px 0;border-bottom:1px solid rgba(255,255,255,.04);
               font-size:13px;vertical-align:middle}
.user-table tr:last-child td{border-bottom:none}
.role-badge{display:inline-flex;align-items:center;padding:2px 10px;border-radius:20px;
            font-size:10px;font-family:'DM Mono',monospace;letter-spacing:1px;text-transform:uppercase}
.role-admin{background:rgba(201,168,76,.12);color:var(--gold);border:1px solid rgba(201,168,76,.25)}
.role-user{background:rgba(255,255,255,.05);color:var(--muted);border:1px solid rgba(255,255,255,.08)}
.alert{border-radius:8px;padding:10px 16px;font-size:13px;margin-bottom:16px}
.alert-success{background:rgba(76,175,122,.08);border:1px solid rgba(76,175,122,.25);color:#4caf7a}
.alert-error{background:rgba(224,90,90,.08);border:1px solid rgba(224,90,90,.25);color:#e05a5a}
.empty{color:var(--muted);font-size:13px;font-family:'DM Mono',monospace;padding:16px 0}
.section-label{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;font-family:'DM Mono',monospace;color:var(--gold);margin-bottom:8px}
</style></head><body>
<div class="header">
  <div class="header-logo">Festival Reviewer</div>
  <div class="header-right">
    <a href="/" class="nav-link">← Back to Reviews</a>
    <a href="/logout" class="nav-link">Sign out</a>
  </div>
</div>
<div class="main">
  <div class="page-title">User Management</div>
  <div class="page-sub">Add and remove portal users. Admins can access this panel.</div>

  {% set msg_success = request.args.get('success') %}
  {% set msg_error   = request.args.get('error') %}
  {% if msg_success %}<div class="alert alert-success">✓ {{ msg_success }}</div>{% endif %}
  {% if msg_error   %}<div class="alert alert-error">✕ {{ msg_error }}</div>{% endif %}

  <!-- Add user -->
  <div class="card">
    <div class="card-head">
      <div class="card-head-icon">➕</div>
      <div class="card-head-title">Add User</div>
    </div>
    <div class="card-body">
      <form method="POST" action="/admin/users">
        <div class="form-row">
          <div>
            <label>Email *</label>
            <input type="email" id="newUserEmail" name="email" placeholder="user@festival.com" required oninput="autoDetectFestival(this.value)">
          </div>
          <div>
            <label>Password *</label>
            <input type="password" name="password" placeholder="••••••••" required minlength="8">
          </div>
          <div>
            <label>Role</label>
            <select name="role" id="newUserRole" onchange="toggleFestivalSelect()">
              <option value="user">User</option>
              <option value="admin">Admin</option>
            </select>
          </div>
          <div id="festivalSelectWrap">
            <label>Festival</label>
            <select name="festival_key" id="newUserFestival">
              <option value="">— any (admin) —</option>
              {% for key, f in festivals.items() %}
              <option value="{{ key }}">{{ f.name }}</option>
              {% endfor %}
            </select>
          </div>
          <div>
            <label>&nbsp;</label>
            <button type="submit" class="btn btn-gold">Add User</button>
          </div>
        </div>
      </form>
      <script>
      const domainMap = {{ domain_festival_map | tojson }};
      function autoDetectFestival(email) {
        const domain = email.split('@')[1] || '';
        const fk = domainMap[domain];
        if (fk) document.getElementById('newUserFestival').value = fk;
      }
      function toggleFestivalSelect() {
        const isAdmin = document.getElementById('newUserRole').value === 'admin';
        document.getElementById('festivalSelectWrap').style.opacity = isAdmin ? '0.3' : '1';
        document.getElementById('newUserFestival').disabled = isAdmin;
      }
      </script>
    </div>
  </div>

  <!-- User list -->
  <div class="card">
    <div class="card-head">
      <div class="card-head-icon">👥</div>
      <div class="card-head-title">All Users ({{ users|length }})</div>
    </div>
    <div class="card-body">
      {% if users %}
      <table class="user-table">
        <thead>
          <tr>
            <th>Email</th>
            <th>Role</th>
            <th>Festival</th>
            <th>Created</th>
            <th></th>
          </tr>
        </thead>
        <tbody>
          {% for u in users %}
          <tr>
            <td>{{ u.email }}</td>
            <td><span class="role-badge role-{{ u.role }}">{{ u.role }}</span></td>
            <td style="color:var(--muted);font-family:'DM Mono',monospace;font-size:11px">
              {% if u.role == 'admin' %}
                <span style="color:var(--gold)">all</span>
              {% elif u.festival_key and u.festival_key in festivals %}
                {{ festivals[u.festival_key].name }}
              {% else %}
                —
              {% endif %}
            </td>
            <td style="color:var(--muted);font-family:'DM Mono',monospace;font-size:11px">
              {{ u.created_at[:10] if u.created_at else '—' }}
            </td>
            <td style="text-align:right">
              {% if u.email != current_user %}
              <form method="POST" action="/admin/users/delete" style="display:inline"
                    onsubmit="return confirm('Delete {{ u.email }}?')">
                <input type="hidden" name="email" value="{{ u.email }}">
                <button type="submit" class="btn btn-red">Delete</button>
              </form>
              {% else %}
              <span style="font-size:11px;color:var(--muted);font-family:'DM Mono',monospace">you</span>
              {% endif %}
            </td>
          </tr>
          {% endfor %}
        </tbody>
      </table>
      {% else %}
      <div class="empty">No users yet. Add one above.</div>
      {% endif %}
    </div>
  </div>

  <!-- Festival management -->
  <div class="card" style="margin-top:32px">
    <div class="card-head">
      <div class="card-head-icon">🎪</div>
      <div class="card-head-title">Add Festival</div>
    </div>
    <div class="card-body">
      <form method="POST" action="/admin/festivals">
        <div class="section-label">Identity</div>
        <div class="form-row">
          <div>
            <label>Key (slug) *</label>
            <input type="text" name="key" placeholder="e.g. sundance" required pattern="[a-z0-9_]+" title="lowercase letters, numbers, underscores — used in URLs and DB">
          </div>
          <div>
            <label>Short Name *</label>
            <input type="text" name="name" placeholder="e.g. Sundance" required>
          </div>
          <div style="flex:2">
            <label>Full Name</label>
            <input type="text" name="full_name" placeholder="e.g. Sundance Film Festival (defaults to Short Name)">
          </div>
        </div>
        <div class="form-row" style="margin-top:12px">
          <div style="flex:2">
            <label>Focus / Tagline *</label>
            <input type="text" name="focus" placeholder="e.g. independent and art-house cinema" required>
          </div>
          <div style="flex:2">
            <label>Review Tone</label>
            <input type="text" name="tone" placeholder="e.g. collegial, honest, encouraging — like a respected peer">
          </div>
        </div>

        <div class="section-label" style="margin-top:20px">AI Behaviour</div>
        <div style="margin-top:4px">
          <label>Analysis Focus <span style="color:var(--muted);font-weight:400">(what Gemini emphasises when scoring)</span></label>
          <textarea name="analysis_focus" rows="4" placeholder="- Prioritise visual storytelling&#10;- Assess thematic depth and distinctive voice&#10;- Evaluate emotional resonance" style="width:100%;background:var(--bg3);border:1px solid rgba(255,255,255,.07);border-radius:8px;color:var(--text);font-family:'DM Mono',monospace;font-size:12px;padding:10px 14px;resize:vertical;line-height:1.6;margin-top:4px"></textarea>
        </div>
        <div style="margin-top:12px">
          <label>Review Writing Prompt <span style="color:var(--muted);font-weight:400">(festival-specific instructions for the written review)</span></label>
          <textarea name="review_prompt" rows="5" placeholder="- Emphasise the film's artistic vision&#10;- Discuss visual language and how it serves the story&#10;- Close with specific festival circuit suggestions" style="width:100%;background:var(--bg3);border:1px solid rgba(255,255,255,.07);border-radius:8px;color:var(--text);font-family:'DM Mono',monospace;font-size:12px;padding:10px 14px;resize:vertical;line-height:1.6;margin-top:4px"></textarea>
          <div style="font-size:11px;color:var(--muted);margin-top:4px;font-family:'DM Mono',monospace">Reviews are capped at 500 words for all festivals.</div>
        </div>

        <div class="section-label" style="margin-top:20px">Access & Integration</div>
        <div class="form-row" style="margin-top:4px">
          <div>
            <label>Email Domains</label>
            <input type="text" name="email_domains" placeholder="festival.com, festival.org">
          </div>
          <div>
            <label>Gemini API Key</label>
            <input type="text" name="gemini_api_key" placeholder="Leave blank to use global key">
          </div>
        </div>
        <div class="form-row" style="margin-top:12px">
          <div style="flex:2">
            <label>WordPress URL</label>
            <input type="url" name="wp_url" placeholder="https://your-festival-blog.com">
          </div>
          <div>
            <label>WP Username</label>
            <input type="text" name="wp_user" placeholder="admin">
          </div>
          <div>
            <label>WP App Password</label>
            <input type="text" name="wp_app_pass" placeholder="xxxx xxxx xxxx xxxx xxxx xxxx">
          </div>
        </div>

        <div style="margin-top:16px">
          <button type="submit" class="btn btn-gold">Create Festival</button>
        </div>
      </form>
    </div>
  </div>

  <!-- Festival list -->
  <div class="card" style="margin-top:16px">
    <div class="card-head">
      <div class="card-head-icon">🗂</div>
      <div class="card-head-title">Festivals ({{ festivals|length }})</div>
    </div>
    <div class="card-body">
      {% if festivals %}
      {% for key, f in festivals.items() %}
      <details style="border:1px solid var(--border);border-radius:8px;margin-bottom:10px;overflow:hidden">
        <summary style="padding:12px 16px;cursor:pointer;list-style:none;display:flex;align-items:center;gap:10px;background:var(--bg3)">
          <span style="font-weight:600;color:var(--text)">{{ f.name }}</span>
          <span style="font-size:11px;font-family:'DM Mono',monospace;color:var(--muted)">{{ key }}</span>
          <span style="font-size:11px;color:var(--muted);margin-left:auto;max-width:360px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">{{ f.focus }}</span>
        </summary>
        <div style="padding:16px;background:var(--bg4)">
          <form method="POST" action="/admin/festivals/edit">
            <input type="hidden" name="key" value="{{ key }}">

            <div class="section-label">Identity</div>
            <div class="form-row">
              <div style="flex:2">
                <label>Focus / Tagline</label>
                <input type="text" name="focus" value="{{ f.focus }}">
              </div>
              <div style="flex:2">
                <label>Review Tone</label>
                <input type="text" name="tone" value="{{ f.tone or '' }}" placeholder="professional, honest, and encouraging">
              </div>
            </div>

            <div class="section-label" style="margin-top:16px">AI Behaviour</div>
            <div style="margin-top:4px">
              <label>Analysis Focus</label>
              <textarea name="analysis_focus" rows="4" style="width:100%;background:var(--bg3);border:1px solid rgba(255,255,255,.07);border-radius:8px;color:var(--text);font-family:'DM Mono',monospace;font-size:12px;padding:10px 14px;resize:vertical;line-height:1.6;margin-top:4px">{{ f.analysis_focus or '' }}</textarea>
            </div>
            <div style="margin-top:12px">
              <label>Review Writing Prompt</label>
              <textarea name="review_prompt" rows="5" style="width:100%;background:var(--bg3);border:1px solid rgba(255,255,255,.07);border-radius:8px;color:var(--text);font-family:'DM Mono',monospace;font-size:12px;padding:10px 14px;resize:vertical;line-height:1.6;margin-top:4px">{{ f.review_prompt or '' }}</textarea>
            </div>

            <div class="section-label" style="margin-top:16px">Access & Integration</div>
            <div class="form-row" style="margin-top:4px">
              <div>
                <label>Email Domains</label>
                <input type="text" name="email_domains" value="{{ f.email_domains | join(', ') }}">
              </div>
              <div>
                <label>Gemini API Key</label>
                <input type="text" name="gemini_api_key" value="{{ f.gemini_api_key or '' }}" placeholder="Leave blank to use global key">
              </div>
            </div>
            <div class="form-row" style="margin-top:12px">
              <div style="flex:2">
                <label>WordPress URL</label>
                <input type="url" name="wp_url" value="{{ f.wp_url or '' }}" placeholder="https://your-festival-blog.com">
              </div>
              <div>
                <label>WP Username</label>
                <input type="text" name="wp_user" value="{{ f.wp_user or '' }}" placeholder="admin">
              </div>
              <div>
                <label>WP App Password</label>
                <input type="text" name="wp_app_pass" value="{{ f.wp_app_pass or '' }}" placeholder="xxxx xxxx xxxx xxxx xxxx xxxx">
              </div>
            </div>

            <div style="display:flex;gap:10px;margin-top:14px">
              <button type="submit" class="btn btn-gold">Save Changes</button>
            </div>
          </form>
          <form method="POST" action="/admin/festivals/delete" style="margin-top:10px"
                onsubmit="return confirm('Delete {{ f.name }}? This cannot be undone.')">
            <input type="hidden" name="key" value="{{ key }}">
            <button type="submit" class="btn btn-red">Delete Festival</button>
          </form>
        </div>
      </details>
      {% endfor %}
      {% else %}
      <div class="empty">No festivals yet. Create one above.</div>
      {% endif %}
    </div>
  </div>
</div>
</body></html>"""


APP_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ festival }} — Festival Reviewer</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;1,300&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
:root{--gold:#C9A84C;--gold-l:#E8C97A;--gold-d:#8A6E2F;
     --bg:#0a0a0f;--bg2:#13131a;--bg3:#1a1a24;--bg4:#1e1e2c;
     --border:rgba(201,168,76,.15);--text:#e0dbd0;--muted:#6a6560;
     --green:#4caf7a;--red:#e05a5a;--blue:#5a8fe0}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;min-height:100vh;
     background-image:radial-gradient(ellipse 80% 40% at 50% 0%,rgba(201,168,76,.06),transparent)}

.header{background:var(--bg2);border-bottom:1px solid var(--border);
        padding:14px 28px;display:flex;align-items:center;justify-content:space-between}
.header-logo{font-family:'Bebas Neue',sans-serif;font-size:22px;color:var(--gold);letter-spacing:2px}
.header-right{display:flex;align-items:center;gap:16px}
.user-badge{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace}
.logout{font-size:11px;color:var(--gold-d);text-decoration:none;font-family:'DM Mono',monospace;letter-spacing:1px;transition:color .2s}
.logout:hover{color:var(--gold)}

.main{max-width:960px;margin:0 auto;padding:32px 20px}
.page-title{font-family:'Bebas Neue',sans-serif;font-size:40px;color:var(--gold);letter-spacing:1px;margin-bottom:4px}
.page-sub{font-size:13px;color:var(--muted);margin-bottom:28px}

.card{background:var(--bg2);border:1px solid var(--border);border-radius:14px;overflow:hidden;margin-bottom:20px}
.card-head{padding:18px 24px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:10px}
.card-head-icon{width:28px;height:28px;border-radius:6px;background:rgba(201,168,76,.1);
                display:flex;align-items:center;justify-content:center;font-size:14px}
.card-head-title{font-size:12px;letter-spacing:2px;color:var(--gold-d);text-transform:uppercase;
                 font-family:'DM Mono',monospace;font-weight:500}
.card-body{padding:20px 24px}

/* ── Film library picker ── */
.library-picker{background:var(--bg3);border:1px solid rgba(201,168,76,.2);
                border-radius:10px;padding:14px 16px;margin-bottom:20px}
.library-label{font-size:10px;color:var(--gold-d);letter-spacing:2px;text-transform:uppercase;
               font-family:'DM Mono',monospace;margin-bottom:8px}
.library-select{width:100%;background:var(--bg2);border:1px solid rgba(255,255,255,.07);
                border-radius:8px;padding:10px 12px;color:var(--text);font-family:'DM Sans',sans-serif;
                font-size:13px;outline:none;transition:border-color .2s}
.library-select:focus{border-color:var(--gold)}
.library-hint{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace;margin-top:6px;min-height:16px}
.cache-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(76,175,122,.1);
             border:1px solid rgba(76,175,122,.25);border-radius:6px;padding:3px 10px;
             font-size:11px;color:var(--green);font-family:'DM Mono',monospace;margin-top:6px}

/* ── Form ── */
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.form-group{display:flex;flex-direction:column;gap:6px}
.form-group.full{grid-column:1/-1}
label{font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;font-family:'DM Mono',monospace}
.optional-tag{font-size:9px;color:var(--muted);opacity:.6;font-family:'DM Mono',monospace;margin-left:4px}
input[type=text],input[type=number],select,textarea{
  background:var(--bg3);border:1px solid rgba(255,255,255,.07);border-radius:8px;
  padding:10px 12px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:13px;
  outline:none;width:100%;transition:border-color .2s;resize:vertical}
input[type=text]:focus,input[type=number]:focus,select:focus,textarea:focus{border-color:var(--gold)}
select option{background:var(--bg3)}
textarea{min-height:80px;line-height:1.5}
.festival-hint{font-size:11px;color:var(--muted);margin-top:4px;font-family:'DM Mono',monospace;min-height:16px}

/* ── Source toggle ── */
.source-toggle{display:flex;gap:0;border:1px solid rgba(255,255,255,.1);border-radius:10px;overflow:hidden;margin-bottom:20px}
.source-btn{flex:1;padding:11px 0;font-family:'DM Sans',sans-serif;font-size:13px;font-weight:600;
            background:transparent;border:none;color:var(--muted);cursor:pointer;transition:all .2s;letter-spacing:.2px}
.source-btn.active{background:var(--gold);color:#000}
.source-btn:not(.active):hover{background:rgba(255,255,255,.05);color:var(--text)}
.link-input-wrap{display:flex;flex-direction:column;gap:8px}
.link-input-wrap input{background:var(--bg3);border:1px solid rgba(255,255,255,.07);border-radius:8px;
  padding:12px 14px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:13px;outline:none;
  width:100%;transition:border-color .2s}
.link-input-wrap input:focus{border-color:var(--gold)}
.link-hint{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace}

/* ── Drop zone ── */
.drop-zone{border:2px dashed rgba(201,168,76,.25);border-radius:12px;padding:40px 24px;
           text-align:center;cursor:pointer;transition:all .2s;position:relative;background:var(--bg3)}
.drop-zone.drag-over{border-color:var(--gold);background:rgba(201,168,76,.05)}
.drop-zone.has-file{border-color:rgba(76,175,122,.4);background:rgba(76,175,122,.04)}
.drop-icon{font-size:36px;margin-bottom:12px;opacity:.6}
.drop-title{font-size:15px;font-weight:600;color:var(--text);margin-bottom:4px}
.drop-sub{font-size:12px;color:var(--muted)}
.file-info{font-size:12px;color:var(--green);font-family:'DM Mono',monospace;margin-top:8px;font-weight:500}
input[type=file]{display:none}

.submit-btn{width:100%;background:var(--gold);color:#000;border:none;border-radius:10px;
            padding:14px;font-family:'DM Sans',sans-serif;font-weight:700;font-size:15px;
            cursor:pointer;margin-top:4px;transition:all .2s;letter-spacing:.3px}
.submit-btn:hover:not(:disabled){background:var(--gold-l);transform:translateY(-1px)}
.submit-btn:disabled{opacity:.4;cursor:not-allowed;transform:none}
.rewrite-btn{width:100%;background:var(--green);color:#000;border:none;border-radius:10px;
             padding:14px;font-family:'DM Sans',sans-serif;font-weight:700;font-size:15px;
             cursor:pointer;margin-top:4px;transition:all .2s;letter-spacing:.3px}
.rewrite-btn:hover{background:#5fe090;transform:translateY(-1px)}

/* ── Progress ── */
.progress-card{display:none}.progress-card.active{display:block}
.progress-status{display:flex;align-items:center;gap:12px;margin-bottom:16px}
.spinner{width:20px;height:20px;border:2px solid rgba(201,168,76,.2);border-top-color:var(--gold);
         border-radius:50%;animation:spin .8s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}
.progress-msg{font-size:13px;color:var(--text)}
.progress-pct{font-family:'DM Mono',monospace;font-size:12px;color:var(--gold);margin-left:auto}
.progress-track{height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--gold-d),var(--gold),var(--gold-l));
               border-radius:2px;transition:width .4s ease;box-shadow:0 0 8px rgba(201,168,76,.4)}
.step-list{display:flex;flex-direction:column;gap:8px;margin-top:16px}
.step{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--muted);font-family:'DM Mono',monospace}
.step.active{color:var(--text)}.step.done{color:var(--green)}
.step-dot{width:6px;height:6px;border-radius:50%;background:currentColor;flex-shrink:0}

/* ── Results ── */
.results-card{display:none}.results-card.active{display:block}
.scores-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}
.score-box{background:var(--bg3);border-radius:10px;padding:14px;text-align:center}
.score-label{font-size:9px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;
             font-family:'DM Mono',monospace;display:block;margin-bottom:6px}
.score-num{font-family:'Bebas Neue',sans-serif;font-size:32px;color:var(--gold);line-height:1}
.score-denom{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace}
.overall{background:linear-gradient(135deg,rgba(201,168,76,.1),rgba(201,168,76,.03));border:1px solid rgba(201,168,76,.25)}
.obs-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}
.obs-item{background:var(--bg3);border-radius:8px;padding:12px}
.obs-label{font-size:9px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;
           font-family:'DM Mono',monospace;display:block;margin-bottom:5px}
.obs-text{font-size:12px;color:var(--text);line-height:1.5}
.standout{border-left:2px solid var(--gold);padding-left:12px}
.weakest{border-left:2px solid rgba(224,90,90,.5);padding-left:12px}
.review-block{background:var(--bg3);border-radius:10px;padding:18px;position:relative}
.review-text{font-size:14px;line-height:1.8;color:var(--text);white-space:pre-wrap;font-family:'DM Sans',sans-serif}
.copy-btn{position:absolute;top:12px;right:12px;background:rgba(201,168,76,.1);border:1px solid var(--border);
          border-radius:6px;padding:6px 12px;color:var(--gold);font-size:11px;font-family:'DM Mono',monospace;
          cursor:pointer;transition:all .2s}
.copy-btn:hover{background:rgba(201,168,76,.2)}.copy-btn.copied{color:var(--green);border-color:rgba(76,175,122,.3)}
.new-btn{width:100%;background:transparent;border:1px solid var(--border);border-radius:10px;padding:12px;
         color:var(--muted);font-family:'DM Sans',sans-serif;font-size:14px;cursor:pointer;margin-top:12px;transition:all .2s}
.new-btn:hover{border-color:var(--gold);color:var(--text)}
.error-msg{background:rgba(224,90,90,.06);border:1px solid rgba(224,90,90,.2);border-radius:8px;
           padding:12px 16px;font-size:13px;color:#e08080;display:none}
.error-msg.active{display:block}
.film-tag{display:inline-flex;align-items:center;gap:6px;background:rgba(201,168,76,.08);
          border:1px solid var(--border);border-radius:20px;padding:4px 12px;font-size:11px;
          color:var(--gold-l);font-family:'DM Mono',monospace;margin-bottom:16px}

@media(max-width:640px){
  .form-grid,.scores-row,.obs-grid{grid-template-columns:1fr}
  .form-group.full{grid-column:1}
}
</style></head><body>

<div class="header">
  <div class="header-logo">Festival Reviewer</div>
  <div class="header-right">
    <span class="user-badge">{{ user }}</span>
    {% if session.get('role') == 'admin' %}<a href="/admin" class="logout" style="color:var(--gold)">Admin</a>{% endif %}
    <a href="/logout" class="logout">Sign out</a>
  </div>
</div>

<div class="main">
  <div class="page-title">Expert Review</div>
  <div class="page-sub">Upload a film and generate an AI-assisted Expert Review for the selected festival</div>

  <!-- ── FORM ── -->
  <div id="formSection">

    <!-- Festival picker -->
    <div class="card">
      <div class="card-head">
        <div class="card-head-icon">🎪</div>
        <div class="card-head-title">Festival</div>
      </div>
      <div class="card-body">
        <div class="form-group">
          {% if user_role == 'admin' %}
          <label>Select Festival *</label>
          <select id="festivalKey" onchange="onFestivalChange(this)">
            <option value="" disabled selected>— Select a festival —</option>
            {% for key, f in festivals.items() %}
            <option value="{{ key }}">{{ f.name }} — {{ f.focus }}</option>
            {% endfor %}
          </select>
          {% else %}
          <input type="hidden" id="festivalKey" value="{{ user_festival }}">
          <div style="font-size:15px;font-weight:600;color:var(--gold)">
            {{ festivals[user_festival].name if user_festival in festivals else user_festival }}
          </div>
          <div class="festival-hint" style="margin-top:4px">
            {{ festivals[user_festival].focus if user_festival in festivals else '' }}
          </div>
          {% endif %}
          <div class="festival-hint" id="festivalHint"></div>
        </div>
      </div>
    </div>

    <!-- Film details -->
    <div class="card" id="formCard" style="display:none">
      <div class="card-head">
        <div class="card-head-icon">🎬</div>
        <div class="card-head-title">Film Details</div>
      </div>
      <div class="card-body">
        <div class="form-grid">
          <div class="form-group">
            <label>Film Title *</label>
            <input type="text" id="title" placeholder="Love Will Set You Free" required>
          </div>
          <div class="form-group">
            <label>Director *</label>
            <input type="text" id="director" placeholder="Jane Smith" required>
          </div>
          <div class="form-group">
            <label>Genre *</label>
            <input type="text" id="genre" placeholder="e.g. Drama, Documentary, Animation..." required>
          </div>
          <div class="form-group full">
            <label>Logline <span class="optional-tag">optional</span></label>
            <input type="text" id="logline" placeholder="A one-sentence summary of the film...">
          </div>
          <div class="form-group full">
            <label>Director's Statement <span class="optional-tag">optional</span></label>
            <textarea id="director_statement" placeholder="The director's artistic intention or context..."></textarea>
          </div>
          <div class="form-group full">
            <label>Synopsis <span class="optional-tag">optional</span></label>
            <textarea id="synopsis" placeholder="Brief description of the film..."></textarea>
          </div>
        </div>
      </div>
    </div>

    <!-- Video source -->
    <div class="card" id="uploadCard" style="display:none">
      <div class="card-head">
        <div class="card-head-icon">🎞</div>
        <div class="card-head-title">Video Source</div>
      </div>
      <div class="card-body">
        <!-- Toggle -->
        <div class="source-toggle">
          <button class="source-btn active" id="btnLink"   onclick="setSource('link')">🔗 &nbsp;Paste Link</button>
          <button class="source-btn"        id="btnUpload" onclick="setSource('upload')">📁 &nbsp;Upload File</button>
        </div>

        <!-- Link panel -->
        <div id="linkPanel">
          <div class="link-input-wrap">
            <input type="url" id="screenerUrl" placeholder="https://www.youtube.com/watch?v=... or Vimeo link">
            <div class="link-hint">Supports YouTube, Vimeo, and most public video links</div>
            <input type="text" id="screenerPassword" placeholder="Password (optional — for password-protected Vimeo links)" autocomplete="off" style="margin-top:4px">
          </div>
        </div>

        <!-- Upload panel -->
        <div id="uploadPanel" style="display:none">
          <div class="drop-zone" id="dropZone">
            <div class="drop-icon">🎞</div>
            <div class="drop-title">Drop video file here</div>
            <div class="drop-sub">or click to browse — MP4, MOV, AVI, WebM, MKV</div>
            <div class="file-info" id="fileInfo"></div>
            <input type="file" id="fileInput" accept=".mp4,.mov,.avi,.webm,.mkv,.mpeg">
          </div>
        </div>

        <div class="error-msg" id="errorMsg"></div>
        <button class="submit-btn" id="submitBtn" onclick="submitReview()" disabled>
          Generate Expert Review
        </button>
      </div>
    </div>

  </div><!-- end formSection -->

  <!-- ── PROGRESS ── -->
  <div class="card progress-card" id="progressCard">
    <div class="card-head">
      <div class="card-head-icon">⚡</div>
      <div class="card-head-title">Processing</div>
    </div>
    <div class="card-body">
      <div class="progress-status">
        <div class="spinner"></div>
        <div class="progress-msg" id="progressMsg">Starting...</div>
        <div class="progress-pct" id="progressPct">0%</div>
      </div>
      <div class="progress-track">
        <div class="progress-fill" id="progressFill" style="width:0%"></div>
      </div>
      <div class="step-list" id="stepList">
        <div class="step" id="step-uploading"><div class="step-dot"></div>Uploading to Gemini</div>
        <div class="step" id="step-processing"><div class="step-dot"></div>Gemini watching the film</div>
        <div class="step" id="step-analysing"><div class="step-dot"></div>Analysing story, direction, technical</div>
        <div class="step" id="step-writing"><div class="step-dot"></div>Writing Expert Review</div>
      </div>
    </div>
  </div>

  <!-- ── RESULTS ── -->
  <div class="results-card" id="resultsCard">
    <div class="card">
      <div class="card-head">
        <div class="card-head-icon">📊</div>
        <div class="card-head-title">Analysis Scores</div>
      </div>
      <div class="card-body">
        <div id="filmTag" class="film-tag"></div>
        <div class="scores-row" id="scoresRow"></div>
        <div class="obs-grid" id="obsGrid"></div>
      </div>
    </div>
    <div class="card">
      <div class="card-head">
        <div class="card-head-icon">✍️</div>
        <div class="card-head-title">Expert Review — Review before delivering</div>
      </div>
      <div class="card-body">
        <div class="review-block">
          <button class="copy-btn" id="copyBtn" onclick="copyReview()">Copy</button>
          <div class="review-text" id="reviewText"></div>
        </div>
        <button class="new-btn" onclick="resetForm()">← Generate another review</button>
      </div>
    </div>
  </div>

</div><!-- main -->

<script>
let selectedFile  = null;
let pollInterval  = null;
let sourceMode    = 'link';  // 'link' | 'upload'

const FESTIVALS = {
  {% for key, f in festivals.items() %}
  "{{ key }}": { name:"{{ f.name }}", focus:"{{ f.focus }}", words:{{ f.word_count }} },
  {% endfor %}
};

function onFestivalChange(sel) {
  if (!sel.value) return;
  const f = FESTIVALS[sel.value];
  document.getElementById('festivalHint').textContent =
    f ? `${f.words}-word review · ${f.focus}` : '';
  document.getElementById('formCard').style.display   = 'block';
  document.getElementById('uploadCard').style.display = 'block';
}

// For non-admin users: festival is fixed, reveal form immediately
(function() {
  const userRole = "{{ user_role }}";
  if (userRole !== 'admin') {
    document.getElementById('formCard').style.display   = 'block';
    document.getElementById('uploadCard').style.display = 'block';
  }
})();

function clearFilmFields() {
  ['title','director','logline','director_statement'].forEach(id =>
    document.getElementById(id).value = '');
  document.getElementById('genre').value    = '';
  document.getElementById('synopsis').value = '';
}

// ── Source toggle ──────────────────────────────────────────
function setSource(mode) {
  sourceMode = mode;
  document.getElementById('btnLink').classList.toggle('active',   mode === 'link');
  document.getElementById('btnUpload').classList.toggle('active', mode === 'upload');
  document.getElementById('linkPanel').style.display   = mode === 'link'   ? 'block' : 'none';
  document.getElementById('uploadPanel').style.display = mode === 'upload' ? 'block' : 'none';
  // Re-evaluate submit button
  _updateSubmitBtn();
  showError('');
}

function _updateSubmitBtn() {
  const ready = sourceMode === 'link'
    ? document.getElementById('screenerUrl').value.trim() !== ''
    : selectedFile !== null;
  document.getElementById('submitBtn').disabled = !ready;
}

document.getElementById('screenerUrl').addEventListener('input', _updateSubmitBtn);

// ── Drop zone ──────────────────────────────────────────────
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

function handleFile(file) {
  selectedFile = file;
  const sizeMb = (file.size/1024/1024).toFixed(1);
  document.getElementById('fileInfo').textContent = `✓ ${file.name}  (${sizeMb} MB)`;
  dropZone.classList.add('has-file');
  document.getElementById('submitBtn').disabled = false;
  showError('');
}

// ── Submit: new video ──────────────────────────────────────
async function submitReview() {
  const festivalKey = document.getElementById('festivalKey').value;
  if (!festivalKey) { showError('Please select a festival before submitting'); return; }
  const title    = document.getElementById('title').value.trim();
  const director = document.getElementById('director').value.trim();
  const genre    = document.getElementById('genre').value.trim();
  if (!title || !director) { showError('Film title and director are required'); return; }
  if (!genre)              { showError('Genre is required'); return; }

  const form = new FormData();
  if (sourceMode === 'link') {
    const url = document.getElementById('screenerUrl').value.trim();
    if (!url) { showError('Please paste a video link'); return; }
    form.append('screener_url', url);
    const pw = document.getElementById('screenerPassword').value.trim();
    if (pw) form.append('screener_password', pw);
  } else {
    if (!selectedFile) { showError('Please upload a video file'); return; }
    form.append('video', selectedFile);
  }
  form.append('festival_key',       document.getElementById('festivalKey').value);
  form.append('title',              title);
  form.append('director',           director);
  form.append('logline',            document.getElementById('logline').value);
  form.append('director_statement', document.getElementById('director_statement').value);
  form.append('genre',              genre);
  form.append('synopsis',           document.getElementById('synopsis').value);

  showProcessing();
  try {
    const res  = await fetch('/upload', { method:'POST', body:form });
    const data = await res.json();
    if (data.error) { showFormError(data.error); return; }
    pollStatus(data.job_id);
  } catch(e) { showFormError('Upload failed. Please try again.'); }
}

// ── Polling ────────────────────────────────────────────────
function pollStatus(jobId) {
  clearInterval(pollInterval);
  let elapsed = 0;
  const MAX_WAIT = 20 * 60 * 1000; // 20 min hard timeout
  pollInterval = setInterval(async () => {
    elapsed += 2000;
    if (elapsed >= MAX_WAIT) {
      clearInterval(pollInterval);
      showFormError('Processing timed out. Please try again.');
      return;
    }
    try {
      const res = await fetch(`/status/${jobId}`);
      if (!res.ok) { clearInterval(pollInterval); showFormError('Server error. Please try again.'); return; }
      const job = await res.json();
      updateProgress(job);
      if (job.status === 'done')  { clearInterval(pollInterval); showResults(job); }
      if (job.status === 'error') { clearInterval(pollInterval); showFormError('Processing failed. Please check your video link and try again.'); }
    } catch(e) { /* network blip — keep polling */ }
  }, 2000);
}

// ── Progress UI ────────────────────────────────────────────
const ALL_STEPS = ['uploading','processing','analysing','writing'];

function updateProgress(job) {
  document.getElementById('progressMsg').textContent  = job.message;
  document.getElementById('progressPct').textContent  = job.progress + '%';
  document.getElementById('progressFill').style.width = job.progress + '%';
  ALL_STEPS.forEach(s => {
    const el = document.getElementById('step-' + s);
    if (!el) return;
    el.className = 'step';
    if (s === job.status) el.classList.add('active');
    if (ALL_STEPS.indexOf(s) < ALL_STEPS.indexOf(job.status) || job.status === 'done')
      el.classList.add('done');
  });
}

// ── Safe DOM helpers ───────────────────────────────────────
function esc(str) {
  const d = document.createElement('div');
  d.textContent = String(str ?? '');
  return d.innerHTML; // entities-escaped, safe to inject into innerHTML
}

function _span(text, style) {
  const s = document.createElement('span');
  s.textContent = text;
  if (style) s.style.cssText = style;
  return s;
}

function _scoreBox(label, score, denom, extra) {
  const box = document.createElement('div');
  box.className = 'score-box' + (extra ? ' ' + extra : '');
  const lbl = document.createElement('span');
  lbl.className = 'score-label'; lbl.textContent = label;
  const num = document.createElement('span');
  num.className = 'score-num'; num.textContent = score;
  const den = document.createElement('span');
  den.className = 'score-denom'; den.textContent = '/' + denom;
  const wrap = document.createElement('div');
  wrap.append(num, den);
  box.append(lbl, wrap);
  return box;
}

function _obsItem(label, text, cls, spanStyle) {
  const item = document.createElement('div');
  item.className = 'obs-item' + (cls ? ' ' + cls : '');
  if (spanStyle) item.style.cssText = spanStyle;
  const lbl = document.createElement('span');
  lbl.className = 'obs-label'; lbl.textContent = label;
  const body = document.createElement('div');
  body.className = 'obs-text'; body.textContent = text;
  item.append(lbl, body);
  return item;
}

// ── Results ────────────────────────────────────────────────
function showResults(job) {
  document.getElementById('progressCard').classList.remove('active');
  document.getElementById('resultsCard').classList.add('active');

  const a    = job.analysis || {};
  const meta = job.meta    || {};

  // Film tag — built with textContent, no innerHTML injection
  const filmTag = document.getElementById('filmTag');
  filmTag.textContent = '';
  const sep = () => { const s = document.createElement('span'); s.innerHTML = ' &nbsp;·&nbsp; '; return s; };
  filmTag.append('🎬 ', _span(meta.title || ''), sep(), _span(meta.director || ''));
  if (meta.genre)        { filmTag.append(sep(), _span(meta.genre)); }
  if (meta.runtime)      { filmTag.append(sep(), _span(meta.runtime)); }
  if (meta.festival_name){ filmTag.append(sep(), _span(meta.festival_name, 'color:var(--gold)')); }
  if (job.from_cache)    { filmTag.append(sep(), _span('⚡ cached', 'color:var(--green);font-size:10px')); }

  // Score boxes
  const scoresRow = document.getElementById('scoresRow');
  scoresRow.textContent = '';
  [{k:'story',l:'Story'},{k:'direction',l:'Direction'},{k:'technical',l:'Technical'},{k:'originality',l:'Originality'}]
    .forEach(c => scoresRow.append(_scoreBox(c.l, (a[c.k] || {}).score ?? '—', 5)));
  scoresRow.append(_scoreBox('Overall', a.overall_score ?? '—', 20, 'overall'));

  // Observation grid
  const obsGrid = document.getElementById('obsGrid');
  obsGrid.textContent = '';
  obsGrid.append(
    _obsItem('Standout Moment', a.standout_moment || '', 'standout'),
    _obsItem('Growth Area',     a.weakest_element || '', 'weakest'),
    _obsItem('Festival Suitability', a.festival_suitability || '', '', 'grid-column:1/-1'),
  );

  // Review text — already uses textContent
  document.getElementById('reviewText').textContent = job.review || '';
}

// ── Copy ───────────────────────────────────────────────────
function copyReview() {
  navigator.clipboard.writeText(document.getElementById('reviewText').textContent).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = '✓ Copied'; btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
  });
}

// ── Helpers ────────────────────────────────────────────────
function showProcessing() {
  document.getElementById('formSection').style.display = 'none';
  document.getElementById('progressCard').classList.add('active');
  document.getElementById('progressFill').style.width = '5%';
}

function showError(msg) {
  const el = document.getElementById('errorMsg');
  el.textContent = msg; el.className = 'error-msg' + (msg ? ' active' : '');
}

function showFormError(msg) {
  document.getElementById('progressCard').classList.remove('active');
  document.getElementById('formSection').style.display = 'block';
  showError(msg);
}

function resetForm() {
  document.getElementById('resultsCard').classList.remove('active');
  document.getElementById('formSection').style.display = 'block';
  clearFilmFields();
  document.getElementById('fileInfo').textContent  = '';
  document.getElementById('screenerUrl').value      = '';
  document.getElementById('screenerPassword').value = '';
  document.getElementById('submitBtn').disabled     = true;
  document.getElementById('festivalKey').value      = '';
  document.getElementById('festivalHint').textContent = '';
  // Hide everything below festival picker until a festival is chosen
  document.getElementById('formCard').style.display   = 'none';
  document.getElementById('uploadCard').style.display = 'none';
  dropZone.classList.remove('has-file');
  selectedFile = null;
  setSource('link');
}
</script>
</body></html>"""


if __name__ == "__main__":
    print(f"Festival Review App — Festival Reviewer")
    print("Local: http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
