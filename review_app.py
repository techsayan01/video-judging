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
from prompts import build_analysis_prompt, build_review_prompt, _cap_words
import db
import wordpress

# Limit concurrent video-processing threads — keeps memory bounded under 5 simultaneous users
_processing_sem = threading.Semaphore(5)

load_dotenv()

app = Flask(__name__)

flask_secret = os.getenv("FLASK_SECRET", "")
if not flask_secret:
    flask_secret = secrets.token_hex(32)
    logging.warning("[security] FLASK_SECRET not set — using ephemeral key; sessions will reset on restart")
app.secret_key = flask_secret

# ── Server-side session storage (MongoDB) ─────────────────
# Only a signed session-ID cookie is sent to the browser.
# Session data lives in the `sessions` MongoDB collection with a TTL index.
# This survives Cloud Run instance restarts, enables server-side invalidation,
# and works correctly across multiple instances.
app.config.update(
    SESSION_TYPE              = "mongodb",
    SESSION_MONGODB           = None,          # set lazily in _init_session_store()
    SESSION_MONGODB_DB        = None,          # set lazily
    SESSION_MONGODB_COLLECT   = "sessions",
    SESSION_PERMANENT         = True,
    SESSION_USE_SIGNER        = True,          # HMAC-sign the session ID cookie
    SESSION_KEY_PREFIX        = "sess:",
    SESSION_COOKIE_HTTPONLY   = True,
    SESSION_COOKIE_SAMESITE   = "Lax",
    SESSION_COOKIE_SECURE     = os.getenv("HTTPS", "false").lower() == "true",
    PERMANENT_SESSION_LIFETIME = 3600 * 8,    # 8-hour sessions
    MAX_CONTENT_LENGTH        = 4096 * 1024 * 1024,  # 4 GB — feature films can be 2–3 GB
)

from flask_session import Session as FlaskSession  # noqa: E402

def _setup_session_store():
    """Wire Flask-Session to MongoDB at startup (before any request is served).
    Must run after the app config is set but the mongo client is lazy-initialised
    on first use, so we force a connection here.
    """
    client  = db.get_mongo_client()
    db_name = db.get_db_name()
    app.config["SESSION_MONGODB"]    = client
    app.config["SESSION_MONGODB_DB"] = db_name
    # Drop any conflicting TTL index left over from a previous deployment
    try:
        client[db_name]["sessions"].drop_index("ttl_sessions")
    except Exception:
        pass
    FlaskSession(app)
    logging.info("[session] Server-side MongoDB session store initialised")

_setup_session_store()

# ── Rate limiter ──────────────────────────────────────────
limiter = Limiter(
    key_func=get_remote_address,
    app=app,
    default_limits=["200 per minute"],
    storage_uri="memory://",
)

# ── Config ────────────────────────────────────────────────
MAX_UPLOAD_MB  = 4000   # 4 GB — feature films can be 2–3 GB
ALLOWED_EXT    = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".mpeg"}
# Script / written-work categories accept documents instead of video
DOC_EXT        = {".pdf", ".doc", ".docx", ".txt"}
MAX_DOC_MB     = 25     # scripts are small; well under Cloud Run's 32 MB request limit
_SCRIPT_RE     = re.compile(r"\b(script|screenplay|stageplay|teleplay|poem|novel|radio script)\b", re.IGNORECASE)


def _is_script_category(category: str) -> bool:
    """True for written-work categories that should accept document uploads."""
    return bool(category and _SCRIPT_RE.search(category))

# Field length caps (prevent storage/API abuse)
MAX_FIELD_LEN  = 500
MAX_TEXTAREA_LEN = 5000

# Every genai.Client call (file upload, poll, generate_content) previously had no
# timeout — a stalled network call (Gemini API hiccup, transient connectivity issue)
# would block its thread forever holding a _processing_sem permit, with no exception
# and no log line. Once enough threads on an instance got stuck this way, new jobs
# would silently hang at progress=0 until the stale-job watchdog killed them 20
# minutes later. A bounded timeout turns that silent hang into a catchable, logged
# error within a few minutes instead.
_GEMINI_HTTP_OPTIONS = types.HttpOptions(timeout=600_000)  # 10 min, generous for multi-GB uploads

# Safety settings for film analysis — Gemini's defaults (BLOCK_MEDIUM_AND_ABOVE)
# are too aggressive for legitimate cinema: horror, body-horror, thriller, drama,
# and experimental films trigger BlockedReason.OTHER even at BLOCK_ONLY_HIGH.
# BLOCK_NONE is required for the analysis pass so Gemini can watch and critique
# real film content (horror, violence, dark themes) without false-positive blocks.
# The review-writing pass keeps BLOCK_ONLY_HIGH as it generates new text.
_FILM_SAFETY = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_NONE"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_NONE"),
]

_REVIEW_SAFETY = [
    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT",        threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_HATE_SPEECH",       threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_SEXUALLY_EXPLICIT", threshold="BLOCK_ONLY_HIGH"),
    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
]

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
    """Sanitise to URL-safe slug: lowercase, alphanumeric, hyphens and underscores only.
    Spaces and other separators are collapsed to a single hyphen.
    Leading/trailing hyphens are stripped.
    """
    s = value.strip().lower()
    s = re.sub(r"[\s]+", "-", s)           # spaces → hyphen
    s = re.sub(r"[^a-z0-9_-]", "", s)      # drop everything else
    s = re.sub(r"-{2,}", "-", s)            # collapse repeated hyphens
    s = s.strip("-")[:64]
    if not s:
        raise ValueError("Slug must contain alphanumeric characters")
    return s

_OVERALL_RE = re.compile(r"overall\s+rating\s*:\s*(\d+(?:\.\d+)?)\s*/\s*10", re.IGNORECASE)

def _extract_overall_rating(review_text: str) -> float | None:
    """Pull the first 'Overall Rating: X/10' from review text; returns None if not found."""
    m = _OVERALL_RE.search(review_text)
    if not m:
        return None
    val = float(m.group(1))
    return val if 0 <= val <= 10 else None


DEFAULT_REVIEW_PROMPT = """\
Write as a professional film critic and seasoned festival juror — the depth and
specificity of a published reviewer, but encouraging and constructive in tone. Every
filmmaker is emotionally invested in their work: be honest about weaknesses while
framing each as a path forward, never brutal or dismissive.

Produce the review in EXACTLY this structure and order:

Overall Rating: [X]/10

Ratings:
- Originality / Creativity: [0–10]
- Direction: [0–10]
- Writing: [0–10]
- Cinematography: [0–10]
- Performances: [0–10]
- Production Value: [0–10]
- Pacing: [0–10]
- Structure: [0–10]
- Sound / Music: [0–10]
- Average: [mean of the nine scores above, one decimal place]

Comments:
[A deep, professional critique. Lead with what genuinely works and why — name the exact
scene, shot, performance, cut, or sound cue (with timestamps where possible). Then move
into growth areas framed constructively, each paired with a concrete, actionable
suggestion. Address the dimensions that matter most for this particular film with real
critical insight, not surface praise. Where it illuminates a point, draw briefly on
professional craft wisdom. Speak to the filmmaker directly and respectfully, as a peer
who wants them to succeed.]

Recommendation: [Pass | Recommend | Award Worthy | Maybe]
Reason: [One encouraging sentence justifying the recommendation.]

Rules:
- All nine scores are integers 0–10; the Average is their arithmetic mean (one decimal).
- Recommendation must align with the Overall Rating: 8.5–10 → Award Worthy,
  7.0–8.4 → Recommend, 5.0–6.9 → Maybe, below 5 → Pass.
- Recommendation must be exactly one of: Pass, Recommend, Award Worthy, Maybe.
- Do not add extra sections or headings beyond those listed above.
- For a film category that has no dialogue, performers, or score, judge the nearest
  equivalent craft fairly rather than scoring it zero.
- Never be harsh, sarcastic, or dismissive. Critique the work, never the filmmaker.\
"""


# ── Predefined category catalog ───────────────────────────────────────────────
# Each category carries a curated judging emphasis that is injected into the
# analysis + review prompts whenever a film is submitted under that category.
# Admins onboard a festival with just identity; categories (with these prompts)
# are picked from this catalog on the Manage page and remain editable per festival.
PREDEFINED_CATEGORIES = {
    "Narrative Short": "A short fiction film. Judge story economy, how efficiently character and stakes are established, and whether the ending lands. Reward tight structure and a clear emotional arc within the runtime.",
    "Narrative Feature": "A feature-length fiction film. Judge sustained narrative momentum, character development across acts, and tonal consistency. Assess whether the runtime is earned.",
    "Documentary": "A non-fiction film. Judge subject access and authenticity, narrative shaping of real material, ethical handling of subjects, and the strength of the central argument or human story. Do not expect scripted drama.",
    "Experimental / Avant-Garde": "A non-narrative or abstract work. Do NOT penalise the absence of plot, dialogue, or conventional characters. Judge formal invention, visual/sonic rhythm, conceptual coherence, and the boldness of its artistic vision.",
    "Animation": "An animated film of any technique. Judge the craft and consistency of the animation style, world cohesion, timing, and how form serves story. Weigh visual inventiveness highly.",
    "Music Video": "A visual piece set to music. Judge the marriage of image and sound, visual concept, editing to rhythm, and overall mood. Narrative is optional; atmosphere and craft are central.",
    "Drama": "A dramatic narrative. Judge emotional authenticity, performance, subtext in dialogue, and the credibility of character choices under pressure.",
    "Comedy": "A comedic film. Judge comic timing, escalation, originality of premise, and whether the humour lands without straining. Tone control matters most.",
    "Horror / Thriller": "A genre film built on tension. Judge atmosphere, pacing of dread or suspense, restraint vs. spectacle, and the effectiveness of payoffs and scares.",
    "Sci-Fi / Fantasy": "A speculative film. Judge world-building coherence, how convincingly the rules are established, and whether spectacle serves theme rather than replacing it.",
    "Student Film": "Work by an emerging student filmmaker. Apply festival-circuit standards but weight potential and ambition generously; frame growth areas as mentorship for a developing voice.",
    "Dance / Poetry Film": "A film fusing movement or verse with cinema. Judge choreography of camera and subject, rhythm, and how the visual language amplifies the text or movement. Non-narrative is expected.",
}


def get_festivals() -> dict:
    """Load festivals from MongoDB, merging env-var API keys by slug.
    Falls back to the static SEED_FESTIVALS name when the DB name is missing
    or still equals the raw key slug (happens when a festival was upserted
    without an explicit display name).
    """
    rows = db.festival_list()
    result = {}
    for f in rows:
        key = f["key"]
        seed = SEED_FESTIVALS.get(key, {})
        # Always prefer the seed display name for known festivals — the Mongo
        # document may store a short slug like "eleiff" rather than "ElegantIFF"
        if seed.get("name"):
            f["name"] = seed["name"]
        elif not f.get("name"):
            f["name"] = key
        env = key.upper()
        f["gemini_api_key"] = (
            f.get("gemini_api_key") or
            os.getenv(f"{env}_GEMINI_KEY") or
            os.getenv("GEMINI_API_KEY", "")
        )
        f.setdefault("gemini_model", "gemini-2.5-flash")
        f.setdefault("word_count", 500)
        # Admins no longer set review prompts — guarantee a sensible default
        if not (f.get("review_prompt") or "").strip():
            f["review_prompt"] = DEFAULT_REVIEW_PROMPT
        result[key] = f
    return result


def get_domain_festival_map() -> dict:
    """Build domain→festival_key map from live MongoDB data."""
    mapping = {}
    for key, f in get_festivals().items():
        for domain in f.get("email_domains", []):
            d = domain.strip().lower()
            if d:
                mapping[d] = key
    return mapping


def festival_for_email(email: str) -> str | None:
    """Return festival_key whose email_domains exactly matches the email's domain, or None."""
    domain = email.split("@")[-1].lower() if "@" in email else ""
    if not domain:
        return None
    return get_domain_festival_map().get(domain)


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
        "connect-src 'self' https://storage.googleapis.com;"  # direct video upload to GCS
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


def _visible_festival_keys() -> list[str]:
    """Festival keys the current user is allowed to read data for.
    Admins see festivals they created; regular users see only their own.
    Used to tenant-scope film/analysis reads and prevent cross-festival IDOR."""
    user = session.get("user", "")
    if session.get("role") == "admin":
        return list(_admin_festivals(user).keys())
    fk = (db.user_get(user) or {}).get("festival_key", "")
    return [fk] if fk else []


# ── Helpers ───────────────────────────────────────────────
def _friendly_error(e: Exception) -> str:
    """Turn a raw exception into an actionable message. In particular, detect
    invalid/expired/revoked Gemini API keys, which otherwise surface as an
    opaque 400/401/403 from the SDK and get mistaken for content-safety blocks."""
    msg = str(e)
    low = msg.lower()
    if any(s in low for s in ("api key not valid", "api_key_invalid", "invalid api key",
                               "unauthenticated", "permission_denied", "403 forbidden")):
        return ("Gemini API key is invalid, expired, or revoked for this festival. "
                "Ask an admin to check/update the key in Manage Festival → Gemini API Key.")
    if "resource_exhausted" in low or "quota" in low or "429" in low:
        return "Gemini API quota exceeded for this festival's key. Please try again later or ask an admin to check the quota."
    if "input token count exceeds" in low or "exceeds the maximum number of tokens" in low:
        return ("This film is too long for Gemini to analyse in one pass (exceeds its 1M-token "
                "context window). This can happen with long feature-length films even under our "
                "120-minute limit. Please contact an admin to have it processed in chunks.")
    # Unrecognised error: don't leak internal details to the client. Log the full
    # exception server-side and hand the user a correlation id to quote in support.
    ref = uuid.uuid4().hex[:8]
    logging.error("[error] ref=%s %s", ref, msg)
    return f"Something went wrong while processing this film (ref: {ref}). Please try again or contact an admin."


def _gemini_generate_with_retry(client, model_id: str, contents, config,
                                 max_retries: int = 3, base_delay: float = 10.0):
    """Call generate_content with exponential-backoff retry for transient OTHER blocks.
    BlockedReason.OTHER can be transient (quota, internal Gemini policy) and often
    succeeds on retry.  Harm-category blocks (SAFETY) are not retried."""
    import time as _time
    last_exc = None
    for attempt in range(max_retries):
        resp = client.models.generate_content(model=model_id, contents=contents, config=config)
        txt = getattr(resp, "text", None)
        if txt and txt.strip():
            return resp
        # Inspect block reason
        try:
            fb = getattr(resp, "prompt_feedback", None)
            br = getattr(fb, "block_reason", None) if fb else None
            if br is None:
                cand = (getattr(resp, "candidates", None) or [None])[0]
                br = getattr(cand, "finish_reason", None) if cand else None
        except Exception:
            br = None
        br_name = str(br) if br else ""
        # Don't retry definitive safety blocks — only transient OTHER / unknown
        if "SAFETY" in br_name or "PROHIBITED" in br_name:
            return resp  # let _response_text raise the proper error
        if attempt < max_retries - 1:
            delay = base_delay * (2 ** attempt)
            print(f"  [retry] Gemini returned empty ({br_name}), attempt {attempt+1}/{max_retries}, "
                  f"retrying in {delay:.0f}s…", flush=True)
            _time.sleep(delay)
    return resp  # final attempt result, callers handle the empty case


def _response_text(resp) -> str:
    """Return the model's text, or raise a clear error explaining why it's empty.
    Gemini returns no text when a request is blocked (safety/recitation), hits the
    token limit, or otherwise stops without content — guard against that here so
    callers never crash on a None response."""
    txt = getattr(resp, "text", None)
    if txt and txt.strip():
        return txt
    # Diagnose the empty response
    detail = "no content returned"
    try:
        fb = getattr(resp, "prompt_feedback", None)
        if fb and getattr(fb, "block_reason", None):
            detail = f"blocked by safety filter ({fb.block_reason})"
        else:
            cand = (getattr(resp, "candidates", None) or [None])[0]
            fr = getattr(cand, "finish_reason", None)
            if fr:
                detail = f"stopped early (reason: {fr})"
    except Exception:
        pass
    raise RuntimeError(
        f"The AI returned no usable output — {detail}. "
        f"The submission may have been declined by Gemini's safety filters or hit a limit. "
        f"Please try again, or adjust the content if it may have triggered a content filter."
    )


def _parse_json(raw: str) -> dict:
    if not raw or not str(raw).strip():
        raise RuntimeError("The AI returned an empty analysis. Please try again.")
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
            try:
                return json.loads(clean[start:end + 1])
            except json.JSONDecodeError:
                pass
        # Final fallback: auto-repair internally malformed JSON (unescaped quotes,
        # trailing commas, truncated strings) that Gemini occasionally produces.
        try:
            from json_repair import repair_json
            repaired = repair_json(clean[start:end + 1] if start != -1 and end > start else clean)
            result = json.loads(repaired)
            logging.warning("[_parse_json] JSON repaired automatically — Gemini returned malformed JSON")
            return result
        except Exception:
            pass
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
            safety_settings=_REVIEW_SAFETY,
        ),
    )
    return _response_text(resp).strip()


# ── Processing thread (new video) ─────────────────────────
def process_video(job_id: str, video_path: str, meta: dict):
    """
    Three-tier video handling based on duration:

    Tier 1 — ≤ 120 min:  direct Gemini Files API upload (full temporal context).
    Tier 2 — > 120 min, GCS_BUCKET set:  stream to GCS → Gemini reads gs:// URI (no size limit).
    Tier 3 — > 120 min, no GCS:  extract keyframes in batches → multi-call analysis → merge.
    """
    if not _processing_sem.acquire(timeout=0):
        # All 5 slots on this instance are busy — say so explicitly instead of
        # leaving the job silently at progress=0, indistinguishable from a hang.
        db.job_update(job_id, {"status": "queued", "progress": 2,
                                "message": "Waiting for a processing slot (server is busy)…"})
        _processing_sem.acquire()  # now block for real
    from downloader import get_video_duration, stream_to_gcs, delete_from_gcs, extract_keyframes
    from config import DIRECT_UPLOAD_MAX_MIN, CHUNK_FRAMES

    GCS_BUCKET = os.getenv("GCS_BUCKET", "")

    # Use MongoDB-backed config so festivals created via admin panel (not in seed) work correctly
    fk       = meta.get("festival_key", DEFAULT_FESTIVAL)
    festival = get_festivals().get(fk) or get_festival(fk)
    client   = genai.Client(api_key=festival["gemini_api_key"], http_options=_GEMINI_HTTP_OPTIONS)
    model_id = festival.get("gemini_model", "gemini-2.5-flash")

    uploaded_files: list = []   # Gemini Files API handles to clean up
    gcs_blob: str        = ""   # GCS blob name to clean up

    gcs_upload_blob = ""   # browser-uploaded blob to clean up afterwards

    try:
        film_id           = meta.get("film_id") or uuid.uuid4().hex
        screener_url_meta = meta.get("screener_url", "")
        password          = meta.get("screener_password", "")
        gcs_upload_blob   = meta.get("gcs_blob", "")

        # ── 0. Acquire video ──────────────────────────────────────────────
        if gcs_upload_blob:
            # Browser uploaded directly to GCS — pull it down to a local temp file
            db.job_update(job_id, {"status": "downloading", "progress": 10,
                                    "message": "Retrieving uploaded video…"})
            from downloader import download_from_gcs
            bucket = os.getenv("GCS_UPLOAD_BUCKET", "")
            tmp_dl = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False).name
            dl = download_from_gcs(bucket, gcs_upload_blob, tmp_dl)
            if not dl["success"]:
                raise RuntimeError(dl.get("error", "Could not retrieve uploaded video"))
            video_path   = dl["path"]
            duration_min = dl["duration_min"]
        elif not video_path:
            db.job_update(job_id, {"status": "downloading", "progress": 8,
                                    "message": "Fetching video from Vimeo — large files can take 3–8 min…"})
            from downloader import download_screener
            # Bump the status every 30 s while yt-dlp runs so the user sees progress
            import threading as _th
            _dl_result    = {}
            _dl_done      = _th.Event()
            def _do_download():
                _dl_result.update(download_screener(screener_url_meta, password, film_id))
                _dl_done.set()
            _th.Thread(target=_do_download, daemon=True).start()
            elapsed_s = 0
            while not _dl_done.wait(timeout=20):
                elapsed_s += 20
                db.job_update(job_id, {
                    "status": "downloading",
                    "progress": min(8 + elapsed_s // 10, 16),
                    "message": f"Downloading from Vimeo… ({elapsed_s}s elapsed)",
                })
            dl = _dl_result
            if not dl["success"]:
                detail = dl.get("error", "")
                msg = f"Could not download video from link: {screener_url_meta}"
                if detail:
                    msg += f" — {detail[:200]}"
                raise RuntimeError(msg)
            video_path   = dl["path"]
            duration_min = dl["duration_min"]
        else:
            duration_min = get_video_duration(video_path)

        if duration_min > DIRECT_UPLOAD_MAX_MIN:
            raise RuntimeError(
                f"Film is {round(duration_min)} minutes long. "
                f"Only films up to {DIRECT_UPLOAD_MAX_MIN} minutes are accepted."
            )

        runtime_str = f"{round(duration_min)} min"
        meta = {**meta, "runtime": runtime_str, "runtime_min": duration_min}
        db.film_update_meta(film_id, {"runtime": runtime_str, "runtime_min": duration_min})

        # ── 1. Upload to Gemini Files API ────────────────────────────────
        heavy = duration_min > 60
        upload_msg = (
            f"Uploading to Gemini — {round(duration_min)} min film, this may take a few minutes…"
            if heavy else
            f"Uploading to Gemini [{festival['name']}]…"
        )
        db.job_update(job_id, {"status": "uploading", "progress": 18, "message": upload_msg})
        f = client.files.upload(file=video_path)
        uploaded_files.append(f)

        # ── 2. Gemini processes the video (can take 1–5 min for feature films) ──
        db.job_update(job_id, {"status": "processing", "progress": 32,
                                "message": "Gemini is watching the film…"})
        poll_count   = 0
        poll_msgs    = [
            "Gemini is watching the film…",
            "Reading every frame…",
            "Following the narrative…",
            "Studying the cinematography…",
            "Listening to the soundtrack…",
            "Absorbing the performances…",
            "Almost done watching…",
        ]
        while f.state.name == "PROCESSING":
            time.sleep(6)
            poll_count += 1
            f = client.files.get(name=f.name)
            uploaded_files[-1] = f
            # Increment from 32 → 62 as polling continues (capped); cycle through messages
            progress = min(32 + poll_count * 2, 62)
            msg      = poll_msgs[min(poll_count - 1, len(poll_msgs) - 1)]
            db.job_update(job_id, {"status": "processing", "progress": progress, "message": msg})
        if f.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini video processing failed: {f.state.name}")
        analysis_contents = [f, build_analysis_prompt(festival, meta)]

        # ── 3. Score & analyse ────────────────────────────────────────────
        db.job_update(job_id, {"status": "analysing", "progress": 65,
                                "message": "Scoring story, direction & craft…"})
        analysis_resp = _gemini_generate_with_retry(
            client, model_id, analysis_contents,
            types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8192,
                # NOTE: response_mime_type="application/json" conflicts with
                # thinking_budget and produces malformed JSON — omitted intentionally.
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                safety_settings=_FILM_SAFETY,
                # Default video resolution tokenises at ~263 tok/s, which blows
                # Gemini's 1,048,576-token context on anything past ~65 min —
                # well under our 120-min direct-upload ceiling. LOW drops that
                # to ~66 tok/s (~4+ hrs fits in-budget) at a modest detail cost,
                # which is an acceptable trade-off for a token-limit hard failure.
                media_resolution=types.MediaResolution.MEDIA_RESOLUTION_LOW,
            ),
        )
        analysis = _parse_json(_response_text(analysis_resp))

        # ── 4. Persist analysis ────────────────────────────────────────────
        db.job_update(job_id, {"status": "scoring", "progress": 75,
                                "message": "Compiling scores and notes…"})
        db.film_save_analysis(film_id, analysis)
        db.film_update_meta(film_id, meta)

        # ── 5. Write review ────────────────────────────────────────────────
        db.job_update(job_id, {"status": "writing", "progress": 82,
                                "message": "Drafting the Expert Review…"})
        review = _write_review(client, meta, analysis, festival)

        # ── 6. Finalise ────────────────────────────────────────────────────
        db.job_update(job_id, {"status": "finalising", "progress": 94,
                                "message": "Saving review and formatting…"})
        db.review_upsert(film_id, meta.get("festival_key", DEFAULT_FESTIVAL), review,
                         season=meta.get("season", ""),
                         overall_rating=_extract_overall_rating(review))

        db.job_update(job_id, {
            "status": "done", "progress": 100, "message": "Review ready",
            "analysis": analysis, "review": review, "film_id": film_id,
            "completed_at": datetime.now(timezone.utc),
        })

    except Exception as e:
        logging.exception("[process] job %s failed", job_id)
        db.job_update(job_id, {"status": "error", "progress": 0, "message": _friendly_error(e)})
    finally:
        if gcs_blob and GCS_BUCKET:
            delete_from_gcs(GCS_BUCKET, gcs_blob)
        # Clean up the browser-uploaded source blob (bucket also auto-expires in 1 day)
        if gcs_upload_blob:
            try:
                from downloader import delete_from_gcs as _del
                _del(os.getenv("GCS_UPLOAD_BUCKET", ""), gcs_upload_blob)
            except Exception: pass
        for uf in uploaded_files:
            try: client.files.delete(name=uf.name)
            except Exception: pass
        if video_path:
            Path(video_path).unlink(missing_ok=True)
        _processing_sem.release()  # always free the slot


# ── Rewrite thread (cached analysis) ──────────────────────
def process_rewrite(job_id: str, film: dict, meta: dict):
    """
    Skip video upload entirely — use stored analysis, only call review writer.
    Cost: ~0.01¢ instead of ~1.6¢.
    """
    fk       = meta.get("festival_key", DEFAULT_FESTIVAL)
    festival = get_festivals().get(fk) or get_festival(fk)
    client   = genai.Client(api_key=festival["gemini_api_key"], http_options=_GEMINI_HTTP_OPTIONS)

    try:
        db.job_update(job_id, {"status": "writing", "progress": 60,
                                "message": f"Rewriting review for {festival['name']} (no video re-upload)..."})

        merged_meta = {**film, **meta}
        review = _write_review(client, merged_meta, film["analysis"], festival)
        db.review_upsert(film["film_id"], meta.get("festival_key", DEFAULT_FESTIVAL), review,
                         season=meta.get("season", ""),
                         overall_rating=_extract_overall_rating(review))

        db.job_update(job_id, {
            "status": "done", "progress": 100, "message": "Review ready",
            "analysis": film["analysis"], "review": review,
            "film_id": film["film_id"],
            "from_cache": True,
            "completed_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        logging.exception("[process] job %s failed", job_id)
        db.job_update(job_id, {"status": "error", "progress": 0, "message": _friendly_error(e)})


# ── Document (script) processing thread ───────────────────
def _read_document_text(path: str, ext: str) -> str:
    """Extract plain text from a non-PDF document (txt / docx / doc)."""
    if ext == ".txt":
        return Path(path).read_text(errors="ignore")[:200000]
    if ext in (".docx", ".doc"):
        try:
            from docx import Document
            return "\n".join(p.text for p in Document(path).paragraphs)[:200000]
        except Exception:
            raise RuntimeError("Could not read this .doc file. Please upload a PDF, DOCX, or TXT.")
    raise RuntimeError("Unsupported document type.")


def process_document(job_id: str, doc_path: str, ext: str, meta: dict):
    """Analyse a written script/document with Gemini and write a review.
    PDFs go to the Gemini Files API directly; txt/docx are extracted to text."""
    if not _processing_sem.acquire(timeout=0):
        db.job_update(job_id, {"status": "queued", "progress": 2,
                                "message": "Waiting for a processing slot (server is busy)…"})
        _processing_sem.acquire()  # now block for real
    fk       = meta.get("festival_key", DEFAULT_FESTIVAL)
    festival = get_festivals().get(fk) or get_festival(fk)
    client   = genai.Client(api_key=festival["gemini_api_key"], http_options=_GEMINI_HTTP_OPTIONS)
    model_id = festival.get("gemini_model", "gemini-2.5-flash")
    uploaded_files: list = []
    try:
        film_id = meta.get("film_id") or uuid.uuid4().hex
        db.job_update(job_id, {"status": "analysing", "progress": 35,
                                "message": "Reading the script…"})
        prompt = build_analysis_prompt(festival, meta, is_document=True)

        if ext == ".pdf":
            f = client.files.upload(file=doc_path)
            uploaded_files.append(f)
            while getattr(f.state, "name", "ACTIVE") == "PROCESSING":
                time.sleep(3)
                f = client.files.get(name=f.name)
                uploaded_files[-1] = f
            if getattr(f.state, "name", "ACTIVE") not in ("ACTIVE", None):
                raise RuntimeError(f"Gemini could not process the PDF: {f.state.name}")
            contents = [f, prompt]
        else:
            text = _read_document_text(doc_path, ext)
            if not text.strip():
                raise RuntimeError("The document appears to be empty or unreadable.")
            contents = [f"SCRIPT / DOCUMENT CONTENT:\n\n{text}", prompt]

        db.job_update(job_id, {"status": "scoring", "progress": 65,
                                "message": "Scoring the writing…"})
        resp = _gemini_generate_with_retry(
            client, model_id, contents,
            types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=8192,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
                safety_settings=_FILM_SAFETY,
            ),
        )
        analysis = _parse_json(_response_text(resp))

        db.film_save_analysis(film_id, analysis)
        db.film_update_meta(film_id, meta)

        db.job_update(job_id, {"status": "writing", "progress": 82,
                                "message": "Drafting the Expert Review…"})
        review = _write_review(client, meta, analysis, festival)

        db.job_update(job_id, {"status": "finalising", "progress": 94,
                                "message": "Saving review…"})
        db.review_upsert(film_id, fk, review,
                         season=meta.get("season", ""),
                         overall_rating=_extract_overall_rating(review))

        db.job_update(job_id, {
            "status": "done", "progress": 100, "message": "Review ready",
            "analysis": analysis, "review": review, "film_id": film_id,
            "completed_at": datetime.now(timezone.utc),
        })
    except Exception as e:
        logging.exception("[process] job %s failed", job_id)
        db.job_update(job_id, {"status": "error", "progress": 0, "message": _friendly_error(e)})
    finally:
        for uf in uploaded_files:
            try: client.files.delete(name=uf.name)
            except Exception: pass
        if doc_path:
            Path(doc_path).unlink(missing_ok=True)
        _processing_sem.release()


# ── Routes ────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
@limiter.limit("5 per minute; 50 per hour", methods=["POST"])
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
                role = user.get("role", "user")
                # Regenerate session ID on login to prevent session fixation
                session.clear()
                session["logged_in"]    = True
                session["user"]         = email
                session["role"]         = role
                # Store festival_key at login so routes don't need an extra DB lookup.
                # Admins have "" here; their festivals are resolved via created_by.
                session["festival_key"] = user.get("festival_key", "")
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
    # Everyone lands on the dashboard; admins get the admin console.
    if session.get("role") == "admin":
        return redirect("/admin")
    return redirect("/dashboard")


@app.route("/new")
@login_required
def new_review():
    """Film upload + review generation form."""
    user_doc = db.user_get(session.get("user", "")) or {}
    user_role        = user_doc.get("role", "user")
    festivals        = get_festivals()
    user_festival    = user_doc.get("festival_key", "") or DEFAULT_FESTIVAL
    if user_festival not in festivals:
        user_festival = next(iter(festivals), DEFAULT_FESTIVAL)
    # Build a categories map keyed by festival slug for the JS dropdown
    categories_map = {k: v.get("categories", []) for k, v in festivals.items()}
    return render_page("new", "New Review", APP_BODY,
                       festival=festivals.get(user_festival, {}).get("name", "Festival Reviewer"),
                       festivals=festivals,
                       categories_map=categories_map,
                       default_festival=DEFAULT_FESTIVAL,
                       user=session.get("user", ""),
                       user_role=user_role,
                       user_festival=user_festival)


@app.route("/dashboard")
@login_required
def dashboard():
    """Overview: stat cards + recent reviews."""
    user       = session.get("user", "")
    user_role  = session.get("role", "user")
    if user_role == "admin":
        my_festivals = _admin_festivals(user)
        reviews = []
        for fk in my_festivals:
            for r in db.review_list_for_festival(fk):
                r["festival_name"] = my_festivals[fk].get("name", fk)
                reviews.append(r)
        reviews.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        scope_name = "All festivals"
    else:
        user_doc      = db.user_get(user) or {}
        festival_key  = user_doc.get("festival_key", "")
        festivals_db  = get_festivals()
        scope_name    = festivals_db.get(festival_key, {}).get("name", festival_key or "—")
        reviews       = db.review_list_for_festival(festival_key)
        for r in reviews:
            r["festival_name"] = scope_name

    rated = [float(r["overall_rating"]) for r in reviews
             if r.get("overall_rating") not in (None, "")]
    stats = {
        "total":     len(reviews),
        "avg":       (f"{sum(rated)/len(rated):.1f}" if rated else "—"),
        "published": sum(1 for r in reviews if r.get("wp_post_id")),
        "seasons":   len({r.get("season") for r in reviews if r.get("season")}),
    }
    return render_page("dashboard", "Dashboard", DASHBOARD_BODY,
                       stats=stats, reviews=reviews[:6], scope_name=scope_name,
                       user_role=user_role, is_admin=(user_role == "admin"))


@app.route("/api/films")
@login_required
def api_films():
    """Return films visible to the current user — scoped to their festival(s)."""
    return jsonify(db.film_list(_visible_festival_keys()))


GCS_UPLOAD_BUCKET = os.getenv("GCS_UPLOAD_BUCKET", "")


@app.route("/api/upload-url", methods=["POST"])
@login_required
@limiter.limit("10 per minute")
def get_upload_url():
    """Return a signed GCS URL so the browser can upload large videos directly,
    bypassing Cloud Run's 32 MB HTTP/1 request-body limit."""
    if not GCS_UPLOAD_BUCKET:
        return jsonify({"error": "Direct upload not configured (GCS_UPLOAD_BUCKET unset)"}), 500
    data     = request.get_json(silent=True) or {}
    filename = _sanitise(data.get("filename", "video.mp4"), 256)
    ext      = Path(Path(filename).name).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXT)}"}), 400
    blob_name = f"uploads/{uuid.uuid4().hex}{ext}"
    try:
        from downloader import generate_upload_url
        url = generate_upload_url(GCS_UPLOAD_BUCKET, blob_name, content_type="video/mp4")
    except Exception as e:
        logging.error("[upload-url] signing failed: %s", e)
        return jsonify({"error": "Could not create upload URL"}), 500
    return jsonify({"upload_url": url, "blob_name": blob_name})


@app.route("/api/check-dedup", methods=["POST"])
@login_required
@limiter.limit("20 per minute")
def check_dedup():
    """Pre-flight: does a cached analysis already exist for this title+director
    within the caller's own festival? Lets the client skip uploading the video
    entirely on a dedup hit. Scoped per-festival, not global."""
    data     = request.get_json(silent=True) or {}
    title    = _sanitise(data.get("title", ""))
    director = _sanitise(data.get("director", ""))
    if not title or not director:
        return jsonify({"found": False})
    festivals_db  = get_festivals()
    user_role     = session.get("role", "user")
    user_festival = db.user_get(session.get("user", "")) or {}
    user_fk       = user_festival.get("festival_key", "")
    festival_key  = data.get("festival_key", "").strip() if user_role == "admin" else ""
    festival_key  = festival_key or user_fk or DEFAULT_FESTIVAL
    if festival_key not in festivals_db:
        festival_key = next(iter(festivals_db), DEFAULT_FESTIVAL)
    existing = db.film_find_by_identity(title, director, festival_key=festival_key)
    found    = bool(existing and existing.get("analysis"))
    return jsonify({"found": found})


@app.route("/upload", methods=["POST"])
@login_required
@limiter.limit("5 per minute; 30 per hour")
def upload():
    """New film — full video analysis + review.
    Accepts a direct file upload, a gcs_blob (already uploaded to GCS), or neither
    when a cached analysis already exists for the same title+director (dedup).
    """
    # ── Sanitise text fields ──────────────────────────────
    title    = _sanitise(request.form.get("title",    ""))
    director = _sanitise(request.form.get("director", ""))
    genre    = _sanitise(request.form.get("genre",    ""))
    logline  = _cap_words(_sanitise(request.form.get("logline",  ""), MAX_TEXTAREA_LEN))
    dir_stmt = _cap_words(_sanitise(request.form.get("director_statement", ""), MAX_TEXTAREA_LEN))
    synopsis = _cap_words(_sanitise(request.form.get("synopsis", ""), MAX_TEXTAREA_LEN))
    season   = _sanitise(request.form.get("season",   ""))

    if not title or not director:
        return jsonify({"error": "Title and director are required"}), 400
    if not genre:
        return jsonify({"error": "Genre / Category is required"}), 400

    gcs_blob = _sanitise(request.form.get("gcs_blob", ""), 256)
    has_file = "video" in request.files and request.files["video"].filename
    # Note: file is required ONLY if no cached analysis exists (checked after dedup below)

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

    # ── Script / written-work categories: document path ───
    if _is_script_category(genre):
        has_doc = "document" in request.files and request.files["document"].filename
        # Dedup: reuse cached analysis if this written work was already judged
        # within this festival (dedup is scoped per-festival, not global)
        existing = db.film_find_by_identity(title, director, festival_key=festival_key)
        if existing and existing.get("analysis"):
            if has_doc:
                try: request.files["document"].stream.close()
                except Exception: pass
            if db.review_get(existing["film_id"], festival_key):
                return jsonify({"error": "A review for this work already exists for your festival. See it under Reviews."}), 409
            meta = {
                "film_id": existing["film_id"], "title": title, "director": director,
                "logline": logline or existing.get("logline", ""),
                "director_statement": dir_stmt or existing.get("director_statement", ""),
                "genre": genre, "synopsis": synopsis,
                "festival_key": festival_key, "festival_name": festival["name"], "season": season,
            }
            job_id = uuid.uuid4().hex
            db.job_create(job_id, {"status": "queued", "progress": 5,
                                   "message": "Found existing analysis — writing review…",
                                   "meta": meta, "analysis": None, "review": None})
            threading.Thread(target=process_rewrite, args=(job_id, existing, meta), daemon=True).start()
            return jsonify({"job_id": job_id, "from_cache": True})

        if not has_doc:
            return jsonify({"error": "Please upload your script (PDF, DOC, DOCX, or TXT)"}), 400
        fdoc = request.files["document"]
        ext  = Path(Path(fdoc.filename).name).suffix.lower()
        if ext not in DOC_EXT:
            return jsonify({"error": f"Unsupported document. Use: {', '.join(sorted(DOC_EXT))}"}), 400
        tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
        fdoc.save(tmp.name)
        size_mb = Path(tmp.name).stat().st_size / (1024 * 1024)
        if size_mb > MAX_DOC_MB:
            Path(tmp.name).unlink(missing_ok=True)
            return jsonify({"error": f"Document too large ({size_mb:.0f}MB). Max {MAX_DOC_MB}MB"}), 400

        film_id = uuid.uuid4().hex
        db.film_create({
            "film_id": film_id, "title": title, "director": director,
            "logline": logline, "director_statement": dir_stmt,
            "genre": genre, "runtime": "", "screener_url": "",
            "festival_key": festival_key,
        })
        meta = {
            "film_id": film_id, "title": title, "director": director,
            "logline": logline, "director_statement": dir_stmt,
            "genre": genre, "synopsis": synopsis,
            "festival_key": festival_key, "festival_name": festival["name"], "season": season,
        }
        job_id = uuid.uuid4().hex
        db.job_create(job_id, {"status": "queued", "progress": 5,
                               "message": "Queued — reading your script…",
                               "meta": meta, "analysis": None, "review": None})
        logging.info("[upload] SCRIPT job queued film_id=%s festival=%s category=%r user=%s",
                     film_id, festival_key, genre, session.get("user"))
        threading.Thread(target=process_document, args=(job_id, tmp.name, ext, meta), daemon=True).start()
        return jsonify({"job_id": job_id})

    # ── Per-festival dedup ─────────────────────────────────
    # If the same film (title + director) was already analysed within THIS
    # festival, reuse the cached Gemini analysis instead of re-running vision.
    # Dedup is scoped per-festival so the same film submitted to different
    # festivals is judged independently.
    existing = db.film_find_by_identity(title, director, festival_key=festival_key)
    if existing and existing.get("analysis"):
        # Discard the just-uploaded video — we don't need to re-analyse it
        if gcs_blob:
            try:
                from downloader import delete_from_gcs
                delete_from_gcs(GCS_UPLOAD_BUCKET, gcs_blob)
            except Exception:
                pass
        elif has_file:
            request.files["video"].stream.close()

        # Guard against re-reviewing the same film for the same festival
        if db.review_get(existing["film_id"], festival_key):
            return jsonify({"error": "A review for this film already exists for your festival. See it under Reviews."}), 409

        meta = {
            "film_id":            existing["film_id"],
            "title":              title,
            "director":           director,
            "logline":            logline or existing.get("logline", ""),
            "director_statement": dir_stmt or existing.get("director_statement", ""),
            "genre":              genre,
            "synopsis":           synopsis,
            "festival_key":       festival_key,
            "festival_name":      festival["name"],
            "season":             season,
        }
        job_id = uuid.uuid4().hex
        db.job_create(job_id, {
            "status": "queued", "progress": 5,
            "message": "Found existing analysis — writing review (no re-upload)…",
            "meta": meta, "analysis": None, "review": None,
        })
        logging.info("[upload] DEDUP hit — reusing film_id=%s for festival=%s (title=%r director=%r)",
                     existing["film_id"], festival_key, title, director)
        threading.Thread(target=process_rewrite,
                         args=(job_id, existing, meta), daemon=True).start()
        return jsonify({"job_id": job_id, "from_cache": True})

    # ── No dedup hit → a video is required ────────────────
    if not gcs_blob and not has_file:
        return jsonify({"error": "Please upload a video file to continue"}), 400

    # ── Resolve video source ──────────────────────────────
    # Large files arrive via GCS (gcs_blob); small ones may still POST directly.
    if gcs_blob:
        if not gcs_blob.startswith("uploads/"):
            return jsonify({"error": "Invalid upload reference"}), 400
        video_path = ""          # process_video will pull it from GCS
    else:
        f   = request.files["video"]
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

    film_id = uuid.uuid4().hex
    db.film_create({
        "film_id":            film_id,
        "title":              title,
        "director":           director,
        "logline":            logline,
        "director_statement": dir_stmt,
        "genre":              genre,
        "runtime":            "",
        "screener_url":       "",
        "festival_key":       festival_key,
    })

    meta = {
        "film_id":            film_id,
        "title":              title,
        "director":           director,
        "logline":            logline,
        "director_statement": dir_stmt,
        "genre":              genre,
        "synopsis":           synopsis,
        "screener_url":       "",
        "screener_password":  "",
        "gcs_blob":           gcs_blob,
        "festival_key":       festival_key,
        "festival_name":      festival["name"],
        "season":             season,
    }
    logging.info("[upload] job queued film_id=%s festival=%s season=%s user=%s", film_id, festival_key, season, session.get("user"))

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


_STALE_JOB_MINUTES = 20   # jobs stuck in a non-terminal state longer than this are auto-failed
_TERMINAL_STATUSES  = {"done", "error"}


@app.route("/api/jobs/<job_id>/cancel", methods=["POST"])
@login_required
def cancel_job(job_id):
    """Called via sendBeacon when the user navigates away mid-job.
    Marks the job as error so it doesn't show as 'in progress' on re-open
    and the stale-job detector doesn't need to wait 20 minutes."""
    job = db.job_get(job_id)
    if not job:
        return "", 204
    if job.get("status") not in _TERMINAL_STATUSES:
        db.job_update(job_id, {
            "status":   "error",
            "progress": 0,
            "message":  "Cancelled — page was closed or refreshed during processing. Please resubmit.",
        })
        logging.info("[job] Cancelled by client unload: %s", job_id)
    return "", 204


@app.route("/status/<job_id>")
@login_required
def status(job_id):
    job = db.job_get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    # ── Stale job detection ────────────────────────────────────────────────
    # Background threads are killed when Cloud Run replaces an instance mid-job.
    # If the job has been in a non-terminal state for > _STALE_JOB_MINUTES,
    # mark it as an error so the client stops polling and shows a clear message.
    if job.get("status") not in _TERMINAL_STATUSES:
        updated_at = job.get("updated_at") or job.get("created_at")
        if updated_at:
            # _clean_job serialises datetimes to ISO strings — parse back if needed
            if isinstance(updated_at, str):
                try:
                    updated_at = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                except ValueError:
                    updated_at = None
            if updated_at and isinstance(updated_at, datetime):
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
                age_minutes = (datetime.now(timezone.utc) - updated_at).total_seconds() / 60
                if age_minutes > _STALE_JOB_MINUTES:
                    db.job_update(job_id, {
                        "status":  "error",
                        "progress": 0,
                        "message": "Processing was interrupted (server restarted mid-job). Please resubmit.",
                    })
                    job = db.job_get(job_id)

    # Strip internal Unix file paths from error messages before sending to client
    if job.get("status") == "error" and job.get("message"):
        msg = re.sub(r"(?<![:/])(/(?:app|tmp|root|home|var|usr|downloads|frames)[^\s]*)", "[path]", job["message"])
        job = {**job, "message": msg}
    return jsonify(job)


@app.route("/reviews")
@login_required
def reviews_list():
    user_doc     = db.user_get(session.get("user", "")) or {}
    festival_key = user_doc.get("festival_key", "")
    user_role    = session.get("role", "user")
    selected_season = request.args.get("season", "").strip()

    if user_role == "admin":
        my_festivals = _admin_festivals(session.get("user", ""))
        all_reviews  = []
        for fk in my_festivals:
            for r in db.review_list_for_festival(fk):
                r["festival_name"] = my_festivals[fk].get("name", fk)
                all_reviews.append(r)
        all_reviews.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        festival_name = "All Festivals"
    else:
        festivals_db  = get_festivals()
        festival_name = festivals_db.get(festival_key, {}).get("name", festival_key)
        all_reviews   = db.review_list_for_festival(festival_key)
        for r in all_reviews:
            r["festival_name"] = festival_name

    # Seasons from the authoritative seasons collection
    if user_role == "admin":
        season_set = set()
        for fk in my_festivals:
            for s in db.season_list(fk):
                season_set.add(s["name"])
        all_seasons = sorted(season_set)
    else:
        all_seasons = [s["name"] for s in db.season_list(festival_key)]

    # Categories: collect from all reviews (genre field) + authoritative category list
    if user_role == "admin":
        cat_set = set()
        for fk in my_festivals:
            for c in db.category_list(fk):
                cat_set.add(c)
        all_categories = sorted(cat_set)
    else:
        all_categories = db.category_list(festival_key)
    # Also include any categories that appear on reviews but aren't in the list
    for r in all_reviews:
        g = r.get("genre", "").strip()
        if g and g not in all_categories:
            all_categories.append(g)

    return render_page("reviews", "Past Reviews", REVIEWS_BODY,
                                  reviews=all_reviews,
                                  festival_name=festival_name,
                                  seasons=all_seasons,
                                  categories=all_categories,
                                  user_role=user_role,
                                  current_user=session.get("user", ""))


@app.route("/reviews/<film_id>")
@login_required
def review_detail(film_id):
    user_doc     = db.user_get(session.get("user", "")) or {}
    festival_key = user_doc.get("festival_key", "")
    user_role    = session.get("role", "user")

    film = db.film_get(film_id)
    if not film:
        return "Film not found", 404

    # Tenant scoping: only reveal a film (and its analysis) if it belongs to a
    # festival the user may read. Return 404 (not 403) to avoid leaking existence.
    if film.get("festival_key") not in _visible_festival_keys():
        return "Film not found", 404

    if user_role == "admin":
        my_festivals = _admin_festivals(session.get("user", ""))
        reviews = db.review_list_for_film(film_id)
        reviews = [r for r in reviews if r.get("festival_key") in my_festivals]
        for r in reviews:
            r["festival_name"] = my_festivals.get(r["festival_key"], {}).get("name", r["festival_key"])
    else:
        review = db.review_get(film_id, festival_key)
        reviews = [review] if review else []
        festivals_db = get_festivals()
        for r in reviews:
            r["festival_name"] = festivals_db.get(festival_key, {}).get("name", festival_key)

    return render_page("reviews", film.get("title", "Review"), REVIEW_DETAIL_BODY,
                                  film=film,
                                  reviews=reviews,
                                  user_role=user_role,
                                  current_user=session.get("user", ""))


@app.route("/publish/<job_id>", methods=["POST"])
@login_required
def publish(job_id):
    return jsonify({"error": "WordPress publishing is disabled"}), 410


@app.route("/publish_live/<job_id>", methods=["POST"])
@login_required
def publish_live(job_id):
    return jsonify({"error": "WordPress publishing is disabled"}), 410


# ── Admin routes ─────────────────────────────────────────

def _admin_festivals(admin_email: str) -> dict:
    """Return only the festivals created by this admin."""
    all_festivals = get_festivals()
    return {
        k: v for k, v in all_festivals.items()
        if v.get("created_by") == admin_email
    }


@app.route("/api/admin/festivals")
@admin_required
def api_admin_festivals():
    """JSON API: return festivals belonging to the current admin."""
    festivals = _admin_festivals(session.get("user", ""))
    # Strip sensitive fields (gemini key) before sending to browser
    safe = {}
    for k, v in festivals.items():
        safe[k] = {fk: fv for fk, fv in v.items() if fk not in ("gemini_api_key", "wp_app_pass")}
    return jsonify(safe)


@app.route("/admin")
@admin_required
def admin():
    current_user = session.get("user", "")
    my_festivals = _admin_festivals(current_user)
    my_festival_keys = set(my_festivals.keys())
    users = [
        u for u in db.user_list()
        if u.get("role") != "admin" and u.get("festival_key", "") in my_festival_keys
    ]
    return render_page("admin", "Admin", ADMIN_BODY,
                                  users=users,
                                  festivals=my_festivals,
                                  default_review_prompt=DEFAULT_REVIEW_PROMPT,
                                  current_user=current_user)


@app.route("/admin/users", methods=["POST"])
@admin_required
@limiter.limit("20 per minute")
def admin_add_user():
    email        = request.form.get("email", "").strip().lower()[:254]
    password     = request.form.get("password", "").strip()
    festival_key = request.form.get("festival_key", "").strip()
    if not email or not password:
        return redirect("/admin?error=Email+and+password+required")
    if not festival_key:
        return redirect("/admin?error=A+festival+must+be+selected")
    if len(password) < 8:
        return redirect("/admin?error=Password+must+be+at+least+8+characters")
    # Verify the festival belongs to this admin
    my_festivals = _admin_festivals(session.get("user", ""))
    if festival_key not in my_festivals:
        return redirect("/admin?error=Invalid+or+unauthorised+festival")
    ok = db.user_create(email, generate_password_hash(password), role="user", festival_key=festival_key)
    if not ok:
        return redirect("/admin?error=User+already+exists")
    logging.info("[admin] User created: %s festival=%s by %s", email, festival_key, session.get("user"))
    return redirect("/admin?success=User+added")


@app.route("/admin/users/delete", methods=["POST"])
@admin_required
@limiter.limit("20 per minute")
def admin_delete_user():
    email = request.form.get("email", "").strip().lower()
    if email == session.get("user"):
        return redirect("/admin?error=Cannot+delete+your+own+account")
    # Never allow deleting another admin
    target = db.user_get(email)
    if target and target.get("role") == "admin":
        return redirect("/admin?error=Cannot+delete+an+admin+account")
    db.user_delete(email)
    logging.info("[admin] User deleted: %s by %s", email, session.get("user"))
    return redirect("/admin?success=User+deleted")


@app.route("/admin/users/password", methods=["POST"])
@admin_required
@limiter.limit("20 per minute")
def admin_change_password():
    email    = request.form.get("email", "").strip().lower()
    password = request.form.get("new_password", "").strip()
    if not email or not password:
        return redirect("/admin?error=Email+and+password+required")
    if len(password) < 8:
        return redirect("/admin?error=Password+must+be+at+least+8+characters")
    target = db.user_get(email)
    if not target:
        return redirect("/admin?error=User+not+found")
    if target.get("role") == "admin":
        return redirect("/admin?error=Cannot+change+admin+password+here")
    # Ensure the target user belongs to one of this admin's festivals
    my_festival_keys = set(_admin_festivals(session.get("user", "")).keys())
    if target.get("festival_key", "") not in my_festival_keys:
        return redirect("/admin?error=Not+authorised+to+manage+this+user")
    db.user_update_password(email, generate_password_hash(password))
    logging.info("[admin] Password changed for %s by %s", email, session.get("user"))
    return redirect("/admin?success=Password+updated")


def _festival_fields_from_form() -> dict:
    """Extract editable festival identity fields from request.form.
    Prompts & categories are NOT set here — they're managed on the Manage page
    via the category catalog. Festivals fall back to global defaults for judging.
    """
    return {
        "name":           _sanitise(request.form.get("name", "")),
        "full_name":      _sanitise(request.form.get("full_name", "")),
        "focus":          _sanitise(request.form.get("focus", ""), MAX_TEXTAREA_LEN),
        "tone":           _sanitise(request.form.get("tone", "")) or "professional, honest, and encouraging",
        "gemini_api_key": request.form.get("gemini_api_key", "").strip()[:256],
        "gemini_model":   "gemini-2.5-flash",
        "word_count":     500,
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
    fields["full_name"]  = fields["full_name"] or fields["name"]
    fields["created_by"] = session.get("user", "")
    # Sensible defaults — categories & per-category prompts are added later via Manage
    fields["analysis_focus"] = ""
    fields["review_prompt"]  = DEFAULT_REVIEW_PROMPT
    fields["categories"]     = []
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
    if existing.get("created_by") and existing["created_by"] != session.get("user"):
        return redirect("/admin?error=Not+authorised+to+edit+this+festival")
    fields = _festival_fields_from_form()
    fields["name"]      = fields["name"]      or existing.get("name", key)
    fields["full_name"] = fields["full_name"] or existing.get("full_name", key)
    fields["focus"]     = fields["focus"]     or existing.get("focus", "")
    # Preserve prompts & categories — they're managed on the Manage page, not here
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
    existing = db.festival_get(key)
    if existing and existing.get("created_by") and existing["created_by"] != session.get("user"):
        return redirect("/admin?error=Not+authorised+to+delete+this+festival")
    db.festival_delete(key)
    logging.info("[admin] Festival deleted: %s by %s", key, session.get("user"))
    return redirect("/admin?success=Festival+deleted")


def _assert_festival_access(key: str):
    """Return error JSON tuple if the current user can't mutate this festival's config.
    Admins must own the festival (created_by). Regular users must be assigned to it."""
    existing = db.festival_get(key)
    if not existing:
        return jsonify({"error": "Festival not found"}), 404
    role = session.get("role", "user")
    user = session.get("user", "")
    if role == "admin":
        if existing.get("created_by") and existing["created_by"] != user:
            return jsonify({"error": "Not authorised"}), 403
    else:
        user_doc = db.user_get(user) or {}
        if user_doc.get("festival_key", "") != key:
            return jsonify({"error": "Not authorised"}), 403
    return None


# Keep old name as alias for admin-only paths
def _assert_festival_owner(key: str):
    return _assert_festival_access(key)


# ── Categories API ────────────────────────────────────────────────────────────

@app.route("/api/predefined-categories", methods=["GET"])
@login_required
def api_predefined_categories():
    """Catalog of selectable categories with their built-in judging prompts."""
    return jsonify({"categories": [{"name": n, "prompt": p} for n, p in PREDEFINED_CATEGORIES.items()]})


@app.route("/api/festivals/<key>/categories", methods=["GET"])
@login_required
def api_categories_list(key):
    err = _assert_festival_access(key)
    if err: return err
    return jsonify({"categories": db.category_list(key)})


@app.route("/api/festivals/<key>/categories/bulk", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
def api_categories_bulk_add(key):
    """Add multiple predefined categories at once, each with its catalog prompt."""
    err = _assert_festival_access(key)
    if err: return err
    names = (request.json or {}).get("names", [])
    if not isinstance(names, list):
        return jsonify({"error": "names must be a list"}), 400
    added = 0
    for raw in names[:50]:
        name = _sanitise(str(raw))
        if not name:
            continue
        if db.category_add(key, name):
            added += 1
        # Attach the predefined prompt if this is a known catalog category
        prompt = PREDEFINED_CATEGORIES.get(name)
        if prompt:
            db.category_set_prompt(key, name, prompt)
    return jsonify({"categories": db.category_list(key), "added": added})


@app.route("/api/festivals/<key>/categories", methods=["POST"])
@login_required
@limiter.limit("60 per minute")
def api_category_add(key):
    err = _assert_festival_access(key)
    if err: return err
    name = _sanitise((request.json or {}).get("name", ""))
    if not name:
        return jsonify({"error": "Name required"}), 400
    if not db.category_add(key, name):
        return jsonify({"error": "Category already exists"}), 409
    return jsonify({"categories": db.category_list(key)})


@app.route("/api/festivals/<key>/categories/<path:name>", methods=["PUT"])
@login_required
@limiter.limit("60 per minute")
def api_category_rename(key, name):
    err = _assert_festival_access(key)
    if err: return err
    new_name = _sanitise((request.json or {}).get("name", ""))
    if not new_name:
        return jsonify({"error": "New name required"}), 400
    if not db.category_rename(key, name, new_name):
        return jsonify({"error": "Rename failed — duplicate or not found"}), 409
    return jsonify({"categories": db.category_list(key)})


@app.route("/api/festivals/<key>/categories/<path:name>", methods=["DELETE"])
@login_required
@limiter.limit("60 per minute")
def api_category_delete(key, name):
    err = _assert_festival_access(key)
    if err: return err
    db.category_delete(key, name)
    return jsonify({"categories": db.category_list(key)})


@app.route("/api/festivals/<key>/categories/<path:name>/prompt", methods=["GET"])
@login_required
def api_category_prompt_get(key, name):
    err = _assert_festival_access(key)
    if err: return err
    return jsonify({"prompt": db.category_get_prompt(key, name)})


@app.route("/api/festivals/<key>/categories/<path:name>/prompt", methods=["PUT"])
@login_required
@limiter.limit("60 per minute")
def api_category_prompt_set(key, name):
    err = _assert_festival_access(key)
    if err: return err
    prompt = _sanitise((request.json or {}).get("prompt", ""), MAX_TEXTAREA_LEN)
    if not db.category_set_prompt(key, name, prompt):
        return jsonify({"error": "Category not found"}), 404
    return jsonify({"ok": True, "prompt": prompt})


# ── Seasons API ───────────────────────────────────────────────────────────────

@app.route("/api/festivals/<key>/seasons", methods=["GET"])
@login_required
def api_seasons_list(key):
    err = _assert_festival_access(key)
    if err: return err
    return jsonify({"seasons": db.season_list(key)})


@app.route("/api/festivals/<key>/seasons", methods=["POST"])
@login_required
@limiter.limit("60 per minute")
def api_season_add(key):
    err = _assert_festival_access(key)
    if err: return err
    name = _sanitise((request.json or {}).get("name", ""))
    if not name:
        return jsonify({"error": "Name required"}), 400
    if not db.season_create(key, name):
        return jsonify({"error": "Season already exists"}), 409
    return jsonify({"seasons": db.season_list(key)})


@app.route("/api/festivals/<key>/seasons/<path:name>", methods=["PUT"])
@login_required
@limiter.limit("60 per minute")
def api_season_rename(key, name):
    err = _assert_festival_access(key)
    if err: return err
    new_name = _sanitise((request.json or {}).get("name", ""))
    if not new_name:
        return jsonify({"error": "New name required"}), 400
    if not db.season_rename(key, name, new_name):
        return jsonify({"error": "Rename failed — duplicate or not found"}), 409
    return jsonify({"seasons": db.season_list(key)})


@app.route("/api/festivals/<key>/seasons/<path:name>", methods=["DELETE"])
@login_required
@limiter.limit("60 per minute")
def api_season_delete(key, name):
    err = _assert_festival_access(key)
    if err: return err
    db.season_delete(key, name)
    return jsonify({"seasons": db.season_list(key)})


# ── WordPress integration ─────────────────────────────────────────────────────

@app.route("/wordpress")
@admin_required
def wordpress_page():
    """Per-festival WordPress config + reference-design template editor."""
    my_festivals = _admin_festivals(session.get("user", ""))
    # Never send the app password to the browser — only whether it is set.
    safe = {}
    for k, f in my_festivals.items():
        safe[k] = {
            "name":            f.get("name", k),
            "wp_url":          f.get("wp_url", ""),
            "wp_user":         f.get("wp_user", ""),
            "wp_configured":   bool(f.get("wp_app_pass")),
            "wp_default_status": f.get("wp_default_status", "draft"),
            "wp_template":     f.get("wp_template", ""),
        }
    return render_page("wordpress", "WordPress", WORDPRESS_BODY,
                       festivals=safe,
                       placeholders=wordpress.PLACEHOLDERS,
                       default_template=wordpress.DEFAULT_WP_TEMPLATE,
                       current_user=session.get("user", ""))


@app.route("/api/festivals/<key>/wp/config", methods=["POST"])
@admin_required
@limiter.limit("20 per minute")
def api_wp_config(key):
    err = _assert_festival_access(key)
    if err: return err
    existing = db.festival_get(key) or {}
    data = request.json or {}
    try:
        wp_url = data.get("wp_url", "").strip()
        if wp_url:
            wp_url = wordpress._safe_base_url(wp_url)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    fields = dict(existing)
    fields["wp_url"]  = wp_url
    fields["wp_user"] = (data.get("wp_user", "") or "").strip()[:128]
    fields["wp_default_status"] = "publish" if data.get("wp_default_status") == "publish" else "draft"
    # Only overwrite the secret when a new non-empty value is supplied
    new_pass = (data.get("wp_app_pass", "") or "").strip()
    if new_pass:
        fields["wp_app_pass"] = new_pass[:256]
    db.festival_upsert(key, fields)
    return jsonify({"ok": True, "wp_configured": bool(fields.get("wp_app_pass"))})


@app.route("/api/festivals/<key>/wp/template", methods=["PUT"])
@admin_required
@limiter.limit("30 per minute")
def api_wp_template_save(key):
    err = _assert_festival_access(key)
    if err: return err
    existing = db.festival_get(key) or {}
    template = (request.json or {}).get("template", "")
    if len(template) > 40000:
        return jsonify({"error": "Template too large"}), 400
    fields = dict(existing)
    fields["wp_template"] = template
    db.festival_upsert(key, fields)
    return jsonify({"ok": True})


@app.route("/api/festivals/<key>/wp/template/generate", methods=["POST"])
@admin_required
@limiter.limit("6 per minute")
def api_wp_template_generate(key):
    err = _assert_festival_access(key)
    if err: return err
    festival = get_festivals().get(key)
    if not festival:
        return jsonify({"error": "Festival not found"}), 404
    data = request.json or {}
    ref_url  = (data.get("reference_url", "") or "").strip()
    ref_html = (data.get("reference_html", "") or "").strip()
    try:
        if ref_url and not ref_html:
            ref_html = wordpress.fetch_reference(ref_url)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Could not fetch the reference URL: {e}"}), 400
    if not ref_html:
        return jsonify({"error": "Provide a reference URL or paste reference HTML."}), 400
    api_key = festival.get("gemini_api_key")
    if not api_key:
        return jsonify({"error": "No Gemini API key configured for this festival."}), 400
    try:
        client = genai.Client(api_key=api_key, http_options=_GEMINI_HTTP_OPTIONS)
        model  = festival.get("gemini_model", "gemini-2.5-flash")
        template = wordpress.generate_template_from_reference(client, model, ref_html[:60000])
    except Exception as e:
        return jsonify({"error": _friendly_error(e)}), 502
    return jsonify({"template": template})


@app.route("/api/festivals/<key>/wp/preview", methods=["POST"])
@login_required
@limiter.limit("30 per minute")
def api_wp_preview(key):
    err = _assert_festival_access(key)
    if err: return err
    festival = get_festivals().get(key) or {}
    template = (request.json or {}).get("template") or festival.get("wp_template") or wordpress.DEFAULT_WP_TEMPLATE
    # Sample data so admins can preview layout without a real film
    sample_film = {
        "title": "Sample Film Title", "director": "A. Director", "genre": "Narrative Short",
        "runtime": "14", "country": "France",
        "analysis": {"ratings": {"originality": 8, "direction": 9, "writing": 7,
                                 "cinematography": 9, "performances": 8, "production_value": 7,
                                 "pacing": 8, "structure": 8, "sound_music": 7},
                     "overall_rating": 8.2,
                     "standout_moment": "A wordless, beautifully composed closing shot.",
                     "weakest_element": "The second act loses a little momentum.",
                     "festival_suitability": "A strong fit for art-house and short-film programmes."}}
    sample_review = {"review_text": "This is a preview of how a published review will look.\n\n"
                                    "Each paragraph of the expert review renders here, styled by the "
                                    "reference design you supplied.", "season": "2025", "overall_rating": 8.2}
    ctx = wordpress.build_context(sample_film, sample_review, festival)
    return jsonify({"html": wordpress.render_template(template, ctx)})


@app.route("/api/reviews/<film_id>/publish", methods=["POST"])
@login_required
@limiter.limit("20 per minute")
def api_publish_review(film_id):
    film = db.film_get(film_id)
    if not film or film.get("festival_key") not in _visible_festival_keys():
        return jsonify({"error": "Review not found"}), 404
    fk = film.get("festival_key")
    festival = get_festivals().get(fk) or {}
    review = db.review_get(film_id, fk)
    if not review:
        return jsonify({"error": "No review exists for this film yet."}), 400
    if not wordpress.wp_configured(festival):
        return jsonify({"error": "WordPress is not configured for this festival. "
                                 "Ask an admin to set it up under WordPress."}), 400
    status = "publish" if (request.json or {}).get("status") == "publish" else "draft"
    template = festival.get("wp_template") or wordpress.DEFAULT_WP_TEMPLATE
    ctx = wordpress.build_context(film, review, festival)
    content = wordpress.render_template(template, ctx)
    title = f"{film.get('title','Untitled')} — {festival.get('name','')} Expert Review"
    post_id = review.get("wp_post_id") or None
    result = wordpress.publish(festival, title, content, status=status, post_id=post_id)
    if not result.get("success"):
        return jsonify({"error": result.get("error", "Publish failed")}), 502
    db.review_set_wp(film_id, fk, str(result.get("post_id", "")), result.get("url", ""))
    logging.info("[wp] Published film=%s festival=%s status=%s by %s", film_id, fk, status, session.get("user"))
    return jsonify({"ok": True, "status": status, "url": result.get("url", ""), "post_id": result.get("post_id")})


@app.route("/manage")
@login_required
def manage():
    """Redirect /manage → /manage/<festival_key> so the slug is in the URL."""
    user_role = session.get("role", "user")
    if user_role == "admin":
        my_fests = _admin_festivals(session.get("user", ""))
        # Redirect to the first owned festival; admin can switch via nav
        first_key = next(iter(my_fests), None)
        if first_key:
            return redirect(f"/manage/{first_key}")
        return redirect("/admin")  # no festivals yet
    else:
        fk = session.get("festival_key", "")
        if not fk:
            # Session predates the festival_key-in-session change — refresh from DB
            user_doc = db.user_get(session.get("user", "")) or {}
            fk = user_doc.get("festival_key", "")
            if fk:
                session["festival_key"] = fk   # patch the live session
        if fk:
            return redirect(f"/manage/{fk}")
        return render_page("manage", "Manage", MANAGE_BODY,
                                      festivals={},
                                      user_role=user_role,
                                      current_user=session.get("user", ""))


@app.route("/manage/<festival_key>")
@login_required
def manage_festival(festival_key: str):
    user_role = session.get("role", "user")
    all_fests = get_festivals()

    if user_role == "admin":
        my_fests = _admin_festivals(session.get("user", ""))
        if festival_key not in my_fests:
            return redirect("/admin")
        festivals = {festival_key: my_fests[festival_key]}
    else:
        assigned = session.get("festival_key", "")
        if not assigned:
            # Stale session — refresh from DB and patch
            user_doc = db.user_get(session.get("user", "")) or {}
            assigned = user_doc.get("festival_key", "")
            if assigned:
                session["festival_key"] = assigned
        if assigned and festival_key != assigned:
            return redirect(f"/manage/{assigned}")
        festivals = {festival_key: all_fests[festival_key]} if festival_key in all_fests else {}

    return render_page("manage", "Manage", MANAGE_BODY,
                                  festivals=festivals,
                                  user_role=user_role,
                                  current_user=session.get("user", ""))


# ── App-shell design system ───────────────────────────────
SHELL_CSS = """
*{margin:0;padding:0;box-sizing:border-box}
:root{
  --gold:#d1af62;--gold-l:#ecca86;--gold-d:#a2854a;
  --bg:#0a0a0d;--bg2:#141419;--bg3:#1c1c24;--bg4:#24242e;
  --sidebar:#0e0e12;
  --border:rgba(255,255,255,.08);--border-gold:rgba(209,175,98,.22);
  --text:#eceae4;--dim:#a29e96;--muted:#78747f;
  --green:#5cbf8a;--red:#e5695f;--blue:#6f9de8;
  --radius:14px;--shadow:0 12px 40px -24px rgba(0,0,0,.8);
}
html,body{height:100%}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;
     -webkit-font-smoothing:antialiased}
::selection{background:rgba(209,175,98,.28);color:#fff}
a{color:inherit}
::-webkit-scrollbar{width:10px;height:10px}
::-webkit-scrollbar-thumb{background:#2a2a34;border-radius:6px;border:2px solid var(--bg)}
::-webkit-scrollbar-thumb:hover{background:#38384a}

/* ── App grid ── */
.app{display:grid;grid-template-columns:250px 1fr;min-height:100vh}
.main-col{display:flex;flex-direction:column;min-width:0}

/* ── Sidebar ── */
.sidebar{background:var(--sidebar);border-right:1px solid var(--border);
         display:flex;flex-direction:column;position:sticky;top:0;height:100vh;z-index:40}
.brand{display:flex;align-items:center;gap:11px;padding:20px 20px 18px;border-bottom:1px solid var(--border)}
.brand-mark{width:34px;height:34px;border-radius:10px;flex-shrink:0;
            background:linear-gradient(145deg,rgba(209,175,98,.2),rgba(209,175,98,.05));
            border:1px solid var(--border-gold);display:flex;align-items:center;justify-content:center;color:var(--gold)}
.brand-mark svg{width:18px;height:18px}
.brand-name{font-family:'Bebas Neue',sans-serif;font-size:21px;letter-spacing:1.5px;color:var(--gold);line-height:1}
.brand-sub{font-size:8.5px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);font-family:'DM Mono',monospace;margin-top:2px}
.sidebar-nav{flex:1;padding:14px 12px;display:flex;flex-direction:column;gap:3px;overflow-y:auto}
.nav-section{font-size:9px;letter-spacing:1.6px;text-transform:uppercase;color:var(--muted);
             font-family:'DM Mono',monospace;padding:14px 12px 6px}
.nav-item{display:flex;align-items:center;gap:11px;padding:10px 12px;border-radius:10px;
          font-size:13.5px;color:var(--dim);text-decoration:none;transition:all .15s;font-weight:500}
.nav-item svg{width:17px;height:17px;flex-shrink:0;opacity:.85}
.nav-item:hover{background:rgba(255,255,255,.04);color:var(--text)}
.nav-item.active{background:linear-gradient(90deg,rgba(209,175,98,.16),rgba(209,175,98,.04));
                 color:var(--gold);box-shadow:inset 2px 0 0 var(--gold)}
.nav-item.active svg{opacity:1;color:var(--gold)}
.sidebar-foot{border-top:1px solid var(--border);padding:14px}
.sidebar-user{display:flex;align-items:center;gap:10px;padding:8px 10px;border-radius:10px}
.avatar{width:32px;height:32px;border-radius:50%;flex-shrink:0;background:linear-gradient(145deg,var(--gold),var(--gold-d));
        color:#1a1408;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:13px;text-transform:uppercase}
.sidebar-user-meta{min-width:0}
.sidebar-user-email{font-size:12px;color:var(--text);white-space:nowrap;overflow:hidden;text-overflow:ellipsis;max-width:140px}
.sidebar-user-role{font-size:9px;letter-spacing:1px;text-transform:uppercase;color:var(--muted);font-family:'DM Mono',monospace}
.sidebar-signout{display:block;text-align:center;margin-top:8px;padding:8px;border-radius:8px;font-size:11px;
                 font-family:'DM Mono',monospace;letter-spacing:1px;color:var(--muted);text-decoration:none;
                 border:1px solid var(--border);transition:all .15s}
.sidebar-signout:hover{color:var(--gold);border-color:var(--border-gold)}

/* ── Topbar ── */
.topbar{position:sticky;top:0;z-index:30;display:flex;align-items:center;gap:14px;
        padding:0 28px;height:60px;background:rgba(10,10,13,.82);backdrop-filter:blur(12px);
        border-bottom:1px solid var(--border)}
.menu-btn{display:none;background:none;border:none;color:var(--text);cursor:pointer;padding:6px}
.topbar-title{font-family:'Bebas Neue',sans-serif;font-size:26px;letter-spacing:.8px;color:var(--text);line-height:1}
.topbar-right{margin-left:auto;display:flex;align-items:center;gap:16px}
.topbar-user{font-size:12px;color:var(--muted);font-family:'DM Mono',monospace}
.topbar-logout{font-size:11px;color:var(--muted);text-decoration:none;font-family:'DM Mono',monospace;letter-spacing:1px;transition:color .15s}
.topbar-logout:hover{color:var(--gold)}

/* ── Content ── */
.content{padding:30px 28px 60px;max-width:1140px;width:100%}
.page-head{display:flex;align-items:flex-end;justify-content:space-between;gap:16px;margin-bottom:24px;flex-wrap:wrap}
.page-title{font-family:'Bebas Neue',sans-serif;font-size:34px;letter-spacing:.6px;color:var(--text);line-height:1}
.page-sub{font-size:13px;color:var(--dim);margin-top:4px}
.section-eyebrow{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--gold-d);font-family:'DM Mono',monospace;margin-bottom:10px}

/* ── Buttons ── */
.btn{display:inline-flex;align-items:center;justify-content:center;gap:8px;border:none;cursor:pointer;
     border-radius:10px;padding:11px 18px;font-family:'DM Sans',sans-serif;font-weight:600;font-size:13px;
     transition:transform .15s,box-shadow .2s,background .2s,border-color .2s;white-space:nowrap;text-decoration:none;line-height:1}
.btn svg{width:15px;height:15px}
.btn-gold{background:linear-gradient(180deg,var(--gold-l),var(--gold));color:#1a1408;box-shadow:0 6px 18px -8px rgba(209,175,98,.55)}
.btn-gold:hover{transform:translateY(-1px);box-shadow:0 10px 24px -10px rgba(209,175,98,.65)}
.btn-green{background:linear-gradient(180deg,#6bd39c,var(--green));color:#08230f;box-shadow:0 6px 18px -8px rgba(92,191,138,.5)}
.btn-green:hover{transform:translateY(-1px)}
.btn-ghost{background:var(--bg3);color:var(--text);border:1px solid var(--border)}
.btn-ghost:hover{border-color:var(--border-gold);color:var(--gold)}
.btn-danger{background:rgba(229,105,95,.12);color:var(--red);border:1px solid rgba(229,105,95,.28)}
.btn-danger:hover{background:rgba(229,105,95,.22)}
.btn-sm{padding:7px 12px;font-size:12px;border-radius:8px}
.btn:disabled{opacity:.4;cursor:not-allowed;transform:none;box-shadow:none}

/* ── Forms ── */
label{font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;font-family:'DM Mono',monospace;display:block;margin-bottom:6px}
input[type=text],input[type=number],input[type=email],input[type=password],input[type=url],select,textarea{
  background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:11px 13px;color:var(--text);
  font-family:'DM Sans',sans-serif;font-size:13px;outline:none;width:100%;transition:border-color .2s,box-shadow .2s;resize:vertical}
input::placeholder,textarea::placeholder{color:#54515c}
input:focus,select:focus,textarea:focus{border-color:var(--gold);box-shadow:0 0 0 3px rgba(209,175,98,.12)}
select option{background:var(--bg3)}
textarea{min-height:88px;line-height:1.55}
.optional-tag{font-size:9px;color:var(--muted);opacity:.6;font-family:'DM Mono',monospace;margin-left:4px}

/* ── Cards ── */
.card{background:linear-gradient(180deg,var(--bg2),#101015);border:1px solid var(--border);border-radius:var(--radius);
      overflow:hidden;margin-bottom:20px;box-shadow:var(--shadow)}
.card-head{padding:17px 22px;border-bottom:1px solid var(--border);display:flex;align-items:center;gap:12px;background:rgba(255,255,255,.015)}
.card-head-icon{width:32px;height:32px;border-radius:9px;flex-shrink:0;color:var(--gold);
                background:linear-gradient(145deg,rgba(209,175,98,.16),rgba(209,175,98,.03));
                border:1px solid var(--border-gold);display:flex;align-items:center;justify-content:center}
.card-head-icon svg{width:16px;height:16px}
.card-head-title{font-size:11px;letter-spacing:2px;color:var(--dim);text-transform:uppercase;font-family:'DM Mono',monospace;font-weight:500}
.card-body{padding:22px}

/* ── Stat cards (dashboard) ── */
.stat-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px}
.statcard{background:linear-gradient(180deg,var(--bg2),#101015);border:1px solid var(--border);border-radius:var(--radius);
          padding:20px;box-shadow:var(--shadow);position:relative;overflow:hidden}
.statcard-icon{width:34px;height:34px;border-radius:9px;color:var(--gold);margin-bottom:14px;
               background:linear-gradient(145deg,rgba(209,175,98,.16),rgba(209,175,98,.03));
               border:1px solid var(--border-gold);display:flex;align-items:center;justify-content:center}
.statcard-icon svg{width:17px;height:17px}
.statcard-value{font-family:'Bebas Neue',sans-serif;font-size:40px;line-height:1;color:var(--text)}
.statcard-value .unit{font-size:16px;color:var(--muted);font-family:'DM Mono',monospace;margin-left:2px}
.statcard-label{font-size:11px;color:var(--muted);letter-spacing:1px;text-transform:uppercase;font-family:'DM Mono',monospace;margin-top:6px}

/* ── Badges ── */
.badge{display:inline-flex;align-items:center;gap:5px;padding:3px 10px;border-radius:20px;font-size:10px;
       font-family:'DM Mono',monospace;letter-spacing:.5px}
.badge-gold{background:rgba(209,175,98,.12);color:var(--gold);border:1px solid var(--border-gold)}
.badge-green{background:rgba(92,191,138,.12);color:var(--green);border:1px solid rgba(92,191,138,.3)}
.badge-muted{background:rgba(255,255,255,.05);color:var(--dim);border:1px solid var(--border)}

/* ── Empty state ── */
.empty-state{text-align:center;padding:56px 20px;color:var(--muted)}
.empty-state svg{width:40px;height:40px;opacity:.4;margin-bottom:14px}
.empty-state h3{font-family:'DM Sans',sans-serif;font-size:16px;color:var(--dim);font-weight:600;margin-bottom:6px}
.empty-state p{font-size:13px;color:var(--muted);margin-bottom:18px}
.empty{color:var(--muted);font-size:13px;font-family:'DM Mono',monospace;padding:40px 0;text-align:center}

/* ── Alerts / toast ── */
.alert{border-radius:10px;padding:11px 16px;font-size:13px;margin-bottom:18px}
.alert-success{background:rgba(92,191,138,.09);border:1px solid rgba(92,191,138,.28);color:var(--green)}
.alert-error{background:rgba(229,105,95,.09);border:1px solid rgba(229,105,95,.28);color:var(--red)}
.toast{position:fixed;bottom:24px;right:24px;background:var(--bg4);border:1px solid var(--border-gold);border-radius:11px;
       padding:12px 20px;font-size:13px;color:var(--gold);font-family:'DM Mono',monospace;opacity:0;transform:translateY(8px);
       transition:all .3s;z-index:999;pointer-events:none;box-shadow:0 16px 40px -18px rgba(0,0,0,.9)}
.toast.show{opacity:1;transform:translateY(0)}
.toast.err{color:var(--red);border-color:rgba(229,105,95,.35)}

/* ── Mobile ── */
.nav-scrim{display:none}
@media(max-width:860px){
  .app{grid-template-columns:1fr}
  .sidebar{position:fixed;left:0;top:0;width:250px;transform:translateX(-100%);transition:transform .25s}
  body.nav-open .sidebar{transform:translateX(0)}
  body.nav-open .nav-scrim{display:block;position:fixed;inset:0;background:rgba(0,0,0,.5);z-index:35}
  .menu-btn{display:flex}
  .topbar{padding:0 16px}
  .content{padding:20px 16px 48px}
  .stat-grid{grid-template-columns:1fr 1fr;gap:12px}
}
@media(max-width:520px){
  .stat-grid{grid-template-columns:1fr}
  .topbar-user{display:none}
}
"""

# Page-specific component CSS (filter bar, review cards, upload, scores, chips, WP)
PAGE_CSS = """
/* Reviews list */
.filter-bar{background:linear-gradient(180deg,var(--bg2),#101015);border:1px solid var(--border);border-radius:var(--radius);padding:20px 22px;margin-bottom:24px;box-shadow:var(--shadow)}
.filter-row{display:flex;align-items:flex-end;gap:12px;flex-wrap:wrap}
.filter-group{display:flex;flex-direction:column;gap:7px;min-width:140px;flex:1}
.filter-group.search-group{flex:2;min-width:200px}
.filter-label{font-size:10px;color:var(--muted);font-family:'DM Mono',monospace;letter-spacing:1px;text-transform:uppercase}
.filter-select,.filter-input{width:100%;background:var(--bg3);border:1px solid var(--border);border-radius:9px;color:var(--text);padding:11px 13px;font-size:13px;outline:none;font-family:'DM Sans',sans-serif;transition:border-color .2s,box-shadow .2s}
.filter-select{appearance:none;cursor:pointer;background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='8' viewBox='0 0 12 8'%3E%3Cpath d='M1 1l5 5 5-5' stroke='%2378747f' stroke-width='1.5' fill='none' stroke-linecap='round'/%3E%3C/svg%3E");background-repeat:no-repeat;background-position:right 12px center;padding-right:32px}
.filter-input::placeholder{color:#54515c}
.filter-select:focus,.filter-input:focus{border-color:var(--gold);box-shadow:0 0 0 3px rgba(209,175,98,.12)}
.filter-select.active{border-color:var(--border-gold);color:var(--gold)}
.rating-group{display:flex;flex-direction:column;gap:7px;min-width:160px}
.rating-filter{display:flex;align-items:center;gap:8px}
.rating-filter input[type=range]{accent-color:var(--gold);flex:1;cursor:pointer;min-width:0}
.rating-val{color:var(--gold);font-weight:600;font-size:12px;font-family:'DM Mono',monospace;white-space:nowrap;min-width:36px}
.clear-btn{background:transparent;border:1px solid var(--border);border-radius:9px;color:var(--muted);padding:11px 16px;font-size:12px;font-family:'DM Mono',monospace;cursor:pointer;white-space:nowrap;transition:all .2s;align-self:flex-end}
.clear-btn:hover{border-color:var(--border-gold);color:var(--gold)}
.result-count{font-size:11px;font-family:'DM Mono',monospace;color:var(--muted);margin-top:14px}
.review-grid{display:grid;gap:14px}
.review-card{background:linear-gradient(180deg,var(--bg2),#101015);border:1px solid var(--border);border-radius:var(--radius);padding:20px 24px;cursor:pointer;transition:border-color .18s,transform .18s,box-shadow .18s;text-decoration:none;display:block;color:inherit}
.review-card:hover{border-color:var(--border-gold);transform:translateY(-2px);box-shadow:0 16px 40px -22px rgba(0,0,0,.9)}
.rc-top{display:flex;align-items:flex-start;justify-content:space-between;gap:16px;margin-bottom:10px}
.rc-title{font-size:16px;font-weight:600;color:var(--text);letter-spacing:.2px}
.rc-director{font-size:12px;color:var(--dim);margin-top:3px}
.rc-meta{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}
.rc-tag{font-size:10px;font-family:'DM Mono',monospace;background:var(--bg4);border:1px solid var(--border);border-radius:6px;padding:3px 9px;color:var(--dim)}
.rc-tag.festival{border-color:var(--border-gold);color:var(--gold);background:rgba(209,175,98,.06)}
.rc-excerpt{font-size:12.5px;color:var(--dim);line-height:1.65;display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden}
.rc-date{font-size:11px;font-family:'DM Mono',monospace;color:var(--muted);white-space:nowrap}
.rc-score{font-family:'Bebas Neue',sans-serif;font-size:26px;color:var(--gold);line-height:1}
.rc-score .d{font-size:12px;color:var(--muted);font-family:'DM Mono',monospace}
.no-results{display:none;color:var(--muted);font-size:13px;font-family:'DM Mono',monospace;padding:60px 0;text-align:center}

/* Upload / new review */
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
.form-group{display:flex;flex-direction:column;gap:7px}
.form-group.full{grid-column:1/-1}
.festival-hint{font-size:11px;color:var(--muted);margin-top:4px;font-family:'DM Mono',monospace;min-height:16px}
.drop-zone{border:1.5px dashed var(--border-gold);border-radius:var(--radius);padding:44px 24px;text-align:center;cursor:pointer;transition:all .2s;position:relative;background:rgba(255,255,255,.015)}
.drop-zone:hover{border-color:rgba(209,175,98,.4);background:rgba(209,175,98,.03)}
.drop-zone.drag-over{border-color:var(--gold);background:rgba(209,175,98,.06)}
.drop-zone.has-file{border-color:rgba(92,191,138,.45);background:rgba(92,191,138,.05)}
.drop-icon{font-size:38px;margin-bottom:12px;opacity:.65}
.drop-title{font-size:15px;font-weight:600;color:var(--text);margin-bottom:4px}
.drop-sub{font-size:12px;color:var(--muted)}
.file-info{font-size:12px;color:var(--green);font-family:'DM Mono',monospace;margin-top:8px;font-weight:500}
input[type=file]{display:none}
.submit-btn{width:100%;background:linear-gradient(180deg,var(--gold-l),var(--gold));color:#1a1408;border:none;border-radius:11px;padding:15px;font-family:'DM Sans',sans-serif;font-weight:700;font-size:15px;cursor:pointer;margin-top:4px;transition:transform .15s,box-shadow .2s;letter-spacing:.3px;box-shadow:0 8px 22px -8px rgba(209,175,98,.55)}
.submit-btn:hover:not(:disabled){transform:translateY(-1px)}
.submit-btn:disabled{opacity:.35;cursor:not-allowed;transform:none;box-shadow:none}
.spinner{width:20px;height:20px;border:2px solid rgba(209,175,98,.2);border-top-color:var(--gold);border-radius:50%;animation:spin .8s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}
.progress-card{display:none}.progress-card.active{display:block}
.progress-status{display:flex;align-items:center;gap:12px;margin-bottom:16px}
.progress-msg{font-size:13px;color:var(--text)}
.progress-pct{font-family:'DM Mono',monospace;font-size:12px;color:var(--gold);margin-left:auto}
.progress-track{height:4px;background:rgba(255,255,255,.06);border-radius:2px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--gold-d),var(--gold),var(--gold-l));border-radius:2px;transition:width .4s ease;box-shadow:0 0 8px rgba(209,175,98,.4)}
.step-list{display:flex;flex-direction:column;gap:8px;margin-top:16px}
.step{display:flex;align-items:center;gap:10px;font-size:12px;color:var(--muted);font-family:'DM Mono',monospace}
.step.active{color:var(--text)}.step.done{color:var(--green)}
.step-dot{width:6px;height:6px;border-radius:50%;background:currentColor;flex-shrink:0}
.results-card{display:none}.results-card.active{display:block}
.scores-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}
.score-box{background:var(--bg3);border:1px solid var(--border);border-radius:11px;padding:14px;text-align:center}
.score-box.overall{background:linear-gradient(160deg,rgba(209,175,98,.14),rgba(209,175,98,.03));border-color:var(--border-gold)}
.score-label{font-size:9px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;font-family:'DM Mono',monospace;display:block;margin-bottom:6px}
.score-num{font-family:'Bebas Neue',sans-serif;font-size:34px;color:var(--gold);line-height:1}
.score-denom{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace}
.obs-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}
.obs-item{background:var(--bg3);border:1px solid var(--border);border-radius:11px;padding:14px 16px}
.obs-label{font-size:9px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;font-family:'DM Mono',monospace;display:block;margin-bottom:5px}
.obs-text{font-size:12.5px;color:var(--text);line-height:1.6}
.standout{border-left:3px solid var(--gold);padding-left:12px}
.weakest{border-left:3px solid rgba(229,105,95,.6);padding-left:12px}
.review-block{background:var(--bg3);border:1px solid var(--border);border-radius:12px;padding:20px;position:relative}
.review-text{font-size:14.5px;line-height:1.9;color:#dedbd3;white-space:pre-wrap}
.copy-btn{position:absolute;top:12px;right:12px;background:rgba(209,175,98,.1);border:1px solid var(--border-gold);border-radius:8px;padding:6px 13px;color:var(--gold);font-size:11px;font-family:'DM Mono',monospace;cursor:pointer;transition:all .2s}
.copy-btn:hover{background:rgba(209,175,98,.2)}.copy-btn.copied{color:var(--green);border-color:rgba(92,191,138,.35)}
.new-btn{width:100%;background:transparent;border:1px solid var(--border);border-radius:11px;padding:13px;color:var(--dim);font-family:'DM Sans',sans-serif;font-size:14px;cursor:pointer;margin-top:12px;transition:all .2s}
.new-btn:hover{border-color:var(--border-gold);color:var(--text)}
.error-msg{background:rgba(229,105,95,.07);border:1px solid rgba(229,105,95,.25);border-radius:9px;padding:12px 16px;font-size:13px;color:#f0a09a;display:none}
.error-msg.active{display:block}
.film-tag{display:inline-flex;align-items:center;gap:6px;background:rgba(209,175,98,.07);border:1px solid var(--border-gold);border-radius:20px;padding:5px 13px;font-size:11px;color:var(--gold-l);font-family:'DM Mono',monospace;margin-bottom:16px}
.library-picker{background:var(--bg3);border:1px solid var(--border-gold);border-radius:12px;padding:16px 18px;margin-bottom:20px}
.library-label{font-size:10px;color:var(--gold-d);letter-spacing:2px;text-transform:uppercase;font-family:'DM Mono',monospace;margin-bottom:8px}
.library-select{width:100%;background:var(--bg2);border:1px solid var(--border);border-radius:9px;padding:11px 13px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:13px;outline:none}
.library-hint{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace;margin-top:6px;min-height:16px}
.cache-badge{display:inline-flex;align-items:center;gap:5px;background:rgba(92,191,138,.1);border:1px solid rgba(92,191,138,.28);border-radius:6px;padding:3px 10px;font-size:11px;color:var(--green);font-family:'DM Mono',monospace;margin-top:6px}
.link-input-wrap{display:flex;flex-direction:column;gap:8px}
.link-hint{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace}

/* Publish (WordPress) actions on review screens */
.publish-bar{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:14px;padding-top:16px;border-top:1px solid var(--border)}
.publish-bar .pb-label{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace;letter-spacing:1px;text-transform:uppercase;margin-right:auto}
.wp-status{font-size:12px;font-family:'DM Mono',monospace;color:var(--green);display:inline-flex;align-items:center;gap:6px}
.wp-status a{color:var(--gold);text-decoration:underline}

/* Manage chips + sections */
.section{background:linear-gradient(180deg,var(--bg2),#101015);border:1px solid var(--border);border-radius:var(--radius);padding:26px 28px;margin-bottom:22px;box-shadow:var(--shadow)}
.section-header{display:flex;align-items:center;gap:12px;margin-bottom:20px;padding-bottom:16px;border-bottom:1px solid var(--border)}
.section-icon{width:34px;height:34px;border-radius:9px;background:linear-gradient(145deg,rgba(209,175,98,.16),rgba(209,175,98,.03));border:1px solid var(--border-gold);color:var(--gold);display:flex;align-items:center;justify-content:center;flex-shrink:0}
.section-icon svg{width:17px;height:17px}
.section-title{font-size:15px;font-weight:600;color:var(--text);letter-spacing:.2px}
.chips{display:flex;flex-wrap:wrap;gap:8px;min-height:32px;margin-bottom:14px}
.chip{display:inline-flex;align-items:center;gap:7px;background:var(--bg4);border:1px solid var(--border-gold);border-radius:8px;padding:5px 12px;font-size:12px;font-family:'DM Mono',monospace;color:var(--gold)}
.chip-lbl{cursor:pointer}.chip-lbl:hover{color:var(--gold-l);text-decoration:underline dotted}
.chip-del{cursor:pointer;color:var(--muted);font-size:15px;line-height:1;transition:color .15s}
.chip-del:hover{color:var(--red)}
.empty-chips{font-size:12px;color:var(--muted);font-family:'DM Mono',monospace;padding:6px 0}
.add-row{display:flex;gap:10px}
.add-row input{flex:1}
.inline-edit{background:var(--bg3);border:1px solid rgba(209,175,98,.45);border-radius:5px;color:var(--text);padding:3px 8px;font-size:12px;font-family:'DM Mono',monospace;width:140px;outline:none}

/* Admin tables */
.form-row{display:grid;grid-template-columns:1fr 1fr auto auto;gap:12px;align-items:end}
.user-table{width:100%;border-collapse:collapse}
.user-table th{text-align:left;font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;font-family:'DM Mono',monospace;padding:0 0 12px;border-bottom:1px solid var(--border)}
.user-table td{padding:14px 0;border-bottom:1px solid rgba(255,255,255,.05);font-size:13px;vertical-align:middle}
.user-table tr:last-child td{border-bottom:none}
.role-badge{display:inline-flex;align-items:center;padding:3px 11px;border-radius:20px;font-size:10px;font-family:'DM Mono',monospace;letter-spacing:1px;text-transform:uppercase}
.role-admin{background:rgba(209,175,98,.14);color:var(--gold);border:1px solid var(--border-gold)}
.role-user{background:rgba(255,255,255,.05);color:var(--dim);border:1px solid var(--border)}
.section-label{font-size:10px;letter-spacing:1.5px;text-transform:uppercase;font-family:'DM Mono',monospace;color:var(--gold-d);margin-bottom:8px}
.modal-overlay{display:none;position:fixed;inset:0;background:rgba(0,0,0,.65);z-index:100;align-items:center;justify-content:center;padding:20px}
.modal-overlay.open{display:flex}
.modal{background:var(--bg2);border:1px solid var(--border);border-radius:14px;max-width:680px;width:100%;padding:24px;max-height:88vh;overflow:auto}
@media(max-width:640px){
  .form-grid{grid-template-columns:1fr}
  .form-row{grid-template-columns:1fr}
  .obs-grid{grid-template-columns:1fr}
  .scores-row{grid-template-columns:1fr 1fr}
  .filter-row{flex-direction:column}
  .filter-group,.rating-group{min-width:0;width:100%}
  .clear-btn{width:100%}
  .add-row{flex-wrap:wrap}
}

/* Review detail page */
.back{display:inline-flex;align-items:center;gap:6px;color:var(--muted);text-decoration:none;font-size:12px;font-family:'DM Mono',monospace;margin-bottom:22px;transition:color .15s}
.back:hover{color:var(--gold)}
.film-header{margin-bottom:24px}
.film-title{font-family:'Bebas Neue',sans-serif;font-size:40px;line-height:1;color:var(--text);letter-spacing:.5px;margin-bottom:8px}
.film-director{font-size:14px;color:var(--dim);margin-bottom:16px}
.film-meta{display:flex;gap:8px;flex-wrap:wrap}
.meta-tag{font-size:11px;font-family:'DM Mono',monospace;background:var(--bg3);border:1px solid var(--border);border-radius:6px;padding:4px 11px;color:var(--dim)}
.review-block-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:22px;padding-bottom:16px;border-bottom:1px solid var(--border)}
.festival-badge{font-size:11px;font-family:'DM Mono',monospace;background:rgba(209,175,98,.09);border:1px solid var(--border-gold);border-radius:6px;padding:4px 11px;color:var(--gold)}
.review-date{font-size:11px;font-family:'DM Mono',monospace;color:var(--muted)}
.section-divider{font-size:10px;font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:2px;color:var(--gold-d);margin:24px 0 14px}
.score-box .lbl{display:block;font-size:9.5px;font-family:'DM Mono',monospace;color:var(--muted);text-transform:uppercase;letter-spacing:1px;margin-bottom:8px}
.score-box .num{font-family:'Bebas Neue',sans-serif;font-size:30px;line-height:1;color:var(--text)}
.score-box.overall .num{color:var(--gold)}
.score-box .den{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace}
.obs-item .obs-lbl{display:block;font-size:10px;font-family:'DM Mono',monospace;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:6px}
.obs-item.standout .obs-lbl{color:var(--green)}
.obs-item.weakest .obs-lbl{color:#e5965f}
.obs-item.weakest{border-color:rgba(229,150,95,.32)}
.obs-item.standout{border-color:rgba(92,191,138,.32)}
.obs-item .obs-body{font-size:13px;line-height:1.65;color:var(--text)}

/* Button aliases / manage add button */
.btn-red{background:rgba(229,105,95,.12);color:var(--red);border:1px solid rgba(229,105,95,.28);border-radius:8px;padding:8px 14px;font-family:'DM Sans',sans-serif;font-weight:600;font-size:12px;cursor:pointer;transition:all .2s}
.btn-red:hover{background:rgba(229,105,95,.22)}
.add-btn{background:linear-gradient(180deg,var(--gold-l),var(--gold));color:#1a1408;border:none;border-radius:9px;padding:11px 20px;font-size:13px;font-weight:600;cursor:pointer;white-space:nowrap;transition:transform .15s;box-shadow:0 5px 16px -6px rgba(209,175,98,.5)}
.add-btn:hover{transform:translateY(-1px)}
.main{width:100%}
"""

# ── Inline icons (used in the sidebar, topbar, stat cards) ─────────────────────
def _svg(path):
    return ('<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" '
            'stroke-linecap="round" stroke-linejoin="round">' + path + '</svg>')

ICONS = {
    "logo":      _svg('<path d="M2 6l4 3-2.5 3M22 6l-4 3 2.5 3M8 4l2 3-2 3M16 4l-2 3 2 3"/><rect x="3" y="12" width="18" height="8" rx="2"/>'),
    "dashboard": _svg('<rect x="3" y="3" width="7" height="9" rx="1.5"/><rect x="14" y="3" width="7" height="5" rx="1.5"/><rect x="14" y="12" width="7" height="9" rx="1.5"/><rect x="3" y="16" width="7" height="5" rx="1.5"/>'),
    "new":       _svg('<rect x="2" y="4" width="20" height="16" rx="2"/><path d="M7 4v16M17 4v16M2 9h5M2 15h5M17 9h5M17 15h5"/>'),
    "reviews":   _svg('<path d="M4 4h16v13H8l-4 3z"/><path d="M8 9h8M8 12.5h5"/>'),
    "manage":    _svg('<path d="M3 6h18M3 12h18M3 18h18"/><circle cx="8" cy="6" r="2" fill="currentColor" stroke="none"/><circle cx="16" cy="12" r="2" fill="currentColor" stroke="none"/><circle cx="9" cy="18" r="2" fill="currentColor" stroke="none"/>'),
    "wordpress": _svg('<circle cx="12" cy="12" r="10"/><path d="M4 9h9a2.5 2.5 0 0 1 0 5H9l3 6M8 5l6 14"/>'),
    "admin":     _svg('<path d="M12 2 4 6v6c0 5 3.4 8.5 8 10 4.6-1.5 8-5 8-10V6z"/><path d="M9 12l2 2 4-4"/>'),
    "star":      _svg('<path d="M12 3l2.6 5.3 5.9.9-4.3 4.1 1 5.8L12 16.9 6.8 19.2l1-5.8L3.5 9.2l5.9-.9z"/>'),
    "published": _svg('<path d="M4 4h16v13H8l-4 3z"/><path d="m9 10 2 2 4-4"/>'),
    "calendar":  _svg('<rect x="3" y="4" width="18" height="18" rx="2"/><path d="M3 9h18M8 2v4M16 2v4"/>'),
    "menu":      _svg('<path d="M3 6h18M3 12h18M3 18h18"/>'),
}

# Sidebar nav definition: (key, label, href, icon, admin_only)
NAV_ITEMS = [
    ("dashboard", "Dashboard",  "/dashboard", "dashboard", False),
    ("new",       "New Review", "/new",       "new",       "user_only"),
    ("reviews",   "Reviews",    "/reviews",   "reviews",   False),
    ("manage",    "Manage",     "/manage",    "manage",    False),
    ("wordpress", "WordPress",  "/wordpress", "wordpress", True),
    ("admin",     "Admin",      "/admin",     "admin",     True),
]


def _sidebar(active: str) -> str:
    items = []
    for key, label, href, icon, gate in NAV_ITEMS:
        cls = "nav-item active" if key == active else "nav-item"
        link = ('<a href="' + href + '" class="' + cls + '">' + ICONS[icon]
                + '<span>' + label + '</span></a>')
        if gate is True:            # admin only
            link = "{% if session.get('role') == 'admin' %}" + link + "{% endif %}"
        elif gate == "user_only":   # hide from admins
            link = "{% if session.get('role') != 'admin' %}" + link + "{% endif %}"
        items.append(link)
    nav = "".join(items)
    return (
        '<aside class="sidebar">'
        '<div class="brand"><div class="brand-mark">' + ICONS["logo"] + '</div>'
        '<div><div class="brand-name">Festival Reviewer</div>'
        '<div class="brand-sub">Expert Review Studio</div></div></div>'
        '<nav class="sidebar-nav">' + nav + '</nav>'
        '<div class="sidebar-foot"><div class="sidebar-user">'
        '<div class="avatar">{{ (session.get("user","?")[:1])|upper }}</div>'
        '<div class="sidebar-user-meta">'
        '<div class="sidebar-user-email">{{ session.get("user","") }}</div>'
        '<div class="sidebar-user-role">{{ session.get("role","user") }}</div>'
        '</div></div>'
        '<a href="/logout" class="sidebar-signout">Sign out</a></div>'
        '</aside>'
    )


SHELL_HEAD = (
    '<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&'
    'family=DM+Sans:ital,wght@0,300;0,400;0,500;0,600;0,700;1,300&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">'
    '<style>' + SHELL_CSS + PAGE_CSS + '</style>'
)


def _shell(active: str, title: str, body: str) -> str:
    """Wrap a page body in the app shell (sidebar + topbar). Returns a Jinja
    template string to be rendered with render_template_string(..., **ctx)."""
    return (
        '<!DOCTYPE html><html lang="en"><head>'
        '<title>' + title + ' — Festival Reviewer</title>'
        + SHELL_HEAD +
        '</head><body>'
        '<div class="app">'
        + _sidebar(active) +
        '<div class="main-col">'
        '<header class="topbar">'
        '<button class="menu-btn" onclick="document.body.classList.toggle(\'nav-open\')" aria-label="Menu">'
        + ICONS["menu"] + '</button>'
        '<div class="topbar-title">' + title + '</div>'
        '<div class="topbar-right"><span class="topbar-user">{{ session.get("user","") }}</span>'
        '<a href="/logout" class="topbar-logout">Sign out</a></div>'
        '</header>'
        '<main class="content">' + body + '</main>'
        '</div>'
        '<div class="nav-scrim" onclick="document.body.classList.remove(\'nav-open\')"></div>'
        '</div></body></html>'
    )


def render_page(active, title, body, **ctx):
    return render_template_string(_shell(active, title, body), **ctx)


# ── HTML ──────────────────────────────────────────────────
DASHBOARD_BODY = """
<div class="page-head">
  <div>
    <div class="section-eyebrow">Overview</div>
    <div class="page-sub" style="margin-top:0">{{ scope_name }}</div>
  </div>
  {% if not is_admin %}
  <a href="/new" class="btn btn-gold">""" + ICONS["new"] + """ New Review</a>
  {% endif %}
</div>

<div class="stat-grid">
  <div class="statcard">
    <div class="statcard-icon">""" + ICONS["reviews"] + """</div>
    <div class="statcard-value">{{ stats.total }}</div>
    <div class="statcard-label">Total Reviews</div>
  </div>
  <div class="statcard">
    <div class="statcard-icon">""" + ICONS["star"] + """</div>
    <div class="statcard-value">{{ stats.avg }}{% if stats.avg != '—' %}<span class="unit">/10</span>{% endif %}</div>
    <div class="statcard-label">Average Rating</div>
  </div>
  <div class="statcard">
    <div class="statcard-icon">""" + ICONS["published"] + """</div>
    <div class="statcard-value">{{ stats.published }}</div>
    <div class="statcard-label">Published to WP</div>
  </div>
  <div class="statcard">
    <div class="statcard-icon">""" + ICONS["calendar"] + """</div>
    <div class="statcard-value">{{ stats.seasons }}</div>
    <div class="statcard-label">Seasons</div>
  </div>
</div>

<div class="card">
  <div class="card-head">
    <div class="card-head-icon">""" + ICONS["reviews"] + """</div>
    <div class="card-head-title">Recent Reviews</div>
    <a href="/reviews" class="btn btn-ghost btn-sm" style="margin-left:auto">View all</a>
  </div>
  <div class="card-body" style="padding:8px 0">
    {% if reviews %}
    <table class="user-table" style="width:100%">
      <tbody>
      {% for r in reviews %}
        <tr onclick="location.href='/reviews/{{ r.film_id }}'" style="cursor:pointer">
          <td style="padding-left:22px">
            <div style="font-weight:600;color:var(--text)">{{ r.title }}</div>
            <div style="font-size:12px;color:var(--dim)">{{ r.director }}</div>
          </td>
          <td style="color:var(--dim);font-family:'DM Mono',monospace;font-size:11px">{{ r.festival_name }}</td>
          <td>{% if r.season %}<span class="badge badge-muted">{{ r.season }}</span>{% endif %}</td>
          <td style="text-align:right">
            {% if r.get('overall_rating') not in (None, '') %}
            <span class="rc-score">{{ '%.1f'|format(r.get('overall_rating')|float) }}<span class="d">/10</span></span>
            {% endif %}
          </td>
          <td style="text-align:right;padding-right:22px">
            {% if r.wp_post_id %}<span class="badge badge-green">Published</span>{% endif %}
          </td>
        </tr>
      {% endfor %}
      </tbody>
    </table>
    {% else %}
    <div class="empty-state">
      """ + ICONS["reviews"] + """
      <h3>No reviews yet</h3>
      <p>Generate your first expert review to see it here.</p>
      {% if not is_admin %}<a href="/new" class="btn btn-gold">Create a review</a>{% endif %}
    </div>
    {% endif %}
  </div>
</div>
"""


WORDPRESS_BODY = """
<div class="page-head">
  <div>
    <div class="section-eyebrow">Integration</div>
    <div class="page-sub" style="margin-top:0">Connect each festival to its WordPress site and design how published reviews look.</div>
  </div>
</div>

{% if not festivals %}
<div class="empty-state">
  <h3>No festivals yet</h3>
  <p>Create a festival in the Admin panel first, then configure WordPress here.</p>
  <a href="/admin" class="btn btn-gold">Go to Admin</a>
</div>
{% endif %}

{% for key, f in festivals.items() %}
<div class="card">
  <div class="card-head">
    <div class="card-head-icon">""" + ICONS["wordpress"] + """</div>
    <div class="card-head-title">{{ f.name }}</div>
    <span class="badge {{ 'badge-green' if f.wp_configured else 'badge-muted' }}" id="wpchip-{{ key }}" style="margin-left:auto">
      {{ 'Connected' if f.wp_configured else 'Not connected' }}
    </span>
  </div>
  <div class="card-body">
    <div class="section-label">Connection</div>
    <div class="form-grid" style="margin-bottom:14px">
      <div class="form-group">
        <label>WordPress Site URL</label>
        <input type="url" id="wpurl-{{ key }}" value="{{ f.wp_url }}" placeholder="https://yourfestival.com">
      </div>
      <div class="form-group">
        <label>WordPress Username</label>
        <input type="text" id="wpuser-{{ key }}" value="{{ f.wp_user }}" placeholder="editor">
      </div>
      <div class="form-group">
        <label>Application Password</label>
        <input type="password" id="wppass-{{ key }}" autocomplete="new-password"
               placeholder="{{ '•••••• saved — leave blank to keep' if f.wp_configured else 'xxxx xxxx xxxx xxxx' }}">
      </div>
      <div class="form-group">
        <label>Default Publish Mode</label>
        <select id="wpstatus-{{ key }}">
          <option value="draft" {{ 'selected' if f.wp_default_status != 'publish' else '' }}>Draft</option>
          <option value="publish" {{ 'selected' if f.wp_default_status == 'publish' else '' }}>Publish live</option>
        </select>
      </div>
    </div>
    <div style="font-size:11px;color:var(--muted);font-family:'DM Mono',monospace;line-height:1.6;margin-bottom:14px">
      In WordPress: <strong style="color:var(--gold)">Users → Profile → Application Passwords</strong> → add one, paste it here. Stored encrypted server-side and never shown again.
    </div>
    <button class="btn btn-gold btn-sm" onclick="wpSaveConfig('{{ key }}')">Save connection</button>

    <div class="section-label" style="margin-top:26px">Reference Design → Template</div>
    <div style="font-size:12px;color:var(--dim);line-height:1.6;margin-bottom:12px">
      Paste a sample post URL or its HTML, and generate a reusable template. Every review published to this festival's site is rendered into it.
    </div>
    <div class="form-grid" style="margin-bottom:12px">
      <div class="form-group">
        <label>Reference post URL <span class="optional-tag">optional</span></label>
        <input type="url" id="wpref-{{ key }}" placeholder="https://yourfestival.com/a-past-review">
      </div>
      <div class="form-group">
        <label>…or paste reference HTML <span class="optional-tag">optional</span></label>
        <textarea id="wprefhtml-{{ key }}" style="min-height:44px" placeholder="<div>…</div>"></textarea>
      </div>
    </div>
    <button class="btn btn-ghost btn-sm" onclick="wpGen('{{ key }}')" id="wpgenbtn-{{ key }}">✨ Generate template</button>

    <div class="form-group" style="margin-top:18px">
      <label>Template HTML <span class="optional-tag">uses {{ '{{title}}' }}, {{ '{{review_body}}' }}, {{ '{{scores_table}}' }}…</span></label>
      <textarea id="wptpl-{{ key }}" style="min-height:180px;font-family:'DM Mono',monospace;font-size:12px">{{ f.wp_template }}</textarea>
    </div>
    <div style="display:flex;gap:10px;flex-wrap:wrap;margin-top:4px">
      <button class="btn btn-gold btn-sm" onclick="wpSaveTpl('{{ key }}')">Save template</button>
      <button class="btn btn-ghost btn-sm" onclick="wpPreview('{{ key }}')">Preview</button>
    </div>
    <div style="margin-top:14px;display:flex;flex-wrap:wrap;gap:6px">
      {% for p in placeholders %}<span class="rc-tag">{{ '{{' }}{{ p }}{{ '}}' }}</span>{% endfor %}
    </div>
  </div>
</div>
{% endfor %}

<!-- Preview modal -->
<div class="modal-overlay" id="wpPreviewModal">
  <div class="modal" style="max-width:820px">
    <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:14px">
      <div class="card-head-title">Publish Preview</div>
      <button class="btn btn-ghost btn-sm" onclick="document.getElementById('wpPreviewModal').classList.remove('open')">Close</button>
    </div>
    <iframe id="wpPreviewFrame" style="width:100%;height:60vh;border:1px solid var(--border);border-radius:10px;background:#fff"></iframe>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
function wpToast(msg, err){const t=document.getElementById('toast');t.textContent=msg;t.className='toast show'+(err?' err':'');clearTimeout(t._t);t._t=setTimeout(()=>t.className='toast',3000);}
function wpJSON(url, method, body){return fetch(url,{method:method,headers:{'Content-Type':'application/json'},body:JSON.stringify(body||{})}).then(r=>r.json().then(d=>({ok:r.ok,d:d})));}

function wpSaveConfig(key){
  const body={wp_url:val('wpurl-'+key),wp_user:val('wpuser-'+key),wp_app_pass:val('wppass-'+key),wp_default_status:val('wpstatus-'+key)};
  wpJSON('/api/festivals/'+key+'/wp/config','POST',body).then(({ok,d})=>{
    if(!ok){wpToast(d.error||'Save failed',true);return;}
    document.getElementById('wppass-'+key).value='';
    const chip=document.getElementById('wpchip-'+key);
    chip.textContent=d.wp_configured?'Connected':'Not connected';
    chip.className='badge '+(d.wp_configured?'badge-green':'badge-muted');
    wpToast('Connection saved');
  });
}
function wpGen(key){
  const btn=document.getElementById('wpgenbtn-'+key);const old=btn.textContent;btn.textContent='Generating…';btn.disabled=true;
  wpJSON('/api/festivals/'+key+'/wp/template/generate','POST',{reference_url:val('wpref-'+key),reference_html:val('wprefhtml-'+key)}).then(({ok,d})=>{
    btn.textContent=old;btn.disabled=false;
    if(!ok){wpToast(d.error||'Generation failed',true);return;}
    document.getElementById('wptpl-'+key).value=d.template;
    wpToast('Template generated — review it, then Save');
  });
}
function wpSaveTpl(key){
  wpJSON('/api/festivals/'+key+'/wp/template','PUT',{template:val('wptpl-'+key)}).then(({ok,d})=>{
    wpToast(ok?'Template saved':(d.error||'Save failed'),!ok);
  });
}
function wpPreview(key){
  wpJSON('/api/festivals/'+key+'/wp/preview','POST',{template:val('wptpl-'+key)}).then(({ok,d})=>{
    if(!ok){wpToast(d.error||'Preview failed',true);return;}
    document.getElementById('wpPreviewFrame').srcdoc='<body style=\"margin:0;padding:24px;background:#fff\">'+d.html+'</body>';
    document.getElementById('wpPreviewModal').classList.add('open');
  });
}
function val(id){const el=document.getElementById(id);return el?el.value:'';}
</script>
"""


LOGIN_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Festival Reviewer — Sign In</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
:root{--gold:#d1af62;--gold-l:#ecca86;--gold-d:#96793a;
      --bg:#0a0a0d;--bg2:#141419;--bg3:#1c1c24;
      --border:rgba(255,255,255,.08);--text:#eceae4;--muted:#78747f;--red:#e5695f}
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;
     min-height:100vh;display:flex;align-items:center;justify-content:center;padding:20px;
     background-image:radial-gradient(ellipse 55% 45% at 50% -5%,rgba(209,175,98,.10),transparent 70%),
                      radial-gradient(ellipse 40% 40% at 90% 110%,rgba(209,175,98,.04),transparent 70%)}
.box{width:400px;max-width:100%;background:linear-gradient(180deg,var(--bg2),#101015);
     border:1px solid var(--border);border-radius:20px;padding:44px 40px 40px;text-align:center;
     box-shadow:0 30px 70px -30px rgba(0,0,0,.85),0 0 0 1px rgba(209,175,98,.04);position:relative;overflow:hidden}
.box::before{content:"";position:absolute;top:0;left:0;right:0;height:1px;
     background:linear-gradient(90deg,transparent,rgba(209,175,98,.5),transparent)}
.crest{width:52px;height:52px;margin:0 auto 18px;border-radius:14px;
     background:linear-gradient(145deg,rgba(209,175,98,.18),rgba(209,175,98,.04));
     border:1px solid rgba(209,175,98,.25);display:flex;align-items:center;justify-content:center;color:var(--gold)}
.logo{font-family:'Bebas Neue',sans-serif;font-size:38px;color:var(--gold);letter-spacing:2px;
     line-height:1;margin-bottom:8px;background:linear-gradient(180deg,var(--gold-l),var(--gold-d));
     -webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.sub{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace;letter-spacing:3px;text-transform:uppercase;margin-bottom:34px}
label{display:block;text-align:left;font-size:10px;color:var(--muted);letter-spacing:1.5px;text-transform:uppercase;font-family:'DM Mono',monospace;margin-bottom:7px}
input{width:100%;background:var(--bg3);border:1px solid var(--border);border-radius:10px;padding:13px 15px;color:var(--text);font-family:'DM Sans',sans-serif;font-size:14px;outline:none;margin-bottom:18px;transition:border-color .2s,box-shadow .2s}
input::placeholder{color:#4f4c57}
input:focus{border-color:var(--gold);box-shadow:0 0 0 3px rgba(209,175,98,.12)}
button[type=submit]{width:100%;background:linear-gradient(180deg,var(--gold-l),var(--gold));color:#1a1408;border:none;border-radius:10px;padding:14px;font-family:'DM Sans',sans-serif;font-weight:700;font-size:14px;cursor:pointer;margin-top:8px;transition:transform .15s,box-shadow .2s;box-shadow:0 6px 18px -6px rgba(209,175,98,.5)}
button[type=submit]:hover{transform:translateY(-1px);box-shadow:0 10px 24px -8px rgba(209,175,98,.6)}
button[type=submit]:active{transform:translateY(0)}
.error{background:rgba(229,105,95,.1);border:1px solid rgba(229,105,95,.3);border-radius:10px;padding:11px;font-size:13px;color:#f0a09a;margin-bottom:18px}
@media(max-width:480px){.box{border-radius:16px;padding:32px 22px}}
.pw-wrap{position:relative;margin-bottom:18px}
.pw-wrap input{margin-bottom:0;padding-right:44px}
.pw-toggle{position:absolute;right:14px;top:50%;transform:translateY(-50%);background:none;border:none;cursor:pointer;color:var(--muted);padding:0;width:auto;font-size:18px;line-height:1;transition:color .2s;display:flex}
.pw-toggle:hover{color:var(--gold);background:none}
</style></head><body>
<div class="box">
  <div class="crest">
    <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round"><path d="M2 6l4 3-2.5 3M22 6l-4 3 2.5 3M8 4l2 3-2 3M16 4l-2 3 2 3"/><rect x="3" y="12" width="18" height="8" rx="2"/></svg>
  </div>
  <div class="logo">Festival Reviewer</div>
  <div class="sub">AI-Powered Film Review</div>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <label>Email</label>
    <input type="email" name="email" placeholder="your@email.com" required autofocus>
    <label>Password</label>
    <div class="pw-wrap">
      <input type="password" id="pwField" name="password" placeholder="••••••••" required>
      <button type="button" class="pw-toggle" onclick="togglePw()" id="pwToggle" aria-label="Show password">👁</button>
    </div>
    <button type="submit">Sign In</button>
  </form>
</div>
<script>
function togglePw() {
  const f = document.getElementById('pwField');
  f.type = f.type === 'password' ? 'text' : 'password';
}
</script>
</body></html>"""


ADMIN_BODY = """<div class="main">
  <div class="page-sub" style="margin-bottom:24px">Add and remove portal users, and configure festivals.</div>

  {% set msg_success = request.args.get('success') %}
  {% set msg_error   = request.args.get('error') %}
  {% if msg_success %}<div class="alert alert-success">✓ {{ msg_success }}</div>{% endif %}
  {% if msg_error   %}<div class="alert alert-error">✕ {{ msg_error }}</div>{% endif %}

  <!-- Add user -->
  <div class="card">
    <div class="card-head">
      <div class="card-head-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M19 8v6M22 11h-6"/></svg></div>
      <div class="card-head-title">Add User</div>
    </div>
    <div class="card-body">
      {% if festivals %}
      <form method="POST" action="/admin/users">
        <div class="form-row" style="align-items:flex-end">
          <div style="flex:2">
            <label>Festival *</label>
            <select name="festival_key" id="newUserFestival" required>
              <option value="" disabled selected>— select a festival —</option>
              {% for key, f in festivals.items() %}
              <option value="{{ key }}">{{ f.name }}</option>
              {% endfor %}
            </select>
          </div>
          <div style="flex:2">
            <label>Email *</label>
            <input type="email" name="email" placeholder="user@festival.com" required>
          </div>
          <div style="flex:2">
            <label>Password *</label>
            <input type="password" name="password" placeholder="••••••••" required minlength="8">
          </div>
          <div>
            <label>&nbsp;</label>
            <button type="submit" class="btn btn-gold">Add User</button>
          </div>
        </div>
      </form>
      {% else %}
      <div class="empty">Create a festival first before adding users.</div>
      {% endif %}
      <script>
      function togglePwRow(key) {
        const row = document.getElementById('pw-row-' + key);
        if (row) row.style.display = row.style.display === 'none' ? 'table-row' : 'none';
      }
      </script>
    </div>
  </div>

  <!-- User list -->
  <div class="card">
    <div class="card-head">
      <div class="card-head-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87M16 3.13a4 4 0 0 1 0 7.75"/></svg></div>
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
            <td style="text-align:right;white-space:nowrap">
              {% if u.email != current_user %}
              <button type="button" class="btn btn-gold" style="font-size:11px;padding:4px 10px"
                      onclick="togglePwRow('{{ u.email|replace('@','__at__') }}')">Change PW</button>
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
          {% if u.email != current_user %}
          <tr id="pw-row-{{ u.email|replace('@','__at__') }}" style="display:none;background:var(--bg3)">
            <td colspan="5" style="padding:10px 16px">
              <form method="POST" action="/admin/users/password" style="display:flex;gap:10px;align-items:center">
                <input type="hidden" name="email" value="{{ u.email }}">
                <input type="password" name="new_password" placeholder="New password (min 8 chars)"
                       style="flex:1;background:var(--bg4);border:1px solid rgba(255,255,255,.1);border-radius:6px;color:var(--text);padding:6px 12px;font-size:13px">
                <button type="submit" class="btn btn-gold" style="font-size:12px;padding:6px 14px">Set Password</button>
                <button type="button" class="btn" style="font-size:12px;padding:6px 14px;background:var(--bg4)"
                        onclick="togglePwRow('{{ u.email|replace('@','__at__') }}')">Cancel</button>
              </form>
            </td>
          </tr>
          {% endif %}
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
      <div class="card-head-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22V4l14 4-14 4"/><path d="M4 4V2"/></svg></div>
      <div class="card-head-title">Add Festival</div>
    </div>
    <div class="card-body">
      <form method="POST" action="/admin/festivals">
        <div class="section-label">Identity</div>
        <div class="form-row">
          <div>
            <label>Key (slug) *</label>
            <input type="text" name="key" id="festKeyInput" placeholder="e.g. sundance" required
                   pattern="[a-z0-9_-]+" title="lowercase letters, numbers, hyphens, underscores — used in URLs and DB"
                   oninput="previewSlug(this.value)">
            <div id="festKeyPreview" style="font-size:11px;font-family:'DM Mono',monospace;color:var(--muted);margin-top:4px;min-height:16px"></div>
            <script>
            function sanitiseSlug(v) {
              return v.trim().toLowerCase()
                .replace(/\s+/g, '-')
                .replace(/[^a-z0-9_-]/g, '')
                .replace(/-{2,}/g, '-')
                .replace(/^-+|-+$/g, '')
                .slice(0, 64);
            }
            function previewSlug(v) {
              const slug = sanitiseSlug(v);
              const el = document.getElementById('festKeyPreview');
              if (!slug) { el.textContent = ''; return; }
              const changed = slug !== v.trim().toLowerCase();
              el.innerHTML = 'Will be stored as: <strong style="color:var(--gold)">' + slug + '</strong>'
                + (changed ? ' <span style="color:var(--red)">(auto-corrected)</span>' : ' ✓');
            }
            // Auto-fill slug from Short Name if user hasn't touched the key field
            document.addEventListener('DOMContentLoaded', () => {
              const nameInput = document.querySelector('input[name="name"]');
              const keyInput  = document.getElementById('festKeyInput');
              if (nameInput && keyInput) {
                nameInput.addEventListener('input', () => {
                  if (!keyInput.dataset.touched) {
                    const slug = sanitiseSlug(nameInput.value);
                    keyInput.value = slug;
                    previewSlug(slug);
                  }
                });
                keyInput.addEventListener('input', () => { keyInput.dataset.touched = '1'; });
              }
            });
            </script>
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

        <div style="font-size:12px;color:var(--muted);margin-top:16px;font-family:'DM Mono',monospace;line-height:1.6">
          Categories &amp; their judging prompts are managed on the <strong style="color:var(--gold)">Manage</strong> page —
          pick them from the built-in catalog after creating the festival. Reviews use smart defaults and are capped at 500 words.
        </div>

        <div class="section-label" style="margin-top:20px">Access & Integration</div>
        <div class="form-row" style="margin-top:4px">
          <div>
            <label>Gemini API Key</label>
            <input type="text" name="gemini_api_key" placeholder="Leave blank to use global key">
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
      <div class="card-head-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 7a2 2 0 0 1 2-2h4l2 2h8a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/></svg></div>
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

            <div style="font-size:12px;color:var(--muted);margin-top:16px;font-family:'DM Mono',monospace;line-height:1.6">
              Categories &amp; judging prompts are managed on the <strong style="color:var(--gold)">Manage</strong> page (catalog picker). Reviews use smart defaults.
            </div>

            <div class="section-label" style="margin-top:16px">Seasons</div>
            <div id="seasons-{{ key }}" style="display:flex;flex-wrap:wrap;gap:8px;margin:8px 0 10px"></div>
            <div style="display:flex;gap:8px">
              <input type="text" id="season-input-{{ key }}" placeholder="Add season… e.g. 2025, Spring 2025" style="flex:1;background:var(--bg3);border:1px solid rgba(255,255,255,.07);border-radius:8px;color:var(--text);padding:8px 12px;font-size:13px">
              <button type="button" class="btn btn-gold" onclick="seasonAdd('{{ key }}')" style="padding:8px 16px;font-size:13px">Add</button>
            </div>

            <div class="section-label" style="margin-top:16px">Access & Integration</div>
            <div class="form-row" style="margin-top:4px">
              <div>
                <label>Gemini API Key</label>
                <input type="text" name="gemini_api_key" value="{{ f.gemini_api_key or '' }}" placeholder="Leave blank to use global key">
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

<!-- Per-category judging prompt modal -->
<div id="promptModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;align-items:center;justify-content:center;padding:20px">
  <div style="background:var(--bg2);border:1px solid var(--border);border-radius:12px;max-width:640px;width:100%;padding:24px">
    <div id="promptModalTitle" style="font-size:15px;font-weight:600;color:var(--gold);margin-bottom:6px">Judging emphasis</div>
    <div style="font-size:12px;color:var(--muted);margin-bottom:12px">Extra judging & writing guidance applied only to films submitted in this category. Leave blank to use the festival default.</div>
    <textarea id="promptModalText" rows="10" style="width:100%;background:var(--bg3);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:'DM Mono',monospace;font-size:12px;padding:10px 12px;resize:vertical;line-height:1.6"></textarea>
    <div style="display:flex;gap:10px;justify-content:flex-end;margin-top:14px">
      <button onclick="closePromptModal()" style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;color:var(--muted);padding:8px 16px;font-size:13px;cursor:pointer">Cancel</button>
      <button id="promptModalSave" style="background:var(--gold);border:none;border-radius:8px;color:#000;padding:8px 18px;font-size:13px;font-weight:600;cursor:pointer">Save</button>
    </div>
  </div>
</div>
<script>
// ── Generic chip renderer ─────────────────────────────────
function renderChips(containerId, items, onRename, onDelete, onPrompt) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '';
  if (!items.length) {
    const em = document.createElement('span');
    em.style.cssText = 'font-size:12px;color:var(--muted)';
    em.textContent = 'None yet';
    el.appendChild(em);
    return;
  }
  items.forEach(item => {
    const name = typeof item === 'string' ? item : item.name;
    const chip = document.createElement('span');
    chip.style.cssText = 'display:inline-flex;align-items:center;gap:6px;background:var(--bg3);border:1px solid rgba(201,168,76,.25);border-radius:6px;padding:4px 10px;font-size:12px;font-family:"DM Mono",monospace;color:var(--gold)';
    const lbl = document.createElement('span');
    lbl.textContent = name;
    lbl.style.cursor = 'pointer';
    lbl.title = 'Click to rename';
    lbl.onclick = () => onRename(name, chip);
    chip.appendChild(lbl);
    if (onPrompt) {
      const pen = document.createElement('span');
      pen.textContent = '✎';
      pen.style.cssText = 'cursor:pointer;color:var(--muted);font-size:12px;line-height:1';
      pen.title = 'Edit judging prompt for this category';
      pen.onclick = () => onPrompt(name);
      chip.appendChild(pen);
    }
    const del = document.createElement('span');
    del.textContent = '×';
    del.style.cssText = 'cursor:pointer;color:var(--muted);font-size:14px;line-height:1';
    del.title = 'Delete';
    del.onclick = () => onDelete(name);
    chip.appendChild(del);
    el.appendChild(chip);
  });
}

// ── Categories CRUD ───────────────────────────────────────
function _catRender(festKey, cats) {
  renderChips('cats-' + festKey, cats || [],
    (n, chip) => catRenameInline(festKey, n, chip),
    (n) => catDelete(festKey, n),
    (n) => catEditPrompt(festKey, n));
}
function catsLoad(festKey) {
  fetch('/api/festivals/' + festKey + '/categories')
    .then(r => r.json()).then(d => _catRender(festKey, d.categories));
}
function catAdd(festKey) {
  const inp = document.getElementById('cat-input-' + festKey);
  const name = inp.value.trim();
  if (!name) return;
  fetch('/api/festivals/' + festKey + '/categories', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name})
  }).then(r => r.json()).then(d => {
    if (d.error) { alert(d.error); return; }
    inp.value = '';
    _catRender(festKey, d.categories);
  });
}
function catDelete(festKey, name) {
  if (!confirm('Delete category "' + name + '"?')) return;
  fetch('/api/festivals/' + festKey + '/categories/' + encodeURIComponent(name), {method:'DELETE'})
    .then(r => r.json()).then(d => _catRender(festKey, d.categories));
}
function catRenameInline(festKey, oldName, chip) {
  const inp = document.createElement('input');
  inp.value = oldName;
  inp.style.cssText = 'background:var(--bg3);border:1px solid rgba(201,168,76,.4);border-radius:4px;color:var(--text);padding:2px 6px;font-size:12px;width:120px';
  chip.replaceChildren(inp);
  inp.focus(); inp.select();
  const commit = () => {
    const newName = inp.value.trim();
    if (!newName || newName === oldName) { catsLoad(festKey); return; }
    fetch('/api/festivals/' + festKey + '/categories/' + encodeURIComponent(oldName), {
      method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name: newName})
    }).then(r => r.json()).then(d => {
      if (d.error) { alert(d.error); catsLoad(festKey); return; }
      _catRender(festKey, d.categories);
    });
  };
  inp.addEventListener('blur', commit);
  inp.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); commit(); } if (e.key === 'Escape') catsLoad(festKey); });
}
// ── Per-category judging prompt editor ────────────────────
function catEditPrompt(festKey, name) {
  const overlay = document.getElementById('promptModal');
  const ta      = document.getElementById('promptModalText');
  const title   = document.getElementById('promptModalTitle');
  title.textContent = 'Judging emphasis — ' + name;
  ta.value = 'Loading…'; ta.disabled = true;
  overlay.style.display = 'flex';
  fetch('/api/festivals/' + festKey + '/categories/' + encodeURIComponent(name) + '/prompt')
    .then(r => r.json()).then(d => { ta.value = d.prompt || ''; ta.disabled = false; ta.focus(); });
  document.getElementById('promptModalSave').onclick = () => {
    fetch('/api/festivals/' + festKey + '/categories/' + encodeURIComponent(name) + '/prompt', {
      method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify({prompt: ta.value})
    }).then(r => r.json()).then(d => {
      if (d.error) { alert(d.error); return; }
      overlay.style.display = 'none';
    });
  };
}
function closePromptModal() { document.getElementById('promptModal').style.display = 'none'; }

// ── Seasons CRUD ──────────────────────────────────────────
function seasonsLoad(festKey) {
  fetch('/api/festivals/' + festKey + '/seasons')
    .then(r => r.json()).then(d => renderChips('seasons-' + festKey, d.seasons || [],
      (n, chip) => seasonRenameInline(festKey, n, chip),
      (n) => seasonDelete(festKey, n)));
}
function seasonAdd(festKey) {
  const inp = document.getElementById('season-input-' + festKey);
  const name = inp.value.trim();
  if (!name) return;
  fetch('/api/festivals/' + festKey + '/seasons', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name})
  }).then(r => r.json()).then(d => {
    if (d.error) { alert(d.error); return; }
    inp.value = '';
    renderChips('seasons-' + festKey, d.seasons || [],
      (n, chip) => seasonRenameInline(festKey, n, chip), (n) => seasonDelete(festKey, n));
  });
}
function seasonDelete(festKey, name) {
  if (!confirm('Delete season "' + name + '"?')) return;
  fetch('/api/festivals/' + festKey + '/seasons/' + encodeURIComponent(name), {method:'DELETE'})
    .then(r => r.json()).then(d => renderChips('seasons-' + festKey, d.seasons || [],
      (n, chip) => seasonRenameInline(festKey, n, chip), (n) => seasonDelete(festKey, n)));
}
function seasonRenameInline(festKey, oldName, chip) {
  const inp = document.createElement('input');
  inp.value = oldName;
  inp.style.cssText = 'background:var(--bg3);border:1px solid rgba(201,168,76,.4);border-radius:4px;color:var(--text);padding:2px 6px;font-size:12px;width:140px';
  chip.replaceChildren(inp);
  inp.focus(); inp.select();
  const commit = () => {
    const newName = inp.value.trim();
    if (!newName || newName === oldName) { seasonsLoad(festKey); return; }
    fetch('/api/festivals/' + festKey + '/seasons/' + encodeURIComponent(oldName), {
      method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name: newName})
    }).then(r => r.json()).then(d => {
      if (d.error) { alert(d.error); seasonsLoad(festKey); return; }
      renderChips('seasons-' + festKey, d.seasons || [],
        (n, c) => seasonRenameInline(festKey, n, c), (n) => seasonDelete(festKey, n));
    });
  };
  inp.addEventListener('blur', commit);
  inp.addEventListener('keydown', e => { if (e.key === 'Enter') { e.preventDefault(); commit(); } if (e.key === 'Escape') seasonsLoad(festKey); });
}

// ── Init: load chips when <details> opens ─────────────────
document.querySelectorAll('details').forEach(det => {
  det.addEventListener('toggle', () => {
    if (!det.open) return;
    const keyEl = det.querySelector('input[name="key"]');
    if (!keyEl) return;
    const fk = keyEl.value;
    seasonsLoad(fk);
  });
});
</script>
"""


REVIEWS_BODY = """<div class="main">
  <div class="page-sub" style="margin-bottom:22px">{{ festival_name }}</div>

  <!-- ── Filter bar ── -->
  <div class="filter-bar">
    <div class="filter-row">
      <div class="filter-group search-group">
        <div class="filter-label">Search</div>
        <input class="filter-input" type="text" id="searchInput" placeholder="Title, director, country…" oninput="applyFilters()">
      </div>
      {% if seasons %}
      <div class="filter-group">
        <div class="filter-label">Season</div>
        <select class="filter-select" id="seasonSelect" onchange="applyFilters()">
          <option value="">All seasons</option>
          {% for s in seasons %}
          <option value="{{ s }}">{{ s }}</option>
          {% endfor %}
        </select>
      </div>
      {% endif %}
      {% if categories %}
      <div class="filter-group">
        <div class="filter-label">Category</div>
        <select class="filter-select" id="catSelect" onchange="applyFilters()">
          <option value="">All categories</option>
          {% for c in categories %}
          <option value="{{ c }}">{{ c }}</option>
          {% endfor %}
        </select>
      </div>
      {% endif %}
      <div class="rating-group">
        <div class="filter-label">Min Rating</div>
        <div class="rating-filter">
          <input type="range" id="ratingSlider" min="0" max="10" step="0.5" value="0" oninput="onRatingSlide(this)">
          <span class="rating-val" id="ratingVal">Any</span>
        </div>
      </div>
      <button class="clear-btn" onclick="clearFilters()">Clear</button>
    </div>
    <div class="result-count" id="resultCount"></div>
  </div>

  {% if reviews %}
  <div class="review-grid" id="reviewGrid">
    {% for r in reviews %}
    <a class="review-card" href="/reviews/{{ r.film_id }}"
       data-title="{{ r.title|lower }}"
       data-director="{{ r.director|lower }}"
       data-country="{{ (r.country or '')|lower }}"
       data-season="{{ r.season or '' }}"
       data-category="{{ r.genre or '' }}"
       data-rating="{{ r.get('overall_rating') if r.get('overall_rating') is not none and r.get('overall_rating') != '' else '' }}">
      <div class="rc-top">
        <div>
          <div class="rc-title">{{ r.title }}</div>
          <div class="rc-director">{{ r.director }}</div>
        </div>
        <div style="display:flex;flex-direction:column;align-items:flex-end;gap:4px">
          {% if r.get('overall_rating') is not none and r.get('overall_rating') != '' %}
          <div style="font-family:'DM Mono',monospace;font-size:18px;font-weight:600;color:var(--gold);line-height:1">{{ '%.1f'|format(r.get('overall_rating')|float) }}<span style="font-size:11px;color:var(--muted)">/10</span></div>
          {% endif %}
          <div class="rc-date">{{ r.created_at[:10] if r.created_at else '' }}</div>
        </div>
      </div>
      <div class="rc-meta">
        <span class="rc-tag festival">{{ r.festival_name }}</span>
        {% if r.season %}<span class="rc-tag" style="border-color:rgba(255,255,255,.15);color:var(--text)">{{ r.season }}</span>{% endif %}
        {% if r.genre %}<span class="rc-tag">{{ r.genre }}</span>{% endif %}
        {% if r.runtime %}<span class="rc-tag">{{ r.runtime }}</span>{% endif %}
        {% if r.country %}<span class="rc-tag">{{ r.country }}</span>{% endif %}
      </div>
      <div class="rc-excerpt">{{ r.review_text[:280] if r.review_text else '' }}</div>
    </a>
    {% endfor %}
  </div>
  <div class="no-results" id="noResults">No reviews match the current filters.</div>
  {% else %}
  <div class="empty">No reviews yet. Submit a film to generate the first one.</div>
  {% endif %}
</div>

<script>
const cards = Array.from(document.querySelectorAll('.review-card'));
let minRating = 0;

function sel(id) { const el = document.getElementById(id); return el ? el.value : ''; }

function onRatingSlide(inp) {
  minRating = parseFloat(inp.value);
  document.getElementById('ratingVal').textContent = minRating === 0 ? 'Any' : minRating.toFixed(1) + '+';
  applyFilters();
}

function applyFilters() {
  const searchQ   = document.getElementById('searchInput').value.toLowerCase().trim();
  const season    = sel('seasonSelect');
  const category  = sel('catSelect');
  let visible = 0;
  cards.forEach(card => {
    const matchSearch   = !searchQ ||
      card.dataset.title.includes(searchQ) ||
      card.dataset.director.includes(searchQ) ||
      card.dataset.country.includes(searchQ);
    const matchSeason   = !season   || card.dataset.season === season;
    const matchCategory = !category || card.dataset.category === category;
    const ratingRaw     = card.dataset.rating;
    const matchRating   = minRating === 0 || (ratingRaw !== '' && parseFloat(ratingRaw) >= minRating);
    const show = matchSearch && matchSeason && matchCategory && matchRating;
    card.style.display = show ? '' : 'none';
    if (show) visible++;
  });
  // Highlight active selects
  ['seasonSelect','catSelect'].forEach(id => {
    const el = document.getElementById(id);
    if (el) el.classList.toggle('active', !!el.value);
  });
  const total   = cards.length;
  const active  = searchQ || season || category || minRating > 0;
  document.getElementById('resultCount').textContent =
    active ? visible + ' of ' + total + ' review' + (total !== 1 ? 's' : '') + ' shown'
           : total + ' review' + (total !== 1 ? 's' : '');
  const noRes = document.getElementById('noResults');
  if (noRes) noRes.style.display = visible === 0 && total > 0 ? 'block' : 'none';
}

function clearFilters() {
  document.getElementById('searchInput').value = '';
  const ss = document.getElementById('seasonSelect');
  const cs = document.getElementById('catSelect');
  if (ss) ss.value = '';
  if (cs) cs.value = '';
  minRating = 0;
  document.getElementById('ratingSlider').value = 0;
  document.getElementById('ratingVal').textContent = 'Any';
  applyFilters();
}

applyFilters();
</script>
"""


MANAGE_BODY = """<div class="main">
  <div class="page-sub" style="margin-bottom:22px">Categories and seasons per festival</div>

  {% if not festivals %}
  <div style="color:var(--muted);font-size:13px">No festivals assigned to your account.</div>
  {% endif %}

  {% for fk, f in festivals.items() %}
  <div style="margin-bottom:8px;padding:4px 0;border-bottom:1px solid var(--border);padding-bottom:12px">
    <span style="font-family:'Bebas Neue',sans-serif;font-size:22px;color:var(--gold);letter-spacing:1px">{{ f.name }}</span>
  </div>

  <!-- Categories -->
  <div class="section">
    <div class="section-header">
      <span class="section-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20.6 13.4 12 22l-8.6-8.6A5 5 0 0 1 3 10V4a1 1 0 0 1 1-1h6a5 5 0 0 1 3.4 1.4l7.2 7.2a1.5 1.5 0 0 1 0 2.8Z"/><circle cx="7.5" cy="7.5" r="1.3"/></svg></span>
      <span class="section-title">Categories</span>
    </div>
    <div class="chips" id="cats-{{ fk }}"><span class="empty-chips">Loading…</span></div>
    <div class="add-row">
      <input type="text" id="cat-input-{{ fk }}" placeholder="Add a category…"
             onkeydown="if(event.key==='Enter'){event.preventDefault();catAdd('{{ fk }}');}">
      <button class="add-btn" onclick="catAdd('{{ fk }}')">Add Category</button>
    </div>
  </div>

  <!-- Seasons -->
  <div class="section">
    <div class="section-header">
      <span class="section-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="4" width="18" height="18" rx="2"/><path d="M3 9h18M8 2v4M16 2v4"/></svg></span>
      <span class="section-title">Seasons</span>
    </div>
    <div class="chips" id="seasons-{{ fk }}"><span class="empty-chips">Loading…</span></div>
    <div class="add-row">
      <input type="text" id="season-input-{{ fk }}" placeholder="e.g. 2025, Spring 2025, Season 3…"
             onkeydown="if(event.key==='Enter'){event.preventDefault();seasonAdd('{{ fk }}');}">
      <button class="add-btn" onclick="seasonAdd('{{ fk }}')">Add Season</button>
    </div>
  </div>
  {% endfor %}
</div>

<div class="toast" id="toast"></div>

<!-- Per-category judging prompt modal -->
<div id="promptModal" style="display:none;position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:100;align-items:center;justify-content:center;padding:20px">
  <div style="background:var(--bg2);border:1px solid var(--border);border-radius:12px;max-width:640px;width:100%;padding:24px">
    <div id="promptModalTitle" style="font-size:15px;font-weight:600;color:var(--gold);margin-bottom:6px">Judging emphasis</div>
    <div style="font-size:12px;color:var(--muted);margin-bottom:12px">Extra judging &amp; writing guidance applied only to films submitted in this category. Leave blank to use the festival default.</div>
    <textarea id="promptModalText" rows="10" style="width:100%;background:var(--bg3);border:1px solid var(--border);border-radius:8px;color:var(--text);font-family:'DM Mono',monospace;font-size:12px;padding:10px 12px;resize:vertical;line-height:1.6"></textarea>
    <div style="display:flex;gap:10px;justify-content:flex-end;margin-top:14px">
      <button onclick="closePromptModal()" style="background:var(--bg3);border:1px solid var(--border);border-radius:8px;color:var(--muted);padding:8px 16px;font-size:13px;cursor:pointer">Cancel</button>
      <button id="promptModalSave" style="background:var(--gold);border:none;border-radius:8px;color:#000;padding:8px 18px;font-size:13px;font-weight:600;cursor:pointer">Save</button>
    </div>
  </div>
</div>

<script>
function showToast(msg, err=false) {
  const t = document.getElementById('toast');
  t.textContent = msg;
  t.className = 'toast show' + (err ? ' err' : '');
  clearTimeout(t._tid);
  t._tid = setTimeout(() => t.className = 'toast', 2800);
}

function renderChips(containerId, items, onRename, onDelete, onPrompt) {
  const el = document.getElementById(containerId);
  if (!el) return;
  el.innerHTML = '';
  if (!items.length) {
    const em = document.createElement('span');
    em.className = 'empty-chips';
    em.textContent = 'None yet — add one below';
    el.appendChild(em);
    return;
  }
  items.forEach(item => {
    const name = typeof item === 'string' ? item : item.name;
    const chip = document.createElement('span');
    chip.className = 'chip';
    const lbl = document.createElement('span');
    lbl.className = 'chip-lbl';
    lbl.textContent = name;
    lbl.title = 'Click to rename';
    lbl.onclick = () => onRename(name, chip);
    chip.appendChild(lbl);
    if (onPrompt) {
      const pen = document.createElement('span');
      pen.className = 'chip-del';
      pen.textContent = '✎';
      pen.title = 'Edit judging prompt for this category';
      pen.onclick = () => onPrompt(name);
      chip.appendChild(pen);
    }
    const del = document.createElement('span');
    del.className = 'chip-del';
    del.textContent = '×';
    del.title = 'Delete';
    del.onclick = () => onDelete(name);
    chip.appendChild(del);
    el.appendChild(chip);
  });
}

// ── Categories ────────────────────────────────────────────
function catsLoad(fk) {
  fetch('/api/festivals/' + fk + '/categories')
    .then(r => r.json())
    .then(d => renderChips('cats-' + fk, d.categories || [],
      (n,c) => catRename(fk,n,c), n => catDelete(fk,n), n => catEditPrompt(fk,n)));
}
function catEditPrompt(fk, name) {
  const overlay = document.getElementById('promptModal');
  const ta      = document.getElementById('promptModalText');
  document.getElementById('promptModalTitle').textContent = 'Judging emphasis — ' + name;
  ta.value = 'Loading…'; ta.disabled = true;
  overlay.style.display = 'flex';
  fetch('/api/festivals/' + fk + '/categories/' + encodeURIComponent(name) + '/prompt')
    .then(r => r.json()).then(d => { ta.value = d.prompt || ''; ta.disabled = false; ta.focus(); });
  document.getElementById('promptModalSave').onclick = () => {
    fetch('/api/festivals/' + fk + '/categories/' + encodeURIComponent(name) + '/prompt', {
      method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify({prompt: ta.value})
    }).then(r => r.json()).then(d => {
      if (d.error) { showToast(d.error, true); return; }
      overlay.style.display = 'none'; showToast('Category prompt saved');
    });
  };
}
function closePromptModal() { document.getElementById('promptModal').style.display = 'none'; }

function catAdd(fk) {
  const inp = document.getElementById('cat-input-' + fk);
  const name = inp.value.trim();
  if (!name) return;
  fetch('/api/festivals/' + fk + '/categories', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name})
  }).then(r=>r.json()).then(d => {
    if (d.error) { showToast(d.error, true); return; }
    inp.value = '';
    renderChips('cats-' + fk, d.categories || [], (n,c)=>catRename(fk,n,c), n=>catDelete(fk,n), n=>catEditPrompt(fk,n));
    showToast('Category added');
  });
}
function catDelete(fk, name) {
  if (!confirm('Delete category "' + name + '"?')) return;
  fetch('/api/festivals/' + fk + '/categories/' + encodeURIComponent(name), {method:'DELETE'})
    .then(r=>r.json()).then(d => {
      renderChips('cats-' + fk, d.categories || [], (n,c)=>catRename(fk,n,c), n=>catDelete(fk,n), n=>catEditPrompt(fk,n));
      showToast('Deleted');
    });
}
function catRename(fk, oldName, chip) {
  const inp = document.createElement('input');
  inp.className = 'inline-edit';
  inp.value = oldName;
  chip.replaceChildren(inp);
  inp.focus(); inp.select();
  const commit = () => {
    const newName = inp.value.trim();
    if (!newName || newName === oldName) { catsLoad(fk); return; }
    fetch('/api/festivals/' + fk + '/categories/' + encodeURIComponent(oldName), {
      method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:newName})
    }).then(r=>r.json()).then(d => {
      if (d.error) { showToast(d.error, true); catsLoad(fk); return; }
      renderChips('cats-' + fk, d.categories || [], (n,c)=>catRename(fk,n,c), n=>catDelete(fk,n), n=>catEditPrompt(fk,n));
      showToast('Renamed');
    });
  };
  inp.addEventListener('blur', commit);
  inp.addEventListener('keydown', e => {
    if (e.key==='Enter') { e.preventDefault(); commit(); }
    if (e.key==='Escape') catsLoad(fk);
  });
}

// ── Seasons ───────────────────────────────────────────────
function seasonsLoad(fk) {
  fetch('/api/festivals/' + fk + '/seasons')
    .then(r => r.json())
    .then(d => renderChips('seasons-' + fk, d.seasons || [],
      (n,c) => seasonRename(fk,n,c), n => seasonDelete(fk,n)));
}
function seasonAdd(fk) {
  const inp = document.getElementById('season-input-' + fk);
  const name = inp.value.trim();
  if (!name) return;
  fetch('/api/festivals/' + fk + '/seasons', {
    method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name})
  }).then(r=>r.json()).then(d => {
    if (d.error) { showToast(d.error, true); return; }
    inp.value = '';
    renderChips('seasons-' + fk, d.seasons || [], (n,c)=>seasonRename(fk,n,c), n=>seasonDelete(fk,n));
    showToast('Season added');
  });
}
function seasonDelete(fk, name) {
  if (!confirm('Delete season "' + name + '"?')) return;
  fetch('/api/festivals/' + fk + '/seasons/' + encodeURIComponent(name), {method:'DELETE'})
    .then(r=>r.json()).then(d => {
      renderChips('seasons-' + fk, d.seasons || [], (n,c)=>seasonRename(fk,n,c), n=>seasonDelete(fk,n));
      showToast('Deleted');
    });
}
function seasonRename(fk, oldName, chip) {
  const inp = document.createElement('input');
  inp.className = 'inline-edit';
  inp.style.width = '160px';
  inp.value = oldName;
  chip.replaceChildren(inp);
  inp.focus(); inp.select();
  const commit = () => {
    const newName = inp.value.trim();
    if (!newName || newName === oldName) { seasonsLoad(fk); return; }
    fetch('/api/festivals/' + fk + '/seasons/' + encodeURIComponent(oldName), {
      method:'PUT', headers:{'Content-Type':'application/json'}, body:JSON.stringify({name:newName})
    }).then(r=>r.json()).then(d => {
      if (d.error) { showToast(d.error, true); seasonsLoad(fk); return; }
      renderChips('seasons-' + fk, d.seasons || [], (n,c)=>seasonRename(fk,n,c), n=>seasonDelete(fk,n));
      showToast('Renamed');
    });
  };
  inp.addEventListener('blur', commit);
  inp.addEventListener('keydown', e => {
    if (e.key==='Enter') { e.preventDefault(); commit(); }
    if (e.key==='Escape') seasonsLoad(fk);
  });
}

// ── Init ──────────────────────────────────────────────────
{% for fk, f in festivals.items() %}
catsLoad('{{ fk }}');
seasonsLoad('{{ fk }}');
{% endfor %}
</script>
"""


REVIEW_DETAIL_BODY = """<div class="main">
  <a href="/reviews" class="back">← Back to reviews</a>

  <div class="film-header">
    <div class="film-title">{{ film.title }}</div>
    <div class="film-director">Directed by {{ film.director }}</div>
    <div class="film-meta">
      {% if film.genre %}<span class="meta-tag">{{ film.genre }}</span>{% endif %}
      {% if film.runtime %}<span class="meta-tag">{{ film.runtime }} min</span>{% endif %}
      {% if film.country %}<span class="meta-tag">{{ film.country }}</span>{% endif %}
    </div>
  </div>

  {% set a = film.analysis %}
  {% if a %}
  <div class="review-block">
    <div class="section-divider" style="margin-top:0">Detailed Assessment</div>
    {% if a.ratings %}
    {% set r = a.ratings %}
    <div class="scores-row">
      <div class="score-box"><span class="lbl">Originality</span><span class="num">{{ r.originality if r.originality is not none else '—' }}</span><span class="den">/10</span></div>
      <div class="score-box"><span class="lbl">Direction</span><span class="num">{{ r.direction if r.direction is not none else '—' }}</span><span class="den">/10</span></div>
      <div class="score-box"><span class="lbl">Writing</span><span class="num">{{ r.writing if r.writing is not none else '—' }}</span><span class="den">/10</span></div>
      <div class="score-box"><span class="lbl">Cinematography</span><span class="num">{{ r.cinematography if r.cinematography is not none else '—' }}</span><span class="den">/10</span></div>
      <div class="score-box"><span class="lbl">Performances</span><span class="num">{{ r.performances if r.performances is not none else '—' }}</span><span class="den">/10</span></div>
      <div class="score-box"><span class="lbl">Production</span><span class="num">{{ r.production_value if r.production_value is not none else '—' }}</span><span class="den">/10</span></div>
      <div class="score-box"><span class="lbl">Pacing</span><span class="num">{{ r.pacing if r.pacing is not none else '—' }}</span><span class="den">/10</span></div>
      <div class="score-box"><span class="lbl">Structure</span><span class="num">{{ r.structure if r.structure is not none else '—' }}</span><span class="den">/10</span></div>
      <div class="score-box"><span class="lbl">Sound / Music</span><span class="num">{{ r.sound_music if r.sound_music is not none else '—' }}</span><span class="den">/10</span></div>
      <div class="score-box overall"><span class="lbl">Overall</span><span class="num">{{ '%.1f'|format(a.get('overall_rating')|float) if a.get('overall_rating') is not none else '—' }}</span><span class="den">/10</span></div>
    </div>
    {% else %}
    <div class="scores-row">
      <div class="score-box"><span class="lbl">Story</span><span class="num">{{ a.story.score if a.story else '—' }}</span><span class="den">/5</span></div>
      <div class="score-box"><span class="lbl">Direction</span><span class="num">{{ a.direction.score if a.direction else '—' }}</span><span class="den">/5</span></div>
      <div class="score-box"><span class="lbl">Technical</span><span class="num">{{ a.technical.score if a.technical else '—' }}</span><span class="den">/5</span></div>
      <div class="score-box"><span class="lbl">Originality</span><span class="num">{{ a.originality.score if a.originality else '—' }}</span><span class="den">/5</span></div>
      <div class="score-box overall"><span class="lbl">Overall</span><span class="num">{{ a.overall_score if a.overall_score is not none else '—' }}</span><span class="den">/20</span></div>
    </div>
    {% endif %}
    <div class="obs-grid">
      {% if a.standout_moment %}<div class="obs-item standout"><span class="obs-lbl">Standout Moment</span><div class="obs-body">{{ a.standout_moment }}</div></div>{% endif %}
      {% if a.weakest_element %}<div class="obs-item weakest"><span class="obs-lbl">Growth Area</span><div class="obs-body">{{ a.weakest_element }}</div></div>{% endif %}
    </div>
    {% if a.festival_suitability %}<div class="obs-item" style="margin-top:12px"><span class="obs-lbl">Festival Suitability</span><div class="obs-body">{{ a.festival_suitability }}</div></div>{% endif %}
  </div>
  {% endif %}

  {% if reviews %}
    {% for r in reviews %}
    <div class="review-block">
      <div class="review-block-header">
        <span class="festival-badge">{{ r.festival_name }}</span>
        <div style="display:flex;align-items:center;gap:16px">
          {% if r.get('overall_rating') is not none and r.get('overall_rating') != '' %}
          <span style="font-family:'DM Mono',monospace;font-size:22px;font-weight:600;color:var(--gold)">{{ '%.1f'|format(r.get('overall_rating')|float) }}<span style="font-size:12px;color:var(--muted)">/10</span></span>
          {% endif %}
          <span class="review-date">{{ r.created_at[:10] if r.created_at else '' }}</span>
        </div>
      </div>
      <div class="section-divider" style="margin-top:0">Expert Review</div>
      <div class="review-text" id="reviewText{{ loop.index }}">{{ r.review_text }}</div>
      <button class="copy-btn" onclick="copyReview({{ loop.index }})">Copy review</button>
      <div class="publish-bar">
        <span class="pb-label">Publish to WordPress</span>
        <span class="wp-status" id="wpstat-{{ loop.index }}" {% if not r.wp_url %}style="display:none"{% endif %}>
          ● Published — <a href="{{ r.wp_url }}" target="_blank" rel="noopener">view post</a>
        </span>
        <button class="btn btn-ghost btn-sm" onclick="wpPublish('{{ film.film_id }}','draft',{{ loop.index }},this)">Save as Draft</button>
        <button class="btn btn-gold btn-sm" onclick="wpPublish('{{ film.film_id }}','publish',{{ loop.index }},this)">Publish live</button>
      </div>
    </div>
    {% endfor %}
  {% else %}
  <div class="empty">No review found for this film.</div>
  {% endif %}
</div>
<div class="toast" id="toast"></div>
<script>
function copyReview(idx) {
  const text = document.getElementById('reviewText' + idx).textContent;
  navigator.clipboard.writeText(text).then(() => {
    const btn = event.target;
    btn.textContent = 'Copied!';
    setTimeout(() => btn.textContent = 'Copy review', 2000);
  });
}
function _toast(msg, err){const t=document.getElementById('toast');t.textContent=msg;t.className='toast show'+(err?' err':'');clearTimeout(t._t);t._t=setTimeout(()=>t.className='toast',3200);}
function wpPublish(filmId, status, idx, btn){
  const old = btn.textContent; btn.textContent = 'Publishing…'; btn.disabled = true;
  fetch('/api/reviews/'+filmId+'/publish',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({status:status})})
    .then(r=>r.json().then(d=>({ok:r.ok,d})))
    .then(({ok,d})=>{
      btn.textContent = old; btn.disabled = false;
      if(!ok){ _toast(d.error||'Publish failed', true); return; }
      _toast(status==='publish'?'Published live to WordPress':'Saved as draft in WordPress');
      const st = document.getElementById('wpstat-'+idx);
      if(st && d.url){ st.style.display=''; st.querySelector('a').href=d.url; }
    });
}
</script>
"""


APP_BODY = """<div class="main">
  <div class="page-sub" style="margin-bottom:22px">Upload a film and generate an AI-assisted Expert Review for the selected festival</div>

  <!-- ── FORM ── -->
  <div id="formSection">

    <!-- Festival picker -->
    <div class="card">
      <div class="card-head">
        <div class="card-head-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M4 22V4l14 4-14 4"/><path d="M4 4V2"/></svg></div>
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
        <div class="card-head-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M20.4 14.5 16 10 4 20"/><rect x="2" y="3" width="20" height="18" rx="2"/><path d="m3.5 8 5-5M8.5 8l5-5M13.5 8l5-5"/></svg></div>
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
            <label>Category *</label>
            <select id="genre" required onchange="updateUploadMode()">
              <option value="" disabled selected>— select a category —</option>
            </select>
          </div>
          <div class="form-group">
            <label>Season <span class="optional-tag">optional</span></label>
            <select id="season">
              <option value="" selected>— select a season —</option>
            </select>
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
        <div class="card-head-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M7 4v16M17 4v16M2 9h5M2 15h5M17 9h5M17 15h5"/></svg></div>
        <div class="card-head-title">Video Source</div>
      </div>
      <div class="card-body">
        <!-- File upload only -->
        <div class="drop-zone" id="dropZone">
          <div class="drop-icon" id="dropIcon">🎞</div>
          <div class="drop-title" id="dropTitle">Drop video file here</div>
          <div class="drop-sub" id="dropSub">or click to browse — MP4, MOV, AVI, WebM, MKV</div>
          <div class="file-info" id="fileInfo"></div>
          <input type="file" id="fileInput" accept=".mp4,.mov,.avi,.webm,.mkv,.mpeg">
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
      <div class="card-head-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M13 2 3 14h8l-1 8 10-12h-8z"/></svg></div>
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
        <div class="step" id="step-downloading"><div class="step-dot"></div>Fetching video</div>
        <div class="step" id="step-uploading"><div class="step-dot"></div>Uploading to Gemini</div>
        <div class="step" id="step-processing"><div class="step-dot"></div>Gemini watching the film</div>
        <div class="step" id="step-analysing"><div class="step-dot"></div>Scoring story, direction &amp; craft</div>
        <div class="step" id="step-scoring"><div class="step-dot"></div>Compiling scores</div>
        <div class="step" id="step-writing"><div class="step-dot"></div>Drafting Expert Review</div>
        <div class="step" id="step-finalising"><div class="step-dot"></div>Saving &amp; formatting</div>
      </div>
    </div>
  </div>

  <!-- ── RESULTS ── -->
  <div class="results-card" id="resultsCard">
    <div class="card">
      <div class="card-head">
        <div class="card-head-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M3 3v18h18"/><rect x="7" y="11" width="3" height="6"/><rect x="12" y="7" width="3" height="10"/><rect x="17" y="13" width="3" height="4"/></svg></div>
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
        <div class="card-head-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round"><path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4z"/></svg></div>
        <div class="card-head-title">Expert Review — Review before delivering</div>
      </div>
      <div class="card-body">
        <div class="review-block">
          <button class="copy-btn" id="copyBtn" onclick="copyReview()">Copy</button>
          <div class="review-text" id="reviewText"></div>
        </div>
        <div class="publish-bar" id="publishBar" style="display:none">
          <span class="pb-label">Publish to WordPress</span>
          <span class="wp-status" id="wpResultStatus" style="display:none">● Published — <a href="#" target="_blank" rel="noopener">view post</a></span>
          <button class="btn btn-ghost btn-sm" id="pubDraftBtn" onclick="publishResult('draft')">Save as Draft</button>
          <button class="btn btn-gold btn-sm" id="pubLiveBtn" onclick="publishResult('publish')">Publish live</button>
        </div>
        <button class="new-btn" onclick="resetForm()">← Generate another review</button>
      </div>
    </div>
  </div>

</div><!-- main -->

<script>
let selectedFile  = null;
let pollInterval  = null;
let activeJobId   = null;   // set while a job is in-flight; cleared on done/error

// ── Unload guard ───────────────────────────────────────────
// Warn the user if they try to refresh or close while a job is running,
// and fire a cancel beacon so the server marks the job as failed (prevents
// orphaned Gemini uploads consuming quota on re-submission).
window.addEventListener('beforeunload', e => {
  if (!activeJobId) return;
  // Ask the browser to show a warning dialog
  e.preventDefault();
  e.returnValue = 'Review generation is in progress. Leaving now will cancel the job and may waste AI tokens. Are you sure?';
  // Best-effort cancel — sendBeacon is the only API that fires reliably on page unload
  navigator.sendBeacon(`/api/jobs/${activeJobId}/cancel`);
});

const FESTIVALS = {
  {% for key, f in festivals.items() %}
  "{{ key }}": { name:"{{ f.name }}", focus:"{{ f.focus }}", words:{{ f.word_count }} },
  {% endfor %}
};

const CATEGORIES_MAP = {{ categories_map | tojson }};

function populateCategories(festivalKey) {
  const sel = document.getElementById('genre');
  const cats = CATEGORIES_MAP[festivalKey] || [];
  sel.innerHTML = '';
  const placeholder = document.createElement('option');
  placeholder.value = ''; placeholder.disabled = true; placeholder.selected = true;
  placeholder.textContent = cats.length ? '— select a category —' : '— no categories configured —';
  sel.appendChild(placeholder);
  cats.forEach(c => {
    const opt = document.createElement('option');
    opt.value = c; opt.textContent = c;
    sel.appendChild(opt);
  });
}

function populateSeasons(festivalKey) {
  const sel = document.getElementById('season');
  sel.innerHTML = '<option value="" disabled selected>Loading…</option>';
  fetch('/api/festivals/' + festivalKey + '/seasons')
    .then(r => r.json())
    .then(d => {
      sel.innerHTML = '';
      const seasons = (d.seasons || []).map(s => s.name);
      const ph = document.createElement('option');
      ph.value = ''; ph.disabled = true; ph.selected = true;
      ph.textContent = seasons.length ? '— select a season —' : '— no seasons configured —';
      sel.appendChild(ph);
      seasons.forEach(name => {
        const opt = document.createElement('option');
        opt.value = name; opt.textContent = name;
        sel.appendChild(opt);
      });
    })
    .catch(() => {
      sel.innerHTML = '<option value="" disabled selected>— could not load seasons —</option>';
    });
}

function onFestivalChange(sel) {
  if (!sel.value) return;
  const f = FESTIVALS[sel.value];
  document.getElementById('festivalHint').textContent =
    f ? `${f.words}-word review · ${f.focus}` : '';
  populateCategories(sel.value);
  populateSeasons(sel.value);
  document.getElementById('formCard').style.display   = 'block';
  document.getElementById('uploadCard').style.display = 'block';
}

// For non-admin users: festival is fixed, reveal form immediately
(function() {
  const userRole = "{{ user_role }}";
  if (userRole !== 'admin') {
    populateCategories("{{ user_festival }}");
    populateSeasons("{{ user_festival }}");
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

function _updateSubmitBtn() {
  document.getElementById('submitBtn').disabled = selectedFile === null;
}

// ── Script / written-work category detection & upload-mode toggle ──
const SCRIPT_RE = /\b(script|screenplay|stageplay|teleplay|poem|novel|radio script)\b/i;
function isScriptCategory() {
  return SCRIPT_RE.test(document.getElementById('genre').value || '');
}
function updateUploadMode() {
  const script = isScriptCategory();
  const fi = document.getElementById('fileInput');
  document.getElementById('dropIcon').textContent  = script ? '📄' : '🎞';
  document.getElementById('dropTitle').textContent = script ? 'Drop your script here' : 'Drop video file here';
  document.getElementById('dropSub').textContent   = script
    ? 'or click to browse — PDF, DOC, DOCX, TXT'
    : 'or click to browse — MP4, MOV, AVI, WebM, MKV';
  fi.setAttribute('accept', script ? '.pdf,.doc,.docx,.txt' : '.mp4,.mov,.avi,.webm,.mkv,.mpeg');
  // Reset any previously selected file when switching modes
  selectedFile = null;
  fi.value = '';
  document.getElementById('fileInfo').textContent = '';
  dropZone.classList.remove('has-file');
  document.getElementById('submitBtn').disabled = true;
}

// no separate listener needed — oninput="_updateSubmitBtn()" is inline on the input

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
  const sizeMb = (file.size/1024/1024).toFixed(1);

  // Script/document categories: accept the file as-is, no video probe
  if (isScriptCategory()) {
    const okExt = /\.(pdf|doc|docx|txt)$/i.test(file.name);
    if (!okExt) {
      showError('Please upload a PDF, DOC, DOCX, or TXT file for this category.');
      selectedFile = null; dropZone.classList.remove('has-file');
      document.getElementById('fileInfo').textContent = ''; document.getElementById('submitBtn').disabled = true;
      return;
    }
    if (file.size > 25 * 1024 * 1024) {
      showError(`Document is too large (${sizeMb} MB). Maximum is 25 MB.`);
      selectedFile = null; dropZone.classList.remove('has-file');
      document.getElementById('submitBtn').disabled = true; return;
    }
    selectedFile = file;
    dropZone.classList.add('has-file');
    showError('');
    document.getElementById('fileInfo').style.color = '';
    document.getElementById('fileInfo').textContent = `✓ ${file.name}  (${sizeMb} MB)`;
    document.getElementById('submitBtn').disabled = false;
    return;
  }

  const MAX_MB = 4000;
  if (file.size > MAX_MB * 1024 * 1024) {
    showError(`File is too large (${sizeMb} MB). Maximum allowed size is ${MAX_MB} MB. Please compress the video or export at a lower bitrate.`);
    selectedFile = null;
    dropZone.classList.remove('has-file');
    document.getElementById('fileInfo').textContent = '';
    document.getElementById('submitBtn').disabled = true;
    return;
  }
  selectedFile = file;
  dropZone.classList.add('has-file');
  showError('');

  // Probe duration using a temporary object URL
  const probe = document.createElement('video');
  probe.preload = 'metadata';
  const objUrl = URL.createObjectURL(file);
  probe.src = objUrl;
  probe.onloadedmetadata = () => {
    URL.revokeObjectURL(objUrl);
    const mins = probe.duration / 60;
    if (mins > 120) {
      showError(`This video is ${Math.round(mins)} minutes long. Only films up to 120 minutes are accepted.`);
      selectedFile = null;
      dropZone.classList.remove('has-file');
      document.getElementById('fileInfo').textContent = '';
      document.getElementById('submitBtn').disabled = true;
      return;
    }
    let info = `✓ ${file.name}  (${sizeMb} MB, ${Math.round(mins)} min)`;
    if (mins > 60) {
      info += '  ⚠ Heavy token usage — analysis may take several minutes';
      document.getElementById('fileInfo').style.color = '#e8c97a';
    }
    document.getElementById('fileInfo').textContent = info;
    document.getElementById('submitBtn').disabled = false;
  };
  probe.onerror = () => {
    URL.revokeObjectURL(objUrl);
    // Can't read duration — allow submission, server will validate
    document.getElementById('fileInfo').textContent = `✓ ${file.name}  (${sizeMb} MB)`;
    document.getElementById('submitBtn').disabled = false;
  };
}

// ── Submit: new video ──────────────────────────────────────
async function submitReview() {
  const festivalKey = document.getElementById('festivalKey').value;
  if (!festivalKey) { showError('Please select a festival before submitting'); return; }
  const title    = document.getElementById('title').value.trim();
  const director = document.getElementById('director').value.trim();
  const genre    = document.getElementById('genre').value.trim();
  if (!title || !director) { showError('Film title and director are required'); return; }
  if (!genre)              { showError('Genre / Category is required'); return; }

  if (!selectedFile) { showError(isScriptCategory() ? 'Please upload your script (PDF, DOC, DOCX, or TXT)' : 'Please upload a video file to continue'); return; }

  // ── Script/document categories: small file → POST directly to /upload ──
  if (isScriptCategory()) {
    showProcessing();
    setProgressMsg('Uploading your script…', 10);
    try {
      const form = new FormData();
      form.append('document',           selectedFile);
      form.append('festival_key',       document.getElementById('festivalKey').value);
      form.append('title',              title);
      form.append('director',           director);
      form.append('logline',            document.getElementById('logline').value);
      form.append('director_statement', document.getElementById('director_statement').value);
      form.append('genre',              genre);
      form.append('season',             document.getElementById('season').value);
      form.append('synopsis',           document.getElementById('synopsis').value);
      const res = await fetch('/upload', { method:'POST', body:form });
      if (!res.ok) {
        let m = `Server error (HTTP ${res.status})`;
        try { const d = await res.json(); m = d.error || m; } catch(_) {}
        showFormError(m); return;
      }
      const data = await res.json();
      if (data.error) { showFormError(data.error); return; }
      pollStatus(data.job_id);
    } catch(e) {
      console.error('[upload] script failed:', e);
      showFormError(`Upload failed: ${e.message || e}`);
    }
    return;
  }

  showProcessing();
  try {
    // ── Step 0: dedup pre-check — skip the upload entirely if this film
    // (title + director) was already analysed within this festival. ──
    setProgressMsg('Checking for an existing analysis of this film…', 4);
    let blob_name = '';
    let isDedup   = false;
    try {
      const ddRes = await fetch('/api/check-dedup', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ title, director, festival_key: document.getElementById('festivalKey').value }),
      });
      if (ddRes.ok) { const dd = await ddRes.json(); isDedup = !!dd.found; }
    } catch(_) { /* non-fatal — fall through to normal upload */ }

    if (!isDedup) {
      // ── Step 1: get a signed GCS URL and upload the file directly ──
      // This bypasses Cloud Run's 32 MB request limit entirely.
      setProgressMsg('Preparing upload…', 6);
      const urlRes = await fetch('/api/upload-url', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ filename: selectedFile.name }),
      });
      if (!urlRes.ok) {
        let m = `Could not start upload (HTTP ${urlRes.status})`;
        try { const d = await urlRes.json(); m = d.error || m; } catch(_) {}
        showFormError(m); return;
      }
      const j = await urlRes.json();
      blob_name = j.blob_name;
      await uploadToGCS(j.upload_url, selectedFile);
    } else {
      setProgressMsg('Existing analysis found — writing your review (no upload needed)…', 8);
    }

    // ── Step 2: tell the server to start processing ──
    const form = new FormData();
    if (blob_name) form.append('gcs_blob', blob_name);
    form.append('festival_key',       document.getElementById('festivalKey').value);
    form.append('title',              title);
    form.append('director',           director);
    form.append('logline',            document.getElementById('logline').value);
    form.append('director_statement', document.getElementById('director_statement').value);
    form.append('genre',              genre);
    form.append('season',             document.getElementById('season').value);
    form.append('synopsis',           document.getElementById('synopsis').value);

    const res = await fetch('/upload', { method:'POST', body:form });
    if (!res.ok) {
      let errMsg = `Server error (HTTP ${res.status})`;
      try { const d = await res.json(); errMsg = d.error || errMsg; } catch(_) {}
      showFormError(errMsg); return;
    }
    const data = await res.json();
    if (data.error) { showFormError(data.error); return; }
    pollStatus(data.job_id);
  } catch(e) {
    console.error('[upload] failed:', e);
    showFormError(`Upload failed: ${e.message || e}`);
  }
}

// Upload a file to a signed GCS URL via PUT, reporting progress on the bar.
function uploadToGCS(url, file) {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open('PUT', url, true);
    xhr.setRequestHeader('Content-Type', 'video/mp4');
    xhr.upload.onprogress = e => {
      if (e.lengthComputable) {
        const pct = Math.round((e.loaded / e.total) * 100);
        // Map upload 0–100% onto the 2–7% slice of the overall bar
        setProgressMsg(`Uploading video… ${pct}%`, 2 + Math.round(pct * 0.05));
      }
    };
    xhr.onload  = () => (xhr.status >= 200 && xhr.status < 300)
      ? resolve()
      : reject(new Error(`GCS upload failed (HTTP ${xhr.status})`));
    xhr.onerror = () => reject(new Error('Network error during upload'));
    xhr.send(file);
  });
}

function setProgressMsg(msg, pct) {
  const m = document.getElementById('progressMsg');
  const p = document.getElementById('progressPct');
  const f = document.getElementById('progressFill');
  if (m) m.textContent = msg;
  if (p) p.textContent = pct + '%';
  if (f) f.style.width = pct + '%';
}

// ── Polling ────────────────────────────────────────────────
function pollStatus(jobId) {
  clearInterval(pollInterval);
  activeJobId = jobId;   // arm the unload guard
  let elapsed = 0;
  const MAX_WAIT = 20 * 60 * 1000; // 20 min hard timeout

  const finish = () => { clearInterval(pollInterval); activeJobId = null; };

  pollInterval = setInterval(async () => {
    elapsed += 2000;
    if (elapsed >= MAX_WAIT) {
      finish();
      showFormError('Processing timed out. Please try again.');
      return;
    }
    try {
      const res = await fetch(`/status/${jobId}`);
      if (!res.ok) { finish(); showFormError('Server error. Please try again.'); return; }
      const job = await res.json();
      updateProgress(job);
      if (job.status === 'done')  { finish(); showResults(job); }
      if (job.status === 'error') { finish(); showFormError(job.message || 'Processing failed. Please check your video link and try again.'); }
    } catch(e) { /* network blip — keep polling */ }
  }, 2000);
}

// ── Progress UI ────────────────────────────────────────────
const ALL_STEPS = ['downloading','uploading','processing','analysing','scoring','writing','finalising'];

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

  // Score boxes — new 9-criteria /10 structure, with legacy fallback
  const scoresRow = document.getElementById('scoresRow');
  scoresRow.textContent = '';
  const RLABELS = [['originality','Originality'],['direction','Direction'],['writing','Writing'],
    ['cinematography','Cinematography'],['performances','Performances'],['production_value','Production'],
    ['pacing','Pacing'],['structure','Structure'],['sound_music','Sound/Music']];
  const ratings = (a.ratings && typeof a.ratings === 'object') ? a.ratings : null;
  if (ratings) {
    const nums = [];
    RLABELS.forEach(([k,l]) => {
      const v = ratings[k];
      if (typeof v === 'number') nums.push(v);
      scoresRow.append(_scoreBox(l, (typeof v === 'number' ? v : '—'), 10));
    });
    let overall = (typeof a.overall_rating === 'number') ? a.overall_rating
                 : (nums.length ? nums.reduce((x,y)=>x+y,0)/nums.length : null);
    scoresRow.append(_scoreBox('Overall', overall != null ? overall.toFixed(1) : '—', 10, 'overall'));
  } else {
    [{k:'story',l:'Story'},{k:'direction',l:'Direction'},{k:'technical',l:'Technical'},{k:'originality',l:'Originality'}]
      .forEach(c => scoresRow.append(_scoreBox(c.l, (a[c.k] || {}).score ?? '—', 5)));
    scoresRow.append(_scoreBox('Overall', a.overall_score ?? '—', 20, 'overall'));
  }

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

  // Publish-to-WordPress actions
  window._resultFilmId = job.film_id || '';
  const pubBar = document.getElementById('publishBar');
  if (pubBar) {
    pubBar.style.display = window._resultFilmId ? 'flex' : 'none';
    const st = document.getElementById('wpResultStatus');
    if (st) st.style.display = 'none';
  }
}

// ── Copy ───────────────────────────────────────────────────
function copyReview() {
  navigator.clipboard.writeText(document.getElementById('reviewText').textContent).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = '✓ Copied'; btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
  });
}

// ── Publish generated review to WordPress ──────────────────
function publishResult(status) {
  const filmId = window._resultFilmId;
  if (!filmId) return;
  const btn = document.getElementById(status === 'publish' ? 'pubLiveBtn' : 'pubDraftBtn');
  const old = btn.textContent; btn.textContent = 'Publishing…'; btn.disabled = true;
  fetch('/api/reviews/' + filmId + '/publish', {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({status: status})
  }).then(r => r.json().then(d => ({ok: r.ok, d})))
    .then(({ok, d}) => {
      btn.textContent = old; btn.disabled = false;
      const err = document.getElementById('errorMsg');
      if (!ok) { err.textContent = d.error || 'Publish failed'; err.classList.add('active'); return; }
      err.classList.remove('active');
      const st = document.getElementById('wpResultStatus');
      if (st && d.url) { st.style.display = ''; st.querySelector('a').href = d.url; }
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
  document.getElementById('submitBtn').disabled     = true;
  selectedFile = null;
  dropZone.classList.remove('has-file');
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
"""


if __name__ == "__main__":
    print(f"Festival Review App — Festival Reviewer")
    print("Local: http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
