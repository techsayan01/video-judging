"""
Festival Expert Review Generator
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

import os, json, uuid, threading, tempfile, time, re
from pathlib import Path
from datetime import datetime, timezone
from functools import wraps
from flask import Flask, request, jsonify, session, redirect, render_template_string
from google import genai
from google.genai import types
from dotenv import load_dotenv

from festivals import FESTIVALS, DEFAULT_FESTIVAL, get_festival
from prompts import build_analysis_prompt, build_review_prompt
from wordpress import publish_review as wp_publish_review, publish_post as wp_publish_post
import db

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", uuid.uuid4().hex)

# ── Config ────────────────────────────────────────────────
MAX_UPLOAD_MB  = 1800
ALLOWED_EXT    = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".mpeg"}

# Lite model used for text-only review writing (cheaper, same prose quality)
REVIEW_MODEL = "gemini-2.5-flash-lite"

USERS = {
    os.getenv("USER1_EMAIL", "employee1@elegantiff.com"): os.getenv("USER1_PASS", "change_me_1"),
    os.getenv("USER2_EMAIL", "employee2@elegantiff.com"): os.getenv("USER2_PASS", "change_me_2"),
}


# ── Auth ──────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper


# ── Helpers ───────────────────────────────────────────────
def _parse_json(raw: str) -> dict:
    clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            return json.loads(m.group())
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
    festival = get_festival(meta.get("festival_key", DEFAULT_FESTIVAL))
    client   = genai.Client(api_key=festival["gemini_api_key"])
    model_id = festival.get("gemini_model", "gemini-2.5-flash")
    uploaded = None

    try:
        # 1 — Upload
        db.job_update(job_id, {"status": "uploading", "progress": 15,
                                "message": f"Uploading to Gemini [{festival['name']}]..."})
        uploaded = client.files.upload(file=video_path)

        # 2 — Wait for processing
        db.job_update(job_id, {"status": "processing", "progress": 35,
                                "message": "Gemini is watching the film..."})
        while uploaded.state.name == "PROCESSING":
            time.sleep(6)
            uploaded = client.files.get(name=uploaded.name)

        if uploaded.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini processing failed: {uploaded.state.name}")

        # 3 — Analyse (video model, festival-specific prompt)
        db.job_update(job_id, {"status": "analysing", "progress": 55,
                                "message": "Analysing story, direction, technical..."})
        analysis_resp = client.models.generate_content(
            model=model_id,
            contents=[uploaded, build_analysis_prompt(festival)],
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024),
        )
        analysis = _parse_json(analysis_resp.text)

        # 4 — Save to DB (before writing review so analysis is never lost)
        film_id = meta.get("film_id") or uuid.uuid4().hex
        db.film_save_analysis(film_id, analysis)
        db.film_update_meta(film_id, meta)

        # 5 — Write review (lite model, text-only)
        db.job_update(job_id, {"status": "writing", "progress": 78,
                                "message": "Writing Expert Review (cached analysis)..."})
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
        try:
            if uploaded:
                client.files.delete(name=uploaded.name)
            Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass


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
def login():
    error = ""
    if request.method == "POST":
        email    = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if USERS.get(email) == password:
            session["logged_in"] = True
            session["user"] = email
            return redirect("/")
        error = "Invalid credentials"
    return render_template_string(LOGIN_HTML, error=error)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


@app.route("/")
@login_required
def index():
    return render_template_string(APP_HTML,
                                  festival=FESTIVALS[DEFAULT_FESTIVAL]["name"],
                                  festivals=FESTIVALS,
                                  default_festival=DEFAULT_FESTIVAL,
                                  user=session.get("user", ""))


@app.route("/api/films")
@login_required
def api_films():
    """Return all films in the DB (for the frontend picker)."""
    return jsonify(db.film_list())


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    """New film — full video analysis + review.

    Before touching the video, check if this film already exists in the DB
    (by title+director). If it does, return a 'duplicate' signal so the
    frontend can switch to the cheaper rewrite path instead.
    """
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    f   = request.files["video"]
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXT)}"}), 400

    title    = request.form.get("title", "").strip()
    director = request.form.get("director", "").strip()

    # Dedup check — if this film is already in the library, skip video upload
    existing = db.film_find_by_identity(title, director)
    if existing and existing.get("analysis"):
        return jsonify({
            "duplicate": True,
            "film_id":   existing["film_id"],
            "title":     existing["title"],
            "director":  existing["director"],
            "analysed_at": existing.get("analysed_at", ""),
            "message":   (
                f'"{existing["title"]}" by {existing["director"]} is already in the film library '
                f'(analysed {existing.get("analysed_at","")[:10]}). '
                "Use the Rewrite path to generate a festival-specific review without re-uploading."
            ),
        }), 409

    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    f.save(tmp.name)
    size_mb = Path(tmp.name).stat().st_size / (1024 * 1024)
    if size_mb > MAX_UPLOAD_MB:
        Path(tmp.name).unlink()
        return jsonify({"error": f"File too large ({size_mb:.0f}MB). Max {MAX_UPLOAD_MB}MB"}), 400

    festival_key = request.form.get("festival_key", DEFAULT_FESTIVAL)
    if festival_key not in FESTIVALS:
        festival_key = DEFAULT_FESTIVAL
    festival = FESTIVALS[festival_key]

    if not festival.get("gemini_api_key"):
        Path(tmp.name).unlink()
        return jsonify({"error": f"No Gemini API key configured for {festival['name']}"}), 400

    # Create DB row now so analysis is persisted even if the job crashes mid-way
    film_id = uuid.uuid4().hex
    db.film_create({
        "film_id":            film_id,
        "title":              title,
        "director":           director,
        "logline":            request.form.get("logline", "").strip(),
        "director_statement": request.form.get("director_statement", "").strip(),
        "genre":              request.form.get("genre", "").strip(),
        "runtime":            request.form.get("runtime", "").strip(),
        "screener_url":       request.form.get("screener_url", "").strip(),
    })

    meta = {
        "film_id":            film_id,
        "title":              title,
        "director":           director,
        "logline":            request.form.get("logline", "").strip(),
        "director_statement": request.form.get("director_statement", "").strip(),
        "genre":              request.form.get("genre", "").strip(),
        "runtime":            request.form.get("runtime", "").strip(),
        "festival_key":       festival_key,
        "festival_name":      festival["name"],
    }

    job_id = uuid.uuid4().hex
    db.job_create(job_id, {
        "status": "queued", "progress": 5,
        "message": "Queued for processing...",
        "meta": meta, "analysis": None, "review": None,
    })
    threading.Thread(target=process_video,
                     args=(job_id, tmp.name, meta), daemon=True).start()
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

    festival_key = request.form.get("festival_key", DEFAULT_FESTIVAL)
    if festival_key not in FESTIVALS:
        festival_key = DEFAULT_FESTIVAL
    festival = FESTIVALS[festival_key]

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
        "runtime":            request.form.get("runtime", film.get("runtime", "")).strip(),
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


# ── HTML ──────────────────────────────────────────────────
LOGIN_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Film Review Portal — Sign In</title>
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
  <div class="logo">Review Portal</div>
  <div class="sub">Film Judging Platform</div>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <label>Email</label>
    <input type="email" name="email" placeholder="your@email.com" required autofocus>
    <label>Password</label>
    <input type="password" name="password" placeholder="••••••••" required>
    <button type="submit">Sign In</button>
  </form>
</div></body></html>"""


APP_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ festival }} — Expert Review Generator</title>
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
  <div class="header-logo">Review Portal</div>
  <div class="header-right">
    <span class="user-badge">{{ user }}</span>
    <a href="/logout" class="logout">Sign out</a>
  </div>
</div>

<div class="main">
  <div class="page-title">Expert Review</div>
  <div class="page-sub">Generate an AI-assisted Expert Review — or rewrite instantly for a different festival using cached analysis</div>

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
          <label>Select Festival *</label>
          <select id="festivalKey" onchange="onFestivalChange(this)">
            <option value="" disabled selected>— Select a festival —</option>
            {% for key, f in festivals.items() %}
            <option value="{{ key }}">{{ f.name }} — {{ f.focus }}</option>
            {% endfor %}
          </select>
          <div class="festival-hint" id="festivalHint"></div>
        </div>
      </div>
    </div>

    <!-- Film library picker -->
    <div class="card" id="libraryCard" style="display:none">
      <div class="card-head">
        <div class="card-head-icon">🗂</div>
        <div class="card-head-title">Film Library — Reuse Cached Analysis</div>
      </div>
      <div class="card-body">
        <div class="form-group">
          <label>Existing film <span class="optional-tag">optional — skips video re-upload</span></label>
          <select id="libraryFilm" onchange="onLibraryChange(this)">
            <option value="">— New film (upload video below) —</option>
          </select>
          <div class="library-hint" id="libraryHint">Select an existing film to reuse its analysis. Only the review text will be regenerated for the chosen festival (~0.01¢ vs ~1.6¢).</div>
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
            <label>Genre</label>
            <select id="genre">
              <option value="">Select genre</option>
              <option>Drama</option><option>Documentary</option><option>Comedy</option>
              <option>Thriller</option><option>Horror</option><option>Animation</option>
              <option>Experimental</option><option>Short Film</option><option>Feature</option>
              <option>Music Video</option><option>Other</option>
            </select>
          </div>
          <div class="form-group">
            <label>Runtime (minutes)</label>
            <input type="number" id="runtime" placeholder="13" min="1">
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

    <!-- Upload (hidden when library film selected) -->
    <div class="card" id="uploadCard" style="display:none">
      <div class="card-head">
        <div class="card-head-icon">📁</div>
        <div class="card-head-title">Upload Film</div>
      </div>
      <div class="card-body">
        <div class="drop-zone" id="dropZone">
          <div class="drop-icon">🎞</div>
          <div class="drop-title">Drop video file here</div>
          <div class="drop-sub">or click to browse — MP4, MOV, AVI, WebM, MKV</div>
          <div class="file-info" id="fileInfo"></div>
          <input type="file" id="fileInput" accept=".mp4,.mov,.avi,.webm,.mkv,.mpeg">
        </div>
        <div class="error-msg" id="errorMsg"></div>
        <button class="submit-btn" id="submitBtn" onclick="submitReview()" disabled>
          Generate Expert Review
        </button>
      </div>
    </div>

    <!-- Rewrite panel (shown when library film selected) -->
    <div class="card" id="rewriteCard" style="display:none">
      <div class="card-head">
        <div class="card-head-icon">⚡</div>
        <div class="card-head-title">Rewrite Review — No Video Upload Needed</div>
      </div>
      <div class="card-body">
        <div class="error-msg" id="rewriteErrorMsg"></div>
        <button class="rewrite-btn" id="rewriteBtn" onclick="submitRewrite()">
          ⚡ Rewrite for Selected Festival (~0.01¢)
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
let selectedFilmId = null;

const FESTIVALS = {
  {% for key, f in festivals.items() %}
  "{{ key }}": { name:"{{ f.name }}", focus:"{{ f.focus }}", words:{{ f.word_count }} },
  {% endfor %}
};

// ── On load ────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  await loadLibrary();
  // Do NOT call onFestivalChange on load — nothing is selected yet
});

function onFestivalChange(sel) {
  if (!sel.value) return;
  const f = FESTIVALS[sel.value];
  document.getElementById('festivalHint').textContent =
    f ? `${f.words}-word review · ${f.focus}` : '';

  // Reveal the rest of the form now that a festival is chosen
  document.getElementById('libraryCard').style.display = 'block';
  document.getElementById('formCard').style.display    = 'block';
  // uploadCard visibility is controlled by library selection; show by default
  if (!selectedFilmId) {
    document.getElementById('uploadCard').style.display  = 'block';
    document.getElementById('rewriteCard').style.display = 'none';
  }
}

// ── Film library ───────────────────────────────────────────
async function loadLibrary() {
  try {
    const res   = await fetch('/api/films');
    const films = await res.json();
    const sel   = document.getElementById('libraryFilm');
    films.forEach(f => {
      const opt   = document.createElement('option');
      opt.value   = f.film_id;
      const date  = f.analysed_at ? new Date(f.analysed_at).toLocaleDateString() : '';
      opt.text    = `${f.title} — ${f.director}${date ? '  (' + date + ')' : ''}`;
      opt.dataset.film = JSON.stringify(f);
      sel.appendChild(opt);
    });
  } catch(e) { console.warn('Could not load library', e); }
}

function onLibraryChange(sel) {
  if (!sel.value) {
    selectedFilmId = null;
    document.getElementById('uploadCard').style.display   = 'block';
    document.getElementById('rewriteCard').style.display  = 'none';
    document.getElementById('submitBtn').disabled = !selectedFile;
    document.getElementById('libraryHint').textContent =
      'Select an existing film to reuse its analysis. Only the review text will be regenerated (~0.01¢ vs ~1.6¢).';
    clearFilmFields();
    return;
  }

  const film     = JSON.parse(sel.options[sel.selectedIndex].dataset.film);
  selectedFilmId = film.film_id;

  // Pre-fill form fields from stored data
  document.getElementById('title').value              = film.title || '';
  document.getElementById('director').value           = film.director || '';
  document.getElementById('logline').value            = film.logline || '';
  document.getElementById('director_statement').value = film.director_statement || '';
  document.getElementById('genre').value              = film.genre || '';
  document.getElementById('runtime').value            = film.runtime || '';

  document.getElementById('uploadCard').style.display  = 'none';
  document.getElementById('rewriteCard').style.display = 'block';

  const a    = film.analysis;
  const date = film.analysed_at ? new Date(film.analysed_at).toLocaleDateString() : '';
  document.getElementById('libraryHint').innerHTML =
    `<span class="cache-badge">✓ Cached analysis — ${date} — Overall ${a.overall_score}/20</span>
     &nbsp; Rewriting only costs ~0.01¢ (no video upload)`;
}

function clearFilmFields() {
  ['title','director','logline','director_statement','runtime'].forEach(id =>
    document.getElementById(id).value = '');
  document.getElementById('genre').value    = '';
  document.getElementById('synopsis').value = '';
}

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
  if (!title || !director) { showError('Film title and director are required'); return; }
  if (!selectedFile)       { showError('Please upload a video file'); return; }

  const form = new FormData();
  form.append('video',              selectedFile);
  form.append('festival_key',       document.getElementById('festivalKey').value);
  form.append('title',              title);
  form.append('director',           director);
  form.append('logline',            document.getElementById('logline').value);
  form.append('director_statement', document.getElementById('director_statement').value);
  form.append('genre',              document.getElementById('genre').value);
  form.append('runtime',            document.getElementById('runtime').value);
  form.append('synopsis',           document.getElementById('synopsis').value);

  showProcessing(false);
  try {
    const res  = await fetch('/upload', { method:'POST', body:form });
    const data = await res.json();
    if (res.status === 409 && data.duplicate) {
      // Film already in library — offer to switch to rewrite mode
      showFormError('');
      document.getElementById('formSection').style.display = 'block';
      document.getElementById('progressCard').classList.remove('active');
      const sel = document.getElementById('libraryFilm');
      // Find and select the matching option, or reload library then select
      let found = false;
      for (const opt of sel.options) {
        if (opt.value === data.film_id) { sel.value = data.film_id; found = true; break; }
      }
      if (!found) { await reloadLibrary(); sel.value = data.film_id; }
      onLibraryChange(sel);
      showError(`"${data.title}" is already in the film library (analysed ${data.analysed_at ? data.analysed_at.slice(0,10) : ''}).\nSwitched to Rewrite mode — no video re-upload needed.`);
      return;
    }
    if (data.error) { showFormError(data.error); return; }
    pollStatus(data.job_id);
  } catch(e) { showFormError('Upload failed. Please try again.'); }
}

// ── Submit: rewrite from cache ─────────────────────────────
async function submitRewrite() {
  const festivalKey = document.getElementById('festivalKey').value;
  if (!festivalKey) { showRewriteError('Please select a festival before submitting'); return; }
  const title    = document.getElementById('title').value.trim();
  const director = document.getElementById('director').value.trim();
  if (!title || !director) { showRewriteError('Film title and director are required'); return; }

  const form = new FormData();
  form.append('film_id',            selectedFilmId);
  form.append('festival_key',       document.getElementById('festivalKey').value);
  form.append('title',              title);
  form.append('director',           director);
  form.append('logline',            document.getElementById('logline').value);
  form.append('director_statement', document.getElementById('director_statement').value);
  form.append('genre',              document.getElementById('genre').value);
  form.append('runtime',            document.getElementById('runtime').value);

  showProcessing(true);
  try {
    const res  = await fetch('/rewrite', { method:'POST', body:form });
    const data = await res.json();
    if (data.error) { showFormError(data.error); return; }
    pollStatus(data.job_id);
  } catch(e) { showFormError('Request failed. Please try again.'); }
}

// ── Polling ────────────────────────────────────────────────
function pollStatus(jobId) {
  clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/status/${jobId}`);
      const job = await res.json();
      updateProgress(job);
      if (job.status === 'done')  { clearInterval(pollInterval); showResults(job); }
      if (job.status === 'error') { clearInterval(pollInterval); showFormError(job.message); }
    } catch(e) {}
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

// ── Results ────────────────────────────────────────────────
function showResults(job) {
  document.getElementById('progressCard').classList.remove('active');
  document.getElementById('resultsCard').classList.add('active');

  const a    = job.analysis;
  const meta = job.meta;

  let tag = `🎬 <span>${meta.title}</span> &nbsp;·&nbsp; <span>${meta.director}</span>`;
  if (meta.genre)        tag += ` &nbsp;·&nbsp; <span>${meta.genre}</span>`;
  if (meta.runtime)      tag += ` &nbsp;·&nbsp; <span>${meta.runtime} min</span>`;
  if (meta.festival_name) tag += ` &nbsp;·&nbsp; <span style="color:var(--gold)">${meta.festival_name}</span>`;
  if (job.from_cache)    tag += ` &nbsp;·&nbsp; <span style="color:var(--green);font-size:10px">⚡ cached</span>`;
  document.getElementById('filmTag').innerHTML = tag;

  const cats = [{k:'story',l:'Story'},{k:'direction',l:'Direction'},{k:'technical',l:'Technical'},{k:'originality',l:'Originality'}];
  let scoresHtml = cats.map(c =>
    `<div class="score-box"><span class="score-label">${c.l}</span>
     <div><span class="score-num">${a[c.k].score}</span><span class="score-denom">/5</span></div></div>`
  ).join('');
  scoresHtml += `<div class="score-box overall"><span class="score-label">Overall</span>
    <div><span class="score-num">${a.overall_score}</span><span class="score-denom">/20</span></div></div>`;
  document.getElementById('scoresRow').innerHTML = scoresHtml;

  document.getElementById('obsGrid').innerHTML = `
    <div class="obs-item standout"><span class="obs-label">Standout Moment</span>
      <div class="obs-text">${a.standout_moment}</div></div>
    <div class="obs-item weakest"><span class="obs-label">Growth Area</span>
      <div class="obs-text">${a.weakest_element}</div></div>
    <div class="obs-item" style="grid-column:1/-1"><span class="obs-label">Festival Suitability</span>
      <div class="obs-text">${a.festival_suitability}</div></div>`;

  document.getElementById('reviewText').textContent = job.review;

  // Reload library in case this was a new film
  if (!job.from_cache) reloadLibrary();
}

async function reloadLibrary() {
  const sel  = document.getElementById('libraryFilm');
  const prev = sel.value;
  while (sel.options.length > 1) sel.remove(1);
  await loadLibrary();
  sel.value = prev;
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
function showProcessing(isCacheMode) {
  document.getElementById('formSection').style.display = 'none';
  document.getElementById('progressCard').classList.add('active');
  document.getElementById('progressFill').style.width = '5%';

  // For cache rewrites, grey out the video steps
  const uploadSteps = ['uploading','processing','analysing'];
  uploadSteps.forEach(s => {
    const el = document.getElementById('step-' + s);
    if (el) el.style.opacity = isCacheMode ? '0.25' : '1';
  });
}

function showError(msg) {
  const el = document.getElementById('errorMsg');
  el.textContent = msg; el.className = 'error-msg' + (msg ? ' active' : '');
}

function showRewriteError(msg) {
  const el = document.getElementById('rewriteErrorMsg');
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
  document.getElementById('synopsis').value       = '';
  document.getElementById('fileInfo').textContent = '';
  document.getElementById('submitBtn').disabled   = true;
  document.getElementById('libraryFilm').value    = '';
  document.getElementById('festivalKey').value    = '';
  document.getElementById('festivalHint').textContent = '';
  // Hide everything below festival picker until a new festival is chosen
  document.getElementById('libraryCard').style.display = 'none';
  document.getElementById('formCard').style.display    = 'none';
  document.getElementById('uploadCard').style.display  = 'none';
  document.getElementById('rewriteCard').style.display = 'none';
  dropZone.classList.remove('has-file');
  selectedFile   = null;
  selectedFilmId = null;
  document.getElementById('libraryHint').textContent =
    'Select an existing film to reuse its analysis (~0.01¢ vs ~1.6¢).';
}
</script>
</body></html>"""


if __name__ == "__main__":
    print(f"Festival Review App — {FESTIVALS[DEFAULT_FESTIVAL]['name']}")
    print("Local: http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
