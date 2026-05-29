"""
Festival Expert Review Generator — Single File App
Deploy to Cloud Run: docker build + gcloud run deploy
Run locally: python review_app.py
"""

import os, json, uuid, threading, tempfile, time, re
from pathlib import Path
from datetime import datetime
from functools import wraps
from flask import Flask, request, jsonify, session, redirect, render_template_string
from google import genai
from google.genai import types
from dotenv import load_dotenv

from festivals import FESTIVALS, DEFAULT_FESTIVAL, get_festival
from prompts import build_analysis_prompt, build_review_prompt
from wordpress import publish_review as wp_publish_review, publish_post as wp_publish_post

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET", uuid.uuid4().hex)

# ── Config ────────────────────────────────────────────────
MAX_UPLOAD_MB = 1800
ALLOWED_EXT   = {".mp4", ".mov", ".avi", ".webm", ".mkv", ".mpeg"}

# Credentials from env (set in GCP Secret Manager / .env)
USERS = {
    os.getenv("USER1_EMAIL", "employee1@elegantiff.com"): os.getenv("USER1_PASS", "change_me_1"),
    os.getenv("USER2_EMAIL", "employee2@elegantiff.com"): os.getenv("USER2_PASS", "change_me_2"),
}

# In-memory job store (fine for 2 concurrent users)
JOBS: dict[str, dict] = {}


# ── Auth ──────────────────────────────────────────────────
def login_required(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not session.get("logged_in"):
            return redirect("/login")
        return f(*args, **kwargs)
    return wrapper


# ── Processing thread ─────────────────────────────────────
def process_video(job_id: str, video_path: str, meta: dict):
    job = JOBS[job_id]
    uploaded_file = None
    festival = get_festival(meta.get("festival_key", DEFAULT_FESTIVAL))

    client = genai.Client(api_key=festival["gemini_api_key"])
    model_id = festival.get("gemini_model", "gemini-2.0-flash")
    uploaded_file = None
    try:
        # Step 1: Upload to Gemini using festival's own API key
        job.update({"status": "uploading", "progress": 15,
                    "message": f"Uploading to Gemini [{festival['name']}]..."})
        uploaded_file = client.files.upload(file=video_path)

        # Step 2: Wait for processing
        job.update({"status": "processing", "progress": 35,
                    "message": "Gemini is watching the film..."})
        while uploaded_file.state.name == "PROCESSING":
            time.sleep(6)
            uploaded_file = client.files.get(name=uploaded_file.name)

        if uploaded_file.state.name != "ACTIVE":
            raise RuntimeError(f"Gemini processing failed: {uploaded_file.state.name}")

        # Step 3: Analyse using festival-specific judging prompt
        job.update({"status": "analysing", "progress": 55,
                    "message": "Analysing story, direction, technical..."})
        analysis_resp = client.models.generate_content(
            model=model_id,
            contents=[uploaded_file, build_analysis_prompt(festival)],
            config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024),
        )
        analysis = _parse_json(analysis_resp.text)

        # Step 4: Generate review using festival tone + guidelines
        job.update({"status": "writing", "progress": 78,
                    "message": "Writing Expert Review..."})
        review_resp = client.models.generate_content(
            model=model_id,
            contents=build_review_prompt(meta, analysis, festival),
            config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=800),
        )

        job.update({
            "status": "done", "progress": 100,
            "message": "Review ready",
            "analysis": analysis,
            "review": review_resp.text.strip(),
            "completed_at": datetime.now().isoformat()
        })

    except Exception as e:
        job.update({"status": "error", "progress": 0, "message": str(e)[:300]})
    finally:
        try:
            if uploaded_file:
                client.files.delete(name=uploaded_file.name)
            Path(video_path).unlink(missing_ok=True)
        except Exception:
            pass


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


# ── Routes ────────────────────────────────────────────────
@app.route("/login", methods=["GET", "POST"])
def login():
    error = ""
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        if USERS.get(email) == password:
            session["logged_in"] = True
            session["user"] = email
            return redirect("/")
        error = "Invalid credentials"
    return render_template_string(LOGIN_HTML,
                                  festival=FESTIVALS[DEFAULT_FESTIVAL]["name"],
                                  error=error)


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


@app.route("/upload", methods=["POST"])
@login_required
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No video file"}), 400

    f = request.files["video"]
    ext = Path(f.filename).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify({"error": f"Unsupported format. Use: {', '.join(ALLOWED_EXT)}"}), 400

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

    meta = {
        "title":         request.form.get("title", "").strip(),
        "director":      request.form.get("director", "").strip(),
        "genre":         request.form.get("genre", "").strip(),
        "runtime":       request.form.get("runtime", "").strip(),
        "synopsis":      request.form.get("synopsis", "").strip(),
        "festival_key":  festival_key,
        "festival_name": festival["name"],
    }

    job_id = uuid.uuid4().hex
    JOBS[job_id] = {
        "status": "queued", "progress": 5,
        "message": "Queued for processing...",
        "meta": meta, "analysis": None, "review": None,
        "created_at": datetime.now().isoformat()
    }

    thread = threading.Thread(target=process_video,
                              args=(job_id, tmp.name, meta), daemon=True)
    thread.start()
    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
@login_required
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/publish/<job_id>", methods=["POST"])
@login_required
def publish(job_id):
    job = JOBS.get(job_id)
    if not job or job.get("status") != "done":
        return jsonify({"error": "Review not ready"}), 400

    festival = get_festival(job["meta"].get("festival_key", DEFAULT_FESTIVAL))
    wp_status = (request.json or {}).get("status", "draft")

    result = wp_publish_review(
        meta=job["meta"],
        analysis=job["analysis"],
        review=job["review"],
        festival=festival,
        status=wp_status,
    )

    if result["success"]:
        JOBS[job_id]["wp_post_id"] = result["post_id"]
        JOBS[job_id]["wp_url"]     = result["url"]
        JOBS[job_id]["wp_status"]  = wp_status

    return jsonify(result)


@app.route("/publish_live/<job_id>", methods=["POST"])
@login_required
def publish_live(job_id):
    job = JOBS.get(job_id)
    post_id = job.get("wp_post_id") if job else None
    if not post_id:
        return jsonify({"error": "Not published to WordPress yet"}), 400
    festival = get_festival(job["meta"].get("festival_key", DEFAULT_FESTIVAL))
    return jsonify(wp_publish_post(post_id, festival))


# ── HTML Templates ────────────────────────────────────────
LOGIN_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{{ festival }} — Sign In</title>
<link href="https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0a0a0f;color:#e0dbd0;font-family:'DM Sans',sans-serif;
     min-height:100vh;display:flex;align-items:center;justify-content:center;
     background-image:radial-gradient(ellipse 60% 50% at 50% 0%,rgba(201,168,76,.07),transparent)}
.box{width:380px;background:#13131a;border:1px solid rgba(201,168,76,.2);
     border-radius:16px;padding:40px;text-align:center}
.logo{font-family:'Bebas Neue',sans-serif;font-size:36px;color:#C9A84C;
      letter-spacing:2px;margin-bottom:4px}
.sub{font-size:12px;color:#6a6560;font-family:'DM Mono',monospace;
     letter-spacing:2px;text-transform:uppercase;margin-bottom:32px}
label{display:block;text-align:left;font-size:11px;color:#6a6560;
      letter-spacing:1.5px;text-transform:uppercase;font-family:'DM Mono',monospace;
      margin-bottom:6px}
input{width:100%;background:#1a1a24;border:1px solid rgba(255,255,255,.08);
      border-radius:8px;padding:12px 14px;color:#e0dbd0;font-family:'DM Sans',sans-serif;
      font-size:14px;outline:none;margin-bottom:16px;transition:border-color .2s}
input:focus{border-color:#C9A84C}
button{width:100%;background:#C9A84C;color:#000;border:none;border-radius:8px;
       padding:13px;font-family:'DM Sans',sans-serif;font-weight:600;font-size:14px;
       cursor:pointer;margin-top:8px;transition:background .2s}
button:hover{background:#e8c97a}
.error{background:rgba(224,90,90,.1);border:1px solid rgba(224,90,90,.3);
       border-radius:8px;padding:10px;font-size:13px;color:#e08080;margin-bottom:16px}
</style></head><body>
<div class="box">
  <div class="logo">{{ festival }}</div>
  <div class="sub">Review Portal</div>
  {% if error %}<div class="error">{{ error }}</div>{% endif %}
  <form method="POST">
    <label>Email</label>
    <input type="email" name="email" placeholder="your@email.com" required autofocus>
    <label>Password</label>
    <input type="password" name="password" placeholder="••••••••" required>
    <button type="submit">Sign In</button>
  </form>
</div>
</body></html>"""


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
body{background:var(--bg);color:var(--text);font-family:'DM Sans',sans-serif;
     min-height:100vh;
     background-image:radial-gradient(ellipse 80% 40% at 50% 0%,rgba(201,168,76,.06),transparent)}

/* ── Header ── */
.header{background:var(--bg2);border-bottom:1px solid var(--border);
        padding:14px 28px;display:flex;align-items:center;justify-content:space-between}
.header-logo{font-family:'Bebas Neue',sans-serif;font-size:22px;
             color:var(--gold);letter-spacing:2px}
.header-right{display:flex;align-items:center;gap:16px}
.user-badge{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace}
.logout{font-size:11px;color:var(--gold-d);text-decoration:none;
        font-family:'DM Mono',monospace;letter-spacing:1px;transition:color .2s}
.logout:hover{color:var(--gold)}

/* ── Layout ── */
.main{max-width:960px;margin:0 auto;padding:32px 20px}
.page-title{font-family:'Bebas Neue',sans-serif;font-size:40px;
            color:var(--gold);letter-spacing:1px;margin-bottom:4px}
.page-sub{font-size:13px;color:var(--muted);margin-bottom:28px}

/* ── Card ── */
.card{background:var(--bg2);border:1px solid var(--border);
      border-radius:14px;overflow:hidden;margin-bottom:20px}
.card-head{padding:18px 24px;border-bottom:1px solid var(--border);
           display:flex;align-items:center;gap:10px}
.card-head-icon{width:28px;height:28px;border-radius:6px;
                background:rgba(201,168,76,.1);display:flex;
                align-items:center;justify-content:center;font-size:14px}
.card-head-title{font-size:12px;letter-spacing:2px;color:var(--gold-d);
                 text-transform:uppercase;font-family:'DM Mono',monospace;font-weight:500}
.card-body{padding:20px 24px}

/* ── Form grid ── */
.form-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
.form-group{display:flex;flex-direction:column;gap:6px}
.form-group.full{grid-column:1/-1}
label{font-size:10px;color:var(--muted);letter-spacing:1.5px;
      text-transform:uppercase;font-family:'DM Mono',monospace}
input[type=text],input[type=number],select,textarea{
  background:var(--bg3);border:1px solid rgba(255,255,255,.07);
  border-radius:8px;padding:10px 12px;color:var(--text);
  font-family:'DM Sans',sans-serif;font-size:13px;
  outline:none;width:100%;transition:border-color .2s;resize:vertical}
input[type=text]:focus,input[type=number]:focus,
select:focus,textarea:focus{border-color:var(--gold)}
select option{background:var(--bg3)}
textarea{min-height:80px;line-height:1.5}

/* ── Festival hint ── */
.festival-hint{font-size:11px;color:var(--muted);margin-top:4px;
               font-family:'DM Mono',monospace;min-height:16px}

/* ── Drop zone ── */
.drop-zone{border:2px dashed rgba(201,168,76,.25);border-radius:12px;
           padding:40px 24px;text-align:center;cursor:pointer;
           transition:all .2s;position:relative;background:var(--bg3)}
.drop-zone.drag-over{border-color:var(--gold);background:rgba(201,168,76,.05)}
.drop-zone.has-file{border-color:rgba(76,175,122,.4);background:rgba(76,175,122,.04)}
.drop-icon{font-size:36px;margin-bottom:12px;opacity:.6}
.drop-title{font-size:15px;font-weight:600;color:var(--text);margin-bottom:4px}
.drop-sub{font-size:12px;color:var(--muted)}
.file-info{font-size:12px;color:var(--green);font-family:'DM Mono',monospace;
           margin-top:8px;font-weight:500}
input[type=file]{display:none}

/* ── Submit btn ── */
.submit-btn{width:100%;background:var(--gold);color:#000;border:none;
            border-radius:10px;padding:14px;font-family:'DM Sans',sans-serif;
            font-weight:700;font-size:15px;cursor:pointer;margin-top:4px;
            transition:all .2s;letter-spacing:.3px}
.submit-btn:hover:not(:disabled){background:var(--gold-l);transform:translateY(-1px)}
.submit-btn:disabled{opacity:.4;cursor:not-allowed;transform:none}

/* ── Progress ── */
.progress-card{display:none}
.progress-card.active{display:block}
.progress-status{display:flex;align-items:center;gap:12px;margin-bottom:16px}
.spinner{width:20px;height:20px;border:2px solid rgba(201,168,76,.2);
         border-top-color:var(--gold);border-radius:50%;
         animation:spin .8s linear infinite;flex-shrink:0}
@keyframes spin{to{transform:rotate(360deg)}}
.progress-msg{font-size:13px;color:var(--text)}
.progress-pct{font-family:'DM Mono',monospace;font-size:12px;
               color:var(--gold);margin-left:auto}
.progress-track{height:4px;background:rgba(255,255,255,.06);
                border-radius:2px;overflow:hidden}
.progress-fill{height:100%;background:linear-gradient(90deg,var(--gold-d),var(--gold),var(--gold-l));
               border-radius:2px;transition:width .4s ease;
               box-shadow:0 0 8px rgba(201,168,76,.4)}
.step-list{display:flex;flex-direction:column;gap:8px;margin-top:16px}
.step{display:flex;align-items:center;gap:10px;font-size:12px;
      color:var(--muted);font-family:'DM Mono',monospace}
.step.active{color:var(--text)}
.step.done{color:var(--green)}
.step-dot{width:6px;height:6px;border-radius:50%;background:currentColor;flex-shrink:0}

/* ── Results ── */
.results-card{display:none}
.results-card.active{display:block}
.scores-row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:20px}
.score-box{background:var(--bg3);border-radius:10px;padding:14px;text-align:center}
.score-label{font-size:9px;color:var(--muted);letter-spacing:1.5px;
              text-transform:uppercase;font-family:'DM Mono',monospace;
              display:block;margin-bottom:6px}
.score-num{font-family:'Bebas Neue',sans-serif;font-size:32px;color:var(--gold);line-height:1}
.score-denom{font-size:11px;color:var(--muted);font-family:'DM Mono',monospace}
.overall{background:linear-gradient(135deg,rgba(201,168,76,.1),rgba(201,168,76,.03));
          border:1px solid rgba(201,168,76,.25)}
.obs-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:20px}
.obs-item{background:var(--bg3);border-radius:8px;padding:12px}
.obs-label{font-size:9px;color:var(--muted);letter-spacing:1.5px;
            text-transform:uppercase;font-family:'DM Mono',monospace;
            display:block;margin-bottom:5px}
.obs-text{font-size:12px;color:var(--text);line-height:1.5}
.standout{border-left:2px solid var(--gold);padding-left:12px}
.weakest{border-left:2px solid rgba(224,90,90,.5);padding-left:12px}
.review-block{background:var(--bg3);border-radius:10px;padding:18px;position:relative}
.review-text{font-size:14px;line-height:1.8;color:var(--text);
              white-space:pre-wrap;font-family:'DM Sans',sans-serif}
.copy-btn{position:absolute;top:12px;right:12px;background:rgba(201,168,76,.1);
           border:1px solid var(--border);border-radius:6px;padding:6px 12px;
           color:var(--gold);font-size:11px;font-family:'DM Mono',monospace;
           cursor:pointer;transition:all .2s}
.copy-btn:hover{background:rgba(201,168,76,.2)}
.copy-btn.copied{color:var(--green);border-color:rgba(76,175,122,.3)}
.new-btn{width:100%;background:transparent;border:1px solid var(--border);
          border-radius:10px;padding:12px;color:var(--muted);
          font-family:'DM Sans',sans-serif;font-size:14px;cursor:pointer;
          margin-top:12px;transition:all .2s}
.new-btn:hover{border-color:var(--gold);color:var(--text)}
.error-msg{background:rgba(224,90,90,.06);border:1px solid rgba(224,90,90,.2);
            border-radius:8px;padding:12px 16px;font-size:13px;color:#e08080;display:none}
.error-msg.active{display:block}
.film-tag{display:inline-flex;align-items:center;gap:6px;
           background:rgba(201,168,76,.08);border:1px solid var(--border);
           border-radius:20px;padding:4px 12px;font-size:11px;
           color:var(--gold-l);font-family:'DM Mono',monospace;margin-bottom:16px}

@media(max-width:640px){
  .form-grid,.scores-row,.obs-grid{grid-template-columns:1fr}
  .form-group.full{grid-column:1}
}
</style></head><body>

<div class="header">
  <div class="header-logo">{{ festival }}</div>
  <div class="header-right">
    <span class="user-badge">{{ user }}</span>
    <a href="/logout" class="logout">Sign out</a>
  </div>
</div>

<div class="main">
  <div class="page-title">Expert Review</div>
  <div class="page-sub">Upload a film submission to generate an AI-assisted Expert Review</div>

  <!-- ── FORM ── -->
  <div class="card" id="formCard">
    <div class="card-head">
      <div class="card-head-icon">🎬</div>
      <div class="card-head-title">Film Details</div>
    </div>
    <div class="card-body">
      <div class="form-grid">

        <div class="form-group full">
          <label>Festival *</label>
          <select id="festivalKey" onchange="onFestivalChange(this)">
            {% for key, f in festivals.items() %}
            <option value="{{ key }}"{% if key == default_festival %} selected{% endif %}>
              {{ f.name }} — {{ f.focus }}
            </option>
            {% endfor %}
          </select>
          <div class="festival-hint" id="festivalHint"></div>
        </div>

        <div class="form-group">
          <label>Film Title *</label>
          <input type="text" id="title" placeholder="The Last Frame" required>
        </div>
        <div class="form-group">
          <label>Director *</label>
          <input type="text" id="director" placeholder="Jane Smith" required>
        </div>
        <div class="form-group">
          <label>Genre</label>
          <select id="genre">
            <option value="">Select genre</option>
            <option>Drama</option><option>Documentary</option>
            <option>Comedy</option><option>Thriller</option>
            <option>Horror</option><option>Animation</option>
            <option>Experimental</option><option>Short Film</option>
            <option>Feature</option><option>Music Video</option>
            <option>Other</option>
          </select>
        </div>
        <div class="form-group">
          <label>Runtime (minutes)</label>
          <input type="number" id="runtime" placeholder="12" min="1">
        </div>
        <div class="form-group full">
          <label>Synopsis</label>
          <textarea id="synopsis" placeholder="Brief description of the film..."></textarea>
        </div>

      </div>
    </div>
  </div>

  <div class="card" id="uploadCard">
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
      <div class="step-list">
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

</div>

<script>
let selectedFile = null;
let pollInterval = null;

const FESTIVALS = {
  {% for key, f in festivals.items() %}
  "{{ key }}": {
    name: "{{ f.name }}",
    focus: "{{ f.focus }}",
    words: {{ f.word_count }},
    tone: "{{ f.tone }}"
  },
  {% endfor %}
};

function onFestivalChange(sel) {
  const f = FESTIVALS[sel.value];
  document.getElementById('festivalHint').textContent =
    f ? `${f.words}-word review · ${f.focus}` : '';
}

window.addEventListener('DOMContentLoaded', () => {
  const sel = document.getElementById('festivalKey');
  if (sel) onFestivalChange(sel);
});

// ── Drop zone ──────────────────────────────────────────────────────
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');

dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault(); dropZone.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});
fileInput.addEventListener('change', e => { if (e.target.files[0]) handleFile(e.target.files[0]); });

function handleFile(file) {
  selectedFile = file;
  const sizeMb = (file.size / 1024 / 1024).toFixed(1);
  document.getElementById('fileInfo').textContent = `✓ ${file.name}  (${sizeMb} MB)`;
  dropZone.classList.add('has-file');
  document.getElementById('submitBtn').disabled = false;
  showError('');
}

// ── Submit ─────────────────────────────────────────────────────────
async function submitReview() {
  const title    = document.getElementById('title').value.trim();
  const director = document.getElementById('director').value.trim();
  if (!title || !director) { showError('Film title and director are required'); return; }
  if (!selectedFile) { showError('Please upload a video file'); return; }

  const form = new FormData();
  form.append('video',       selectedFile);
  form.append('title',       title);
  form.append('director',    director);
  form.append('genre',       document.getElementById('genre').value);
  form.append('runtime',     document.getElementById('runtime').value);
  form.append('synopsis',    document.getElementById('synopsis').value);
  form.append('festival_key', document.getElementById('festivalKey').value);

  showProcessing();
  try {
    const res  = await fetch('/upload', { method: 'POST', body: form });
    const data = await res.json();
    if (data.error) { showFormError(data.error); return; }
    pollStatus(data.job_id);
  } catch(e) {
    showFormError('Upload failed. Please try again.');
  }
}

// ── Polling ────────────────────────────────────────────────────────
function pollStatus(jobId) {
  clearInterval(pollInterval);
  pollInterval = setInterval(async () => {
    try {
      const res = await fetch(`/status/${jobId}`);
      const job = await res.json();
      updateProgress(job);
      if (job.status === 'done') { clearInterval(pollInterval); showResults(job); }
      else if (job.status === 'error') { clearInterval(pollInterval); showFormError(job.message); }
    } catch(e) { /* keep polling */ }
  }, 2500);
}

// ── Progress UI ────────────────────────────────────────────────────
const STEPS = ['uploading','processing','analysing','writing'];

function updateProgress(job) {
  document.getElementById('progressMsg').textContent  = job.message;
  document.getElementById('progressPct').textContent  = job.progress + '%';
  document.getElementById('progressFill').style.width = job.progress + '%';
  STEPS.forEach(s => {
    const el = document.getElementById('step-' + s);
    el.className = 'step';
    if (s === job.status) el.classList.add('active');
    if (STEPS.indexOf(s) < STEPS.indexOf(job.status) || job.status === 'done')
      el.classList.add('done');
  });
}

// ── Results ────────────────────────────────────────────────────────
function showResults(job) {
  document.getElementById('progressCard').classList.remove('active');
  document.getElementById('resultsCard').classList.add('active');

  const a    = job.analysis;
  const meta = job.meta;
  const festivalLabel = meta.festival_name || '';

  document.getElementById('filmTag').innerHTML =
    `🎬 <span>${meta.title}</span> &nbsp;·&nbsp; <span>${meta.director}</span>` +
    (meta.genre   ? ` &nbsp;·&nbsp; <span>${meta.genre}</span>`    : '') +
    (meta.runtime ? ` &nbsp;·&nbsp; <span>${meta.runtime} min</span>` : '') +
    (festivalLabel ? ` &nbsp;·&nbsp; <span style="color:var(--gold)">${festivalLabel}</span>` : '');

  const cats = [
    {k:'story',l:'Story'},{k:'direction',l:'Direction'},
    {k:'technical',l:'Technical'},{k:'originality',l:'Originality'}
  ];
  let scoresHtml = cats.map(c =>
    `<div class="score-box">
      <span class="score-label">${c.l}</span>
      <div><span class="score-num">${a[c.k].score}</span><span class="score-denom">/5</span></div>
    </div>`).join('');
  scoresHtml += `<div class="score-box overall">
    <span class="score-label">Overall</span>
    <div><span class="score-num">${a.overall_score}</span><span class="score-denom">/20</span></div>
  </div>`;
  document.getElementById('scoresRow').innerHTML = scoresHtml;

  document.getElementById('obsGrid').innerHTML = `
    <div class="obs-item standout">
      <span class="obs-label">Standout Moment</span>
      <div class="obs-text">${a.standout_moment}</div>
    </div>
    <div class="obs-item weakest">
      <span class="obs-label">Growth Area</span>
      <div class="obs-text">${a.weakest_element}</div>
    </div>
    <div class="obs-item" style="grid-column:1/-1">
      <span class="obs-label">Festival Suitability</span>
      <div class="obs-text">${a.festival_suitability}</div>
    </div>`;

  document.getElementById('reviewText').textContent = job.review;
}

// ── Copy ───────────────────────────────────────────────────────────
function copyReview() {
  navigator.clipboard.writeText(document.getElementById('reviewText').textContent).then(() => {
    const btn = document.getElementById('copyBtn');
    btn.textContent = '✓ Copied'; btn.classList.add('copied');
    setTimeout(() => { btn.textContent = 'Copy'; btn.classList.remove('copied'); }, 2000);
  });
}

// ── Helpers ────────────────────────────────────────────────────────
function showProcessing() {
  document.getElementById('formCard').style.display   = 'none';
  document.getElementById('uploadCard').style.display = 'none';
  document.getElementById('progressCard').classList.add('active');
  document.getElementById('progressFill').style.width = '5%';
}

function showError(msg) {
  const el = document.getElementById('errorMsg');
  el.textContent = msg; el.className = 'error-msg' + (msg ? ' active' : '');
}

function showFormError(msg) {
  document.getElementById('progressCard').classList.remove('active');
  document.getElementById('formCard').style.display   = 'block';
  document.getElementById('uploadCard').style.display = 'block';
  showError(msg);
}

function resetForm() {
  document.getElementById('resultsCard').classList.remove('active');
  document.getElementById('formCard').style.display   = 'block';
  document.getElementById('uploadCard').style.display = 'block';
  ['title','director','runtime','synopsis'].forEach(id =>
    document.getElementById(id).value = '');
  document.getElementById('genre').value = '';
  document.getElementById('fileInfo').textContent = '';
  document.getElementById('submitBtn').disabled = true;
  dropZone.classList.remove('has-file');
  selectedFile = null;
}
</script>
</body></html>"""


if __name__ == "__main__":
    print(f"Festival Review App — {FESTIVALS[DEFAULT_FESTIVAL]['name']}")
    print("Local: http://localhost:8080")
    app.run(host="0.0.0.0", port=8080, debug=False)
