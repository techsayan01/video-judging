"""
Festival Review Approval UI
Run: python app.py
Open: http://localhost:5000
"""

import json
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template_string, request, jsonify, redirect
from config import QUEUE_DIR, APPROVED_DIR

app = Flask(__name__)

# ── HTML Template ─────────────────────────────────────────────────────────────

TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Festival Review Queue</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;600&family=DM+Mono:wght@400;500&display=swap');
  :root {
    --gold:#C9A84C; --bg:#0f0f13; --bg2:#16161d; --bg3:#1e1e28;
    --border:rgba(201,168,76,0.15); --text:#e0dbd0; --muted:#6a6560;
    --green:#4caf7a; --red:#e05a5a;
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body { background:var(--bg); color:var(--text); font-family:'DM Sans',sans-serif;
         min-height:100vh; }
  .header { background:var(--bg2); border-bottom:1px solid var(--border);
             padding:16px 24px; display:flex; align-items:center;
             justify-content:space-between; }
  .header h1 { font-family:'DM Mono',monospace; font-size:14px;
                letter-spacing:2px; color:var(--gold); text-transform:uppercase; }
  .stats { display:flex; gap:16px; font-size:12px; color:var(--muted);
            font-family:'DM Mono',monospace; }
  .stat strong { color:var(--text); }
  .container { max-width:1200px; margin:0 auto; padding:24px; }

  /* Queue list */
  .queue { display:flex; flex-direction:column; gap:10px; margin-bottom:32px; }
  .queue-item { background:var(--bg2); border:1px solid var(--border);
                 border-radius:10px; padding:16px 20px;
                 display:grid; grid-template-columns:1fr auto;
                 gap:12px; align-items:center; cursor:pointer;
                 transition:border-color 0.2s; }
  .queue-item:hover { border-color:var(--gold); }
  .queue-item.approved { border-color:rgba(76,175,122,0.3);
                          background:rgba(76,175,122,0.04); }
  .qi-title { font-weight:600; font-size:14px; margin-bottom:3px; }
  .qi-meta { font-size:12px; color:var(--muted);
              font-family:'DM Mono',monospace; }
  .qi-status { font-size:11px; font-family:'DM Mono',monospace;
                padding:4px 10px; border-radius:4px; font-weight:600; }
  .status-pending { background:rgba(201,168,76,0.12); color:var(--gold); }
  .status-approved { background:rgba(76,175,122,0.12); color:var(--green); }
  .status-error { background:rgba(224,90,90,0.12); color:var(--red); }

  /* Review editor */
  .review-editor { background:var(--bg2); border:1px solid var(--border);
                    border-radius:12px; overflow:hidden; }
  .editor-header { padding:20px 24px; border-bottom:1px solid var(--border);
                    display:flex; justify-content:space-between; align-items:center; }
  .film-title { font-size:18px; font-weight:600; }
  .film-meta { font-size:12px; color:var(--muted); font-family:'DM Mono',monospace;
                margin-top:4px; }
  .editor-body { display:grid; grid-template-columns:1fr 1fr; }
  .panel-left { padding:20px 24px; border-right:1px solid var(--border); }
  .panel-right { padding:20px 24px; }
  .section-label { font-size:10px; letter-spacing:2px; color:var(--gold);
                    text-transform:uppercase; font-family:'DM Mono',monospace;
                    margin-bottom:12px; }
  .scores { display:grid; grid-template-columns:1fr 1fr; gap:8px;
             margin-bottom:16px; }
  .score-box { background:var(--bg3); border-radius:8px; padding:10px 12px; }
  .score-box label { font-size:10px; color:var(--muted); display:block;
                      margin-bottom:3px; font-family:'DM Mono',monospace;
                      letter-spacing:1px; text-transform:uppercase; }
  .score-val { font-family:'DM Mono',monospace; font-size:20px;
                color:var(--gold); font-weight:600; }
  .score-val small { font-size:11px; color:var(--muted); }
  .obs-item { margin-bottom:12px; }
  .obs-item label { font-size:10px; color:var(--muted); display:block;
                     margin-bottom:3px; font-family:'DM Mono',monospace;
                     letter-spacing:1px; text-transform:uppercase; }
  .obs-item p { font-size:12px; line-height:1.5; color:var(--text); }
  .highlight { background:rgba(201,168,76,0.08); border-left:2px solid var(--gold);
                padding:8px 12px; border-radius:0 6px 6px 0; margin:8px 0; 
                font-size:12px; line-height:1.5; }
  textarea { width:100%; background:var(--bg3); border:1px solid var(--border);
              border-radius:8px; color:var(--text); padding:14px;
              font-family:'DM Sans',sans-serif; font-size:13px;
              line-height:1.7; resize:vertical; min-height:280px; outline:none; }
  textarea:focus { border-color:var(--gold); }
  .word-count { font-size:11px; color:var(--muted); font-family:'DM Mono',monospace;
                 margin-top:6px; text-align:right; }
  .editor-footer { padding:16px 24px; border-top:1px solid var(--border);
                    display:flex; justify-content:space-between; align-items:center; }
  .btn { padding:10px 24px; border-radius:8px; border:none; cursor:pointer;
          font-family:'DM Sans',sans-serif; font-size:13px; font-weight:600;
          transition:all 0.2s; }
  .btn-approve { background:var(--green); color:#000; }
  .btn-approve:hover { background:#5fe090; }
  .btn-back { background:transparent; color:var(--muted); border:1px solid var(--border); }
  .btn-back:hover { color:var(--text); border-color:var(--text); }
  .synopsis-box { background:var(--bg3); border-radius:8px; padding:12px;
                   font-size:12px; color:var(--muted); line-height:1.6;
                   margin-bottom:12px; max-height:100px; overflow-y:auto; }
  .empty { text-align:center; padding:60px 24px; color:var(--muted); }
  .empty h2 { font-size:24px; color:var(--green); margin-bottom:8px; }
  @media(max-width:768px) {
    .editor-body { grid-template-columns:1fr; }
    .panel-left { border-right:none; border-bottom:1px solid var(--border); }
  }
</style>
</head>
<body>

<div class="header">
  <h1>🎬 Review Queue</h1>
  <div class="stats">
    <span>Pending: <strong>{{ pending }}</strong></span>
    <span>Approved: <strong>{{ approved }}</strong></span>
  </div>
</div>

<div class="container">
{% if view == 'list' %}

  {% if items %}
  <div class="queue">
  {% for item in items %}
    <div class="queue-item {% if item.approved %}approved{% endif %}"
         onclick="window.location='/review/{{ item.entry_id }}'">
      <div>
        <div class="qi-title">{{ item.film.title or 'Untitled' }}</div>
        <div class="qi-meta">
          {{ item.film.director }} · {{ item.film.genre }} · 
          {{ item.film.runtime }} min · {{ item.film.country }}
          {% if item.error %} · ⚠ {{ item.error[:50] }}{% endif %}
        </div>
      </div>
      <span class="qi-status 
        {% if item.approved %}status-approved
        {% elif item.error %}status-error
        {% else %}status-pending{% endif %}">
        {% if item.approved %}✓ Approved
        {% elif item.error %}Error
        {% else %}Pending{% endif %}
      </span>
    </div>
  {% endfor %}
  </div>
  {% else %}
  <div class="empty">
    <h2>✓ All clear</h2>
    <p>No reviews pending. Run the pipeline to process new submissions.</p>
  </div>
  {% endif %}

{% elif view == 'editor' %}

  {% set a = item.analysis %}
  <div class="review-editor">
    <div class="editor-header">
      <div>
        <div class="film-title">{{ item.film.title }}</div>
        <div class="film-meta">
          {{ item.film.director }} · {{ item.film.genre }} · 
          {{ item.film.runtime }} min · {{ item.film.country }} · 
          {{ item.film.email }}
        </div>
      </div>
      {% if a %}
      <div style="font-family:'DM Mono',monospace; font-size:28px; color:var(--gold);">
        {{ a.overall_score }}<small style="font-size:14px;color:var(--muted)">/20</small>
      </div>
      {% endif %}
    </div>

    <div class="editor-body">
      <!-- LEFT: Analysis observations -->
      <div class="panel-left">
        <div class="section-label">Film Analysis</div>

        {% if item.film.synopsis %}
        <div class="obs-item">
          <label>Synopsis</label>
          <div class="synopsis-box">{{ item.film.synopsis }}</div>
        </div>
        {% endif %}

        {% if a %}
        <div class="scores">
          <div class="score-box">
            <label>Story</label>
            <div class="score-val">{{ a.story.score }}<small>/5</small></div>
          </div>
          <div class="score-box">
            <label>Direction</label>
            <div class="score-val">{{ a.direction.score }}<small>/5</small></div>
          </div>
          <div class="score-box">
            <label>Technical</label>
            <div class="score-val">{{ a.technical.score }}<small>/5</small></div>
          </div>
          <div class="score-box">
            <label>Originality</label>
            <div class="score-val">{{ a.originality.score }}<small>/5</small></div>
          </div>
        </div>

        <div class="obs-item">
          <label>Standout Moment</label>
          <div class="highlight">{{ a.standout_moment }}</div>
        </div>
        <div class="obs-item">
          <label>Weakest Element</label>
          <p>{{ a.weakest_element }}</p>
        </div>
        <div class="obs-item">
          <label>Festival Suitability</label>
          <p>{{ a.festival_suitability }}</p>
        </div>
        <div class="obs-item">
          <label>Story Notes</label>
          <p>{{ a.story.notes }}</p>
        </div>
        <div class="obs-item">
          <label>Technical Notes</label>
          <p>{{ a.technical.notes }}</p>
        </div>
        {% else %}
        <p style="color:var(--red);font-size:13px;">
          Analysis unavailable{% if item.error %}: {{ item.error }}{% endif %}
        </p>
        {% endif %}
      </div>

      <!-- RIGHT: Editable review draft -->
      <div class="panel-right">
        <div class="section-label">Expert Review Draft — Edit before approving</div>
        <form method="POST" action="/approve/{{ item.entry_id }}" id="reviewForm">
          <textarea name="review_final" id="reviewText"
                    placeholder="Review draft will appear here..."
                    oninput="updateWordCount(this)">{{ item.review_draft }}</textarea>
          <div class="word-count" id="wordCount">0 words</div>
        </form>
      </div>
    </div>

    <div class="editor-footer">
      <a href="/" class="btn btn-back">← Back to queue</a>
      <button class="btn btn-approve" 
              onclick="document.getElementById('reviewForm').submit()">
        ✓ Approve & Mark Complete
      </button>
    </div>
  </div>

{% endif %}
</div>

<script>
function updateWordCount(el) {
  const words = el.value.trim().split(/\s+/).filter(w => w).length;
  document.getElementById('wordCount').textContent = words + ' words';
}
// Init on load
window.onload = () => {
  const ta = document.getElementById('reviewText');
  if (ta) updateWordCount(ta);
};
</script>
</body>
</html>
"""


# ── Routes ────────────────────────────────────────────────────────────────────

def load_all_items():
    """Load all items from queue dir, sorted pending first."""
    items = []
    for f in sorted(Path(QUEUE_DIR).glob("*.json")):
        try:
            items.append(json.loads(f.read_text()))
        except Exception:
            pass
    # Pending first, then approved
    return sorted(items, key=lambda x: x.get("approved", False))


@app.route("/")
def index():
    items = load_all_items()
    pending = sum(1 for i in items if not i.get("approved") and not i.get("error"))
    approved = sum(1 for i in items if i.get("approved"))
    return render_template_string(TEMPLATE, view="list", items=items,
                                  pending=pending, approved=approved)


@app.route("/review/<entry_id>")
def review(entry_id):
    path = Path(QUEUE_DIR) / f"{entry_id}.json"
    if not path.exists():
        return redirect("/")
    item = json.loads(path.read_text())
    pending = sum(1 for f in Path(QUEUE_DIR).glob("*.json")
                  if not json.loads(f.read_text()).get("approved"))
    approved = sum(1 for f in Path(QUEUE_DIR).glob("*.json")
                   if json.loads(f.read_text()).get("approved"))
    return render_template_string(TEMPLATE, view="editor", item=item,
                                  pending=pending, approved=approved)


@app.route("/approve/<entry_id>", methods=["POST"])
def approve(entry_id):
    """Save approved review and mark as done."""
    path = Path(QUEUE_DIR) / f"{entry_id}.json"
    if not path.exists():
        return redirect("/")

    item = json.loads(path.read_text())
    item["review_final"] = request.form.get("review_final", "").strip()
    item["approved"] = True
    item["approved_at"] = datetime.now().isoformat()

    # Save approved copy
    approved_path = Path(APPROVED_DIR) / f"{entry_id}.json"
    approved_path.write_text(json.dumps(item, indent=2, ensure_ascii=False))

    # Remove from pending queue
    path.unlink()

    print(f"[approved] {entry_id} — {item['film']['title']}")
    return redirect("/")


@app.route("/api/pending")
def api_pending():
    """JSON API — count of pending reviews."""
    items = load_all_items()
    pending = [i for i in items if not i.get("approved") and not i.get("error")]
    return jsonify({"count": len(pending), "items": [
        {"entry_id": i["entry_id"], "title": i["film"]["title"]}
        for i in pending
    ]})


if __name__ == "__main__":
    print("Festival Review Approval UI")
    print("Open: http://localhost:5000")
    app.run(debug=True, port=5000)
