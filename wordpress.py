"""
wordpress.py — WordPress REST API integration + reference-design templating.

Each festival publishes to its own WordPress site using its own credentials
(stored on the festival document in MongoDB, not env vars):
    wp_url            — e.g. https://myfestival.com
    wp_user           — WordPress username
    wp_app_pass       — Application Password (Users > Profile > Application Passwords)
    wp_default_status — "draft" | "publish"
    wp_template       — HTML template using the placeholder tokens below

Publishing renders the generated review + analysis into the festival's template
(a reference design the admin supplies once, optionally AI-generated) and pushes
it to the site via the REST API using Basic-auth Application Passwords.
"""

import re
import html
import requests
from base64 import b64encode
from datetime import datetime
from urllib.parse import urlparse

# ── Placeholder tokens available in a festival's WP template ───────────────────
PLACEHOLDERS = [
    "title", "director", "genre", "runtime", "country",
    "festival_name", "season", "overall_rating",
    "scores_table", "standout", "growth_area",
    "festival_suitability", "review_body", "date",
]

# Human-readable labels for the 9-criteria ratings schema (order preserved)
_RATING_LABELS = [
    ("originality",      "Originality"),
    ("direction",        "Direction"),
    ("writing",          "Writing"),
    ("cinematography",   "Cinematography"),
    ("performances",     "Performances"),
    ("production_value", "Production"),
    ("pacing",           "Pacing"),
    ("structure",        "Structure"),
    ("sound_music",      "Sound / Music"),
]

# Default template used when a festival hasn't defined its own. Self-contained
# inline styles so it renders correctly inside any WordPress theme.
DEFAULT_WP_TEMPLATE = """\
<div style="max-width:720px;margin:0 auto;font-family:Georgia,'Times New Roman',serif;color:#1a1a1a">
  <p style="font-size:12px;letter-spacing:2px;text-transform:uppercase;color:#9a7d3c;font-family:Arial,sans-serif;margin:0 0 6px">
    {{festival_name}} — Official Expert Review</p>
  <h2 style="font-size:30px;line-height:1.15;margin:0 0 6px">{{title}}</h2>
  <p style="color:#666;font-size:15px;margin:0 0 4px">Directed by {{director}}</p>
  <p style="color:#888;font-size:13px;font-family:Arial,sans-serif;margin:0 0 24px">
    {{genre}} · {{runtime}} · {{country}}</p>
  {{scores_table}}
  <blockquote style="border-left:3px solid #d1af62;padding:4px 0 4px 18px;margin:24px 0;font-style:italic;color:#333">
    <strong style="font-style:normal">Standout:</strong> {{standout}}</blockquote>
  <div style="line-height:1.85;font-size:17px">{{review_body}}</div>
  <p style="margin-top:36px;padding-top:16px;border-top:1px solid #eee;font-size:12px;color:#aaa;font-family:Arial,sans-serif">
    Reviewed by the {{festival_name}} programming team · {{date}}</p>
</div>"""


# ── URL safety ────────────────────────────────────────────────────────────────
def _safe_base_url(url: str) -> str:
    """Return a normalised http/https base URL, or raise ValueError.
    Blocks non-web schemes (file://, gopher://, etc.) to limit SSRF surface."""
    url = (url or "").strip().rstrip("/")
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("WordPress URL must be a full http(s) URL, e.g. https://example.com")
    return url


def _auth_header(festival: dict) -> dict:
    token = b64encode(
        f"{festival.get('wp_user','')}:{festival.get('wp_app_pass','')}".encode()
    ).decode()
    return {"Authorization": f"Basic {token}"}


def wp_configured(festival: dict) -> bool:
    return bool(festival.get("wp_url") and festival.get("wp_user") and festival.get("wp_app_pass"))


# ── Context assembly ──────────────────────────────────────────────────────────
def _paras(text: str) -> str:
    """Split review text into HTML paragraphs, escaping each block."""
    text = (text or "").strip()
    if not text:
        return ""
    blocks = re.split(r"\n\s*\n", text) if "\n\n" in text else text.split("\n")
    return "".join(
        f"<p>{html.escape(b.strip())}</p>" for b in blocks if b.strip()
    )


def _scores_table(analysis: dict) -> str:
    """Render the 9-criteria ratings (with legacy fallback) as an HTML table."""
    if not analysis:
        return ""
    ratings = analysis.get("ratings") or {}
    rows = []
    if ratings:
        for key, label in _RATING_LABELS:
            val = ratings.get(key)
            if val is None:
                continue
            rows.append((label, f"{val}/10"))
    else:  # legacy /5 schema
        for key, label in (("story", "Story"), ("direction", "Direction"),
                            ("technical", "Technical"), ("originality", "Originality")):
            sec = analysis.get(key) or {}
            if sec.get("score") is not None:
                rows.append((label, f"{sec['score']}/5"))
    overall = analysis.get("overall_rating")
    if overall is None:
        overall = analysis.get("overall_score")
    if not rows and overall is None:
        return ""
    body = "".join(
        f"<tr><td style='padding:7px 0;border-bottom:1px solid #eee'>{html.escape(l)}</td>"
        f"<td style='padding:7px 0;border-bottom:1px solid #eee;text-align:right;font-weight:600'>{html.escape(v)}</td></tr>"
        for l, v in rows
    )
    overall_denom = "/10" if ratings else "/20"
    overall_row = ""
    if overall is not None:
        try:
            ov = f"{float(overall):.1f}"
        except (TypeError, ValueError):
            ov = str(overall)
        overall_row = (
            f"<tr><td style='padding:10px 0 0;font-weight:700'>Overall</td>"
            f"<td style='padding:10px 0 0;text-align:right;font-weight:700;color:#9a7d3c'>{ov}{overall_denom}</td></tr>"
        )
    return (
        "<table style=\"width:100%;border-collapse:collapse;margin:20px 0;"
        "font-family:Arial,sans-serif;font-size:14px\"><tbody>"
        f"{body}{overall_row}</tbody></table>"
    )


def build_context(film: dict, review: dict, festival: dict) -> dict:
    """Assemble placeholder values from a film + review + festival config."""
    analysis = (film or {}).get("analysis") or {}
    runtime = (film or {}).get("runtime") or ""
    runtime_str = f"{runtime} min" if runtime else ""
    overall = review.get("overall_rating")
    if overall is None:
        overall = analysis.get("overall_rating")
    try:
        overall_str = f"{float(overall):.1f}" if overall is not None else ""
    except (TypeError, ValueError):
        overall_str = str(overall or "")
    return {
        "title":                html.escape((film or {}).get("title", "") or ""),
        "director":             html.escape((film or {}).get("director", "") or ""),
        "genre":                html.escape((film or {}).get("genre", "") or ""),
        "runtime":              html.escape(runtime_str),
        "country":              html.escape((film or {}).get("country", "") or ""),
        "festival_name":        html.escape((festival or {}).get("name", "") or ""),
        "season":               html.escape(review.get("season", "") or ""),
        "overall_rating":       html.escape(overall_str),
        "scores_table":         _scores_table(analysis),
        "standout":             html.escape(analysis.get("standout_moment", "") or ""),
        "growth_area":          html.escape(analysis.get("weakest_element", "") or ""),
        "festival_suitability": html.escape(analysis.get("festival_suitability", "") or ""),
        "review_body":          _paras(review.get("review_text", "")),
        "date":                 datetime.now().strftime("%B %Y"),
    }


def render_template(template_html: str, ctx: dict) -> str:
    """Substitute {{token}} placeholders literally. Unknown tokens → blank.
    No eval/Jinja — the template may be admin- or AI-authored, so it is treated
    as untrusted for control flow (values are already escaped in build_context)."""
    out = template_html or DEFAULT_WP_TEMPLATE
    for token in PLACEHOLDERS:
        out = out.replace("{{" + token + "}}", str(ctx.get(token, "")))
        out = out.replace("{{ " + token + " }}", str(ctx.get(token, "")))
    # Strip any leftover unknown placeholders
    out = re.sub(r"\{\{\s*[a-zA-Z0-9_]+\s*\}\}", "", out)
    return out


# ── Publishing ────────────────────────────────────────────────────────────────
def publish(festival: dict, title: str, content_html: str,
            status: str = "draft", post_id=None) -> dict:
    """Create a new post (or update an existing one when post_id is given).
    status: "draft" | "publish". Returns {success, post_id, url, error}."""
    if not wp_configured(festival):
        return {"success": False, "post_id": None, "url": "",
                "error": "WordPress is not configured for this festival."}
    try:
        base = _safe_base_url(festival["wp_url"])
    except ValueError as e:
        return {"success": False, "post_id": None, "url": "", "error": str(e)}

    status = "publish" if status == "publish" else "draft"
    payload = {"title": title, "content": content_html, "status": status}
    endpoint = f"{base}/wp-json/wp/v2/posts"
    if post_id:
        endpoint = f"{endpoint}/{post_id}"
    try:
        r = requests.post(
            endpoint, json=payload,
            headers={**_auth_header(festival), "Content-Type": "application/json"},
            timeout=20,
        )
        r.raise_for_status()
        d = r.json()
        return {"success": True, "post_id": d.get("id"),
                "url": d.get("link", ""), "error": None}
    except requests.HTTPError as e:
        code = getattr(e.response, "status_code", "?")
        detail = ""
        try:
            detail = e.response.json().get("message", "")
        except Exception:
            pass
        hint = ""
        if code in (401, 403):
            hint = " Check the WordPress username and Application Password."
        return {"success": False, "post_id": None, "url": "",
                "error": f"WordPress rejected the request (HTTP {code}). {detail}{hint}".strip()}
    except Exception as e:
        return {"success": False, "post_id": None, "url": "",
                "error": f"Could not reach the WordPress site: {e}"}


# ── AI template generation from a reference design ────────────────────────────
def fetch_reference(url: str) -> str:
    """Fetch an admin-supplied reference post's HTML (http/https only)."""
    base = _safe_base_url(url)
    r = requests.get(base, timeout=15, headers={"User-Agent": "FestivalReviewer/1.0"})
    r.raise_for_status()
    return r.text[:60000]  # cap to keep the model prompt bounded


def generate_template_from_reference(client, model_id: str, reference_html: str) -> str:
    """Ask Gemini to turn a reference design into a reusable HTML template that
    uses ONLY our placeholder tokens. Returns the template HTML string."""
    tokens = ", ".join("{{" + p + "}}" for p in PLACEHOLDERS)
    prompt = f"""You are converting a reference web design into a reusable HTML template for a film-review blog post.

Below is the reference HTML. Produce a SINGLE self-contained HTML fragment (no <html>, <head>, or <body> tags, no markdown fences) that reproduces the reference's visual style using inline CSS, and inserts the review content using ONLY these placeholder tokens:
{tokens}

Rules:
- Use each relevant placeholder exactly as written, e.g. {{{{title}}}}. Do not invent new tokens.
- {{{{scores_table}}}} and {{{{review_body}}}} already contain HTML — insert them as-is (do not wrap their contents in extra <p> tags).
- All other tokens are plain text.
- Keep all styling inline so it renders inside any WordPress theme.
- Output ONLY the HTML fragment.

REFERENCE HTML:
{reference_html}
"""
    resp = client.models.generate_content(model=model_id, contents=prompt)
    text = (getattr(resp, "text", "") or "").strip()
    # Strip accidental code fences
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text).strip()
    return text or DEFAULT_WP_TEMPLATE
