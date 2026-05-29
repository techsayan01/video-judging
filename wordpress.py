"""
WordPress REST API integration for Festival Review App.
Publishes approved reviews to elegantiff.com automatically.

Setup on WordPress side:
1. Users > Your Profile > Application Passwords > Add New
2. Name it "Festival Review App" -> copy the generated password
3. Add to GCP Secret Manager as 'wp-app-password'
"""

import os, requests
from base64 import b64encode
from datetime import datetime

WP_URL      = os.getenv("WP_URL", "https://elegantiff.com")
WP_USER     = os.getenv("WP_USER", "admin")
WP_APP_PASS = os.getenv("WP_APP_PASS", "")

def _auth_header() -> dict:
    token = b64encode(f"{WP_USER}:{WP_APP_PASS}".encode()).decode()
    return {"Authorization": f"Basic {token}"}

def _build_content(meta: dict, analysis: dict, review: str, festival: str) -> str:
    cats = [
        ("Story",       analysis["story"]["score"],       analysis["story"]["notes"]),
        ("Direction",   analysis["direction"]["score"],   analysis["direction"]["notes"]),
        ("Technical",   analysis["technical"]["score"],   analysis["technical"]["notes"]),
        ("Originality", analysis["originality"]["score"], analysis["originality"]["notes"]),
    ]
    rows = "".join(
        f"<tr><td><strong>{c}</strong></td><td>{s}/5</td>"
        f"<td style='color:#666;font-size:13px'>{n}</td></tr>"
        for c,s,n in cats
    )
    paragraphs = "".join(f"<p>{p}</p>" for p in review.strip().split("\n") if p.strip())
    return f"""
<div class="festival-review">
<p style="font-size:13px;color:#888;letter-spacing:1px;
   text-transform:uppercase;font-family:monospace">
  {festival} &mdash; Official Expert Review</p>
<p style="font-size:13px;color:#888">
  {meta.get('genre','')}
  {'&nbsp;&middot;&nbsp;' + meta.get('runtime','') + ' min' if meta.get('runtime') else ''}
</p>
<table style="width:100%;border-collapse:collapse;margin:24px 0;
   font-family:monospace;font-size:14px">
  <thead><tr style="border-bottom:1px solid #eee">
    <th style="text-align:left;padding:8px 0">Category</th>
    <th style="text-align:left;padding:8px 0">Score</th>
    <th style="text-align:left;padding:8px 0">Notes</th>
  </tr></thead>
  <tbody>{rows}</tbody>
  <tfoot><tr style="border-top:2px solid #eee">
    <td><strong>Overall</strong></td>
    <td><strong>{analysis['overall_score']}/20</strong></td>
    <td></td>
  </tr></tfoot>
</table>
<blockquote style="border-left:3px solid #C9A84C;padding-left:16px;
   margin:20px 0;font-style:italic;color:#444">
  <strong>Standout:</strong> {analysis['standout_moment']}
</blockquote>
<div style="line-height:1.8;font-size:15px">{paragraphs}</div>
<p style="margin-top:32px;font-size:12px;color:#aaa;font-family:monospace">
  Reviewed by {festival} Programming Team &nbsp;&middot;&nbsp;
  {datetime.now().strftime('%B %Y')}</p>
</div>"""

def publish_review(meta, analysis, review, festival, status="draft"):
    payload = {
        "title":   f"{meta.get('title','Untitled')} \u2014 {festival} Expert Review",
        "content": _build_content(meta, analysis, review, festival),
        "status":  status,
    }
    try:
        r = requests.post(
            f"{WP_URL}/wp-json/wp/v2/posts", json=payload,
            headers={**_auth_header(), "Content-Type": "application/json"},
            timeout=15)
        r.raise_for_status()
        d = r.json()
        return {"success": True, "post_id": d["id"],
                "url": d.get("link",""), "error": None}
    except Exception as e:
        return {"success": False, "post_id": None, "url": "", "error": str(e)}

def publish_post(post_id):
    try:
        r = requests.post(
            f"{WP_URL}/wp-json/wp/v2/posts/{post_id}",
            json={"status": "publish"},
            headers={**_auth_header(), "Content-Type": "application/json"},
            timeout=10)
        r.raise_for_status()
        return {"success": True, "url": r.json().get("link",""), "error": None}
    except Exception as e:
        return {"success": False, "url": "", "error": str(e)}
