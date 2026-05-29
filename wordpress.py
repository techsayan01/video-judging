"""
wordpress.py — WordPress REST API integration.

All functions accept a `festival` config dict from festivals.py
so each festival publishes to its own WordPress site with its own credentials.

WordPress setup per site:
  Users > Profile > Application Passwords > Add New
  Copy the generated password → set as SLUG_WP_PASS env var
"""

import requests
from base64 import b64encode
from datetime import datetime


def _auth_header(festival: dict) -> dict:
    token = b64encode(
        f"{festival['wp_user']}:{festival['wp_app_pass']}".encode()
    ).decode()
    return {"Authorization": f"Basic {token}"}


def _build_content(meta: dict, analysis: dict, review: str, festival: dict) -> str:
    cats = [
        ("Story",       analysis["story"]["score"],       analysis["story"]["notes"]),
        ("Direction",   analysis["direction"]["score"],   analysis["direction"]["notes"]),
        ("Technical",   analysis["technical"]["score"],   analysis["technical"]["notes"]),
        ("Originality", analysis["originality"]["score"], analysis["originality"]["notes"]),
    ]
    rows = "".join(
        f"<tr><td><strong>{c}</strong></td><td>{s}/5</td>"
        f"<td style='color:#666;font-size:13px'>{n}</td></tr>"
        for c, s, n in cats
    )
    paragraphs = "".join(
        f"<p>{p}</p>" for p in review.strip().split("\n") if p.strip()
    )
    festival_name = festival["name"]
    return f"""
<div class="festival-review">
<p style="font-size:13px;color:#888;letter-spacing:1px;
   text-transform:uppercase;font-family:monospace">
  {festival_name} &mdash; Official Expert Review</p>
<p style="font-size:13px;color:#888">
  {meta.get('genre', '')}
  {'&nbsp;&middot;&nbsp;' + str(meta.get('runtime', '')) + ' min' if meta.get('runtime') else ''}
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
  Reviewed by {festival_name} Programming Team &nbsp;&middot;&nbsp;
  {datetime.now().strftime('%B %Y')}</p>
</div>"""


def publish_review(meta: dict, analysis: dict, review: str,
                   festival: dict, status: str = "draft") -> dict:
    """
    Publish a review to the festival's WordPress site.
    festival: config dict from festivals.py
    status: "draft" or "publish"
    """
    if not festival.get("wp_url"):
        return {"success": False, "post_id": None, "url": "",
                "error": f"No WP_URL configured for {festival['name']}"}

    payload = {
        "title":   f"{meta.get('title', 'Untitled')} — {festival['name']} Expert Review",
        "content": _build_content(meta, analysis, review, festival),
        "status":  status,
    }
    try:
        r = requests.post(
            f"{festival['wp_url']}/wp-json/wp/v2/posts",
            json=payload,
            headers={**_auth_header(festival), "Content-Type": "application/json"},
            timeout=15,
        )
        r.raise_for_status()
        d = r.json()
        return {"success": True, "post_id": d["id"],
                "url": d.get("link", ""), "error": None}
    except Exception as e:
        return {"success": False, "post_id": None, "url": "", "error": str(e)}


def publish_post(post_id: int, festival: dict) -> dict:
    """Flip an existing draft post to published."""
    if not festival.get("wp_url"):
        return {"success": False, "url": "",
                "error": f"No WP_URL configured for {festival['name']}"}
    try:
        r = requests.post(
            f"{festival['wp_url']}/wp-json/wp/v2/posts/{post_id}",
            json={"status": "publish"},
            headers={**_auth_header(festival), "Content-Type": "application/json"},
            timeout=10,
        )
        r.raise_for_status()
        return {"success": True, "url": r.json().get("link", ""), "error": None}
    except Exception as e:
        return {"success": False, "url": "", "error": str(e)}
