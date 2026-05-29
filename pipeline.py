"""
pipeline.py — batch processing of FilmFreeway CSV submissions.

Usage:
    python pipeline.py submissions.csv elegantiff
    python pipeline.py submissions.csv shortwave --limit 5

Each run is scoped to one festival. The festival key controls which
Gemini key, WordPress credentials, and judging prompts are used.
"""

import os
import json
import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

from config import DOWNLOADS_DIR, FRAMES_DIR, QUEUE_DIR, APPROVED_DIR, SHORT_FILM_MAX_MIN
from festivals import FESTIVALS, DEFAULT_FESTIVAL, get_festival
from downloader import download_screener, extract_keyframes, transcribe_audio
from analyzer import analyse_film

os.makedirs(QUEUE_DIR, exist_ok=True)
os.makedirs(APPROVED_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)


# ── FilmFreeway CSV column mapping ────────────────────────────────────────────

FF_COLUMNS = {
    "entry_id":           "Entry #",
    "title":              "Project Title",
    "director":           "Director",
    "genre":              "Primary Genre",
    "runtime":            "Runtime (minutes)",
    "country":            "Country",
    "language":           "Primary Language",
    "synopsis":           "Synopsis",
    "director_statement": "Director's Statement",
    "screener_url":       "Screener URL",
    "screener_password":  "Screener Password",
    "status":             "Current Status",
    "email":              "Contact Email",
}


def load_filmfreeway_csv(csv_path: str, filter_status: str = None) -> list[dict]:
    df = pd.read_csv(csv_path, dtype=str).fillna("")
    col_map = {v: k for k, v in FF_COLUMNS.items() if v in df.columns}
    df = df.rename(columns=col_map)
    if filter_status and "status" in df.columns:
        df = df[df["status"] == filter_status]
    submissions = df.to_dict("records")
    print(f"[csv] Loaded {len(submissions)} submissions from {csv_path}")
    return submissions


def submission_already_processed(entry_id: str, festival_key: str) -> bool:
    """Check if this submission was already processed for this festival."""
    slug = f"{festival_key}__{entry_id}"
    return (
        (Path(QUEUE_DIR) / f"{slug}.json").exists() or
        (Path(APPROVED_DIR) / f"{slug}.json").exists()
    )


def process_submission(sub: dict, festival: dict) -> dict:
    """
    Full pipeline for one submission under a specific festival.
    Returns the result dict saved to QUEUE_DIR.
    """
    festival_key = next(k for k, v in FESTIVALS.items() if v is festival)
    entry_id = sub.get("entry_id", "unknown")
    title = sub.get("title", "Unknown Title")
    print(f"\n[{festival['name']}][{entry_id}] Processing: {title}")

    if submission_already_processed(entry_id, festival_key):
        print(f"  [skip] Already in queue/approved for {festival['name']}")
        return {"skipped": True, "entry_id": entry_id}

    if not sub.get("screener_url"):
        print(f"  [skip] No screener URL")
        return {"skipped": True, "entry_id": entry_id, "reason": "no screener"}

    dl = download_screener(
        screener_url=sub.get("screener_url", ""),
        password=sub.get("screener_password", ""),
        entry_id=entry_id,
    )

    if not dl["success"]:
        return save_to_queue(entry_id, sub, {}, "",
                             festival_key=festival_key,
                             error=f"Download failed for {entry_id}")

    video_path = dl["path"]
    duration_min = dl["duration_min"]
    frame_paths = []
    transcript = ""

    if duration_min > SHORT_FILM_MAX_MIN:
        print(f"  [long film] {duration_min:.1f} min — extracting frames + transcript")
        frame_paths = extract_keyframes(video_path, entry_id)
        transcript = transcribe_audio(video_path, entry_id)
    else:
        print(f"  [short film] {duration_min:.1f} min — direct Gemini upload")

    result = analyse_film(
        video_path=video_path,
        duration_min=duration_min,
        film_meta=sub,
        festival=festival,
        frame_paths=frame_paths,
        transcript=transcript,
    )

    if result["error"]:
        return save_to_queue(entry_id, sub, {}, "",
                             festival_key=festival_key, error=result["error"])

    return save_to_queue(
        entry_id=entry_id,
        film_meta=sub,
        analysis=result["analysis"],
        review_draft=result["review_draft"],
        festival_key=festival_key,
        festival_name=festival["name"],
    )


def save_to_queue(entry_id: str, film_meta: dict,
                  analysis: dict, review_draft: str,
                  festival_key: str = DEFAULT_FESTIVAL,
                  festival_name: str = "",
                  error: str = None) -> dict:
    """Save processed result to QUEUE_DIR for the approval UI."""
    slug = f"{festival_key}__{entry_id}"
    record = {
        "entry_id":      entry_id,
        "slug":          slug,
        "festival_key":  festival_key,
        "festival_name": festival_name or FESTIVALS.get(festival_key, {}).get("name", ""),
        "processed_at":  datetime.now().isoformat(),
        "film": {
            "title":              film_meta.get("title", ""),
            "director":           film_meta.get("director", ""),
            "genre":              film_meta.get("genre", ""),
            "runtime":            film_meta.get("runtime", ""),
            "country":            film_meta.get("country", ""),
            "synopsis":           film_meta.get("synopsis", ""),
            "director_statement": film_meta.get("director_statement", ""),
            "email":              film_meta.get("email", ""),
        },
        "analysis":      analysis,
        "review_draft":  review_draft,
        "review_final":  "",
        "approved":      False,
        "approved_at":   None,
        "error":         error,
    }

    path = Path(QUEUE_DIR) / f"{slug}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    status = "error" if error else "queued"
    print(f"  [{status}] Saved → {path}")
    return record


# ── Main entry point ──────────────────────────────────────────────────────────

def run_pipeline(csv_path: str, festival_key: str = DEFAULT_FESTIVAL,
                 filter_status: str = None, limit: int = None):
    """
    Process all submissions from a FilmFreeway CSV export for one festival.

    Args:
        csv_path:      Path to FilmFreeway CSV export
        festival_key:  Key from festivals.FESTIVALS (e.g. "elegantiff")
        filter_status: Only process submissions with this status (e.g. "Submitted")
        limit:         Process only first N submissions (for testing)
    """
    festival = get_festival(festival_key)
    submissions = load_filmfreeway_csv(csv_path, filter_status)

    if limit:
        submissions = submissions[:limit]

    print(f"\n{'='*55}")
    print(f"Festival:   {festival['full_name']}")
    print(f"Gemini key: {'set' if festival['gemini_api_key'] else 'MISSING'}")
    print(f"WP site:    {festival['wp_url'] or 'not configured'}")
    print(f"Processing: {len(submissions)} submissions")
    print(f"{'='*55}")

    results = {"queued": 0, "skipped": 0, "errors": 0}

    for sub in submissions:
        try:
            result = process_submission(sub, festival)
            if result.get("skipped"):
                results["skipped"] += 1
            elif result.get("error"):
                results["errors"] += 1
            else:
                results["queued"] += 1
        except Exception as e:
            print(f"  [fatal] {sub.get('entry_id')}: {e}")
            results["errors"] += 1

    print(f"\n{'='*55}")
    print(f"DONE — Queued: {results['queued']} | "
          f"Skipped: {results['skipped']} | Errors: {results['errors']}")
    print(f"Open http://localhost:5000 to review and approve")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    csv_file    = sys.argv[1] if len(sys.argv) > 1 else "submissions.csv"
    fest_key    = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_FESTIVAL
    run_pipeline(csv_file, fest_key)
