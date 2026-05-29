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
from analyzer import analyse_film, generate_expert_review
import db

db.init_db()

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

    Dedup strategy (cheapest first):
      1. Queue/approved file already exists for this festival → skip entirely
      2. Film found in DB by screener_url or title+director AND analysis exists
         → skip video download + Gemini upload, regenerate review text only
      3. Otherwise → full download + Gemini video analysis
    """
    festival_key = next(k for k, v in FESTIVALS.items() if v is festival)
    entry_id = sub.get("entry_id", "unknown")
    title    = sub.get("title", "Unknown Title")
    print(f"\n[{festival['name']}][{entry_id}] Processing: {title}")

    if submission_already_processed(entry_id, festival_key):
        print(f"  [skip] Already in queue/approved for {festival['name']}")
        return {"skipped": True, "entry_id": entry_id}

    if not sub.get("screener_url"):
        print(f"  [skip] No screener URL")
        return {"skipped": True, "entry_id": entry_id, "reason": "no screener"}

    # ── Fast dedup on screener URL (no download needed) ───────────────────
    existing = db.film_find_by_screener(sub.get("screener_url", ""))
    if existing and existing.get("analysis"):
        film_id = existing["film_id"]
        print(f"  [cache hit] URL match in DB (id={film_id[:8]}) — skipping video upload")
        existing_review = db.review_get(film_id, festival_key)
        if existing_review and existing_review.get("review_text"):
            print(f"  [cache hit] Review already exists for {festival['name']} — skipping entirely")
            return save_to_queue(
                entry_id=entry_id, film_meta=sub,
                analysis=existing["analysis"],
                review_draft=existing_review["review_text"],
                festival_key=festival_key, festival_name=festival["name"],
            )
        review_draft = generate_expert_review(sub, existing["analysis"], festival)
        db.review_upsert(film_id, festival_key, review_draft)
        return save_to_queue(
            entry_id=entry_id, film_meta=sub,
            analysis=existing["analysis"], review_draft=review_draft,
            festival_key=festival_key, festival_name=festival["name"],
        )

    # ── Download video to extract runtime for dedup ────────────────────────
    dl = download_screener(
        screener_url=sub.get("screener_url", ""),
        password=sub.get("screener_password", ""),
        entry_id=entry_id,
    )

    if not dl["success"]:
        return save_to_queue(entry_id, sub, {}, "",
                             festival_key=festival_key,
                             error=f"Download failed for {entry_id}")

    video_path   = dl["path"]
    duration_min = dl["duration_min"]
    runtime_str  = f"{round(duration_min)} min"

    # ── Full dedup: title + director + runtime (±2 min tolerance) ─────────
    existing = db.film_find_by_identity(
        sub.get("title", ""), sub.get("director", ""), duration_min
    )
    if existing and existing.get("analysis"):
        film_id = existing["film_id"]
        print(f"  [cache hit] title+director+runtime match (id={film_id[:8]}) — skipping analysis")
        existing_review = db.review_get(film_id, festival_key)
        if existing_review and existing_review.get("review_text"):
            print(f"  [cache hit] Review already exists for {festival['name']} — skipping entirely")
            return save_to_queue(
                entry_id=entry_id, film_meta={**sub, "runtime": runtime_str},
                analysis=existing["analysis"],
                review_draft=existing_review["review_text"],
                festival_key=festival_key, festival_name=festival["name"],
            )
        review_draft = generate_expert_review(
            {**sub, "runtime": runtime_str}, existing["analysis"], festival
        )
        db.review_upsert(film_id, festival_key, review_draft)
        return save_to_queue(
            entry_id=entry_id, film_meta={**sub, "runtime": runtime_str},
            analysis=existing["analysis"], review_draft=review_draft,
            festival_key=festival_key, festival_name=festival["name"],
        )

    # ── Full analysis path ─────────────────────────────────────────────────
    frame_paths = []
    transcript  = ""

    if duration_min > SHORT_FILM_MAX_MIN:
        print(f"  [long film] {duration_min:.1f} min — extracting frames + transcript")
        frame_paths = extract_keyframes(video_path, entry_id)
        transcript  = transcribe_audio(video_path, entry_id)
    else:
        print(f"  [short film] {duration_min:.1f} min — direct Gemini upload")

    result = analyse_film(
        video_path=video_path,
        duration_min=duration_min,
        film_meta={**sub, "runtime": runtime_str},
        festival=festival,
        frame_paths=frame_paths,
        transcript=transcript,
    )

    if result["error"]:
        return save_to_queue(entry_id, sub, {}, "",
                             festival_key=festival_key, error=result["error"])

    # Persist film + review to DB so future festivals can skip the video
    film_id = db.film_create({
        "title":              sub.get("title", ""),
        "director":           sub.get("director", ""),
        "director_statement": sub.get("director_statement", ""),
        "genre":              sub.get("genre", ""),
        "runtime":            runtime_str,
        "runtime_min":        duration_min,
        "country":            sub.get("country", ""),
        "screener_url":       sub.get("screener_url", ""),
        "analysis":           result["analysis"],
    })

    db.review_upsert(film_id, festival_key, result["review_draft"])

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
