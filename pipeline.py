import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime
from config import (DOWNLOADS_DIR, FRAMES_DIR, QUEUE_DIR,
                    APPROVED_DIR, SHORT_FILM_MAX_MIN)
from downloader import download_screener, extract_keyframes, transcribe_audio
from analyzer import analyse_film

os.makedirs(QUEUE_DIR, exist_ok=True)
os.makedirs(APPROVED_DIR, exist_ok=True)
os.makedirs(DOWNLOADS_DIR, exist_ok=True)


# ── FilmFreeway CSV column mapping ────────────────────────────────────────────
# Adjust these if your FilmFreeway export uses different column names

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


def load_filmfreeway_csv(csv_path: str, 
                          filter_status: str = None) -> list[dict]:
    """
    Parse FilmFreeway CSV export into list of submission dicts.
    filter_status: e.g. "Submitted" to only process new submissions
    """
    df = pd.read_csv(csv_path, dtype=str).fillna("")

    # Remap columns
    col_map = {v: k for k, v in FF_COLUMNS.items() if v in df.columns}
    df = df.rename(columns=col_map)

    if filter_status and "status" in df.columns:
        df = df[df["status"] == filter_status]

    submissions = df.to_dict("records")
    print(f"[csv] Loaded {len(submissions)} submissions from {csv_path}")
    return submissions


def submission_already_processed(entry_id: str) -> bool:
    """Check if review already exists in queue or approved."""
    queue_file = Path(QUEUE_DIR) / f"{entry_id}.json"
    approved_file = Path(APPROVED_DIR) / f"{entry_id}.json"
    return queue_file.exists() or approved_file.exists()


def process_submission(sub: dict) -> dict:
    """
    Full pipeline for one submission.
    Returns result dict saved to QUEUE_DIR for your approval.
    """
    entry_id = sub.get("entry_id", "unknown")
    title = sub.get("title", "Unknown Title")
    print(f"\n[{entry_id}] Processing: {title}")

    # Skip if already done
    if submission_already_processed(entry_id):
        print(f"  [skip] Already in queue/approved")
        return {"skipped": True, "entry_id": entry_id}

    # Check if add-on was purchased (you can filter by CSV column)
    # Adjust logic based on how FilmFreeway marks add-on purchases
    # For now: process all submissions that have a screener URL
    if not sub.get("screener_url"):
        print(f"  [skip] No screener URL")
        return {"skipped": True, "entry_id": entry_id, "reason": "no screener"}

    # 1. Download
    dl = download_screener(
        screener_url=sub.get("screener_url", ""),
        password=sub.get("screener_password", ""),
        entry_id=entry_id
    )

    if not dl["success"]:
        return save_to_queue(entry_id, sub, {}, "", 
                             error=f"Download failed for {entry_id}")

    video_path = dl["path"]
    duration_min = dl["duration_min"]

    # 2. Extract frames / transcribe if long film
    frame_paths = []
    transcript = ""

    if duration_min > SHORT_FILM_MAX_MIN:
        print(f"  [long film] {duration_min:.1f} min — extracting frames + transcript")
        frame_paths = extract_keyframes(video_path, entry_id)
        transcript = transcribe_audio(video_path, entry_id)
    else:
        print(f"  [short film] {duration_min:.1f} min — direct Gemini upload")

    # 3. Analyse
    result = analyse_film(
        video_path=video_path,
        duration_min=duration_min,
        film_meta=sub,
        frame_paths=frame_paths,
        transcript=transcript
    )

    if result["error"]:
        return save_to_queue(entry_id, sub, {}, "", error=result["error"])

    # 4. Save to approval queue
    return save_to_queue(
        entry_id=entry_id,
        film_meta=sub,
        analysis=result["analysis"],
        review_draft=result["review_draft"]
    )


def save_to_queue(entry_id: str, film_meta: dict, 
                   analysis: dict, review_draft: str,
                   error: str = None) -> dict:
    """Save processed result to QUEUE_DIR for approval UI."""
    record = {
        "entry_id": entry_id,
        "processed_at": datetime.now().isoformat(),
        "film": {
            "title": film_meta.get("title", ""),
            "director": film_meta.get("director", ""),
            "genre": film_meta.get("genre", ""),
            "runtime": film_meta.get("runtime", ""),
            "country": film_meta.get("country", ""),
            "synopsis": film_meta.get("synopsis", ""),
            "director_statement": film_meta.get("director_statement", ""),
            "email": film_meta.get("email", ""),
        },
        "analysis": analysis,
        "review_draft": review_draft,
        "review_final": "",       # filled in approval UI
        "approved": False,
        "approved_at": None,
        "error": error
    }

    path = Path(QUEUE_DIR) / f"{entry_id}.json"
    path.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    status = "error" if error else "queued"
    print(f"  [{status}] Saved to queue: {path}")
    return record


# ── Main entry point ──────────────────────────────────────────────────────────

def run_pipeline(csv_path: str, filter_status: str = None,
                 limit: int = None, add_on_only: bool = False):
    """
    Process all submissions from a FilmFreeway CSV export.

    Args:
        csv_path:      Path to FilmFreeway CSV export
        filter_status: Only process submissions with this status (e.g. "Submitted")
        limit:         Process only first N submissions (for testing)
        add_on_only:   Only process submissions that purchased Expert Review add-on
    """
    submissions = load_filmfreeway_csv(csv_path, filter_status)

    if limit:
        submissions = submissions[:limit]

    print(f"\n{'='*55}")
    print(f"Processing {len(submissions)} submissions")
    print(f"{'='*55}")

    results = {"queued": 0, "skipped": 0, "errors": 0}

    for sub in submissions:
        try:
            result = process_submission(sub)
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
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "submissions.csv"
    run_pipeline(csv_file)
