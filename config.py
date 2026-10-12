"""
config.py — global infrastructure settings shared across all festivals.

Festival-specific settings (Gemini keys, WordPress creds, judging prompts,
word counts, tones) live in festivals.py — not here.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Gemini (global fallback — override per festival in festivals.py) ──────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL     = "gemini-3.8-flash"       # film analysis (video)
REVIEW_MODEL     = "gemini-3.1-flash-lite"  # text-only review writing
# Models Google has closed to new projects. Festival records saved before the
# switch still name them, so they are mapped to the current model on load.
RETIRED_MODELS   = {"gemini-2.5-flash": GEMINI_MODEL, "gemini-2.5-flash-lite": REVIEW_MODEL,
                    "gemini-1.5-pro": GEMINI_MODEL, "gemini-1.5-flash": GEMINI_MODEL}
GEMINI_MODEL_PRO = "gemini-1.5-pro"

# ── Film length thresholds ────────────────────────────────────────────────────
DIRECT_UPLOAD_MAX_MIN = 120    # ≤ 120 min → direct Gemini Files API upload
FILM_MAX_MIN          = 240    # YouTube links and browser uploads: longer films are
                               # watched in 120-min batches with a shared recap
GEMINI_FILE_MAX_MB    = 1900   # Gemini Files API takes up to 2 GB per file
FEATURE_MIN           = 120    # > 120 min → GCS gs:// URI (or chunked keyframes fallback)
LONG_FILM_FRAME_FPS   = "1/10" # 1 keyframe every 10s for chunked fallback
CHUNK_FRAMES          = 80     # max frames per Gemini call in chunked mode
SHORT_FILM_MAX_MIN    = 40     # kept for backward compat with analyzer.py

# ── Upload limits ─────────────────────────────────────────────────────────────
MAX_UPLOAD_MB = 1800           # stay under 2 GB Gemini limit

# ── Local directory paths ─────────────────────────────────────────────────────
DOWNLOADS_DIR = "downloads"
FRAMES_DIR    = "frames"
QUEUE_DIR     = "reviews_queue"
APPROVED_DIR  = "approved_reviews"
