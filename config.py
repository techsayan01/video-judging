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
GEMINI_MODEL     = "gemini-2.5-flash"
GEMINI_MODEL_PRO = "gemini-1.5-pro"

# ── Film length thresholds ────────────────────────────────────────────────────
SHORT_FILM_MAX_MIN  = 40       # under 40 min → direct Gemini video upload
LONG_FILM_FRAME_FPS = "1/10"  # 1 keyframe every 10s for features

# ── Upload limits ─────────────────────────────────────────────────────────────
MAX_UPLOAD_MB = 1800           # stay under 2 GB Gemini limit

# ── Local directory paths ─────────────────────────────────────────────────────
DOWNLOADS_DIR = "downloads"
FRAMES_DIR    = "frames"
QUEUE_DIR     = "reviews_queue"
APPROVED_DIR  = "approved_reviews"
