import os
from dotenv import load_dotenv
load_dotenv()

# ── Gemini ────────────────────────────────────────────────
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL     = "gemini-2.0-flash"          # cost-effective for video
GEMINI_MODEL_PRO = "gemini-1.5-pro"            # fallback for long features

# ── Limits ────────────────────────────────────────────────
SHORT_FILM_MAX_MIN  = 40      # under 40 min → direct Gemini upload
LONG_FILM_FRAME_FPS = "1/10" # 1 keyframe every 10s for features
MAX_UPLOAD_MB       = 1800    # stay under 2GB Gemini limit

# ── Paths ─────────────────────────────────────────────────
DOWNLOADS_DIR  = "downloads"
FRAMES_DIR     = "frames"
QUEUE_DIR      = "reviews_queue"
APPROVED_DIR   = "approved_reviews"

# ── Review settings ───────────────────────────────────────
REVIEW_WORD_COUNT = 300
FESTIVAL_NAME     = os.getenv("FESTIVAL_NAME", "ElegantIFF")
REVIEWER_NAME     = os.getenv("REVIEWER_NAME", "Festival Director")
