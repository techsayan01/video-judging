# Festival Review Pipeline

Automates Expert Review generation for film festival submissions.
Built for Apple M4 Pro · Uses Gemini 2.0 Flash for video analysis.

## What it does

1. Reads your FilmFreeway CSV export
2. Downloads each screener (Vimeo/Drive/Dropbox via yt-dlp)
3. Sends video to Gemini for structured analysis
4. Generates Expert Review draft
5. Queues everything in a local approval UI
6. You spend 2 minutes reading, editing, and approving

For films > 40 min: extracts keyframes (ffmpeg) + transcribes audio (whisper.cpp)

---

## Setup

### 1. Install dependencies
```bash
cd festival_pipeline
pip install -r requirements.txt
brew install ffmpeg yt-dlp   # if not already installed
```

### 2. Create .env file
```
GEMINI_API_KEY=your_key_here
FESTIVAL_NAME=ElegantIFF
REVIEWER_NAME=Your Name
```

### 3. Export CSV from FilmFreeway
FilmFreeway → your festival → Entries → Export CSV
Save as `submissions.csv` in the festival_pipeline folder.

---

## Running the pipeline

### Process all new submissions:
```bash
python pipeline.py submissions.csv
```

### Process only first 5 (test run):
```python
from pipeline import run_pipeline
run_pipeline("submissions.csv", limit=5)
```

### Process only add-on purchasers:
Filter by checking which entries have "Expert Review" in their add-ons column.
Adjust the `add_on_only` logic in `pipeline.py` based on your CSV structure.

---

## Approval UI

```bash
python app.py
```
Open http://localhost:5000

**For each review you see:**
- Film details (title, director, genre, country)
- Overall score /20 with breakdown
- Key observations (standout moment, weakest element)
- Editable review draft in a text area
- Word count live counter
- Approve button → moves to approved_reviews/

**Your 2 minutes per film:**
1. Scan the analysis scores on the left
2. Read the draft on the right
3. Adjust any emotional or cultural observations that feel generic
4. Click Approve

---

## File structure after running

```
festival_pipeline/
  downloads/          # Downloaded screeners (keep until approved)
  frames/             # Keyframes for long films (auto-cleaned)  
  reviews_queue/      # Pending your approval (JSON)
  approved_reviews/   # Done — copy review text to deliver to filmmaker
  submissions.csv     # Your FilmFreeway export
```

---

## FilmFreeway CSV columns

The pipeline expects these columns (adjust FF_COLUMNS in pipeline.py if yours differ):
- Entry #
- Project Title  
- Director
- Primary Genre
- Runtime (minutes)
- Country
- Primary Language
- Synopsis
- Director's Statement
- Screener URL
- Screener Password
- Current Status
- Contact Email

---

## Cost estimate (Gemini API)

| Film length | Approx cost |
|------------|-------------|
| Short < 10 min | ~$0.05–0.15 |
| Short 10–40 min | ~$0.20–0.50 |
| Feature (keyframes) | ~$0.30–0.80 |

At 89 submissions/month: **~$15–45/month total API cost**

---

## Extending

**To add certificate generation:**
```python
from prompts import certificate_prompt
from analyzer import generate_expert_review
# Use same pattern — prompt → Gemini → approve
```

**To connect with your Canva API pipeline:**
Pass `entry_id` from approved_reviews/ to your existing certificate builder.

**To auto-deliver via FilmFreeway:**
FilmFreeway doesn't have an outbound API for sending messages, but you can 
copy the approved review text into the status change message template.
For bulk delivery, export approved reviews and use your email tool of choice.
