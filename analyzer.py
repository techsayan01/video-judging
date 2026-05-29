import time
import json
import re
import google.generativeai as genai
from pathlib import Path
from config import (GEMINI_API_KEY, GEMINI_MODEL, GEMINI_MODEL_PRO,
                    SHORT_FILM_MAX_MIN, REVIEW_WORD_COUNT,
                    FESTIVAL_NAME, REVIEWER_NAME)
from prompts import (ANALYSIS_PROMPT, LONG_FILM_ANALYSIS_PROMPT,
                     expert_review_prompt)

genai.configure(api_key=GEMINI_API_KEY)


# ── Short film: direct video upload ──────────────────────────────────────────

def analyse_short_film(video_path: str, film_meta: dict) -> dict:
    """
    Direct Gemini video upload for films under SHORT_FILM_MAX_MIN.
    Returns structured analysis dict.
    """
    print(f"  [gemini] Uploading {Path(video_path).name}...")
    video_file = genai.upload_file(path=video_path, mime_type="video/mp4")

    # Poll until Gemini finishes processing
    while video_file.state.name == "PROCESSING":
        print("  [gemini] Processing video...", end="\r")
        time.sleep(8)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini video processing failed: {video_file.state.name}")

    print(f"  [gemini] Video ready — analysing...")
    model = genai.GenerativeModel(GEMINI_MODEL)

    response = model.generate_content(
        [video_file, ANALYSIS_PROMPT],
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=2048,
        )
    )

    # Clean up uploaded file to save Gemini storage
    try:
        genai.delete_file(video_file.name)
    except Exception:
        pass

    return _parse_analysis(response.text)


# ── Long film: keyframes + transcript ────────────────────────────────────────

def analyse_long_film(frame_paths: list[str], transcript: str,
                      film_meta: dict) -> dict:
    """
    For features > SHORT_FILM_MAX_MIN: analyse keyframes + whisper transcript.
    """
    print(f"  [gemini] Analysing {len(frame_paths)} keyframes + transcript...")
    model = genai.GenerativeModel(GEMINI_MODEL_PRO)

    # Upload keyframes (sample max 120 frames to control cost)
    sampled = frame_paths[::max(1, len(frame_paths)//120)]
    print(f"  [gemini] Using {len(sampled)} sampled frames")

    parts = []
    uploaded_files = []

    for frame_path in sampled:
        f = genai.upload_file(path=frame_path, mime_type="image/jpeg")
        parts.append(f)
        uploaded_files.append(f)

    # Add transcript if available
    prompt = LONG_FILM_ANALYSIS_PROMPT
    if transcript:
        prompt += f"\n\nTRANSCRIPT (first 4000 chars):\n{transcript[:4000]}"

    parts.append(prompt)

    response = model.generate_content(
        parts,
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=2048,
        )
    )

    # Clean up uploaded frames
    for f in uploaded_files:
        try:
            genai.delete_file(f.name)
        except Exception:
            pass

    return _parse_analysis(response.text)


# ── Expert Review generation ─────────────────────────────────────────────────

def generate_expert_review(film_meta: dict, analysis: dict) -> str:
    """
    Generate polished Expert Review from film metadata + analysis.
    This is what the filmmaker receives after your 2-min approval.
    """
    model = genai.GenerativeModel(GEMINI_MODEL)
    prompt = expert_review_prompt(
        film_meta, analysis,
        word_count=REVIEW_WORD_COUNT,
        festival_name=FESTIVAL_NAME
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,       # slightly higher for natural prose
            max_output_tokens=800,
        )
    )

    return response.text.strip()


# ── Full pipeline for one submission ─────────────────────────────────────────

def analyse_film(video_path: str, duration_min: float,
                 film_meta: dict, frame_paths: list = None,
                 transcript: str = "") -> dict:
    """
    Route to correct analyser based on film duration.
    Returns: {"analysis": dict, "review_draft": str}
    """
    try:
        if duration_min <= SHORT_FILM_MAX_MIN:
            analysis = analyse_short_film(video_path, film_meta)
        else:
            if not frame_paths:
                raise ValueError("Frame paths required for long film analysis")
            analysis = analyse_long_film(frame_paths, transcript, film_meta)

        review_draft = generate_expert_review(film_meta, analysis)
        return {"analysis": analysis, "review_draft": review_draft, "error": None}

    except Exception as e:
        print(f"  [error] Analysis failed: {e}")
        return {"analysis": {}, "review_draft": "", "error": str(e)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_analysis(raw_text: str) -> dict:
    """Parse Gemini JSON response — handles markdown code fences."""
    clean = raw_text.strip()
    # Strip markdown fences if present
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)

    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        # Last resort: extract JSON block
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse Gemini response as JSON:\n{raw_text[:500]}")
