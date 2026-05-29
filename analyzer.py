"""
analyzer.py — Gemini-powered film analysis and review generation.

Every public function accepts a `festival` config dict from festivals.py
so each festival can use its own Gemini API key and judging criteria.
"""

import time
import json
import re
import google.generativeai as genai
from pathlib import Path
from config import SHORT_FILM_MAX_MIN
from prompts import (
    build_analysis_prompt,
    build_long_film_analysis_prompt,
    build_review_prompt,
)


def _get_model(festival: dict, model_override: str = None) -> genai.GenerativeModel:
    """Configure Gemini with the festival's API key and return a model instance."""
    genai.configure(api_key=festival["gemini_api_key"])
    model_name = model_override or festival.get("gemini_model", "gemini-2.0-flash")
    return genai.GenerativeModel(model_name)


# ── Short film: direct video upload ──────────────────────────────────────────

def analyse_short_film(video_path: str, film_meta: dict, festival: dict) -> dict:
    """
    Direct Gemini video upload for films under SHORT_FILM_MAX_MIN.
    Uses the festival's Gemini API key and analysis prompt.
    """
    print(f"  [gemini:{festival['name']}] Uploading {Path(video_path).name}...")
    genai.configure(api_key=festival["gemini_api_key"])
    video_file = genai.upload_file(path=video_path, mime_type="video/mp4")

    while video_file.state.name == "PROCESSING":
        print("  [gemini] Processing video...", end="\r")
        time.sleep(8)
        video_file = genai.get_file(video_file.name)

    if video_file.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini video processing failed: {video_file.state.name}")

    print(f"  [gemini] Video ready — analysing for {festival['name']}...")
    model = _get_model(festival)
    prompt = build_analysis_prompt(festival)

    response = model.generate_content(
        [video_file, prompt],
        generation_config=genai.types.GenerationConfig(
            temperature=0.2,
            max_output_tokens=2048,
        )
    )

    try:
        genai.delete_file(video_file.name)
    except Exception:
        pass

    return _parse_analysis(response.text)


# ── Long film: keyframes + transcript ────────────────────────────────────────

def analyse_long_film(frame_paths: list[str], transcript: str,
                      film_meta: dict, festival: dict) -> dict:
    """
    For features > SHORT_FILM_MAX_MIN: analyse keyframes + whisper transcript.
    Uses the festival's Gemini API key and analysis prompt.
    """
    print(f"  [gemini:{festival['name']}] Analysing {len(frame_paths)} keyframes + transcript...")
    genai.configure(api_key=festival["gemini_api_key"])
    model = _get_model(festival, model_override="gemini-1.5-pro")

    sampled = frame_paths[::max(1, len(frame_paths) // 120)]
    print(f"  [gemini] Using {len(sampled)} sampled frames")

    parts = []
    uploaded_files = []

    for frame_path in sampled:
        f = genai.upload_file(path=frame_path, mime_type="image/jpeg")
        parts.append(f)
        uploaded_files.append(f)

    prompt = build_long_film_analysis_prompt(festival)
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

    for f in uploaded_files:
        try:
            genai.delete_file(f.name)
        except Exception:
            pass

    return _parse_analysis(response.text)


# ── Expert Review generation ──────────────────────────────────────────────────

def generate_expert_review(film_meta: dict, analysis: dict, festival: dict) -> str:
    """
    Generate the polished Expert Review using the festival's Gemini key,
    word count, tone, and review guidelines.
    """
    model = _get_model(festival)
    prompt = build_review_prompt(film_meta, analysis, festival)

    response = model.generate_content(
        prompt,
        generation_config=genai.types.GenerationConfig(
            temperature=0.7,
            max_output_tokens=800,
        )
    )
    return response.text.strip()


# ── Full pipeline for one submission ─────────────────────────────────────────

def analyse_film(video_path: str, duration_min: float,
                 film_meta: dict, festival: dict,
                 frame_paths: list = None, transcript: str = "") -> dict:
    """
    Route to correct analyser based on film duration.
    festival: config dict from festivals.py
    Returns: {"analysis": dict, "review_draft": str, "error": str|None}
    """
    try:
        if duration_min <= SHORT_FILM_MAX_MIN:
            analysis = analyse_short_film(video_path, film_meta, festival)
        else:
            if not frame_paths:
                raise ValueError("Frame paths required for long film analysis")
            analysis = analyse_long_film(frame_paths, transcript, film_meta, festival)

        review_draft = generate_expert_review(film_meta, analysis, festival)
        return {"analysis": analysis, "review_draft": review_draft, "error": None}

    except Exception as e:
        print(f"  [error] Analysis failed: {e}")
        return {"analysis": {}, "review_draft": "", "error": str(e)}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _parse_analysis(raw_text: str) -> dict:
    """Parse Gemini JSON response, stripping markdown fences if present."""
    clean = raw_text.strip()
    clean = re.sub(r"^```(?:json)?\s*", "", clean)
    clean = re.sub(r"\s*```$", "", clean)
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        match = re.search(r'\{.*\}', clean, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise ValueError(f"Could not parse Gemini response as JSON:\n{raw_text[:500]}")
