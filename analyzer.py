"""
analyzer.py — Gemini-powered film analysis and review generation.

Uses the new google-genai SDK (google.generativeai is deprecated).
Every public function accepts a `festival` config dict from festivals.py
so each festival can use its own Gemini API key and judging criteria.
"""

import time
import json
import re
from pathlib import Path
from google import genai
from google.genai import types

from config import SHORT_FILM_MAX_MIN
from prompts import build_analysis_prompt, build_long_film_analysis_prompt, build_review_prompt


def _client(festival: dict) -> genai.Client:
    return genai.Client(api_key=festival["gemini_api_key"])


def _model(festival: dict) -> str:
    return festival.get("gemini_model", "gemini-2.0-flash")


# ── Short film: direct video upload ──────────────────────────────────────────

def analyse_short_film(video_path: str, film_meta: dict, festival: dict) -> dict:
    client = _client(festival)
    print(f"  [gemini:{festival['name']}] Uploading {Path(video_path).name}...")

    uploaded = client.files.upload(file=video_path)

    print("  [gemini] Waiting for video processing...")
    while uploaded.state.name == "PROCESSING":
        time.sleep(8)
        uploaded = client.files.get(name=uploaded.name)

    if uploaded.state.name != "ACTIVE":
        raise RuntimeError(f"Gemini video processing failed: {uploaded.state.name}")

    print(f"  [gemini] Analysing for {festival['name']}...")
    response = client.models.generate_content(
        model=_model(festival),
        contents=[uploaded, build_analysis_prompt(festival)],
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=1024),
    )

    try:
        client.files.delete(name=uploaded.name)
    except Exception:
        pass

    return _parse_analysis(response.text)


# ── Long film: keyframes + transcript ────────────────────────────────────────

def analyse_long_film(frame_paths: list[str], transcript: str,
                      film_meta: dict, festival: dict) -> dict:
    client = _client(festival)
    print(f"  [gemini:{festival['name']}] Analysing {len(frame_paths)} keyframes + transcript...")

    sampled = frame_paths[::max(1, len(frame_paths) // 120)]
    print(f"  [gemini] Using {len(sampled)} sampled frames")

    uploaded_files = []
    parts = []
    for frame_path in sampled:
        f = client.files.upload(file=frame_path)
        parts.append(f)
        uploaded_files.append(f)

    prompt = build_long_film_analysis_prompt(festival)
    if transcript:
        prompt += f"\n\nTRANSCRIPT (first 4000 chars):\n{transcript[:4000]}"
    parts.append(prompt)

    response = client.models.generate_content(
        model="gemini-1.5-pro",
        contents=parts,
        config=types.GenerateContentConfig(temperature=0.2, max_output_tokens=2048),
    )

    for f in uploaded_files:
        try:
            client.files.delete(name=f.name)
        except Exception:
            pass

    return _parse_analysis(response.text)


# ── Expert Review generation ──────────────────────────────────────────────────

def generate_expert_review(film_meta: dict, analysis: dict, festival: dict) -> str:
    client = _client(festival)
    response = client.models.generate_content(
        model=_model(festival),
        contents=build_review_prompt(film_meta, analysis, festival),
        config=types.GenerateContentConfig(temperature=0.7, max_output_tokens=800),
    )
    return response.text.strip()


# ── Full pipeline for one submission ─────────────────────────────────────────

def analyse_film(video_path: str, duration_min: float,
                 film_meta: dict, festival: dict,
                 frame_paths: list = None, transcript: str = "") -> dict:
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
