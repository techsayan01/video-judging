"""
prompts.py — AI prompt builders for analysis and review generation.

All prompts accept a festival config dict from festivals.py so that
scoring emphasis and review tone can differ per festival.
"""


def _build_document_analysis_prompt(festival, genre, logline, dir_stmt, focus) -> str:
    """Analysis prompt for written submissions (scripts, screenplays, poems, novels).
    Produces the SAME JSON keys as the video analysis so downstream review/display
    code is unchanged, but the four scored dimensions are reinterpreted for the page."""
    cat_prompts  = festival.get("category_prompts", {}) or {}
    cat_emphasis = (cat_prompts.get(genre, "") or "").strip()
    category_block = (
        f"\nCATEGORY-SPECIFIC JUDGING EMPHASIS for the '{genre}' category"
        f" (takes priority):\n{cat_emphasis}"
    ) if cat_emphasis else ""
    ctx = []
    if genre:    ctx.append(f"Category: {genre}")
    if logline:  ctx.append(f"Logline: {logline}")
    if dir_stmt: ctx.append(f"Writer's statement: {dir_stmt}")
    context_block = ("\nWRITER-PROVIDED CONTEXT:\n" + "\n".join(ctx)) if ctx else ""

    return f"""
You are a professional script analyst and senior development executive with 15 years
of experience evaluating screenplays, teleplays, stage plays, and literary writing for
{festival['full_name']}. You are reading a WRITTEN SUBMISSION (a script / screenplay /
poem / novel / written work), NOT a finished film. Judge the writing ON THE PAGE with
the rigour of professional script coverage — cite specific pages, scenes, or lines, and
back every score with evidence. Never penalise it for lacking realised visuals or sound.

FESTIVAL JUDGING EMPHASIS:
{focus}
{category_block}
{context_block}

Identify the written form first (feature screenplay, short script, pilot, stage play,
radio/audio script, poem, novel, etc.) and judge it on the conventions of THAT form.

Score each of the NINE criteria 0–10. Four of them describe a finished film, so judge
the script's POTENTIAL for them as written (say so in the note):
- cinematography  → visual storytelling potential implied by the action/description
- performances    → how actable, distinct, and playable the roles/voices are
- production_value → ambition, scope, and producibility of what's on the page
- sound_music     → how sound, silence, music, or (for poetry) meter and cadence are used

Return ONLY valid JSON — no markdown, no preamble:
{{
  "form": "<the written form you identified>",
  "ratings": {{
    "originality": <0-10>,
    "direction": <0-10>,
    "writing": <0-10>,
    "cinematography": <0-10>,
    "performances": <0-10>,
    "production_value": <0-10>,
    "pacing": <0-10>,
    "structure": <0-10>,
    "sound_music": <0-10>
  }},
  "notes": {{
    "originality": "<2-3 sentences: freshness of premise, voice, what sets it apart>",
    "direction": "<2-3 sentences: directability — how clearly the writing implies staging and vision>",
    "writing": "<2-3 sentences: dialogue, subtext, prose/line craft, theme — the core of the assessment>",
    "cinematography": "<2-3 sentences: visual potential implied by description; cite a page/scene>",
    "performances": "<2-3 sentences: how actable and distinct the roles/voices are>",
    "production_value": "<2-3 sentences: scope, ambition, producibility as written>",
    "pacing": "<2-3 sentences: momentum and rhythm on the page; where it drags or sings>",
    "structure": "<2-3 sentences: architecture, escalation, act/scene/stanza construction>",
    "sound_music": "<2-3 sentences: use of sound/silence/music, or meter & cadence for poetry>"
  }},
  "overall_rating": <0-10, your holistic professional verdict — close to but not mechanically the average>,
  "standout_moment": "<the strongest scene, line, page, or passage (cite a page/scene)>",
  "weakest_element": "<most constructive area for improvement, be specific>",
  "festival_suitability": "<1-2 sentences on fit for {festival['name']}>",
  "recommendation": "<Pass | Recommend | Award Worthy | Maybe>"
}}

CALIBRATION (0-10): 0-2 broken; 3-4 early-stage; 5-6 competent/producible; 7-8 strong &
distinctive; 9-10 exceptional, award-calibre writing. Most solid submissions land 6-8.
RECOMMENDATION must follow overall_rating: 8.5-10 → Award Worthy, 7.0-8.4 → Recommend,
5.0-6.9 → Maybe, below 5 → Pass. Be specific; reference actual lines, scenes, or pages.
"""


def build_analysis_prompt(festival: dict, film_meta: dict | None = None, is_document: bool = False) -> str:
    """
    Build the Gemini analysis prompt for a given festival.
    Returns structured JSON instructions with festival-specific scoring focus.
    Form-aware (judges experimental/art films on their own terms) and intent-aware
    (uses the director's stated intentions where provided).
    When is_document=True, judges a written script/screenplay/literary work on the
    page rather than a video.
    """
    film_meta = film_meta or {}
    focus     = festival.get("analysis_focus", "").strip()
    genre     = (film_meta.get("genre") or "").strip()
    logline   = _cap_words((film_meta.get("logline") or "").strip())
    dir_stmt  = _cap_words((film_meta.get("director_statement") or "").strip())

    if is_document:
        return _build_document_analysis_prompt(festival, genre, logline, dir_stmt, focus)

    # Per-category (genre) judging emphasis for THIS festival, if configured
    cat_prompts   = festival.get("category_prompts", {}) or {}
    cat_emphasis  = (cat_prompts.get(genre, "") or "").strip()
    category_block = (
        f"\nCATEGORY-SPECIFIC JUDGING EMPHASIS for the '{genre}' category"
        f" (takes priority for this submission):\n{cat_emphasis}"
    ) if cat_emphasis else ""

    ctx = []
    if genre:    ctx.append(f"Stated genre/category: {genre}")
    if logline:  ctx.append(f"Logline: {logline}")
    if dir_stmt: ctx.append(f"Director's statement: {dir_stmt}")
    context_block = ("\nFILMMAKER-PROVIDED CONTEXT (use to understand intent):\n" + "\n".join(ctx)) if ctx else ""

    return f"""
You are a professional film critic and senior festival juror with 15 years of
experience across narrative, documentary, experimental, animation, and music-driven
work — the calibre of reviewer published in Variety or Sight & Sound. You are
evaluating a submission for {festival['full_name']}, which focuses on {festival['focus']}.
Watch the ENTIRE film closely and judge it with the rigour, specificity, and craft
vocabulary of a working professional reviewer — cite concrete moments with timestamps,
name techniques precisely, and back every score with observable evidence.

FESTIVAL JUDGING EMPHASIS:
{focus}
{category_block}
{context_block}

STEP 1 — IDENTIFY THE FORM before scoring (narrative short/feature, documentary,
experimental/art film, music video, animation, dance/poetry film, or hybrid), then
judge the film ON ITS OWN TERMS. Do NOT penalise an intentionally non-narrative or
dialogue-free work for lacking plot or dialogue — judge what it is attempting and how
well it achieves it. Where a director's statement is given, assess realisation of the
STATED intent.

STEP 2 — SCORE EACH OF THE NINE CRITERIA from 0 to 10 (whole numbers). For any
criterion that genuinely does not apply to this film's form, award a fair score based
on the closest equivalent craft rather than a low score (and say so in the note).

Return ONLY valid JSON — no markdown, no preamble:
{{
  "form": "<the film type you identified>",
  "ratings": {{
    "originality": <0-10>,
    "direction": <0-10>,
    "writing": <0-10>,
    "cinematography": <0-10>,
    "performances": <0-10>,
    "production_value": <0-10>,
    "pacing": <0-10>,
    "structure": <0-10>,
    "sound_music": <0-10>
  }},
  "notes": {{
    "originality": "<2-3 sentences: freshness of concept, voice, what sets it apart>",
    "direction": "<2-3 sentences: directorial command, staging, visual storytelling, tonal control>",
    "writing": "<2-3 sentences: script/concept — dialogue, subtext, theme; for non-narrative, conceptual writing>",
    "cinematography": "<2-3 sentences: composition, lighting, camera movement, colour — cite a timestamp>",
    "performances": "<2-3 sentences: acting truth/range/presence; 'N/A — no performers' for abstract work, judge the nearest equivalent>",
    "production_value": "<2-3 sentences: scope, design, polish relative to evident resources>",
    "pacing": "<2-3 sentences: rhythm, momentum, where it drags or sings — cite timestamps>",
    "structure": "<2-3 sentences: architecture, escalation, how beginning/middle/end (or formal arc) hold>",
    "sound_music": "<2-3 sentences: score, sound design, mix, dialogue clarity>"
  }},
  "overall_rating": <0-10, your holistic professional verdict — close to but not mechanically the average>,
  "standout_moment": "<the single strongest scene or moment, with an approximate timestamp>",
  "weakest_element": "<the most important, constructive area to improve, be specific>",
  "festival_suitability": "<1-2 sentences on audience and circuit fit for {festival['name']}>",
  "recommendation": "<Pass | Recommend | Award Worthy | Maybe>"
}}

CALIBRATION (0-10, festival-submission context — be fair and discerning, not stingy):
0-2 = fundamentally broken craft (rare)   3-4 = notable weaknesses, early-stage
5-6 = competent and screenable baseline    7-8 = strong, distinctive, programme-worthy
9-10 = exceptional, award-calibre execution of its form.
Most accomplished festival films land at 6-8. Reserve 0-4 for genuinely deficient
craft, and award 9-10 only when a film truly excels at what it sets out to do —
including abstract or experimental excellence.
RECOMMENDATION must follow the overall_rating: 8.5-10 → Award Worthy, 7.0-8.4 →
Recommend, 5.0-6.9 → Maybe, below 5 → Pass.
"""


def build_long_film_analysis_prompt(festival: dict) -> str:
    """
    Analysis prompt for feature films processed via keyframes + transcript.
    """
    focus = festival.get("analysis_focus", "").strip()
    return f"""
You are a professional film critic evaluating a submission for {festival['full_name']}.
You are reviewing sampled keyframes and a transcript — not the full film.
Reflect this appropriately; focus on what is clearly observable.

FESTIVAL JUDGING EMPHASIS:
{focus}

Score each of the NINE criteria 0–10 from the available keyframes + transcript.
Return ONLY valid JSON — no markdown, no preamble:
{{
  "form": "<film type>",
  "ratings": {{
    "originality": <0-10>, "direction": <0-10>, "writing": <0-10>,
    "cinematography": <0-10>, "performances": <0-10>, "production_value": <0-10>,
    "pacing": <0-10>, "structure": <0-10>, "sound_music": <0-10>
  }},
  "notes": {{
    "originality": "<...>", "direction": "<...>", "writing": "<...>",
    "cinematography": "<...>", "performances": "<...>", "production_value": "<...>",
    "pacing": "<...>", "structure": "<...>", "sound_music": "<...>"
  }},
  "overall_rating": <0-10>,
  "standout_moment": "<most striking moment observed>",
  "weakest_element": "<most constructive area for improvement>",
  "festival_suitability": "<1-2 sentences on fit for {festival['name']}>",
  "recommendation": "<Pass | Recommend | Award Worthy | Maybe>"
}}
"""


MAX_REVIEW_WORDS = 500
MAX_CONTEXT_WORDS = 500   # cap on logline / director statement / synopsis fed to the prompt


def _cap_words(text: str, limit: int = MAX_CONTEXT_WORDS) -> str:
    """Truncate text to at most `limit` words."""
    if not text:
        return ""
    words = text.split()
    if len(words) <= limit:
        return text
    return " ".join(words[:limit]) + " …"


def build_review_prompt(film_meta: dict, analysis: dict, festival: dict) -> str:
    """
    Build the expert review generation prompt for a given festival.
    Word count is capped at MAX_REVIEW_WORDS globally.
    Logline, director statement, and synopsis are included only when present,
    each capped at MAX_CONTEXT_WORDS words.
    """
    word_count    = min(int(festival.get("word_count", 300) or 300), MAX_REVIEW_WORDS)
    custom_prompt = festival.get("review_prompt", "").strip()
    guidelines    = custom_prompt or festival.get("review_guidelines", "").strip()
    tone          = festival.get("tone", "professional, honest, and encouraging")

    # Per-category writing emphasis for this festival, if configured
    genre        = (film_meta.get("genre") or "").strip()
    cat_prompts  = festival.get("category_prompts", {}) or {}
    cat_emphasis = (cat_prompts.get(genre, "") or "").strip()
    if cat_emphasis:
        guidelines = (
            f"{guidelines}\n\nCATEGORY-SPECIFIC GUIDANCE for '{genre}' "
            f"(prioritise this for this submission):\n{cat_emphasis}"
        )

    logline   = _cap_words(film_meta.get("logline", "").strip())
    dir_stmt  = _cap_words(film_meta.get("director_statement", "").strip())
    synopsis  = _cap_words(film_meta.get("synopsis", "").strip())

    # Build optional context block — only include lines that have content
    context_lines = []
    if logline:
        context_lines.append(f"Logline:            {logline}")
    if synopsis:
        context_lines.append(f"Synopsis:           {synopsis}")
    if dir_stmt:
        context_lines.append(
            f"Director Statement: {dir_stmt}\n"
            f"  → Use the director's stated intentions to contextualise your critique — "
            f"acknowledge their vision while honestly assessing how well the film achieves it."
        )
    context_block = "\n".join(context_lines)

    # Defensive accessors — read the new 9-criteria structure, falling back to the
    # legacy 4-dimension structure so old cached analyses still produce a review.
    a = analysis if isinstance(analysis, dict) else {}
    ratings = a.get("ratings") if isinstance(a.get("ratings"), dict) else {}
    notes   = a.get("notes")   if isinstance(a.get("notes"), dict)   else {}

    CRITERIA = [
        ("originality",      "Originality / Creativity"),
        ("direction",        "Direction"),
        ("writing",          "Writing"),
        ("cinematography",   "Cinematography"),
        ("performances",     "Performances"),
        ("production_value", "Production Value"),
        ("pacing",           "Pacing"),
        ("structure",        "Structure"),
        ("sound_music",      "Sound / Music"),
    ]

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    score_lines, valid_scores = [], []
    for key, label in CRITERIA:
        s = _num(ratings.get(key))
        note = (notes.get(key) or "").strip()
        shown = f"{s:.0f}" if s is not None else "—"
        if s is not None:
            valid_scores.append(s)
        score_lines.append(f"- {label}: {shown}/10" + (f"  — {note}" if note else ""))
    scores_block = "\n".join(score_lines)

    overall = _num(a.get("overall_rating"))
    if overall is None and "overall_score" in a:          # legacy /20 → /10
        legacy = _num(a.get("overall_score"))
        overall = (legacy / 2) if legacy is not None else None
    if overall is None and valid_scores:
        overall = sum(valid_scores) / len(valid_scores)
    overall = overall if overall is not None else 0.0
    avg = (sum(valid_scores) / len(valid_scores)) if valid_scores else overall

    standout    = a.get("standout_moment", "")
    weakest     = a.get("weakest_element", "")
    suitability = a.get("festival_suitability", "")

    return f"""
You are a professional film critic and senior programmer at {festival['full_name']},
writing an Expert Review for a filmmaker who paid for professional, mentor-grade
feedback. Write with the depth, specificity, and craft vocabulary of a published
critic — but stay encouraging and respectful. The filmmaker is emotionally invested;
be honest about weaknesses while framing every criticism as a path forward. Never
brutal, never dismissive.
The "Comments" section must be approximately {word_count} words of substantive,
insight-rich prose (the scores and headings are additional, not counted).

FILM DETAILS:
Title:    {film_meta.get('title', 'Unknown')}
Director: {film_meta.get('director', 'Unknown')}
Genre:    {film_meta.get('genre', '')}
Runtime:  {film_meta.get('runtime', '')} min
Country:  {film_meta.get('country', '')}
{context_block}

DETAILED ANALYSIS (use these EXACT scores in your Ratings block — do not invent new numbers):
Overall Rating: {overall:.1f}/10
{scores_block}
Standout moment:  {standout}
Weakest element:  {weakest}
Festival fit:     {suitability}

OUTPUT FORMAT — reproduce EXACTLY this structure and order:

Overall Rating: {overall:.1f}/10

Ratings:
- Originality / Creativity: [0–10]
- Direction: [0–10]
- Writing: [0–10]
- Cinematography: [0–10]
- Performances: [0–10]
- Production Value: [0–10]
- Pacing: [0–10]
- Structure: [0–10]
- Sound / Music: [0–10]
- Average: [mean of the nine scores above, one decimal place]

Comments:
[Your deep, professional critique here — see writing rules below.]

Recommendation: [Pass | Recommend | Award Worthy | Maybe]
Reason: [One encouraging sentence justifying the recommendation.]

RATING RULES:
- Use the EXACT nine scores and Overall Rating from the DETAILED ANALYSIS above.
- Average = arithmetic mean of the nine criterion scores, to one decimal place.
- Recommendation must follow the Overall Rating: 8.5–10 → Award Worthy,
  7.0–8.4 → Recommend, 5.0–6.9 → Maybe, below 5 → Pass.

FESTIVAL-SPECIFIC WRITING GUIDELINES (for tone & emphasis only — if these mention any
rating categories or a different scoring structure, IGNORE that and use ONLY the
nine-criteria OUTPUT FORMAT above):
{guidelines}

WRITING RULES FOR THE COMMENTS:
- Open with a specific, evocative observation about the film, not a generic compliment.
- Lead with genuine strengths (cite exact scenes, shots, performances, cuts, sound)
  before moving to growth areas — earn the filmmaker's trust first.
- Address each major dimension that matters for this film with real critical insight,
  not surface praise; reference concrete moments and name techniques precisely.
- Pair every weakness with a concrete, actionable suggestion ("try…", "consider…").
- Where it illuminates a point, weave in a brief piece of craft wisdom (how editors,
  DPs, or directors handle a similar challenge) — relevant, never name-dropping.
- Critique the work, never the person; no sarcasm. The filmmaker should finish
  motivated to keep creating.
- Do NOT start with "This film" or "The film". Avoid: compelling, captivating,
  masterful, stunning. Vary sentence length; include at least one short punchy line.
- Tone: {tone}
"""


def build_certificate_prompt(film_meta: dict, award: str, festival: dict) -> str:
    return f"""
Write a formal certificate citation (2 sentences max) for:
Film:     {film_meta.get('title')}
Director: {film_meta.get('director')}
Award:    {award}
Festival: {festival['full_name']}

Formal, celebratory tone. Reference the film's genre or theme briefly.
"""
