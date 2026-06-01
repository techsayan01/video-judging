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
You are a seasoned development executive and script reader with 15 years of experience
evaluating screenplays, teleplays, stage plays, and literary writing for {festival['full_name']}.
You are reading a WRITTEN SUBMISSION (a script / screenplay / poem / novel / written work),
NOT a finished film. Judge the writing ON THE PAGE — never penalise it for lacking
visuals, sound, or performances.

FESTIVAL JUDGING EMPHASIS:
{focus}
{category_block}
{context_block}

Identify the written form first (feature screenplay, short script, pilot, stage play,
radio/audio script, poem, novel, etc.) and judge it on the conventions of THAT form.

Return ONLY valid JSON — no markdown, no preamble. Use these exact keys; the four
scored dimensions are reinterpreted for written work as noted:
{{
  "form": "<the written form you identified>",
  "story": {{
    "score": <1-5>,
    "clarity": "<concept, premise, and thematic clarity>",
    "character_depth": "<character development & voice; 'N/A' for non-narrative poems>",
    "emotional_impact": "<emotional or intellectual resonance>",
    "notes": "<2-3 specific observations on story/concept>"
  }},
  "direction": {{
    "score": <1-5>,
    "visual_language": "<structure & architecture of the writing (acts, scenes, stanzas)>",
    "pacing": "<momentum and rhythm on the page>",
    "notes": "<2-3 observations on structure & pacing>"
  }},
  "technical": {{
    "score": <1-5>,
    "cinematography": "<format & presentation discipline (industry formatting, layout)>",
    "sound_design": "<dialogue craft & subtext, or language/meter for poetry>",
    "editing": "<scene construction, economy, clarity of action lines>",
    "notes": "<2-3 observations on craft & format>"
  }},
  "originality": {{
    "score": <1-5>,
    "distinctive_element": "<what makes this writing stand apart>",
    "thematic_freshness": "<angle on subject matter>",
    "notes": "<2-3 observations on voice & originality>"
  }},
  "standout_moment": "<the strongest scene, line, page, or passage (cite a page/scene if possible)>",
  "weakest_element": "<most constructive area for improvement, be specific>",
  "festival_suitability": "<1-2 sentences on fit for {festival['name']}>",
  "overall_score": <sum of 4 scores, 4-20>
}}

CALIBRATION (be fair, not stingy): 3 = competent, screenable/producible writing;
4 = strong, distinctive work; 5 = exceptional, award-calibre writing. Reserve 1-2 for
genuinely deficient craft. Be specific and reference actual lines, scenes, or pages.
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
You are a seasoned film festival programmer and juror with 15 years of experience
across narrative, documentary, experimental, animation, and music-driven work.
You are evaluating a submission for {festival['full_name']}, which focuses on {festival['focus']}.

FESTIVAL JUDGING EMPHASIS:
{focus}
{category_block}
{context_block}

STEP 1 — IDENTIFY THE FORM before scoring. Determine what KIND of film this is:
narrative short/feature, experimental/art film, music video or visual-music piece,
documentary, animation, dance/poetry film, or hybrid. Many strong festival films are
NOT conventional narratives.

STEP 2 — JUDGE THE FILM ON ITS OWN TERMS, not against a template:
- Do NOT penalise a film for lacking plot, dialogue, or character arcs if it is
  intentionally non-narrative (experimental, mood/atmosphere, music-driven, poetic).
  For such films, "story" means concept, thematic coherence, and emotional/sensory
  journey — score that, not the absence of a three-act plot.
- A film with no dialogue is not deficient in "dialogue"; judge its sound design and
  music instead, and treat dialogue as N/A rather than a low score.
- Reward distinctive artistic vision, atmosphere, and craft even when the work is
  abstract or minimal. Ambitious art films that fully achieve their intent can and
  should score 4–5.
- Where a director's statement is provided, assess how well the film realises its
  STATED intentions — not your assumption of what it should have been.

Analyse carefully and return ONLY valid JSON — no markdown, no preamble:
{{
  "form": "<the film type you identified in Step 1>",
  "story": {{
    "score": <1-5>,
    "clarity": "<narrative clarity, OR conceptual/thematic coherence for non-narrative work>",
    "character_depth": "<character development, OR 'N/A — non-narrative' if not applicable>",
    "emotional_impact": "<emotional or sensory resonance>",
    "notes": "<2-3 specific observations, judged on the film's own form>"
  }},
  "direction": {{
    "score": <1-5>,
    "visual_language": "<one sentence on visual storytelling / composition>",
    "pacing": "<one sentence on rhythm and pacing>",
    "notes": "<2-3 specific observations about direction>"
  }},
  "technical": {{
    "score": <1-5>,
    "cinematography": "<specific observation with timestamp if notable>",
    "sound_design": "<music, ambient sound; dialogue clarity only if dialogue exists>",
    "editing": "<cut rhythm, transitions, overall flow>",
    "notes": "<2-3 specific technical observations>"
  }},
  "originality": {{
    "score": <1-5>,
    "distinctive_element": "<what makes this film stand apart>",
    "thematic_freshness": "<angle on subject matter>",
    "notes": "<2-3 observations on creative voice>"
  }},
  "standout_moment": "<specific scene or moment with approximate timestamp>",
  "weakest_element": "<most constructive area for improvement, be specific>",
  "festival_suitability": "<1-2 sentences on audience and circuit fit for {festival['name']}>",
  "overall_score": <sum of 4 scores, 4-20>
}}

CALIBRATION (festival-submission context — be fair, not stingy):
1 = fundamentally broken craft (rare)
2 = notable weaknesses, early-stage
3 = competent and screenable — solid festival-circuit baseline
4 = strong, distinctive work that earns a place in a programme
5 = exceptional, award-calibre execution of its form
Most accomplished festival films land at 3–4. Reserve 1–2 for genuinely deficient
craft, and do not hesitate to award 5 when a film excels at what it sets out to do —
including abstract or experimental excellence. Score the achievement of intent, not
conformity to convention. Be specific and reference actual moments in the film.
"""


def build_long_film_analysis_prompt(festival: dict) -> str:
    """
    Analysis prompt for feature films processed via keyframes + transcript.
    """
    focus = festival.get("analysis_focus", "").strip()
    return f"""
You are a professional film festival programmer evaluating a submission for {festival['full_name']}.
You are reviewing sampled keyframes and a transcript — not the full film.
Reflect this appropriately; focus on what is clearly observable.

FESTIVAL JUDGING EMPHASIS:
{focus}

Return ONLY valid JSON — no markdown, no preamble:

{{
  "story": {{
    "score": <1-5>,
    "clarity": "<based on transcript and visual narrative>",
    "character_depth": "<from dialogue and visible character arcs>",
    "emotional_impact": "<from observable emotional beats>",
    "notes": "<2-3 observations>"
  }},
  "direction": {{
    "score": <1-5>,
    "visual_language": "<from keyframe composition and lighting>",
    "pacing": "<from scene transitions and transcript rhythm>",
    "notes": "<2-3 observations>"
  }},
  "technical": {{
    "score": <1-5>,
    "cinematography": "<from keyframe quality and composition>",
    "sound_design": "<from transcript clarity and audio notes>",
    "editing": "<from visible cuts and scene structure>",
    "notes": "<2-3 observations>"
  }},
  "originality": {{
    "score": <1-5>,
    "distinctive_element": "<visual or narrative signature>",
    "thematic_freshness": "<subject matter and treatment>",
    "notes": "<2-3 observations>"
  }},
  "standout_moment": "<most visually striking or narratively compelling moment observed>",
  "weakest_element": "<most constructive area for improvement>",
  "festival_suitability": "<1-2 sentences on fit for {festival['name']}>",
  "overall_score": <sum of 4 scores, 4-20>
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

    # Defensive accessors — Gemini occasionally omits a key or returns null,
    # which previously crashed with "NoneType is not subscriptable".
    a = analysis if isinstance(analysis, dict) else {}
    def _dim(key):
        d = a.get(key) or {}
        if not isinstance(d, dict):
            d = {}
        score = d.get("score", "—")
        notes = d.get("notes") or d.get("clarity") or ""
        return score, notes
    story_s, story_n   = _dim("story")
    dir_s, dir_n       = _dim("direction")
    tech_s, tech_n     = _dim("technical")
    orig_s, orig_n     = _dim("originality")
    standout    = a.get("standout_moment", "")
    weakest     = a.get("weakest_element", "")
    suitability = a.get("festival_suitability", "")
    try:
        overall20 = float(a.get("overall_score") or 0)
    except (TypeError, ValueError):
        overall20 = 0.0
    overall10 = overall20 / 2

    return f"""
You are a senior programmer at {festival['full_name']} and a working filmmaker
yourself, writing an Expert Review for a filmmaker who paid for professional,
mentor-grade feedback. They are emotionally invested in this film — your feedback
must be encouraging and respectful while still being honest about what needs work.
Never deliver a brutal or dismissive verdict; instead, write the kind of note a
generous mentor gives: warm, specific, and genuinely useful.
Write exactly {word_count} words — no more, no fewer.

FILM DETAILS:
Title:    {film_meta.get('title', 'Unknown')}
Director: {film_meta.get('director', 'Unknown')}
Genre:    {film_meta.get('genre', '')}
Runtime:  {film_meta.get('runtime', '')} min
Country:  {film_meta.get('country', '')}
{context_block}

ANALYSIS SCORES:
Story ({story_s}/5): {story_n}
Direction ({dir_s}/5): {dir_n}
Technical ({tech_s}/5): {tech_n}
Originality ({orig_s}/5): {orig_n}
Standout moment: {standout}
Weakest element: {weakest}
Festival suitability: {suitability}

RATING CONSISTENCY (IMPORTANT):
The detailed analysis above gives an overall score of {overall20:.0f}/20,
which is equivalent to {overall10:.1f}/10.
Your "Overall Rating: X/10" MUST stay within ±0.5 of {overall10:.1f}
so the headline rating matches the detailed assessment. Your six category sub-scores
should also broadly reflect the four analysis scores above — do not contradict them
(e.g. don't praise direction with a 9/10 if Direction analysis scored 2/5).

FESTIVAL-SPECIFIC WRITING GUIDELINES:
{guidelines}

UNIVERSAL RULES:
- Open with a specific observation about the film, not a generic compliment
- Lead with genuine strengths before growth areas — earn the filmmaker's trust first
- Reference concrete moments, not vague impressions
- Pair every weakness with a concrete, actionable suggestion ("try…", "consider…")
- Where it illuminates a point, weave in a short piece of craft wisdom or a brief
  professional anecdote (how editors, DPs, or directors handle a similar challenge) —
  but keep it relevant and never name-drop for its own sake
- Be encouraging, never brutal: critique the work, not the person; no sarcasm or
  dismissiveness; the filmmaker should finish reading motivated to keep creating
- Do NOT start with "This film" or "The film"
- Do NOT use: compelling, captivating, masterful, stunning
- Sound human — vary sentence length, include at least one short punchy sentence
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
