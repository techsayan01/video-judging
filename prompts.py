"""
prompts.py — AI prompt builders for analysis and review generation.

All prompts accept a festival config dict from festivals.py so that
scoring emphasis and review tone can differ per festival.
"""


def build_analysis_prompt(festival: dict) -> str:
    """
    Build the Gemini analysis prompt for a given festival.
    Returns structured JSON instructions with festival-specific scoring focus.
    """
    focus = festival.get("analysis_focus", "").strip()
    return f"""
You are a professional film festival programmer with 15 years of experience.
You are evaluating a submission for {festival['full_name']}, which focuses on {festival['focus']}.

FESTIVAL JUDGING EMPHASIS:
{focus}

Analyse this film submission carefully and return ONLY valid JSON — no markdown, no preamble.

Return this exact structure:
{{
  "story": {{
    "score": <1-5>,
    "clarity": "<one sentence on narrative clarity>",
    "character_depth": "<one sentence on character development>",
    "emotional_impact": "<one sentence on emotional resonance>",
    "notes": "<2-3 specific observations about storytelling>"
  }},
  "direction": {{
    "score": <1-5>,
    "visual_language": "<one sentence on visual storytelling>",
    "pacing": "<one sentence on rhythm and pacing>",
    "notes": "<2-3 specific observations about direction>"
  }},
  "technical": {{
    "score": <1-5>,
    "cinematography": "<specific observation with timestamp if notable>",
    "sound_design": "<dialogue clarity, music, ambient sound>",
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

Score guide: 1=needs significant work, 2=developing, 3=competent, 4=strong, 5=exceptional
Be specific. Reference actual moments in the film. Avoid generic praise.
Apply the festival judging emphasis above when weighting your scores.
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


def build_review_prompt(film_meta: dict, analysis: dict, festival: dict) -> str:
    """
    Build the expert review generation prompt for a given festival.
    """
    word_count = festival.get("word_count", 300)
    guidelines = festival.get("review_guidelines", "").strip()

    return f"""
You are a senior programmer at {festival['full_name']}, writing an Expert Review
for a filmmaker who paid for professional feedback.
Write exactly {word_count} words.

FILM DETAILS:
Title:     {film_meta.get('title', 'Unknown')}
Director:  {film_meta.get('director', 'Unknown')}
Genre:     {film_meta.get('genre', '')}
Runtime:   {film_meta.get('runtime', '')} min
Country:   {film_meta.get('country', '')}
Synopsis:  {film_meta.get('synopsis', '')}
Director Statement: {film_meta.get('director_statement', '')}

ANALYSIS SCORES:
Story ({analysis['story']['score']}/5): {analysis['story']['notes']}
Direction ({analysis['direction']['score']}/5): {analysis['direction']['notes']}
Technical ({analysis['technical']['score']}/5): {analysis['technical']['notes']}
Originality ({analysis['originality']['score']}/5): {analysis['originality']['notes']}
Standout moment: {analysis['standout_moment']}
Weakest element: {analysis['weakest_element']}
Festival suitability: {analysis['festival_suitability']}

FESTIVAL-SPECIFIC WRITING GUIDELINES:
{guidelines}

UNIVERSAL RULES:
- Open with a specific observation about the film, not a generic compliment
- Reference concrete moments, not vague impressions
- Balance strengths with one clear, actionable growth area
- Do NOT start with "This film" or "The film"
- Do NOT use: compelling, captivating, masterful, stunning
- Sound human — vary sentence length, include at least one short punchy sentence
- Tone: {festival['tone']}
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
