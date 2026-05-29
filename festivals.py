"""
festivals.py — single source of truth for all festival configurations.

To add a new festival: copy an entry in FESTIVALS and set the env vars.

Env var pattern per festival (replace SLUG with the festival key in uppercase):
  SLUG_GEMINI_KEY   — Gemini API key
  SLUG_WP_URL       — WordPress site URL
  SLUG_WP_USER      — WordPress username
  SLUG_WP_PASS      — WordPress application password

Example for "elegantiff":
  ELEGANTIFF_GEMINI_KEY, ELEGANTIFF_WP_URL, ELEGANTIFF_WP_USER, ELEGANTIFF_WP_PASS

A global GEMINI_API_KEY env var is used as fallback if a festival-specific key is not set.
DEFAULT_FESTIVAL env var controls which festival is pre-selected in the UI.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────────────────────────

def _cfg(slug: str, **kwargs) -> dict:
    """Build a festival config, auto-resolving env vars by slug."""
    env = slug.upper()
    return {
        **kwargs,
        "gemini_api_key": os.getenv(f"{env}_GEMINI_KEY", os.getenv("GEMINI_API_KEY", "")),
        "gemini_model":   kwargs.get("gemini_model", "gemini-2.5-flash"),
        "wp_url":         os.getenv(f"{env}_WP_URL", ""),
        "wp_user":        os.getenv(f"{env}_WP_USER", "admin"),
        "wp_app_pass":    os.getenv(f"{env}_WP_PASS", ""),
        "email_domains":  kwargs.get("email_domains", []),
        "review_prompt":  kwargs.get("review_prompt", ""),
        "word_count":     500,
    }

# ─────────────────────────────────────────────────────────────────────────────

FESTIVALS: dict[str, dict] = {

    "elegantiff": _cfg(
        slug="elegantiff",
        name="ElegantIFF",
        full_name="Elegant International Film Festival",
        focus="art-house and independent cinema",
        circuit="art-house and independent film circuits",
        word_count=300,
        tone="collegial, honest, encouraging — like a respected peer",

        # What the AI focuses on when scoring the film
        analysis_focus="""
- Prioritise visual storytelling and how the camera serves the narrative
- Assess thematic depth and whether the film has a distinctive artistic voice
- Evaluate emotional resonance and the filmmaker's command of tone
- Note any influences from world cinema traditions
""",

        # Additional writing rules appended to the review prompt
        review_guidelines="""
- Emphasise the film's artistic vision and thematic ambition
- Discuss visual language and how it serves the story
- Situate the work within the independent/art-house landscape
- Close with specific festival circuit suggestions (e.g. Tribeca, SXSW, BFI)
""",
    ),

    "shortwave": _cfg(
        slug="shortwave",
        name="ShortWave",
        full_name="ShortWave Short Film Festival",
        focus="short films under 20 minutes",
        circuit="short film festivals and curated streaming platforms",
        word_count=250,
        tone="energetic, direct, nurturing of emerging talent",

        analysis_focus="""
- Reward economy of storytelling — how much is achieved in the runtime
- Assess the clarity and strength of the single central idea
- Evaluate how well the ending lands relative to the premise
- Note replay value and shareability for online platforms
""",

        review_guidelines="""
- Acknowledge the challenge of the short format and how the filmmaker meets it
- Focus on one strong idea and its execution rather than broad critique
- Give concrete suggestions on which short film festivals suit this work
- Keep the tone energetic — short film makers often work with low budgets and high ambition
""",
    ),

    "docuverse": _cfg(
        slug="docuverse",
        name="DocuVerse",
        full_name="DocuVerse Documentary Festival",
        focus="documentary and non-fiction filmmaking",
        circuit="documentary festivals and broadcast markets",
        word_count=350,
        tone="analytical, socially aware, constructive",

        analysis_focus="""
- Assess subject access and the filmmaker's relationship with their subjects
- Evaluate research depth and how it shapes the narrative
- Consider the balance between information and emotional engagement
- Note the film's argument or thesis and how clearly it is articulated
- Assess broadcast and streaming market potential
""",

        review_guidelines="""
- Discuss the strength of the central argument or story
- Address subject access and ethical approach where observable
- Note potential for broadcast pre-sales, streaming acquisition, or educational use
- Be specific about which documentary festivals match the film's ambition and topic
""",
    ),

    "newwave": _cfg(
        slug="newwave",
        name="NewWave Cinema",
        full_name="NewWave Cinema Festival",
        focus="experimental and avant-garde cinema",
        circuit="experimental and avant-garde film circuits",
        word_count=280,
        tone="intellectually curious, open to risk-taking, challenging conventions",

        analysis_focus="""
- Prioritise formal innovation over conventional storytelling metrics
- Assess whether the film extends or subverts the language of cinema
- Evaluate the relationship between form and content
- Do NOT penalise unconventional structure if the intention is clear
- Risk-taking and ambitious failure outweigh safe, competent execution
""",

        review_guidelines="""
- Engage with the film's formal or structural ambition directly
- Situate the work within experimental cinema traditions where relevant
- Do not use conventional narrative film as the benchmark
- Encourage the filmmaker to push further into their formal instincts
""",
    ),

    "horizon": _cfg(
        slug="horizon",
        name="Horizon FF",
        full_name="Horizon Film Festival",
        focus="emerging voices and debut features",
        circuit="debut and emerging filmmaker circuits",
        word_count=300,
        tone="mentoring, supportive, growth-oriented",

        analysis_focus="""
- Assess the filmmaker's potential as much as the finished film
- Look for a distinctive voice even if execution is uneven
- Give weight to ambition and risk-taking from first/second-time directors
- Identify the single most important skill the filmmaker should develop next
""",

        review_guidelines="""
- Lead with what the filmmaker does well and shows clear promise in
- Frame every criticism as a specific actionable step for their next project
- Be honest about weaknesses but situate them in the context of early-career growth
- Close with encouragement — name the circuit or opportunity that fits their stage
""",
    ),
}

# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_FESTIVAL = os.getenv("DEFAULT_FESTIVAL", "elegantiff")
if DEFAULT_FESTIVAL not in FESTIVALS:
    DEFAULT_FESTIVAL = "elegantiff"


def get_festival(key: str) -> dict:
    """Return festival config by key, falling back to default."""
    return FESTIVALS.get(key, FESTIVALS[DEFAULT_FESTIVAL])


# domain → festival_key lookup (e.g. "elegantiff.com" → "elegantiff")
DOMAIN_FESTIVAL_MAP: dict[str, str] = {
    domain: key
    for key, cfg in FESTIVALS.items()
    for domain in cfg.get("email_domains", [])
}
