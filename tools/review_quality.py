#!/usr/bin/env python3
"""
review_quality.py — measure whether the AI reviewer agrees with your human jury.

Three commands:

  init      Write a starter CSV listing every film that has an AI analysis, with
            blank human columns for you to fill in.
  golden    Compare your human verdicts against the AI scores: rank correlation,
            band agreement, and the films where the AI disagrees most.
  coverage  Audit where in each film's runtime the AI's cited timestamps fall —
            the check that exposed the "only watches the opening" problem.

Reads through the deployed app's admin API because Atlas is not reachable from
a local machine. Credentials come from $FR_ADMIN_EMAIL / $FR_ADMIN_PASS, or are
pulled from Secret Manager via gcloud if those are unset.

  python3 tools/review_quality.py init  golden_set.csv
  python3 tools/review_quality.py golden golden_set.csv
  python3 tools/review_quality.py coverage
"""
import csv
import html
import http.cookiejar
import json
import os
import re
import statistics
import subprocess
import sys
import urllib.parse
import urllib.request

BASE_URL = os.getenv("FR_BASE_URL", "https://festival-reviewer-e53sualg4a-el.a.run.app")
PROJECT  = os.getenv("FR_GCP_PROJECT", "personal-workspace-490012")

# AI overall_rating → recommendation band (must match prompts.py calibration)
BANDS = [(8.5, "Award Worthy"), (7.0, "Recommend"), (5.0, "Maybe"), (0.0, "Pass")]


def band_for(score):
    for floor, name in BANDS:
        if score >= floor:
            return name
    return "Pass"


# ── Data access ───────────────────────────────────────────────────────────────

def _secret(name):
    out = subprocess.run(["gcloud", "secrets", "versions", "access", "latest",
                          "--secret", name, "--project", PROJECT],
                         capture_output=True, text=True)
    if out.returncode != 0:
        sys.exit(f"Could not read secret {name!r}. Set FR_ADMIN_EMAIL / FR_ADMIN_PASS "
                 f"instead, or check gcloud auth.\n{out.stderr.strip()}")
    return out.stdout.strip()


def fetch_films():
    """Log in as admin and return every film visible to that account."""
    email = os.getenv("FR_ADMIN_EMAIL") or _secret("admin-1-email")
    pw    = os.getenv("FR_ADMIN_PASS")  or _secret("admin-1-pass")

    opener = urllib.request.build_opener(
        urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))
    body = urllib.parse.urlencode({"email": email, "password": pw}).encode()
    opener.open(f"{BASE_URL}/login", body).read()

    raw = opener.open(f"{BASE_URL}/api/films").read().decode()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        sys.exit("Login failed — /api/films returned HTML, not JSON. Check credentials.")


def ai_score(analysis):
    """The AI's 0-10 verdict, tolerating older analysis shapes.
    Current analyses carry overall_rating; pre-rubric ones carry overall_score
    out of 20, and some only have the individual criterion ratings."""
    a = analysis or {}
    if isinstance(a.get("overall_rating"), (int, float)):
        return float(a["overall_rating"])
    if isinstance(a.get("overall_score"), (int, float)):
        return float(a["overall_score"]) / 2          # legacy /20 scale
    nums = [v for v in (a.get("ratings") or {}).values() if isinstance(v, (int, float))]
    return statistics.mean(nums) if nums else None


def clean_title(t):
    return html.unescape(t or "").strip()


def scored_films():
    """Films that carry an AI verdict in any supported analysis shape."""
    return [f for f in fetch_films() if ai_score(f.get("analysis")) is not None]


# ── Statistics (no scipy dependency) ──────────────────────────────────────────

def _ranks(values):
    """Average-rank transform, so ties share a rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        shared = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = shared
        i = j + 1
    return ranks


def spearman(xs, ys):
    """Spearman rank correlation: Pearson computed over average ranks."""
    if len(xs) < 3:
        return None
    rx, ry = _ranks(xs), _ranks(ys)
    mx, my = statistics.mean(rx), statistics.mean(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return (num / den) if den else None


def interpret(rho):
    if rho is None:
        return "not enough data"
    r = abs(rho)
    strength = ("very strong" if r >= .8 else "strong" if r >= .6 else
                "moderate" if r >= .4 else "weak" if r >= .2 else "negligible")
    return f"{strength} {'agreement' if rho >= 0 else 'DISAGREEMENT (inverted!)'}"


# ── Commands ──────────────────────────────────────────────────────────────────

def cmd_init(path):
    films = scored_films()
    if not films:
        sys.exit("No films with an AI analysis yet — nothing to benchmark.")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["title", "director", "human_score", "human_verdict", "ai_score_readonly"])
        for f in sorted(films, key=lambda f: clean_title(f.get("title"))):
            w.writerow([clean_title(f.get("title")), f.get("director", ""), "", "",
                        ai_score(f.get("analysis"))])
    print(f"Wrote {len(films)} films to {path}\n")
    print("Now fill in, for the films your jury actually judged:")
    print("  human_score    0-10, what your jury would score it (best signal)")
    print("  human_verdict  Award Worthy | Recommend | Maybe | Pass")
    print("Leave rows you have no human judgement for blank — they are skipped.")
    print(f"\nThen run:  python3 tools/review_quality.py golden {path}")


def cmd_golden(path):
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run:  python3 tools/review_quality.py init {path}")

    by_key = {}
    for f in scored_films():
        t = clean_title(f.get("title")).lower()
        by_key[(t, f.get("director", "").strip().lower())] = f
        by_key.setdefault((t, ""), f)

    pairs, verdicts, missing = [], [], []
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            title = (row.get("title") or "").strip()
            hs    = (row.get("human_score") or "").strip()
            hv    = (row.get("human_verdict") or "").strip()
            if not title or (not hs and not hv):
                continue
            film = (by_key.get((title.lower(), (row.get("director") or "").strip().lower()))
                    or by_key.get((title.lower(), "")))
            if not film:
                missing.append(title)
                continue
            ai = ai_score(film.get("analysis"))
            if hs:
                try:
                    pairs.append((title, float(hs), float(ai)))
                except ValueError:
                    pass
            if hv:
                verdicts.append((title, hv, band_for(float(ai))))

    if missing:
        print(f"Not found in the system ({len(missing)}): {', '.join(missing[:8])}\n")
    if not pairs and not verdicts:
        sys.exit("No usable rows — fill in human_score or human_verdict first.")

    if pairs:
        rho = spearman([p[1] for p in pairs], [p[2] for p in pairs])
        print("=== AI vs HUMAN JURY — score agreement ===\n")
        print(f"{'FILM':38}{'HUMAN':>7}{'AI':>7}{'DELTA':>8}")
        print("-" * 60)
        for t, h, a in sorted(pairs, key=lambda p: -abs(p[1] - p[2])):
            print(f"{t[:36]:38}{h:>7.1f}{a:>7.1f}{a - h:>+8.1f}")
        deltas = [abs(a - h) for _, h, a in pairs]
        print("-" * 60)
        print(f"films compared      : {len(pairs)}")
        print(f"mean absolute error : {statistics.mean(deltas):.2f} points")
        print(f"AI bias             : {statistics.mean([a - h for _, h, a in pairs]):+.2f} "
              f"({'AI scores higher' if statistics.mean([a - h for _, h, a in pairs]) > 0 else 'AI scores lower'})")
        if rho is not None:
            print(f"Spearman rho        : {rho:+.3f}  — {interpret(rho)}")
            print("\n  rho is the number that matters: it asks whether the AI RANKS films")
            print("  the way your jury does. Above +0.6 means you can trust it to shortlist.")
        else:
            print("Spearman rho        : need at least 3 scored films")

    if verdicts:
        agree = sum(1 for _, h, a in verdicts if h.lower() == a.lower())
        print(f"\n=== Verdict band agreement: {agree}/{len(verdicts)} "
              f"({agree / len(verdicts) * 100:.0f}%) ===")
        for t, h, a in verdicts:
            print(f"  {'OK ' if h.lower() == a.lower() else 'MISS'}  {t[:34]:36}"
                  f"human={h:<14} ai={a}")


def cmd_coverage():
    """Where in the runtime does the AI actually cite evidence?"""
    TS = re.compile(r"\b(\d{1,2}):(\d{2})(?::(\d{2}))?\b")
    rows, all_fracs = [], []

    for f in fetch_films():
        a, rt = f.get("analysis") or {}, f.get("runtime_min")
        if not a or not rt:
            continue
        text = " ".join([json.dumps(a.get("notes", {})), str(a.get("standout_moment", "")),
                         str(a.get("weakest_element", "")), str(a.get("arc", ""))])
        runtime_s = rt * 60
        secs = []
        for m in TS.finditer(text):
            g = m.groups()
            s = (int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2])) if g[2] else int(g[0]) * 60 + int(g[1])
            if 0 < s <= runtime_s:
                secs.append(s)
        if not secs:
            rows.append((clean_title(f.get("title")) or "?", rt, 0, None, None))
            continue
        fracs = [s / runtime_s for s in secs]
        all_fracs.extend(fracs)
        rows.append((clean_title(f.get("title")) or "?", rt, len(secs),
                     round(max(fracs) * 100), round(statistics.mean(fracs) * 100)))

    print(f"{'FILM':38}{'RUNTIME':>9}{'#TS':>5}{'DEEPEST':>9}{'MEAN':>7}")
    print("-" * 68)
    for t, rt, n, deepest, mean in sorted(rows, key=lambda r: -(r[1] or 0)):
        print(f"{t[:36]:38}{round(rt):>7}m{n:>5}"
              f"{(f'{deepest}%' if deepest is not None else '—'):>9}"
              f"{(f'{mean}%' if mean is not None else '—'):>7}")

    if all_fracs:
        buckets = [0] * 5
        for fr in all_fracs:
            buckets[min(int(fr * 5), 4)] += 1
        tot = len(all_fracs)
        print("\n=== Distribution of cited moments across runtime ===")
        for lab, c in zip(["1st fifth (0-20%)", "2nd fifth (20-40%)", "3rd fifth (40-60%)",
                           "4th fifth (60-80%)", "final fifth (80-100%)"], buckets):
            print(f"  {lab:24}{c:>4}  {c / tot * 100:5.1f}%  {'#' * round(c / tot * 40)}")
        print(f"\n  A healthy reviewer spreads citations across all five bands.")
        print(f"  Heavy skew to the first fifth means it is not watching the whole film.")


def main():
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    cmd = args[0]
    if cmd == "init":
        cmd_init(args[1] if len(args) > 1 else "golden_set.csv")
    elif cmd == "golden":
        cmd_golden(args[1] if len(args) > 1 else "golden_set.csv")
    elif cmd == "coverage":
        cmd_coverage()
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
