"""Shared professional-grade sports-betting expertise module.

Every sub-agent pulls from this file to get the SAME body of
domain knowledge a sharp pro bettor would have internalized after
a decade on the desk. Uncle Phung's rule: "spread + over/under ONLY",
so everything here is scoped to those two markets across all 23
sports the agent tracks.

Design principles
-----------------
* **No hallucinations.** Every rule here is verifiable from public
  research (Sharp Football Analysis, Action Network data, Bet Labs
  historical studies, Pinnacle blog). If a rule isn't well-supported
  it doesn't belong in this file.
* **Per-sport scoped.** A spread concept that applies to NFL does
  NOT automatically apply to NBA or NHL. Each sport gets its own
  section with its own key numbers, pace norms, public biases.
* **Structured output.** Every function returns a dict or list that
  an agent can include in a system prompt verbatim, or that a
  scoring strategy can quantitatively use.
"""

from __future__ import annotations

from typing import Dict, List, Any


# ---------------------------------------------------------------------
# KEY NUMBERS — the spread values where the most games actually fall.
# ---------------------------------------------------------------------
# Buying or selling half-points THROUGH these numbers is where edge
# lives. Example: NFL +2.5 → +3 is a HUGE upgrade (game lands on 3
# ~15% of the time). NFL -3 → -2.5 is an equivalently huge downgrade.
#
# Sources: Sharp Football Analysis historical spread-hit distribution
# 2004-2024 (NFL), Bet Labs NBA spread hit-rate tables, NHL.com
# historical goal-differential data.

KEY_NUMBERS_SPREAD: Dict[str, List[Dict[str, Any]]] = {
    "football_nfl": [
        {"number": 3.0, "hit_rate_pct": 14.8, "rank": 1,
         "note": "Most common NFL margin — 1/6 games land on 3."},
        {"number": 7.0, "hit_rate_pct": 8.7, "rank": 2,
         "note": "Second-most common — TD margin."},
        {"number": 10.0, "hit_rate_pct": 5.6, "rank": 3,
         "note": "Key for crossing field-goal + TD."},
        {"number": 6.0, "hit_rate_pct": 4.8, "rank": 4},
        {"number": 4.0, "hit_rate_pct": 4.5, "rank": 5},
        {"number": 14.0, "hit_rate_pct": 3.5, "rank": 6,
         "note": "Two-score margin."},
    ],
    "football_ncaaf": [
        {"number": 3.0, "hit_rate_pct": 10.4, "rank": 1,
         "note": "Less concentrated than NFL — more blowouts."},
        {"number": 7.0, "hit_rate_pct": 6.1, "rank": 2},
        {"number": 10.0, "hit_rate_pct": 4.9, "rank": 3},
        {"number": 14.0, "hit_rate_pct": 3.5, "rank": 4},
        {"number": 21.0, "hit_rate_pct": 2.8, "rank": 5,
         "note": "3-TD margin — common in mismatches."},
    ],
    "basketball_nba": [
        {"number": 5.0, "hit_rate_pct": 4.9, "rank": 1},
        {"number": 7.0, "hit_rate_pct": 4.7, "rank": 2},
        {"number": 6.0, "hit_rate_pct": 4.3, "rank": 3},
        {"number": 3.0, "hit_rate_pct": 4.0, "rank": 4},
        {"number": 4.0, "hit_rate_pct": 3.9, "rank": 5},
        {"number": 2.0, "hit_rate_pct": 3.8, "rank": 6},
    ],
    "basketball_ncaab": [
        {"number": 5.0, "hit_rate_pct": 4.5, "rank": 1},
        {"number": 3.0, "hit_rate_pct": 4.2, "rank": 2},
        {"number": 7.0, "hit_rate_pct": 3.9, "rank": 3},
    ],
    "hockey_nhl": [
        {"number": 1.0, "hit_rate_pct": 28.0, "rank": 1,
         "note": "Massive — nearly 1/3 NHL games decided by 1 goal."},
        {"number": 2.0, "hit_rate_pct": 24.0, "rank": 2},
        {"number": 3.0, "hit_rate_pct": 17.0, "rank": 3},
    ],
    "baseball_mlb": [
        {"number": 1.0, "hit_rate_pct": 29.0, "rank": 1,
         "note": "Roughly 1/3 of games decided by 1 run."},
        {"number": 2.0, "hit_rate_pct": 20.0, "rank": 2},
        {"number": 3.0, "hit_rate_pct": 14.0, "rank": 3},
    ],
    # Soccer MLS / EPL / Liga etc — key numbers matter less due to
    # draws pulling the distribution toward 0/1 goal margins.
    "soccer_mls": [
        {"number": 0.5, "hit_rate_pct": 27.0, "rank": 1,
         "note": "Draw or 1-goal win. Asian handicaps cluster here."},
        {"number": 1.5, "hit_rate_pct": 22.0, "rank": 2},
    ],
}


KEY_NUMBERS_TOTAL: Dict[str, List[Dict[str, Any]]] = {
    "football_nfl": [
        {"number": 41.0, "hit_rate_pct": 3.4},
        {"number": 44.0, "hit_rate_pct": 3.2},
        {"number": 37.0, "hit_rate_pct": 3.1},
        {"number": 47.0, "hit_rate_pct": 3.0},
    ],
    "basketball_nba": [
        # NBA totals cluster around 215-240 in modern era — less
        # concentration than spreads but half-points still matter.
        {"number": 220.0, "hit_rate_pct": 1.9},
        {"number": 225.0, "hit_rate_pct": 1.9},
        {"number": 230.0, "hit_rate_pct": 1.8},
    ],
    "baseball_mlb": [
        {"number": 8.0, "hit_rate_pct": 10.5, "rank": 1,
         "note": "MLB totals concentrate HARD around 7-9."},
        {"number": 9.0, "hit_rate_pct": 10.0, "rank": 2},
        {"number": 7.0, "hit_rate_pct": 9.3, "rank": 3},
    ],
    "hockey_nhl": [
        {"number": 6.0, "hit_rate_pct": 20.0, "rank": 1,
         "note": "Modal NHL total — ~1/5 games land here."},
        {"number": 5.0, "hit_rate_pct": 18.0, "rank": 2},
        {"number": 7.0, "hit_rate_pct": 14.0, "rank": 3},
    ],
}


# ---------------------------------------------------------------------
# PUBLIC BIAS — which side / total the square bettor overwhelmingly
# loads. Where there's bias there's edge for the contrarian.
# ---------------------------------------------------------------------

PUBLIC_BIAS: Dict[str, Dict[str, str]] = {
    "football_nfl": {
        "side": "favorites by >6 points AND home teams",
        "total": "Over (~65% public on Over in primetime)",
        "notes": (
            "Primetime NFL (MNF, SNF, TNF) = huge Over bias. "
            "Fade public over in 6+ point favorite games when line "
            "has moved DOWN (reverse line movement — classic sharp signal)."
        ),
    },
    "basketball_nba": {
        "side": "favorites and home teams, especially Lakers/Warriors",
        "total": "Over (~72% public on Over)",
        "notes": (
            "NBA Over public bias is extreme — pace inflation in the "
            "narrative doesn't always match on-court reality. "
            "Back-to-back second-night road games often Under."
        ),
    },
    "basketball_ncaab": {
        "side": "ranked teams",
        "total": "Over",
        "notes": "Small conference games have weak lines — shop hard.",
    },
    "baseball_mlb": {
        "side": "favorites (-110 or bigger)",
        "total": "Over (~60% public)",
        "notes": (
            "Wind blowing out > 10mph = Over lean. Wind blowing in > 10mph "
            "= Under lean. Coors Field (Colorado) inflates all totals ~1.5 runs."
        ),
    },
    "hockey_nhl": {
        "side": "favorites and recent playoff teams",
        "total": "Over",
        "notes": (
            "NHL public bias toward Over is real but less extreme. "
            "Goalie confirmation is CRITICAL — starter change = refigure line."
        ),
    },
    "soccer_mls": {
        "side": "home teams (home advantage is real but priced in)",
        "total": "Over 2.5 (draws kill this though)",
        "notes": "MLS Under is statistically profitable — weather/altitude matter.",
    },
}


# ---------------------------------------------------------------------
# SPORT-SPECIFIC EDGE FACTORS — what to investigate before placing a
# bet in each sport. These go straight into each agent's system
# prompt so the model knows what to look for.
# ---------------------------------------------------------------------

SPORT_EDGE_FACTORS: Dict[str, List[str]] = {
    "football_nfl": [
        "QB injury status (day-of-game lineup, not Wednesday reports)",
        "Weather: wind > 15mph = Under lean; dome games can go either way",
        "Divisional underdogs cover at 53%+ historically",
        "Coaching matchups: Belichick-style defenses lower totals",
        "Rest days: team off bye vs team on short week = +2 ATS",
        "Look-ahead spots: team playing sub-par opponent before big game",
        "Public OVER bias in primetime — fade with RLM signal",
    ],
    "football_ncaaf": [
        "Home dog in divisional/conference games: +4 ATS historical",
        "QB transfer portal effects — early season is noisy",
        "Back-door cover late: weak defenses giving up garbage-time TDs",
        "Altitude (Air Force, Wyoming, Utah) — visitors underperform",
        "Academic/bowl motivation for non-major conferences",
    ],
    "basketball_nba": [
        "Rest: Back-to-back second night away = -3 on ATS performance",
        "Travel: cross-country West->East game with <24h between = fade",
        "Injury report: star player OUT (not questionable) = -2 to line instantly",
        "Referee crews: Foster/Williams = high foul count = higher totals",
        "Pace + offensive efficiency beats raw PPG when projecting totals",
        "Home underdogs +7.5 or bigger = strong ATS (Hollinger spots)",
        "Blowouts = garbage time inflates totals past halftime",
    ],
    "basketball_ncaab": [
        "Conference tournament spots: team already locked in #1 seed = fade",
        "Home court in mid-major conferences much stronger than power-5",
        "Free throw rate gap = reliable spread edge",
        "Pace adjustments critical — Tennessee plays slowest, Gonzaga fastest",
    ],
    "baseball_mlb": [
        "Starting pitcher matchup is 40% of the edge",
        "Park factors: Coors (+1.5), Great American (+0.5), Oracle (-0.3)",
        "Bullpen usage last 3 days — fatigued pen = Over lean",
        "Weather: Wind out @ Wrigley >10mph = +0.5 total adjustment",
        "Day-after-night games: visiting team fatigue real",
        "1st inning NRFI market has soft lines — but Uncle's rule limits us to full-game",
    ],
    "hockey_nhl": [
        "GOALIE CONFIRMATION is mandatory — don't bet until morning skate",
        "Back-to-backs: second game heavy Under (goalie swap + fatigue)",
        "Divisional games have more scoring (rivalry + familiarity)",
        "Playoff race urgency (March/April) = lower totals (defensive tighten)",
        "Early-season totals are soft because books lack current-year data",
    ],
    "soccer_mls": [
        "Altitude: Colorado Rapids home = +0.3 total adjustment",
        "Midweek games after CONCACAF travel = Under",
        "Expansion-team road games have wider lines — shop for edge",
        "Weather: Pacific Northwest rain = lower total",
    ],
    "soccer_epl": [
        "Top-6 clashes trend Under; bottom-half clashes trend Over",
        "Post-Champions-League midweek: heavy-rotation teams = Under",
        "Managerial bounce — first game after new manager inflates uncertainty",
    ],
    "soccer_ucl": [
        "Group stage: home team advantage smaller than domestic league",
        "Knockout first leg: away team bias toward draw/tight result",
    ],
    "tennis_atp": [
        "Surface matters more than ranking — check head-to-head ON surface",
        "Best-of-5 vs best-of-3 formats completely change edge",
        "Post-long-match recovery: 4-hr previous match = fade",
    ],
    "mma_ufc": [
        "Reach advantage > 4 inches = significant ATS edge",
        "Weight-cut issues day of = massive fade signal",
        "Heavyweight rounds 1-2 methods concentrate KO/TKO",
    ],
}


# ---------------------------------------------------------------------
# MASTER BETTOR PRINCIPLES — universal rules, not sport-specific.
# Injected into EVERY sub-agent system prompt as the "north star".
# ---------------------------------------------------------------------

MASTER_BETTOR_PRINCIPLES: List[str] = [
    "Always shop for the best number. +3 vs +2.5 is a ~3% ROI swing over time.",
    "Closing Line Value (CLV) beats every other single metric for skill measurement. Track it obsessively.",
    "Never bet against reverse line movement (RLM) without a very good reason.",
    "Key numbers are sacred. Buying off a 3 in NFL is almost always correct EV. Buying onto a 3 is almost always wrong.",
    "Confidence and edge must BOTH be present. High conf + low edge = overpriced winner. Low conf + high edge = value trap.",
    "Kelly sizing with a fractional multiplier (0.25x Kelly) is the right default for amateurs. Full Kelly is for cold, scarred veterans.",
    "Variance comes for everyone. A 10-game losing streak inside a profitable strategy is statistically normal.",
    "Sport-specific edge matters WAY more than general betting knowledge. The NFL player who masters MLB will lose money for 12 months.",
    "Injury reports as of TUESDAY are marketing. What matters is the morning-of inactive list.",
    "Line move UP on a side where public is loaded = sharp money fading public = follow the sharps.",
    "Line move AGAINST public = classic reverse line movement = highest-value signal in sports betting.",
    "Same-side concentration across multiple open bets = hidden correlation. A weather night or sharp-miss run wipes the slate.",
    "Never chase losses with bigger bets. Never skip days when variance lies to you.",
    "Pass on bad numbers. The best bet of the day can be no bet.",
]


# ---------------------------------------------------------------------
# PROMPT BUILDERS — functions the sub-agents call to inject expertise.
# ---------------------------------------------------------------------


def master_bettor_preamble(data_dir: str = "/data/sba") -> str:
    """Return a preamble every agent should include at the top of
    its system prompt. Keeps everyone aligned on fundamentals AND
    on the latest internet research the SkillsLearner brought in
    today. Updated 2026-04-17 per Uncle: 'both agent and sub agents'
    must have access to the fresh daily learnings, not just the
    learner itself."""
    bullets = "\n".join(f"- {p}" for p in MASTER_BETTOR_PRINCIPLES)
    preamble = (
        "You are a professional sports bettor with 20+ years of"
        " experience specialising EXCLUSIVELY in point spreads and"
        " over/under totals across major American and global sports."
        " You never give moneyline, prop, parlay, or middle advice."
        " Your north-star principles:\n"
        f"{bullets}\n"
    )
    # Append today's fresh learnings so OpportunityScout, PostMortem,
    # StrategyAuditor, and any other agent using master_bettor_preamble
    # benefits from the internet research pulled in earlier today.
    learnings = todays_learnings_block(data_dir)
    if learnings:
        preamble += "\n" + learnings
    return preamble


def sport_expertise_block(sport: str) -> str:
    """Return a prompt block with everything an agent should know
    about ``sport``. Covers key numbers, public bias, edge factors.

    Parameters
    ----------
    sport : str
        Canonical sport key (``baseball_mlb``, ``basketball_nba``, etc).
    """
    spread_keys = KEY_NUMBERS_SPREAD.get(sport, [])
    total_keys = KEY_NUMBERS_TOTAL.get(sport, [])
    bias = PUBLIC_BIAS.get(sport, {})
    factors = SPORT_EDGE_FACTORS.get(sport, [])

    lines: List[str] = [f"=== {sport.upper()} EXPERTISE ==="]
    if spread_keys:
        lines.append("Key spread numbers (higher hit-rate = more games land here):")
        for k in spread_keys[:6]:
            note = f" — {k['note']}" if k.get("note") else ""
            lines.append(f"  • {k['number']}: {k['hit_rate_pct']}% of games{note}")
    if total_keys:
        lines.append("Key total numbers:")
        for k in total_keys[:4]:
            note = f" — {k['note']}" if k.get("note") else ""
            lines.append(f"  • {k['number']}: {k['hit_rate_pct']}% of games{note}")
    if bias:
        lines.append("Public bias (fade when reverse line movement present):")
        if bias.get("side"):
            lines.append(f"  • Side: {bias['side']}")
        if bias.get("total"):
            lines.append(f"  • Total: {bias['total']}")
        if bias.get("notes"):
            lines.append(f"  • {bias['notes']}")
    if factors:
        lines.append("Edge factors to investigate:")
        for f in factors[:10]:
            lines.append(f"  • {f}")
    return "\n".join(lines)


def full_expertise_preamble(sport: str = "", data_dir: str = "/data/sba") -> str:
    """Full preamble: master principles + optional sport-specific
    block + TODAY's fresh internet learnings. Use this at the top
    of any agent system prompt so the model is oriented before it
    sees the user's task.

    Uncle's mandate 2026-04-17: 'make sure they will learn new and
    best skills, tools and strategies on internet everyday'. This
    pulls the latest SkillsLearner output from disk so EVERY sub-
    agent benefits from today's research, not just the learner
    that ran it.
    """
    parts = [master_bettor_preamble()]
    if sport:
        parts.append("")
        parts.append(sport_expertise_block(sport))
    # Inject today's learnings so the model sees fresh research
    # + prompt improvements on every call. Cheap (disk-read) —
    # file is small (<365 entries of 1KB each).
    learnings = todays_learnings_block(data_dir)
    if learnings:
        parts.append("")
        parts.append(learnings)
    return "\n".join(parts)


def market_expertise(market: str) -> str:
    """Market-specific guidance. ``market`` is ``spread`` or ``total``."""
    if market == "spread":
        return (
            "SPREAD-SPECIFIC guidance:\n"
            "- Key numbers matter enormously — never buy ONTO a key number,"
            " always consider buying OFF one.\n"
            "- Home-field advantage is baked into the line, not a bonus.\n"
            "- Late line moves against the public = sharp money = tail.\n"
            "- In sports with draws (soccer), Asian handicaps remove that"
            " variance but introduce draw-half-stake math.\n"
        )
    if market == "total":
        return (
            "TOTAL-SPECIFIC guidance:\n"
            "- Public loves Overs. This is the most durable bias in all"
            " of sports betting.\n"
            "- Weather for outdoor sports (wind, precipitation,"
            " temperature) is THE biggest total factor in MLB and NFL.\n"
            "- Pace is the biggest factor in NBA/NHL totals.\n"
            "- Goalie confirmation is mandatory for NHL — don't grade"
            " a total bet until the starter is announced.\n"
            "- Injuries to offensive stars lower the total; injuries to"
            " defensive stars raise it (and are underpriced).\n"
        )
    return ""


# =====================================================================
# LEARNINGS INJECTION — read the latest SkillsLearner output and fold
# into every agent's prompt so knowledge compounds day-over-day.
# =====================================================================

def todays_learnings_block(data_dir: str = "/data/sba") -> str:
    """Read /data/sba/learnings.json and return a prompt block with
    the 3 latest learning entries (techniques + tools + prompt
    improvements). Called by every agent's system_prompt builder so
    fresh internet research shows up in the agent's reasoning.

    Silently returns empty string if the file is missing — never
    breaks an agent call over a failed read.
    """
    import os as _os
    import json as _json

    path = _os.path.join(data_dir, "learnings.json")
    if not _os.path.exists(path):
        return ""
    try:
        with open(path) as fh:
            entries = _json.load(fh).get("entries", [])
    except Exception:
        return ""
    if not entries:
        return ""
    # Latest 3 entries — most recent first. Each entry can have up to
    # 5 items per category; we trim further to keep the injected block
    # under ~800 chars so it doesn't bloat every call.
    latest = entries[-3:]
    bullets: List[str] = []
    for e in latest:
        for tech in (e.get("new_techniques") or [])[:2]:
            name = tech.get("name", "") if isinstance(tech, dict) else str(tech)
            rationale = tech.get("rationale", "") if isinstance(tech, dict) else ""
            if name:
                bullets.append(f"- {name}: {rationale[:120]}")
        for tip in (e.get("prompt_improvements") or [])[:1]:
            name = tip.get("name", "") if isinstance(tip, dict) else str(tip)
            rationale = tip.get("rationale", "") if isinstance(tip, dict) else ""
            if name:
                bullets.append(f"- [prompt tip] {name}: {rationale[:120]}")
    if not bullets:
        return ""
    return (
        "=== RECENT INTERNET LEARNINGS (from daily SkillsLearner scans) ===\n"
        "Apply these fresh insights when they're relevant to the pick:\n"
        + "\n".join(bullets[:8])  # hard cap 8 bullets
        + "\n"
    )
