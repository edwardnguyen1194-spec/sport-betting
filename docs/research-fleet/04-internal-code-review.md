# Internal Code Review — Strategy Audit

## 1. Bias audit

**`total_projection.py`** — Market-anchored regression (lines 121–153) works but is **incomplete**:
- `park_factor` (line 84) and `weather_delta` (line 91) apply *before* regression
- Coors' +1.15 on 9.5 baseline → +1.4 runs of Over pressure that 50% blend only shrinks halfway
- **Fix:** apply park/weather to the book-line side of the blend, OR subtract from comparison baseline. Better: `adjusted_book = total_num - expected_park_weather_delta`, compare `projected` vs `adjusted_book`

**`public_fade.py`** — dog-bias on spreads (line 104: `l.line >= 0`). Intentional but asymmetric. Document explicitly.

**`steam_follow.py`** — dog-bias on spreads (lines 136–141 skip `line < 0`) combined with `shorten`-only (line 90) = strategy **cannot pick a favorite spread**. Document in docstring.

**`spread_value.py` / `total_value.py`** — genuinely symmetric. No bias concern.

## 2. Threshold calibration (MIN_LINE_EDGE, lines 37–45)

Rough data-driven targets (1-sigma of projection):

| Sport | Current | Team σ/game | Suggested 1·SE |
|---|---|---|---|
| MLB | 0.5 | ~3.0 runs | **0.95** |
| NBA | 2.5 | ~12 pts | **3.8** |
| NCAAB | 3.0 | ~14 pts | **4.4** |
| NHL | 0.4 | ~1.4 goals | **0.45** |
| NFL | 2.0 | ~10 pts | **3.2** |
| NCAAF | 2.5 | ~14 pts | **4.4** |

**Current thresholds are too loose** for MLB/NBA/NCAAB/NFL/NCAAF — firing inside noise band.

Concrete: compute per-sport `std(scored_per_game)` from tracker, replace dict with `min_edge = max(line_increment, 0.3 * scoring_std)` at runtime.

## 3. Confidence cap (line 171, 0.62)

**Proposal: tighten to 0.57**.
- 60% regression toward book admits the book is mostly right — claiming 62% is inconsistent
- `tanh` ramp reaches 0.62 only at ~4×min_edge — over-confident when it hits
- Kelly at 0.57 vs 0.62 on +100 line = 14% vs 24% of bankroll — huge variance reduction for minimal EV loss

## 4. Dead code audit

`__init__.py` exports 12 strategies. Recommendations:

| Strategy | Verdict |
|---|---|
| `MiddleDetectorStrategy` | **Keep, wire in** — middles are real money |
| `EloEdgeStrategy` | **Gate to spreads only OR delete** (moneyline-first now) |
| `ContrarianStrategy` | **Delete one of {Contrarian, PublicFade}** — overlap, double-sizes |
| `PythagoreanStrategy` | **Delete** — already captured by TeamScoringTracker |
| `HeavyFavoriteStrategy` | **Delete** — moneyline contradicts Uncle's rule |
| `ValueBetStrategy` | **Delete if moneyline-only**, else collapse into Spread/TotalValue |

Net: drop 3–4 files, revive Middle, gate Elo.

## 5. Missing edge — build this next

**Reverse Line Movement detector (~150 lines).**

Already have `LineMovementStore` + `detect_steam` infra. RLM is the inverse: **line moves against heavy public % is the sharpest signal** (56–58% ATS historically).

Implementation:
1. Join `LineMovementStore` history with `game.meta.public_spread_*_pct`
2. Flag games where public ≥60% on side A but line moved ≥0.5 pts toward side B at 2+ sharp books within 6 hours
3. Bet side B. Reuse `public_fade.py` candidate logic

Why this over alternatives:
- NFL key-number buying: dormant 4 months (April 16) — ship year-round edge first
- NBA B2B fade: only 2–3 games/night qualify; low volume
- Umpire-adjusted MLB totals: great but >200 lines of infra before model runs

RLM is live year-round across MLB+NBA+NHL, reuses existing code, signal stronger than public_fade or steam_follow alone.

## Key file references
- `total_projection.py`: lines 37–45, 84–91, 121–153, 171
- `public_fade.py`: line 104
- `steam_follow.py`: lines 90, 136–141
- `strategies/__init__.py`: lines 9–22
