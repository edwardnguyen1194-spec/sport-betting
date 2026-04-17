# Sports-Betting Agent — Post-Research Roadmap

Synthesized from 4 parallel research agents (2026-04-17). Four independent perspectives agreed on the top priorities — this is high-signal consensus.

## 🥇 Tier 1 — Ship Immediately (year-round edges, reuse existing code)

### 1. Reverse Line Movement Strategy (RLM)
- **Where it came from:** Academic + GitHub + Internal reviewer all independently flagged this as #1
- **Effect size:** 56–58% ATS historically, 2–5% ROI per bet when aligned with steam
- **Complexity:** ~150 lines — reuses `LineMovementStore`, `public_fade.py` candidate logic
- **Status:** Code-ready once public-% data source found (see Tier 2 item below)
- **Reference:** [maccam912/sports-reverse-line-movement](https://github.com/maccam912/sports-reverse-line-movement), [nikhilkichili/nba-analytics-mcp](https://github.com/nikhilkichili/nba-analytics-mcp)

### 2. Middle-Bet Scanner
- **Where:** GitHub scout's #4 and internal reviewer both flagged
- **Effect:** 1–4% per middle (book A spread 4.5 + book B spread 5.5 = both can win)
- **Complexity:** ~400 lines; reuses multi-book aggregation
- **Reference:** [DeliciousPipe1326/edge-scanner](https://github.com/DeliciousPipe1326/edge-scanner)

### 3. Fix `total_projection.py` park/weather-before-regression bug
- **Where:** Internal reviewer
- **Effect:** removes residual Over bias at Coors/Fenway/summer heat
- **Complexity:** ~10 lines — subtract park/weather from book side of blend
- **Priority:** **Ship this first** — fixes Uncle's ongoing "why always Over" complaint at the root

### 4. Dead code purge
- **Where:** Internal reviewer
- **Drop:** `HeavyFavoriteStrategy`, `PythagoreanStrategy`, `ContrarianStrategy` (overlaps PublicFade), `ValueBetStrategy` (moneyline)
- **Revive:** `MiddleDetectorStrategy`
- **Gate:** `EloEdgeStrategy` to spreads only
- **Complexity:** cleanup + 1 rewire

### 5. Tighten confidence cap 0.62 → 0.57
- **Where:** Internal reviewer
- **Effect:** Kelly sizing 14% vs 24% — massive variance reduction
- **Complexity:** 1-line change

---

## 🥈 Tier 2 — 1-2 week builds, high ROI

### 6. Recalibrate `MIN_LINE_EDGE` to data-driven 1-sigma
- Current NBA 2.5 → suggested 3.8, current MLB 0.5 → 0.95
- Compute `std(scored_per_game)` per sport at runtime
- Complexity: ~50 lines

### 7. MLB Umpire-Adjusted Totals
- **Where:** Academic brief
- **Effect:** ±0.15–0.30 runs/game on extreme umpires (Hernández vs Hoberg)
- **Data:** [umpscorecards.com](https://umpscorecards.com) free API + daily umpire assignment scrape
- **Complexity:** ~200 lines + infra

### 8. NHL Goalie B2B Fade
- **Where:** Academic brief
- **Effect:** Backup goalie in B2B = 0.25 goals ≈ 4-5 cents juice on puckline
- **Data:** [MoneyPuck CSV](https://moneypuck.com/data.htm) — free
- **Complexity:** ~150 lines

### 9. MLS Home-Field-Advantage + Travel
- **Where:** Soccer specialist
- **Effect:** ~3% edge 2022–2024 when visiting team crossed 2+ timezones in <4 days
- **Data:** Already have ESPN scoreboard + static timezone matrix
- **Complexity:** ~100 lines

---

## 🥉 Tier 3 — Seasonal or bigger lifts

### 10. NFL Key-Number Half-Point Buying (dormant until Aug 29)
- **Where:** Academic + GitHub + Internal all flagged as durable
- **Effect:** half-point across 3 worth ~10-12¢ juice; across 7 ~6-8¢
- **Data:** [nflverse/nflfastR](https://github.com/nflverse/nflfastR)
- **Complexity:** ~200 lines — set reminder for mid-August

### 11. Soccer Dixon-Coles Totals Model
- **Where:** Soccer specialist + GitHub
- **Effect:** ~1.5% over closing O/U 2.5 on soft books
- **Port from:** [martineastwood/penaltyblog](https://github.com/martineastwood/penaltyblog) or [octosport/octopy](https://github.com/octosport/octopy)
- **Data:** [clubelo.com/API](http://clubelo.com/API) (EU leagues) + FiveThirtyEight SPI archive (MLS)
- **Complexity:** ~600 lines

### 12. Public-% data source (unblocks RLM + PublicFade)
- ActionNetwork's public fields are auth-gated
- Options: scrape scoresandodds.com (fragile HTML), VSIN, or pay The Odds API
- **Complexity:** ~300 lines scraping + resilience

---

## 🚫 Explicitly deprioritized (research agreed)

- NBA Back-to-Back fade — ~60% arbitraged by markets (SSRN Nutting & Price 2020)
- NBA referee foul tendencies — thin edge, data lags
- Heavy ML ensemble (XGBoost/NN) — marginal over existing projection
- Arbitrage-only scanning — depleted by HFT aggregators

---

## Execution order recommendation

**Week 1:**
1. Ship Tier 1 items 3, 4, 5 (park/weather fix, dead code purge, confidence cap) — 1 day
2. Ship Tier 1 item 1 RLM — 2 days (once public-% source decided)
3. Ship Tier 1 item 2 Middle scanner — 2 days

**Week 2:**
1. Tier 2 item 6 threshold recalibration — 1 day
2. Tier 2 item 8 NHL goalie B2B — 2 days
3. Tier 2 item 9 MLS travel — 1 day

**Reminder for August:**
- Tier 3 item 10 NFL key numbers — block 3 days

---

## Key research insights (context)

- **Market-efficiency half-life:** ~24 months post-public-documentation (Croxson & Reade, Ramirez et al.)
- **Realistic ROI expectation:** 1–3% pre-juice, 0–1.5% post-juice on disciplined execution
- **CLV > win rate:** our current +1.89% CLV at 73.7% win rate is the leading indicator — protect it
- **Sharp-book line absorption:** 60–180 seconds on Pinnacle/Circa; 3-25 min lag on soft books (steam window)
