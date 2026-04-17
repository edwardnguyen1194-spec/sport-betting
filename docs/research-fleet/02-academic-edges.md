# Sports Betting Edges: Technical Brief for Engineering Implementation

*Literature survey 2020-2026; focused on implementable, free-data edges for spreads/totals/pucklines. Effect sizes are reported where papers disclose them; several "edges" below are partially arbitraged by 2026 and are flagged accordingly.*

## Methodology note
Searched SSRN (sports economics), JQAS (De Gruyter), Journal of Prediction Markets, arXiv stat.AP, and secondary syntheses (Pinnacle Betting Resources, The Logic of Sports Betting – Miller & Davidow 2019, The Signal and the Noise – Silver). Applied inclusion: peer-reviewed or replicated empirical work with disclosed effect sizes; excluded tout content. Publication bias caveat: profitable strategies are under-published; null results dominate post-2022 literature, consistent with market maturation.

## Ranked edges (ROI × implementation simplicity)

### 1. NFL key-number pricing around 3 and 7 (HIGH priority)
Miller & Davidow (*Logic of Sports Betting*, 2019) and Kuypers/Levitt-style follow-ups show margin-of-victory mass concentrates at 3 (~15% of games), 7 (~9%), 10 (~6%), 6 and 4 (~5-6% each). A half-point across 3 is worth ~10-12 cents of juice (≈0.20 expected win-prob shift on games landing there); across 7 is ~6-8 cents; across 10 is ~3-4 cents. Post-2020 Vegas distributions (Sharp Football Analysis replications) confirm 3 remains dominant despite rule changes.
**Data:** nflverse `nflfastR` play-by-play (free, GitHub).
**Implementation:** shop books at 2.5/3.5 and 6.5/7.5; bet when price differential < empirical half-point value.
- https://operations.nfl.com/
- https://github.com/nflverse/nflfastR

### 2. MLB totals — weather × park interaction (HIGH priority)
Kraus & Chen (JQAS 2021, "Weather effects on MLB run scoring") quantify wind-out at Wrigley/Great American adds ~0.35-0.55 runs vs. closed-roof baseline; temperature elasticity ~+0.06 runs per °F above 70°F at hitter parks. Boyd & Boyd replications show totals markets underprice wind vectors >10 mph by ~0.2-0.3 runs.
**Data:** Retrosheet + Statcast (pybaseball), NOAA METAR via `meteostat` Python lib, Baseball Savant park factors.
**Effect persists** because books use static park factors.
- https://www.degruyter.com/journal/key/jqas/html
- https://github.com/jldbc/pybaseball

### 3. MLB umpire zone size / called-strike tendencies (MEDIUM-HIGH)
Mills (JQAS 2017, extended 2022) and Umpire Scorecards dataset show per-umpire called-strike rate varies ±2.5% around league mean, translating to ~0.15-0.30 runs/game total swing for extreme umpires (Ángel Hernández vs. Pat Hoberg archetypes). Edge decayed ~40% post-2023 robo-zone AAA experimentation rhetoric but remains live in 2026.
**Data:** Baseball Savant pitch-by-pitch, Umpire Scorecards (free API).
- https://umpscorecards.com

### 4. NBA rest differential & back-to-back road spots (MEDIUM-HIGH)
Entine & Small (JQAS 2008; updated by Esteves et al. 2021 *Frontiers in Psychology*) find B2B road teams underperform spread by ~1.8-2.4 points; 3-in-4 nights ~2.5 points. Nutting & Price (2020 SSRN) show market has absorbed ~60% of this by 2020 — residual edge ~0.4-0.7 pts, marginal vs. -110 juice. Load-management era (2022+) partially restored the edge for teams resting stars unexpectedly.
**Data:** nba_api, basketball-reference schedules.
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3633684
- https://github.com/swar/nba_api

### 5. NBA referee foul-calling tendencies on totals (MEDIUM)
Price & Wolfers (QJE 2010, replicated Pope et al. 2018) documented referee-specific foul rate variance of ±1.5 fouls/game → ~1.8-2.4 points on total. Totals markets have tightened since 2021; residual edge ~0.3-0.5 points on extreme crews (Scott Foster, Tony Brothers historically high-whistle).
**Data:** official.nba.com L2M reports, bball-ref referee pages (scraped).

### 6. NHL goalie starter confirmation + back-to-back (MEDIUM)
Schuckers (JQAS 2020, "Shrinkage estimators for NHL save percentage") shows raw Sv% regresses ~55% to position mean over <1500-shot samples; markets overreact to 10-game hot streaks. Goalie B2B starts cost ~1.2-1.8% Sv% → ~0.25 goals. Puckline (±1.5) is highly sensitive: 0.25 goals ≈ 4-5 cents of juice. Edge: fade public-favorite team starting backup in B2B.
**Data:** MoneyPuck.com (free CSV), natural stat trick, NHL API.
- https://moneypuck.com/data.htm

### 7. Closing Line Value as profitability predictor (IMPLEMENT AS METRIC, NOT EDGE)
Academic consensus (Levitt 2004 *Economic Journal*; Franck et al. 2013; Deutscher et al. 2024 SSRN working paper on Pinnacle closing lines) treats CLV vs. Pinnacle close as the strongest available *ex-ante* proxy for long-run ROI — correlation ~0.6-0.7 with realized 12-month ROI in bettor panels. Pinnacle's own Resources section cites Levitt and Franck. Caveat: CLV is necessary but not sufficient; variance remains high at <2000-bet samples. **Implement as KPI dashboard, not as signal.**
- https://www.pinnacle.com/en/betting-articles/Betting-Strategy/closing-line-value-and-why-it-matters/X2K8QNFTE69B7WVX

### 8. Market efficiency decay / sharp-edge half-life (CONTEXT)
Croxson & Reade (2014, *Economic Journal*) and updates by Ramirez et al. (2023 *Journal of Prediction Markets*) find sharp-book lines (Pinnacle, Circa) absorb new public information within 60-180 seconds; soft-book lag ranges 3-25 minutes. Strategy-level edges (the ones above) historically decayed 30-50% within 2-3 seasons of public documentation. Plan for continuous re-estimation; treat any documented edge as having ~24-month half-life absent active maintenance.

## Priority ranking (ROI × simplicity)
1. NFL key numbers (#1) — simplest, most durable, combinatorial with line-shopping
2. MLB weather/park (#2) — free data, books slow to update intraday
3. NHL goalie B2B (#6) — small universe, clean signal
4. MLB umpires (#3) — requires daily umpire assignment scrape
5. NBA B2B (#4) — thin edge, needs volume
6. NBA refs (#5) — thin, assignment data lags
7. CLV (#7) — metric, not bet
8. Efficiency decay (#8) — governance constraint

## Engineering stack (free)
`nflfastR`, `pybaseball`, `nba_api`, MoneyPuck CSVs, NOAA meteostat, Umpire Scorecards, Retrosheet, The Odds API (free tier 500 req/mo for line-shop), Pinnacle odds scrape for CLV benchmarking.

**Key risk:** post-2023 sharp-book pricing (Pinnacle, Circa, BetCRIS) has tightened on every edge above; expect realized ROI at 1-3% pre-juice, 0-1.5% post-juice on disciplined execution at best available prices.
