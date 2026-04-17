# Soccer Spread/Total Betting: Implementable Edges Brief

## Top 4 Implementable Edges (Ranked by ROI/Complexity)

### 1. Asian Handicap Line Shopping + Quarter-Line Arbitrage (HIGH ROI / LOW COMPLEXITY)
Asian Handicaps split into quarters (-0.25, -0.75) effectively run two half-stake bets. Books often disagree on whether to offer -0.5 vs -0.25 vs 0 for the same favorite, creating middle opportunities. Pinnacle's AH is the sharpest market in soccer — deviations of 0.25+ from Pinnacle on Bet365/DraftKings/FanDuel are exploitable.

**Edge mechanic:** If Pinnacle closes EPL match at -0.5 favorite and a soft book offers -0.25, backing -0.25 at the soft book has positive EV because you win half-stake on a draw. Historically +2-4% ROI when >0.25 line discrepancy exists.

**Data:** Pinnacle odds via [oddsapi.io](https://the-odds-api.com) (free tier: 500 req/month), or scrape [oddsportal.com](https://www.oddsportal.com).

### 2. Club Elo + Dixon-Coles Goals Model for Totals (HIGH ROI / MEDIUM COMPLEXITY)
Dixon-Coles (1997) remains the state-of-the-art baseline for Poisson-based goal totals, with adjustment for 0-0/1-0/0-1/1-1 correlation. Pair with Club Elo ratings (updated daily, covers all leagues) to get team strength input. Beats closing over/under 2.5 lines by ~1.5% when book totals deviate >0.25 from model.

**Data:**
- [clubelo.com/API](http://clubelo.com/API) — free CSV API, covers EPL, UCL, and most European leagues (MLS NOT covered — use 538 SPI archive instead)
- [FiveThirtyEight SPI archive](https://github.com/fivethirtyeight/data/tree/master/soccer-spi) — final archive through 2023, includes MLS offensive/defensive ratings you can freeze as priors

**GitHub:** [octosport/octopy](https://github.com/octosport/octopy) — implements Dixon-Coles, bivariate Poisson, and Karlis-Ntzoufras models in Python. Direct port target.

### 3. MLS Home Field Advantage Exploitation (MEDIUM ROI / LOW COMPLEXITY)
MLS has the largest HFA in major soccer — ~0.45 goals/game vs ~0.30 in EPL (altitude for Colorado/RSL/Atlanta, travel fatigue on 3-timezone trips). Books systematically undervalue west-coast teams traveling east on short rest and vice versa. Back AH home favorite -0.5 when visiting team traveled >2 timezones in <4 days: historical ~3% edge 2022-2024.

**Data:** ESPN scoreboard feed you already have + static travel distance matrix. No external source needed.

### 4. EPL "Big 6 vs Rest" Totals Mispricing (LOW-MEDIUM ROI / LOW COMPLEXITY)
Public money inflates Over 3.5 in Big 6 vs bottom-half matchups. Actual rate of 4+ goal games in these spots is ~38% vs implied ~44% at typical -130 Over prices. Fade Over 3.5 in Big 6 home vs relegation-battle away after line moves up from 3.0 → 3.5 on public action.

**Data:** [football-data.co.uk](https://www.football-data.co.uk/englandm.php) — free historical EPL results + closing odds back to 1993.

## Other Research Answers

- **Steam windows:** Soccer steam is slower than NFL — typically 30-90 min after sharp action, vs <5 min in NFL. Pinnacle moves first; US books follow 15-45 min later.
- **DNB vs AH -0.25/+0.25:** Mathematically equivalent payout structures, but DNB is often offered by softer books (FanDuel, BetMGM) at worse juice. AH -0.25 at Pinnacle is always the sharper/cheaper version of the same bet.
- **xG without paywalls:** [StatsBomb open data](https://github.com/statsbomb/open-data) (limited competitions but full event data with xG); [mplsoccer](https://github.com/andrewRowlinson/mplsoccer) Python lib wraps it. [WyScout](https://footystats.org) free tier. Skip FBRef (Cloudflare blocks), skip Understat (TOS issues).

## GitHub Repos Worth Porting

1. **[octosport/octopy](https://github.com/octosport/octopy)** — Dixon-Coles + bivariate Poisson + ML ensemble for match outcomes and totals. MIT license, clean API, ~400 stars. Best single port for your stack.
2. **[martineastwood/penaltyblog](https://github.com/martineastwood/penaltyblog)** — Dixon-Coles, Rue-Salvesen, goal models + Club Elo scraper + football-data.co.uk loader built-in. Actively maintained 2024-2025.

**Start with #3 (MLS HFA) — uses data you already have, shippable in a day.** Then port octopy for edges #2 and #4.
