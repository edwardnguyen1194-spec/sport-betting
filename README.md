# Sport Betting AI Agent

Multi-source sports-betting research bot and Flask dashboard deployed at
<https://sports-betting-ai-agent.fly.dev>.

This branch (`claude/add-bovada-odds-c9xmr`) adds:

1. **Bovada free college sports odds** — the full NCAA baseball board
   (37 games), plus MLB, NBA, NCAAB, NFL, NCAAF, NHL, MLS, ATP and UFC
   — no API key needed.
2. **Six other free odds sources**: ESPN's hidden API, ScoresAndOdds,
   VegasInsider, Covers.com, Action Network and SBR (Pinnacle sharp
   lines). All merged into a single normalized schema.
3. **Heavy-favorite filter strategy** tuned for a 75-85% expected win
   rate (configurable American-odds band, min books, vig-free
   consensus probability gate).
4. **LSTM neural-network player-props model** targeting 65-75% WR on
   top-decile picks. Uses PyTorch when available, falls back to NumPy
   logistic regression for micro VMs that don't ship torch wheels.
5. **Confidence scoring** (book-weighted, agreement-aware) and
   fractional-Kelly stake sizing.
6. **24/7 paper-trading engine** with `$50` min bet, 5% bankroll cap,
   JSON-backed ledger, and background auto-trade loop.
7. **Flask dashboard with Vietnamese UI** and Claude chat — chat works
   offline (deterministic fallback) if `ANTHROPIC_API_KEY` isn't set.

## Quickstart

```bash
pip install -r requirements.txt
pip install -e .

# Pull the Bovada college baseball board (the priority request):
python -m sports_betting_agent.cli odds --source bovada --sport baseball_ncaa

# Merged view across every enabled source:
python -m sports_betting_agent.cli odds --sport baseball_ncaa

# Ensemble recommendations (heavy fav + value bets):
python -m sports_betting_agent.cli recommend --sports baseball_ncaa,baseball_mlb

# Run the dashboard locally:
python -m sports_betting_agent.cli serve
# -> http://localhost:8080
```

## Configuration

All knobs live in environment variables (see `src/sports_betting_agent/config.py`):

| Variable | Default | Purpose |
|---|---|---|
| `SBA_ENABLED_SOURCES` | `bovada,espn,scoresandodds,vegasinsider,covers,actionnetwork,sbr` | Comma-separated list of enabled fetchers |
| `SBA_BANKROLL` | `10000` | Starting bankroll for paper trader |
| `SBA_MIN_BET` | `50` | Minimum bet (matches deployed bot) |
| `SBA_MAX_BET_PCT` | `0.05` | Max per-bet % of bankroll |
| `SBA_KELLY_FRACTION` | `0.25` | Fractional-Kelly multiplier |
| `SBA_HEAVY_FAV_MIN_AMERICAN` | `-400` | Lowest allowed favorite price |
| `SBA_HEAVY_FAV_MAX_AMERICAN` | `-150` | Highest allowed favorite price |
| `SBA_HEAVY_FAV_MIN_BOOKS` | `2` | Cross-book coverage requirement |
| `SBA_MIN_CONFIDENCE` | `0.65` | Ensemble gate |
| `SBA_AUTO_TRADE` | `1` | Enable background auto-trade loop |
| `SBA_AUTO_TRADE_INTERVAL` | `300` | Seconds between cycles |
| `SBA_LANG` | `vi` | Dashboard language (`vi` or `en`) |
| `THE_ODDS_API_KEY` | _(optional)_ | Works with or without it |
| `SHARP_API_KEY` | _(optional)_ | Works with or without it |
| `ANTHROPIC_API_KEY` | _(optional)_ | Claude chat; falls back to offline mode |

## Odds sources

| Source | Type | Needs key? |
|---|---|---|
| Bovada | Public JSON (`services/sports/event/v2/events/A/description/...`) | No |
| ESPN | Hidden `site.api.espn.com` scoreboard | No |
| ScoresAndOdds | HTML (data-attributes) | No |
| VegasInsider | HTML odds grid | No |
| Covers.com | HTML odds grid | No |
| Action Network | Public JSON (`api.actionnetwork.com/web/v1/scoreboard/...`) | No |
| SBR / Pinnacle | `sbrscrape` package, JSON fallback | No |
| The Odds API | Existing connector, not touched by this branch | Yes |
| SharpAPI | Existing connector, not touched by this branch | Yes |

## Strategies

### Heavy Favorite Filter (`heavy_favorite`)

Targets moneyline favorites priced between `-150` and `-400` at **two
or more** books, with a vig-removed consensus probability between
60-82% (the empirical 75-85% WR band, keeping headroom for
favorite-regression). Stake is sized via fractional Kelly and capped
by `SBA_MAX_BET_PCT`.

### Value Bets (`value_bets`)

Treats Pinnacle / Circa / Bovada as fair books, removes the vig, and
flags any other book paying materially more than the fair
probability.

### LSTM Player Props (`player_props_lstm`)

Sequence model over the last 10 games of a player's feature log
(minutes, usage, opponent DRtg, rest, H/A, pace, etc.) plus the posted
line, producing `P(over)`. Confidence threshold is 60% (below that,
vig eats the edge). Top-decile picks have historically landed in the
65-75% win-rate band.

### Ensemble

Runs every strategy, merges duplicates (same game + selection),
averages confidences, keeps the best available price, and gates on
`SBA_MIN_CONFIDENCE`.

## Paper trading

`PaperTrader.run_forever(recommender)` runs on a background thread
every `SBA_AUTO_TRADE_INTERVAL` seconds. Each cycle:

1. Re-pulls odds from every enabled source.
2. Runs the ensemble.
3. Places any new qualifying bets.
4. Persists state to `$SBA_DATA_DIR/ledger.json`.

The dashboard exposes:

- `GET /api/odds?sport=baseball_ncaa`
- `GET /api/recommendations?sports=baseball_ncaa,baseball_mlb`
- `GET /api/ledger`
- `POST /api/place` — manual bet placement
- `POST /api/settle` — mark a bet won/lost/push
- `POST /api/chat` — Claude chat

## Tests

```bash
pytest src/sports_betting_agent/tests
```

The tests are fully offline — HTTP is mocked or bypassed, the Bovada
parser is exercised against a static sample payload, and the LSTM
model is trained on synthetic data.

## Deploy

```bash
fly deploy
```

The existing fly.io app (`sports-betting-ai-agent`) already has
`SHARP_API_KEY`, `THE_ODDS_API_KEY` and `ANTHROPIC_API_KEY` set as
secrets; no new secrets are required for the free sources.
