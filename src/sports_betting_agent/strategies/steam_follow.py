"""Steam-follow strategy.

When multiple sharp books (Pinnacle, Circa, Bovada, Bookmaker,
BetOnline) move a line the same direction inside a short window,
that is professional money hitting a side. Following those moves —
before the rest of the market catches up — is one of the most
consistently profitable plays documented in the modern betting
literature (Pinnacle's own research, ThunderBet, PicktheOdds,
TheSharpEdge).

This strategy consumes the output of ``LineMovementStore.detect_steam``
and turns each qualifying move into a BetRecommendation on the same
selection, priced at the best currently-posted sharp price.

Rules
-----

* Spreads and totals only. Uncle does not want moneyline bets.
* We only bet in the direction of ``shorten`` moves (the selection
  is getting more expensive because sharps are hammering it). A
  ``lengthen`` move means the market is *selling* that side — we
  do not bet the other side automatically, because we do not know
  whether the move is a sharp lay or simply soft money running the
  other way.
* We need the selection to still be quoted at a sharp book right
  now so we know a real, honest price to bet.
* Confidence scales with the number of sharp books moving and the
  magnitude of the move, capped conservatively at 63%.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..line_movement import LineMovementStore, SteamMove
from ..models_schema import GameOdds, american_to_implied, kelly_fraction
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


SHARP_BOOKS = {"pinnacle", "circa", "bovada", "bookmaker", "betonline"}
# Confidence cap tightened from 0.63 -> 0.53. A line shortening at a
# sharp book is *not* the same as a 63%-probability event — it could
# just as easily be public money piling in. Without public-betting-%
# data we can't distinguish sharp steam from square steam. Capping at
# 0.53 keeps Kelly sizing near the minimum bet, so one bad call can't
# wipe out a good run.
CONFIDENCE_CAP = 0.53
# Require at least 3 sharp books moving together for a cleaner signal
# (single-book moves are often just that book rebalancing its own book).
MIN_SHARP_BOOKS = 3
# Minimum edge in probability space before we emit the pick.
MIN_EDGE = 0.02


class SteamFollowStrategy(Strategy):
    name = "steam_follow"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        line_store: Optional[LineMovementStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.line_store = line_store or LineMovementStore(self.settings.data_dir)

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        games_list = list(games)
        games_by_key = {g.game_key: g for g in games_list}
        try:
            moves = self.line_store.detect_steam()
        except Exception as exc:
            logger.warning("steam_follow: detect_steam failed: %s", exc)
            return []

        recs: List[BetRecommendation] = []
        cfg = self.settings

        for move in moves:
            # Uncle's rule: spreads + totals only, no moneyline.
            if move.market not in ("spread", "total"):
                continue
            # Only follow true sharp hammers — books buying the side,
            # not selling it off.
            if move.direction != "shorten":
                continue
            # Require multi-book consensus — a single sharp moving alone
            # is often just that book adjusting its own position, not
            # real professional action.
            if len(move.sharp_books_moving) < MIN_SHARP_BOOKS:
                continue
            # Skip -1.5 favorite runlines/pucklines at plus money: these
            # are the single largest variance bucket in the dataset and
            # steam direction is too ambiguous to justify the swing.
            if (
                move.market == "spread"
                and move.selection
                and "under" not in move.selection.lower()
                and "over" not in move.selection.lower()
            ):
                # Defer the -1.5 filter to the candidate line — we don't
                # have the handicap on the SteamMove object itself.
                pass

            # Game-matching strategy. First try exact game_key (same
            # matchup + same day). If the steam move was stored for a
            # different day's game, fall back to matching by the teams
            # named in the move — this lets a real sharp signal carry
            # across days, e.g. a Royals spread that saw steam last
            # night still applies to Royals' next spread tonight only
            # if we still trust the read; we take that risk for spreads
            # (team-level), not for totals (game-level only).
            game = games_by_key.get(move.game_key)
            if game is None and move.market == "spread":
                game = self._find_game_by_team(games_list, move.selection)
            if game is None:
                continue

            # Find the cheapest currently-posted sharp price on the
            # selection the sharps were buying.
            candidate = self._best_current_line(game, move)
            if candidate is None:
                continue

            # Skip -1.5 (or more negative) favorite runlines/pucklines.
            # The dataset shows these are the biggest variance bucket —
            # we hit them 33% at plus money and 33% is break-even at
            # around +200 but not at +135, and Kelly sizing was
            # magnifying every miss. Prefer +1.5 dog runlines where the
            # base rate is closer to 65-70% cover.
            if (
                candidate.market == "spread"
                and candidate.line is not None
                and candidate.line < 0
            ):
                continue

            confidence = self._confidence(move)
            implied = american_to_implied(candidate.american)
            edge = confidence - implied
            if edge < MIN_EDGE:
                continue

            stake_frac = min(
                kelly_fraction(confidence, candidate.decimal, cfg.kelly_fraction),
                cfg.max_bet_pct,
            )

            display_sel = self._display_selection(game, move, candidate)

            recs.append(
                BetRecommendation(
                    game_key=game.game_key,
                    sport=game.sport,
                    league=game.league,
                    home_team=game.home_team,
                    away_team=game.away_team,
                    market=move.market,
                    selection=display_sel,
                    american=candidate.american,
                    decimal=candidate.decimal,
                    line=candidate.line,
                    book=candidate.book,
                    strategy=self.name,
                    confidence=round(confidence, 4),
                    edge=round(edge, 4),
                    stake_fraction=round(stake_frac, 4),
                    reasoning=(
                        f"Steam follow: {len(move.sharp_books_moving)} sharp books "
                        f"({', '.join(move.sharp_books_moving)}) shortened {display_sel} "
                        f"by {abs(move.juice_delta):.0f}¢; riding the move at "
                        f"{candidate.american:+.0f} on {candidate.book}."
                    ),
                    sources=sorted(move.sharp_books_moving),
                )
            )

        logger.info("steam_follow: %d recs from %d steam moves", len(recs), len(moves))
        return recs

    # -- helpers -----------------------------------------------------

    def _find_game_by_team(self, games, team_norm: str):
        """Locate today's game whose home or away team matches ``team_norm``.

        The steam store lowercases selections, so we compare normalized.
        This is the fallback when the steam record's game_key belongs to
        a past matchup of the same team.
        """
        needle = team_norm.strip().lower()
        for g in games:
            if g.home_team.lower() == needle or g.away_team.lower() == needle:
                return g
        return None

    def _best_current_line(self, game: GameOdds, move: SteamMove):
        """Return the cheapest-to-buy sharp line on ``move.selection``."""
        candidates = []
        for line in game.lines:
            if line.market != move.market:
                continue
            if line.american is None or line.decimal is None:
                continue
            if line.book.lower() not in SHARP_BOOKS:
                continue
            if line.selection.lower().strip() != move.selection.lower().strip():
                continue
            candidates.append(line)
        if not candidates:
            return None
        # Cheapest to buy = highest decimal odds.
        return max(candidates, key=lambda l: l.decimal or 0.0)

    def _confidence(self, move: SteamMove) -> float:
        """Map a steam move to a capped confidence.

        The signal combines two things: how many sharp books moved
        together (consensus) and how far the summed juice moved
        (magnitude). Each contributes a diminishing-returns share,
        and we cap the sum at CONFIDENCE_CAP so Kelly sizing stays
        disciplined even on monster moves.
        """
        consensus = len(move.sharp_books_moving)
        consensus_score = 1.0 - math.exp(-0.35 * (consensus - 1))   # 0 at 1 book, ~0.5 at 3 books
        magnitude = abs(move.juice_delta) / 100.0                    # cents of juice
        magnitude_score = 1.0 - math.exp(-magnitude / 2.0)            # ~0.4 at 1 unit, ~0.86 at 4

        raw = 0.5 + 0.5 * (consensus_score * 0.6 + magnitude_score * 0.4)
        return min(CONFIDENCE_CAP, raw)

    @staticmethod
    def _display_selection(game: GameOdds, move: SteamMove, line) -> str:
        """Prefer the original cased selection from the live odds line."""
        if line.selection:
            return line.selection
        # Fall back to the normalized name from the steam move — map back
        # to the actual team casing if we can.
        normalized = move.selection.strip()
        for team in (game.home_team, game.away_team):
            if team.lower() == normalized:
                return team
        return normalized
