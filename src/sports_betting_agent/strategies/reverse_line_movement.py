"""Reverse Line Movement (RLM) strategy.

The most consistently documented edge in modern sports betting
literature: when a large share of public tickets (>60%) backs one
side but the line moves AGAINST that side at 2+ sharp books, the
sharp money is on the unpopular side. Historical ATS win-rate sits
at 56-58% (Pinnacle research + SSRN replications).

RLM is the intersection of two signals:
  1. Public-betting % strongly lopsided (from game.meta).
  2. Line movement opposite to the public side across sharp books
     (from LineMovementStore).

Both together is a much stronger signal than either alone, which is
why this strategy's confidence cap (0.60) is higher than the
PublicFadeStrategy (0.58) and SteamFollowStrategy (0.63) used in
isolation.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from ..config import Settings, get_settings
from ..line_movement import LineMovementStore, SteamMove
from ..models_schema import GameOdds, american_to_implied, kelly_fraction
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


SHARP_BOOKS = {"pinnacle", "circa", "bovada", "bookmaker", "betonline"}
PUBLIC_THRESHOLD = 0.60      # ≥60% of tickets on one side = lopsided
CONFIDENCE_CAP = 0.60        # higher than single-signal strategies
MIN_EDGE = 0.02


class ReverseLineMovementStrategy(Strategy):
    """Bet against heavy public action when the line moves the other way."""

    name = "reverse_line_movement"

    def __init__(
        self,
        settings: Optional[Settings] = None,
        line_store: Optional[LineMovementStore] = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.line_store = line_store or LineMovementStore(self.settings.data_dir)

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        games_list = list(games)
        try:
            steam_moves = self.line_store.detect_steam()
        except Exception as exc:
            logger.warning("rlm: detect_steam failed: %s", exc)
            return []

        # Index steam moves by (game_key, market) for fast lookup.
        moves_by_game: dict[tuple[str, str], list[SteamMove]] = {}
        for m in steam_moves:
            if m.market not in ("spread", "total"):
                continue
            moves_by_game.setdefault((m.game_key, m.market), []).append(m)

        recs: List[BetRecommendation] = []
        cfg = self.settings

        for game in games_list:
            meta = game.meta or {}
            # RLM on spreads
            recs.extend(self._spread_pick(game, meta, moves_by_game, cfg))
            # RLM on totals
            recs.extend(self._total_pick(game, meta, moves_by_game, cfg))

        logger.info("rlm: %d recs", len(recs))
        return recs

    # -- spreads -----------------------------------------------------

    def _spread_pick(self, game, meta, moves_by_game, cfg):
        home_pct = meta.get("public_spread_home_pct")
        away_pct = meta.get("public_spread_away_pct")
        if home_pct is None or away_pct is None:
            return []

        # Identify the PUBLIC-heavy side (which we'll fade).
        if home_pct >= PUBLIC_THRESHOLD:
            public_side = game.home_team
            public_pct = home_pct
            fade_side = game.away_team
        elif away_pct >= PUBLIC_THRESHOLD:
            public_side = game.away_team
            public_pct = away_pct
            fade_side = game.home_team
        else:
            return []

        # Check for line movement AGAINST the public side. A sharp
        # "shorten" move on the fade_side means sharps are buying
        # that unpopular side — exactly the RLM pattern.
        moves = moves_by_game.get((game.game_key, "spread"), [])
        rlm_move = None
        for m in moves:
            if m.direction != "shorten":
                continue
            if m.selection.lower().strip() == fade_side.lower().strip():
                rlm_move = m
                break
        if rlm_move is None:
            return []

        # Find the best currently-posted sharp price on fade_side.
        candidate = self._best_sharp_line(game, "spread", fade_side)
        if candidate is None:
            return []

        confidence = self._confidence(public_pct, rlm_move)
        implied = american_to_implied(candidate.american)
        edge = confidence - implied
        if edge < MIN_EDGE:
            return []

        stake_frac = min(
            kelly_fraction(confidence, candidate.decimal, cfg.kelly_fraction),
            cfg.max_bet_pct,
        )

        return [BetRecommendation(
            game_key=game.game_key,
            sport=game.sport,
            league=game.league,
            home_team=game.home_team,
            away_team=game.away_team,
            market="spread",
            selection=fade_side,
            american=candidate.american,
            decimal=candidate.decimal,
            line=candidate.line,
            book=candidate.book,
            strategy=self.name,
            confidence=round(confidence, 4),
            edge=round(edge, 4),
            stake_fraction=round(stake_frac, 4),
            reasoning=(
                f"RLM: public {public_pct:.0%} on {public_side} but "
                f"{len(rlm_move.sharp_books_moving)} sharp books "
                f"({', '.join(rlm_move.sharp_books_moving)}) moved the line "
                f"toward {fade_side} — taking {fade_side} {candidate.line:+.1f} "
                f"at {candidate.american:+.0f} on {candidate.book}."
            ),
            sources=sorted(set(rlm_move.sharp_books_moving) | {candidate.book}),
        )]

    # -- totals ------------------------------------------------------

    def _total_pick(self, game, meta, moves_by_game, cfg):
        over_pct = meta.get("public_total_over_pct")
        under_pct = meta.get("public_total_under_pct")
        if over_pct is None or under_pct is None:
            return []

        if over_pct >= PUBLIC_THRESHOLD:
            public_sel = "Over"
            public_pct = over_pct
            fade_sel = "Under"
        elif under_pct >= PUBLIC_THRESHOLD:
            public_sel = "Under"
            public_pct = under_pct
            fade_sel = "Over"
        else:
            return []

        moves = moves_by_game.get((game.game_key, "total"), [])
        rlm_move = None
        for m in moves:
            if m.direction != "shorten":
                continue
            if m.selection.lower().strip() == fade_sel.lower():
                rlm_move = m
                break
        if rlm_move is None:
            return []

        candidate = self._best_sharp_line(game, "total", fade_sel)
        if candidate is None:
            return []

        confidence = self._confidence(public_pct, rlm_move)
        implied = american_to_implied(candidate.american)
        edge = confidence - implied
        if edge < MIN_EDGE:
            return []

        stake_frac = min(
            kelly_fraction(confidence, candidate.decimal, cfg.kelly_fraction),
            cfg.max_bet_pct,
        )

        return [BetRecommendation(
            game_key=game.game_key,
            sport=game.sport,
            league=game.league,
            home_team=game.home_team,
            away_team=game.away_team,
            market="total",
            selection=fade_sel,
            american=candidate.american,
            decimal=candidate.decimal,
            line=candidate.line,
            book=candidate.book,
            strategy=self.name,
            confidence=round(confidence, 4),
            edge=round(edge, 4),
            stake_fraction=round(stake_frac, 4),
            reasoning=(
                f"RLM: public {public_pct:.0%} on {public_sel} but "
                f"{len(rlm_move.sharp_books_moving)} sharp books moved "
                f"toward {fade_sel} — taking {fade_sel} {candidate.line} "
                f"at {candidate.american:+.0f} on {candidate.book}."
            ),
            sources=sorted(set(rlm_move.sharp_books_moving) | {candidate.book}),
        )]

    # -- helpers -----------------------------------------------------

    def _best_sharp_line(self, game, market, selection):
        candidates = []
        for l in game.lines:
            if l.market != market:
                continue
            if l.american is None or l.decimal is None or l.line is None:
                continue
            if l.book.lower() not in SHARP_BOOKS:
                continue
            if l.selection.lower().strip() != selection.lower().strip():
                continue
            candidates.append(l)
        if not candidates:
            return None
        return max(candidates, key=lambda l: l.decimal or 0.0)

    @staticmethod
    def _confidence(public_pct: float, move: SteamMove) -> float:
        """Blend public-lopsidedness and sharp-move strength into a
        single confidence. Both signals pointing the same way is the
        whole point of RLM — so we reward joint strength.
        """
        # Public lopsidedness: ramp 0 at threshold to 0.08 at 85%.
        pub_score = max(0.0, (public_pct - PUBLIC_THRESHOLD)) / (0.85 - PUBLIC_THRESHOLD)
        pub_score = min(1.0, pub_score) * 0.08
        # Sharp consensus: +0.03 per sharp book beyond the first.
        sharp_score = 0.03 * max(0, len(move.sharp_books_moving) - 1)
        raw = 0.52 + pub_score + sharp_score
        return min(CONFIDENCE_CAP, raw)
