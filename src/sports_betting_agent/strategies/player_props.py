"""Player-props strategy backed by the LSTM model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from ..config import Settings, get_settings
from ..models.lstm_player_props import LSTMPlayerPropsModel, PlayerGameLog
from ..models_schema import GameOdds, OddsLine, american_to_decimal, kelly_fraction
from .base import BetRecommendation, Strategy


logger = logging.getLogger(__name__)


@dataclass
class PropMarket:
    """An outstanding player prop posted at a sportsbook."""

    player: str
    prop: str
    line: float
    over_price: float = -110.0
    under_price: float = -110.0
    book: str = ""
    game_key: str = ""
    sport: str = ""
    league: str = ""
    home_team: str = ""
    away_team: str = ""


class PlayerPropsStrategy(Strategy):
    """Consumes posted player props + recent game logs and emits picks."""

    name = "player_props_lstm"

    def __init__(
        self,
        model: LSTMPlayerPropsModel,
        logs_by_player: Dict[str, PlayerGameLog],
        settings: Optional[Settings] = None,
    ) -> None:
        self.model = model
        self.logs = logs_by_player
        self.settings = settings or get_settings()

    def generate_from_props(self, props: Iterable[PropMarket]) -> List[BetRecommendation]:
        recs: List[BetRecommendation] = []
        for market in props:
            log = self.logs.get(market.player)
            if log is None:
                continue
            decision = self.model.predict_recommendation(
                log,
                market.line,
                market.player,
                over_price=market.over_price,
                under_price=market.under_price,
            )
            if decision is None:
                continue
            stake_frac = min(
                kelly_fraction(decision["confidence"], decision["decimal"], self.settings.kelly_fraction),
                self.settings.max_bet_pct,
            )
            recs.append(
                BetRecommendation(
                    game_key=market.game_key,
                    sport=market.sport,
                    league=market.league,
                    home_team=market.home_team,
                    away_team=market.away_team,
                    market="player_prop",
                    selection=f"{market.player} {decision['selection']} {market.line} {market.prop}",
                    american=decision["american"],
                    decimal=decision["decimal"],
                    line=market.line,
                    book=market.book,
                    strategy=self.name,
                    confidence=decision["confidence"],
                    edge=decision["edge"],
                    stake_fraction=round(stake_frac, 4),
                    reasoning=(
                        f"LSTM predicts {decision['confidence']:.1%} {decision['selection']} "
                        f"{market.line} {market.prop} for {market.player}. "
                        f"Target WR band 65-75%."
                    ),
                    meta={"prop": market.prop, "player": market.player},
                )
            )
        return recs

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        # The generic Strategy.generate contract doesn't carry props.
        # Most callers should use generate_from_props. We also expose
        # a "derive-from-games" path for sources (like Bovada) whose
        # GameOdds objects may include player-prop OddsLine entries.
        props: List[PropMarket] = []
        for game in games:
            for line in game.lines:
                if line.market != "player_prop" or not line.player or line.line is None:
                    continue
                side = line.selection or ""
                if side.lower().startswith("o"):
                    # Skip props with no real juice — previously we
                    # defaulted to -110 which inflated EV math and
                    # produced phantom prop recommendations at prices
                    # the book never actually offered.
                    if line.american is None:
                        continue
                    props.append(
                        PropMarket(
                            player=line.player,
                            prop=line.prop or "",
                            line=line.line,
                            over_price=line.american,
                            book=line.book,
                            game_key=game.game_key,
                            sport=game.sport,
                            league=game.league,
                            home_team=game.home_team,
                            away_team=game.away_team,
                        )
                    )
        return self.generate_from_props(props)
