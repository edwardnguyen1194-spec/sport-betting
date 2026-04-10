"""Strategy contract + recommendation dataclass."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Dict, Iterable, List, Optional

from ..models_schema import GameOdds, OddsLine


@dataclass
class BetRecommendation:
    """A single bet a strategy wants to place."""

    game_key: str
    sport: str
    league: str
    home_team: str
    away_team: str
    market: str                 # moneyline / spread / total / player_prop
    selection: str              # team or player name, Over/Under, etc.
    american: float
    decimal: float
    line: Optional[float] = None
    book: str = ""
    strategy: str = ""
    confidence: float = 0.0     # 0..1 -- predicted win probability
    edge: float = 0.0           # predicted p - implied p (no-vig)
    stake_fraction: float = 0.0 # of bankroll
    stake_amount: float = 0.0
    reasoning: str = ""
    sources: List[str] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return asdict(self)


class Strategy:
    """Base class for a betting strategy."""

    name: str = "base"

    def generate(self, games: Iterable[GameOdds]) -> List[BetRecommendation]:
        raise NotImplementedError

    # ------------------------------------------------------------------

    @staticmethod
    def _best_line(game: GameOdds, market: str, selection: str) -> Optional[OddsLine]:
        best: Optional[OddsLine] = None
        for line in game.lines:
            if line.market != market:
                continue
            if line.selection.lower() != selection.lower():
                continue
            if line.decimal is None:
                continue
            if best is None or (line.decimal or 0.0) > (best.decimal or 0.0):
                best = line
        return best
