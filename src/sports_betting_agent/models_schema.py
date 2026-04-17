"""Shared normalized odds schema used by every fetcher.

Every source (Bovada, ESPN, SBR, SharpAPI, The Odds API, ...) returns
a list of :class:`GameOdds` objects so the aggregator, strategies, and
dashboard all speak one language.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional


# ---------------------------------------------------------------------------
# Odds conversion helpers
# ---------------------------------------------------------------------------


def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""

    if american == 0:
        return 1.0
    if american > 0:
        return 1.0 + american / 100.0
    return 1.0 + 100.0 / abs(american)


def decimal_to_american(decimal: float) -> int:
    """Convert decimal odds to American odds (rounded)."""

    if decimal <= 1.0:
        return 0
    if decimal >= 2.0:
        return int(round((decimal - 1.0) * 100))
    return int(round(-100.0 / (decimal - 1.0)))


def american_to_implied(american: float) -> float:
    """Implied probability from American odds (no vig removal)."""

    if american == 0:
        return 0.0
    if american > 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def remove_vig_two_way(p_home: float, p_away: float) -> tuple[float, float]:
    """Remove the overround on a two-way market."""

    total = p_home + p_away
    if total <= 0:
        return p_home, p_away
    return p_home / total, p_away / total


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class OddsLine:
    """A single posted line at one sportsbook."""

    book: str                           # e.g. "bovada", "pinnacle"
    market: str                         # moneyline / spread / total / player_prop
    selection: str                      # team or player name, Over/Under, etc.
    american: Optional[float] = None
    decimal: Optional[float] = None
    line: Optional[float] = None        # handicap / total line
    player: Optional[str] = None        # for player props
    prop: Optional[str] = None          # e.g. "points", "hits", "strikeouts"
    last_update: Optional[datetime] = None

    def __post_init__(self) -> None:
        # Keep both american and decimal populated so downstream code can
        # pick whichever is cheaper to work with.
        if self.american is not None and self.decimal is None:
            self.decimal = american_to_decimal(self.american)
        elif self.decimal is not None and self.american is None:
            self.american = decimal_to_american(self.decimal)

    @property
    def implied_probability(self) -> float:
        if self.american is None:
            return 0.0
        return american_to_implied(self.american)

    def to_dict(self) -> Dict:
        data = asdict(self)
        if self.last_update is not None:
            data["last_update"] = self.last_update.isoformat()
        return data


@dataclass
class GameOdds:
    """Normalized odds for a single game across one or many books."""

    sport: str                              # baseball_ncaa, basketball_nba, ...
    league: str                             # "college-baseball", "nba", ...
    home_team: str
    away_team: str
    commence_time: Optional[datetime] = None
    source: str = "unknown"                 # origin fetcher
    event_id: Optional[str] = None
    lines: List[OddsLine] = field(default_factory=list)
    meta: Dict = field(default_factory=dict)

    @property
    def game_key(self) -> str:
        """Stable key used to merge the same game across sources.

        Uses only league + normalized team names (no time) so that
        sources with and without commence_time merge correctly.
        The date portion uses only the date (not time) when available,
        to handle doubleheaders where the same teams play twice.
        """
        date_part = ""
        if self.commence_time is not None:
            date_part = self.commence_time.astimezone(timezone.utc).strftime("%Y%m%d")
        raw = f"{self.league}|{_norm(self.home_team)}|{_norm(self.away_team)}|{date_part}"
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def moneyline(self) -> Dict[str, List[OddsLine]]:
        return {
            "home": [l for l in self.lines if l.market == "moneyline" and _norm(l.selection) == _norm(self.home_team)],
            "away": [l for l in self.lines if l.market == "moneyline" and _norm(l.selection) == _norm(self.away_team)],
        }

    def best_moneyline(self, side: str) -> Optional[OddsLine]:
        """Return the best (highest-paying) ML line for side in {home, away}."""

        ml = self.moneyline()[side]
        if not ml:
            return None
        # "Best" for the bettor == highest decimal payout.
        return max(ml, key=lambda l: l.decimal or 0.0)

    def consensus_implied(self, side: str) -> Optional[float]:
        """Vig-removed consensus implied probability for side."""

        ml = self.moneyline()
        homes = ml["home"]
        aways = ml["away"]
        if not homes or not aways:
            return None
        p_home = sum(l.implied_probability for l in homes) / len(homes)
        p_away = sum(l.implied_probability for l in aways) / len(aways)
        p_home, p_away = remove_vig_two_way(p_home, p_away)
        return p_home if side == "home" else p_away

    def spread_lines_grouped(self) -> Dict[tuple, List[OddsLine]]:
        """Group spread lines by (normalized_selection, handicap)."""
        groups: Dict[tuple, List[OddsLine]] = {}
        for l in self.lines:
            if l.market != "spread" or l.line is None or l.american is None:
                continue
            key = (_norm(l.selection), l.line)
            groups.setdefault(key, []).append(l)
        return groups

    def total_lines_grouped(self) -> Dict[tuple, List[OddsLine]]:
        """Group total lines by (Over/Under, total_number)."""
        groups: Dict[tuple, List[OddsLine]] = {}
        for l in self.lines:
            if l.market != "total" or l.line is None or l.american is None:
                continue
            key = (l.selection.lower(), l.line)
            groups.setdefault(key, []).append(l)
        return groups

    def best_line_for(self, market: str, selection: str,
                      handicap: Optional[float] = None) -> Optional[OddsLine]:
        """Best (highest decimal payout) line for a market/selection/handicap."""
        candidates = [
            l for l in self.lines
            if l.market == market
            and _norm(l.selection) == _norm(selection)
            and l.decimal is not None
            and (handicap is None or l.line == handicap)
        ]
        return max(candidates, key=lambda l: l.decimal or 0.0) if candidates else None

    def to_dict(self) -> Dict:
        data = {
            "sport": self.sport,
            "league": self.league,
            "home_team": self.home_team,
            "away_team": self.away_team,
            "commence_time": self.commence_time.isoformat() if self.commence_time else None,
            "source": self.source,
            "event_id": self.event_id,
            "game_key": self.game_key,
            "lines": [l.to_dict() for l in self.lines],
            "meta": self.meta,
        }
        return data


_MASCOT_SUFFIXES = (
    "blue devils", "cavaliers", "crimson tide", "tar heels", "fighting irish",
    "red raiders", "sun devils", "golden bears", "golden eagles", "golden knights",
    "horned frogs", "blue raiders", "ragin cajuns", "demon deacons",
    "mountaineers", "volunteers", "commodores", "wolverines", "spartans",
    "buckeyes", "longhorns", "sooners", "razorbacks", "gators", "seminoles",
    "hurricanes", "hoyas", "jayhawks", "hoosiers", "boilermakers", "gamecocks",
    "tigers", "bulldogs", "wildcats", "eagles", "hawks", "lions", "bears",
    "panthers", "ducks", "beavers", "cougars", "cowboys", "rams", "mustangs",
    "knights", "cardinals", "cardinal", "huskies", "owls", "mavericks",
    "bobcats", "broncos", "bulls", "bison", "falcons", "flames", "grizzlies",
    "hornets", "jaguars", "mavericks", "musketeers", "orange", "pirates",
    "terrapins", "toreros", "utes", "vandals", "yellow jackets",
)


def _norm(name: str) -> str:
    """Loose team-name normalization for cross-book merging.

    Lowercases, strips common college/league suffix words, strips
    well-known mascot words, and finally reduces to an alphanumeric
    string. This lets "Duke" and "Duke Blue Devils" hash to the same
    key without exploding into a huge alias table.
    """

    if not name:
        return ""
    cleaned = name.lower().strip()
    for token in (
        "university of ",
        " university",
        " college",
        " baseball",
        " football",
        " basketball",
        " hockey",
        " state",
    ):
        cleaned = cleaned.replace(token, "")
    for mascot in _MASCOT_SUFFIXES:
        if cleaned.endswith(" " + mascot) or cleaned == mascot:
            cleaned = cleaned[: -len(mascot)].strip()
        elif cleaned.endswith(mascot):
            cleaned = cleaned[: -len(mascot)].strip()
    return "".join(ch for ch in cleaned if ch.isalnum())


def merge_games(games: Iterable[GameOdds]) -> List[GameOdds]:
    """Merge `GameOdds` from multiple sources by game_key.

    Preserves per-source meta fields across sources — first non-None
    value wins. Critical for public-betting percentages: only
    ActionNetwork populates ``public_spread_home_pct`` etc., and prior
    to this fix those keys were dropped during merge, silently starving
    RLM and PublicFade of their input signal.
    """

    bucket: Dict[str, GameOdds] = {}
    for g in games:
        key = g.game_key
        if key not in bucket:
            merged_meta = {"sources": [g.source]}
            for mk, mv in (g.meta or {}).items():
                if mk != "sources":
                    merged_meta[mk] = mv
            bucket[key] = GameOdds(
                sport=g.sport,
                league=g.league,
                home_team=g.home_team,
                away_team=g.away_team,
                commence_time=g.commence_time,
                source="merged",
                event_id=g.event_id,
                lines=list(g.lines),
                meta=merged_meta,
            )
        else:
            existing = bucket[key]
            existing.lines.extend(g.lines)
            sources = existing.meta.setdefault("sources", [])
            if g.source not in sources:
                sources.append(g.source)
            # Propagate commence_time if the existing entry lacks one.
            if existing.commence_time is None and g.commence_time is not None:
                existing.commence_time = g.commence_time
            # Propagate any meta key the first-seen source lacked.
            # First non-None value wins; e.g. Bovada arrives first with
            # no public-%, AN arrives second with them populated.
            for mk, mv in (g.meta or {}).items():
                if mk == "sources" or mv is None:
                    continue
                if mk not in existing.meta or existing.meta.get(mk) is None:
                    existing.meta[mk] = mv
    return list(bucket.values())


def kelly_fraction(p: float, decimal_odds: float, fraction: float = 0.25) -> float:
    """Fractional Kelly stake (as a proportion of bankroll).

    Returns 0 on negative-EV bets. ``fraction`` applies the classic
    "quarter Kelly" risk dampening (or any other multiplier).
    """

    b = decimal_odds - 1.0
    if b <= 0:
        return 0.0
    q = 1.0 - p
    edge = (b * p - q) / b
    if edge <= 0 or math.isnan(edge):
        return 0.0
    return max(0.0, edge * fraction)
