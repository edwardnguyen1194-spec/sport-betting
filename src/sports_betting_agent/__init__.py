"""Sports Betting AI Agent.

A multi-source odds aggregator, strategy engine, and 24/7 paper-trading
bot for sports betting research. Includes:

- Free, unlimited odds from Bovada, ESPN, ScoresAndOdds, VegasInsider,
  Covers.com and Action Network (in addition to existing SBR / SharpAPI
  / The Odds API connectors).
- A heavy-favorite filter strategy tuned for 75-85% expected win rate.
- An LSTM neural-network player-props model (65-75% target WR).
- Confidence scoring and Kelly-criterion stake sizing.
- Flask dashboard with Vietnamese UI and Claude AI chat.

See :mod:`sports_betting_agent.config` for runtime configuration.
"""

from .config import Settings, get_settings

__all__ = ["Settings", "get_settings", "__version__"]

__version__ = "0.3.0"
