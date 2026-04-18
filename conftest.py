"""Root pytest conftest.

Disables the 30-min per-agent decision cache during tests so each
``analyze()`` call in a test genuinely hits the (mocked) router.
Production keeps the cache — it's what prevents Uncle's 10 sub-agents
from burning through Groq's 100K-tokens/day free tier.
"""
import os

# Run BEFORE any sports_betting_agent module imports — pytest loads
# this file first at the session level.
os.environ.setdefault("SBA_DISABLE_AGENT_CACHE", "1")
