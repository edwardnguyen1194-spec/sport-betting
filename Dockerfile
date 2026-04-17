FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# System dependencies (lxml/etree parsing for BeautifulSoup is optional; the
# html.parser backend is enough for our HTML scrapers).
RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install gunicorn

COPY pyproject.toml ./
COPY src ./src
RUN pip install -e .

EXPOSE 8080

#
# CRITICAL: single worker only. The PaperTrader keeps the paper-ledger
# as in-memory Python state and periodically saves it to /data/sba/
# ledger.json. With N>1 workers, each process loads a snapshot then
# writes its own version over the file, silently stomping the void
# acks / trade placements made by the other process. Symptom: bets
# voided via /api/settle re-appear after 30-90s because the OTHER
# worker's stale in-memory copy got flushed to disk. Staying at 1
# worker is the correct fix until we move to a real transactional
# store (SQLite + per-bet row locking, or Postgres). 4 threads are
# enough — dashboard requests are I/O-bound (aggregator fetches +
# Anthropic chat) and the GIL release on I/O keeps concurrency fine.
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "1", "--threads", "8", \
     "--timeout", "120", "sports_betting_agent.dashboard:create_app()"]
