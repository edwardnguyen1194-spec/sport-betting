"""Flask dashboard package.

Call :func:`create_app` to obtain a Flask application with Vietnamese
UI, Claude chat endpoint, and JSON APIs for odds and the paper-trading
ledger.
"""

from .app import create_app

__all__ = ["create_app"]
