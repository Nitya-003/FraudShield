"""
FastAPI entrypoint re-exporting ml.api:app.
"""
from ml.api import app

__all__ = ["app"]

