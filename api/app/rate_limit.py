"""Rate limiting via slowapi (Starlette-compatible flask-limiter port)."""

from __future__ import annotations

import os

from slowapi import Limiter
from slowapi.util import get_remote_address

# Allow tests to disable rate limiting.
_disabled = os.environ.get("ECG_DISABLE_RATE_LIMIT", "").lower() in ("1", "true", "yes")

limiter = Limiter(
    key_func=get_remote_address,
    default_limits=[],
    enabled=not _disabled,
)
