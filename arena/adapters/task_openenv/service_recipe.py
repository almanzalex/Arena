"""Fail-loud recipe for separately operated OpenEnv qualification (R-05).

Loopback servers started inside the Arena client/pytest process are not enough
for a 1.0 stable claim. Operators must run the pilot as a separate service and
point the client at ``packaging.base_url`` / ``ARENA_OPENENV_BASE_URL``.
"""

from __future__ import annotations

import os
from urllib.error import URLError
from urllib.request import urlopen

OPENENV_BASE_URL_ENV = "ARENA_OPENENV_BASE_URL"

SEPARATE_SERVICE_RECIPE = """OpenEnv separate-service qualification requires a live service.

Recipe:
  # Docker (preferred for R-05):
  docker compose -f docker/openenv/docker-compose.yml up --build -d
  export ARENA_OPENENV_BASE_URL=http://127.0.0.1:8000

  # Or process (still separate from the Arena client):
  ./examples/openenv/separate_service/run_service.sh --daemon
  export ARENA_OPENENV_BASE_URL=http://127.0.0.1:8000

  # Then qualify / test:
  .venv/bin/python scripts/qualify_openenv_separate_service.py \\
    --out docs/qualifications/openenv
  .venv/bin/pytest -m docker tests/integrations/test_openenv_separate_service.py -q

Never treat an unset/unhealthy endpoint as success, and never flip support-matrix
to stable without recorded evidence under docs/qualifications/openenv/.
"""


def resolve_openenv_base_url(explicit: str | None = None) -> str | None:
    """Return a trimmed base URL from ``explicit`` or the environment."""
    value = (explicit if explicit is not None else os.environ.get(OPENENV_BASE_URL_ENV)) or ""
    value = value.strip().rstrip("/")
    return value or None


def openenv_service_healthy(base_url: str, *, timeout_seconds: float = 2.0) -> bool:
    """Return True when ``{base_url}/health`` responds with HTTP 200."""
    try:
        with urlopen(f"{base_url.rstrip('/')}/health", timeout=timeout_seconds) as response:  # noqa: S310
            return int(response.status) == 200
    except (URLError, TimeoutError, OSError, ValueError):
        return False


def require_openenv_separate_service(explicit: str | None = None) -> str:
    """Return a healthy separate-service base URL or raise RuntimeError with recipe."""
    base_url = resolve_openenv_base_url(explicit)
    if base_url is None:
        raise RuntimeError(SEPARATE_SERVICE_RECIPE)
    if not openenv_service_healthy(base_url):
        raise RuntimeError(
            f"OpenEnv service at {base_url!r} is not healthy.\n\n{SEPARATE_SERVICE_RECIPE}"
        )
    return base_url
