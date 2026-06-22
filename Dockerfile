# syntax=docker/dockerfile:1
# ── base: install runtime package only (zero third-party deps) ──────────────
FROM python:3.9-slim AS base

WORKDIR /app

# Copy only the packaging manifests first so layer is cached on dep changes
COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir .

# ── test: add dev extras and run pytest ─────────────────────────────────────
FROM base AS test

RUN pip install --no-cache-dir ".[dev]"

COPY tests/ ./tests/

CMD ["python", "-m", "pytest", "-q"]

# ── runtime: non-root user, ballq as entrypoint ─────────────────────────────
FROM base AS runtime

# Create a non-root user
RUN useradd --create-home --shell /bin/bash ballq

USER ballq

# ballq is installed by pip into /usr/local/bin (root-owned, world-executable)
ENTRYPOINT ["ballq"]
CMD ["--help"]
