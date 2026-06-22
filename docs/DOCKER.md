---
title: Docker Guide — ball-quant
updated: 2026-06-14
tldr: |
  Build: make docker-build. Tests: make docker-test.
  Dev shell with live-mount: make docker-dev.
  Pass API key via API_FOOTBALL_KEY env var — never baked into image.
---

# Docker Guide

## Quick start

```bash
# Build the runtime image
make docker-build

# Run the full pytest suite in a container
make docker-test

# Drop into an interactive dev shell with src/tests/data/reports live-mounted
make docker-dev

# Run backtest against ./data
make docker-backtest
```

## Image stages

| Stage      | FROM     | Purpose                                      |
|------------|----------|----------------------------------------------|
| `base`     | python:3.9-slim | `pip install .` — zero third-party deps |
| `test`     | base     | adds `.[dev]` (pytest, coverage) + tests/    |
| `runtime`  | base     | non-root user `ballq`, `ENTRYPOINT ["ballq"]`|

## Passing the API key

`API_FOOTBALL_KEY` is optional and never baked into the image. Pass it at runtime:

```bash
# one-shot
docker run --rm -e API_FOOTBALL_KEY=xxx ball-quant:latest capture

# via .env file (docker compose picks this up automatically)
echo "API_FOOTBALL_KEY=xxx" > .env
make docker-dev
```

## Live-editing (dev service)

`make docker-dev` (or `docker compose run --rm dev`) mounts:

- `./src`     → `/app/src`     (read-only)
- `./tests`   → `/app/tests`   (read-only)
- `./data`    → `/app/data`
- `./reports` → `/app/reports`

Inside the container you can run `python -m pytest -q`, `ballq backtest`, etc. and
edits to `./src` on the host are reflected immediately (package is installed
editable-style via the mounted volume — re-install with `pip install -e .` inside
the shell if needed after structural changes).

## Running the harness loop in-container

```bash
docker compose run --rm dev bash -c "ballq run"
```

Or override CMD in compose:

```yaml
command: ["run"]
```

## Manual docker build without Makefile

```bash
# runtime image
docker build --target runtime -t ball-quant:latest .

# test image
docker build --target test -t ball-quant:test .
docker run --rm ball-quant:test          # runs pytest -q

# check help
docker run --rm ball-quant:latest --help
```
