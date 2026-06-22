.PHONY: install test cov backtest optimize capture settle run clean \
        docker-build docker-test docker-dev docker-backtest

install:
	python3 -m pip install -e ".[dev]"

test:
	python3 -m pytest -q

cov:
	coverage run -m pytest
	coverage report

backtest:
	ballq backtest

optimize:
	ballq optimize

capture:
	ballq capture

settle:
	ballq settle

run:
	ballq run

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf .coverage htmlcov dist build *.egg-info

# ── Docker targets ────────────────────────────────────────────────────────────
docker-build:
	docker build --target runtime -t ball-quant:latest .

docker-test:
	docker compose run --rm test

docker-dev:
	docker compose run --rm dev

docker-backtest:
	docker compose run --rm backtest
