.PHONY: data demo backtest test lint

data:
	uv run wholesale-forecast download

demo:
	uv run wholesale-forecast backtest --demo --trees 200

backtest:
	uv run wholesale-forecast backtest

test:
	uv run pytest -q

lint:
	uv run ruff check src tests
