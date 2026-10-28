.PHONY: setup data train bench figures test lint all clean

setup:
	uv sync

data:
	uv run python scripts/fetch_data.py

train:
	uv run python scripts/run_train.py --config configs/train_50m.toml

bench:
	uv run python scripts/run_bench.py --config configs/bench.toml

figures:
	uv run python scripts/plot.py

test:
	uv run pytest -q

lint:
	uv run ruff check src scripts

all: data train bench figures

clean:
	rm -rf checkpoints results/metrics/*.json results/metrics/*.csv
