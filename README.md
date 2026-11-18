# nofe-seismic

Exploratory dimensionality reduction and spatial-structure evaluation on 3D seismic volumes using **NOFE** ([`laueb4818/workshop-skibotn`](https://github.com/laueb4818/workshop-skibotn)).

## Setup

Requires Python `3.11+` and [`uv`](https://docs.astral.sh/uv/):

```bash
uv sync
uv run pytest -v
uv run ruff check .
```

## Datasets

### 1. Full Netherlands F3 Benchmark (`f3-full`)
With `f3-full` extracted in `./f3-full` (shape `601 x 901 x 255`, zero-padded boundary planes `0` and `254` automatically trimmed to `601 x 901 x 253`):

```bash
uv run python scripts/inspect_data.py --extract-dir f3-full
uv run python scripts/run_experiments.py --dataset f3 --data-dir f3-full
```

Or if starting from `f3-full-ready.zip`:

```bash
uv run python scripts/inspect_data.py --zip-path f3-full-ready.zip --extract-dir f3-full
uv run python scripts/run_experiments.py --dataset f3 --data-dir f3-full
```

### 2. Synthetic 3D seismic cube
Includes dipping reflectors, prograding clinoforms, a buried sinuous channel, a chaotic salt dome, and a normal fault plane with 5 ground-truth facies classes:

```bash
uv run python scripts/run_experiments.py --dataset synthetic --shape 64 64 96 --stride 2 2 2 --epochs 80
```

## Outputs
All experiment runs write structured metrics (`metrics_summary.json`, `benchmark.csv`, `ablations.csv`, `generalization.csv`, `clustering.csv`, `heldout_clustering.csv`) and figures (`figures/*.png`) under `outputs/`.
