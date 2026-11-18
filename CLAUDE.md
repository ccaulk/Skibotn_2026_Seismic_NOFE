# CLAUDE.md

Guidance for working in this repository.

## What this is

Research codebase for NOFE dimensionality reduction experiments on 3D seismic volumes (`f3-full` and synthetic cubes).
Covers 11-channel 3D seismic attribute extraction, 2D depth-log compression, baseline comparisons (PCA, UMAP and t-SNE) and operator generalisation evaluations.

## Commands

Always invoke Python via `uv run` (never bare `python` or `pytest`).

```bash
just check       # ruff format (rewrites) + ty check + ruff check + full pytest
just verify      # read-only gate (format --check, skips -m slow); used before committing
just fix         # ruff format + ruff check --fix + ty check --fix
just test *args  # pytest; forwards args (e.g. `just test -k "data"`)
just cov         # coverage report (diagnostic, not a gate)
```

Do not write `just test -- -k foo`: `just` forwards `--` and pytest treats `-k` as a path.

## Style and tooling

- **Package layout:** keep `__init__.py` empty and use explicit module imports.
- **Types:** annotate all production and script functions, including return types.
  Prefer frozen dataclasses (`PointCloud`, `TraceLogs`, `ExperimentConfig`, `DepthCompressionConfig`) over untyped dictionaries, and validate third-party unions at boundaries instead of using `Any`, `cast()` or `# type: ignore`.
- **Control flow:** keep functions flat and single-purpose with early returns (trivial case first, non-trivial logic last).
  Avoid nested local `def` functions inside other functions; hoist helpers to module level.
- **Functional default:** prefer pure helpers and comprehensions over accumulator loops where clear.
- **Prose:** use British spelling (`normalise`, `standardise`, `visualisation`, `neighbour`, `artefact`) in comments, docstrings and docs while keeping existing American identifiers (`SharedStandardizer`, `normalize`, `neighborhood_metrics`).
  Avoid em dashes and Oxford commas.
- **Linting:** `pyproject.toml` uses Ruff `select = ["ALL"]` with the shared `probekit` ignore list plus `ty` for static type checking.
- **Tests:** keep tests lean and focused on our logic and invariants.
