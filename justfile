# Shared core: keep this block identical across repos. Repo-specific recipes go under
# the divider at the bottom.
#
# Arguments reach pytest with quoting intact, so all of these work:
#     just test tests/a.py tests/b.py
#     just test -k "foo and not bar"
#     just test -m "not slow" -x
#     just test 'tests/x.py::test_y[case-1]'
# Do NOT write `just test -- -k foo`: just forwards the `--` as an argument and pytest
# then reads `-k` as a path. The test path and -q come from [tool.pytest.ini_options],
# not from the recipe, so `just test <path>` overrides cleanly.

SRC := "src scripts tests"

default:
    @just --list

[doc("Rewrite src, scripts and tests with ruff format")]
format:
    uv run ruff format {{SRC}}

[doc("ruff check")]
lint:
    uv run ruff check {{SRC}}

[doc("ty check")]
typecheck:
    uv run ty check {{SRC}}

[doc("pytest; forwards all arguments (see the header for what that allows)")]
[positional-arguments]
test *args:
    uv run pytest "$@"

[doc("Dev loop: format, typecheck, lint, test. REWRITES files. Stops at first failure")]
check: format typecheck lint test

[doc("Same checks, writes nothing. What the commit hook runs")]
verify:
    uv run ruff format --check {{SRC}}
    uv run ruff check {{SRC}}
    uv run ty check {{SRC}}
    uv run pytest -m "not slow"

[doc("Apply every available autofix; each step runs even if an earlier one fails")]
fix:
    -uv run ruff format {{SRC}}
    -uv run ruff check --fix {{SRC}}
    -uv run ty check --fix {{SRC}}

[doc("Coverage report. A diagnostic, not a gate")]
cov:
    uv run pytest --cov=src --cov-report=term
