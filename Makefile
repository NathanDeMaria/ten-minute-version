# `uv run` syncs the environment from uv.lock before it runs anything, so
# these targets work from a fresh checkout with no install step in front of
# them. That is also why nothing here activates a venv: uv owns .venv.
lint:
	uv run ruff check --fix .
	uv run ruff format .
	uv run ty check .


# Same checks as `lint`, but reports instead of fixing (what CI runs)
check:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check .


test:
	uv run pytest .


# A local copy of endgame's processed NCAABB play-by-play, which is what
# `stints` reads by default. BUCKET comes from the same config endgame's
# stores read.
BUCKET ?= $(shell jq -r .bucket.value $(HOME)/.aws-batch/config.json)
plays:
	aws s3 sync s3://${BUCKET}/processed/basketball_plays ./plays


# One season's possession tables, and the checks that say how far to trust
# them: `make stints ARGS="--league mens --season 2025"`. Writes under
# ./stints; `--root` and `--out` move either end.
ARGS ?=
stints:
	uv run python build.py stints $(ARGS)


.PHONY: lint check test plays stints
