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


# How good the estimated lineups are, on games whose real ones are known:
#   make evaluate ARGS="--league mens --season 2025 --train 2024"
# `--train` fits the probabilities on another season first, for an honest
# score; `--old_feed` strips ids the way the feed before 2022-23 lacks them.
evaluate:
	uv run python build.py evaluate $(ARGS)


# Refit the on-floor probabilities that ship in ten_minute_version/releases,
# on every season with known lineups: commit what this changes.
#   make fit ARGS="--league mens --seasons 2024,2025"
fit:
	uv run python build.py fit $(ARGS)


.PHONY: lint check test plays stints evaluate fit
