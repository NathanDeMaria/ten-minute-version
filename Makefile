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


# A local copy of the game summaries endgame stores, which say who played
# each game and where (`build.py context`): every season it has, 2010-11 on,
# ~3 GB.
summaries:
	aws s3 sync --only-show-errors s3://${BUCKET}/summaries/ncaabb ./summaries


# One season's possession tables, and the checks that say how far to trust
# them: `make stints ARGS="--league mens --season 2025"`. Writes under
# ./stints; `--root` and `--out` move either end.
ARGS ?=
stints:
	uv run python build.py stints $(ARGS)


# Who played each game and where, from the summaries, next to the stint
# tables: make context ARGS="--league mens --season 2025"
context:
	uv run python build.py context $(ARGS)


# Both, for every league and season in ./plays, with their checks.
seasons:
	for dir in plays/league=*/season=*; do \
		league=$${dir#plays/league=}; league=$${league%%/*}; season=$${dir##*=}; \
		echo "$$league $$season"; \
		uv run python build.py stints --league $$league --season $$season && \
		uv run python build.py context --league $$league --season $$season || exit 1; \
	done


# Player ratings: every model and target scored on held-out games, then
# the best fitted and written to ratings.parquet next to the season's tables.
#   make ratings ARGS="--league mens --season 2025"
ratings:
	uv run python build.py ratings $(ARGS)


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


.PHONY: lint check test plays summaries stints context seasons ratings evaluate fit
