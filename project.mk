# This repo's own make targets. The Makefile beside this one belongs to the
# mirrorball template and includes this file; this one is never touched by
# `copier update`.

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


# so-high-school's college basketball recruits -- ESPN's top 100 of each high
# school class, matched to athlete ids -- which `recruit-model` and
# `carryovers` read: {league}/recruits.csv.
recruits:
	aws s3 sync --only-show-errors --exclude "*/raw/*" --exclude "*/players/*" \
		s3://${BUCKET}/so-high-school/ncaabb ./recruits


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


# Both, and the estimated possessions, for every league and season in
# ./plays, with their checks.
seasons:
	for dir in plays/league=*/season=*; do \
		league=$${dir#plays/league=}; league=$${league%%/*}; season=$${dir##*=}; \
		echo "$$league $$season"; \
		uv run python build.py stints --league $$league --season $$season && \
		uv run python build.py context --league $$league --season $$season && \
		uv run python build.py estimated --league $$league --season $$season && \
		uv run python build.py minutes --league $$league --season $$season || exit 1; \
	done


# Player ratings: every model and target scored on held-out games, then
# the best fitted and written to ratings.parquet next to the season's tables.
#   make ratings ARGS="--league mens --season 2025"
ratings:
	uv run python build.py ratings $(ARGS)


# Possession rows for a season's games without substitutions, from
# estimated lineups: make estimated ARGS="--league mens --season 2024"
estimated:
	uv run python build.py estimated $(ARGS)


# Priors for the ratings, scored against none on held-out games, and last
# season's ratings scored on this season's early games:
#   make priors ARGS="--league mens --season 2025"
#   make carryover ARGS="--league mens --season 2025"
priors:
	uv run python build.py priors $(ARGS)

carryover:
	uv run python build.py carryover $(ARGS)

# ...and for every season, each from the one before, after `make seasons`:
#   make carryovers ARGS="--league mens"
carryovers:
	uv run python build.py carryovers $(ARGS)


# Every season rated from all its games, pulled toward the box score:
#   make season-ratings ARGS="--league mens"
# and the box-score prior it uses, learned on a season of real lineups --
# commit what this changes:
#   make box-model ARGS="--league mens --season 2025"
season-ratings:
	uv run python build.py season_ratings $(ARGS)

box-model:
	uv run python build.py box_model $(ARGS)

# What a newcomer's recruiting rank says of his first season, learned from
# every season's season_ratings (run season-ratings first) -- commit what
# this changes: make recruit-model ARGS="--league mens"
recruit-model:
	uv run python build.py recruit_model $(ARGS)

# How well each season's teams are known before it starts: rosters, projected
# minutes and last season's ratings (run season-ratings first):
#   make preseasons ARGS="--league mens"
preseasons:
	uv run python build.py preseasons $(ARGS)


# Chained ratings: each player's rating carried from season to season with
# how sure it is, tested week by week (see ten_minute_version/chain.py).
# After season-ratings, for a league:
#   make chain-weeks ARGS="--league mens"      # cache each season's weeks
#   make newcomer-fits ARGS="--league mens"    # newcomers, nearly unshrunk
#   make newcomer-prior ARGS="--league mens"   # their prior by rank: commit what this changes
#   make chain ARGS="--league mens --variant baseline"
#   make chain ARGS="--league mens"
# and whether the posterior variances are right, on a season's two halves:
#   make calibrate ARGS="--league mens --season 2025 --side defense"
chain-weeks:
	uv run python build.py chain_weeks $(ARGS)

newcomer-fits:
	uv run python build.py newcomer_fits $(ARGS)

newcomer-prior:
	uv run python build.py newcomer_prior $(ARGS)

chain:
	uv run python build.py chain $(ARGS)

calibrate:
	uv run python build.py calibrate $(ARGS)


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


.PHONY: plays summaries stints context seasons ratings estimated priors carryover carryovers season-ratings box-model recruits recruit-model preseasons chain-weeks newcomer-fits newcomer-prior chain calibrate evaluate fit
