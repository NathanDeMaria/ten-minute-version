# ten-minute-version

Player ratings for college basketball, from who was on the floor.

The name is the ten: every possession is five players against five, and the
point of this package is to split what happened on it between them.

First the possession table: each game broken into lineup-vs-lineup stints,
and each side's possessions in a stint counted by how they ended. Then the
ratings fitted to it (see [Ratings](#ratings)).

```bash
make plays                                         # endgame's processed plays, locally
make summaries                                     # endgame's game summaries, every season
make stints ARGS="--league mens --season 2025"     # -> ./stints/league=mens/season=2025/
make context ARGS="--league mens --season 2025"    # who played and where, next to them
make ratings ARGS="--league mens --season 2025"    # score the models, write ratings.parquet
make estimated ARGS="--league mens --season 2024"  # possessions for games with no substitutions
make priors ARGS="--league mens --season 2025"     # what to shrink players toward
make carryover ARGS="--league mens --season 2025"  # last season's ratings on this season's games
```

`make seasons` builds the stint tables, context and estimated possessions for
every league and season in `./plays`, 2013 (2013-14) on.

## What the data allows

The plays are [endgame][endgame]'s processed NCAABB play-by-play, ESPN's feed,
from 2013-14 on: 87-95% of each season's mens games, and of womens games a
quarter before 2019-20 and all of them since. The summaries, which say who
played and where, cover every game from 2010-11 on.

**ESPN only logs substitutions from mid-February 2025** -- every game before
2025-02-15 has none, in ESPN's own API as well as in what endgame stored -- so
lineups exist for the last weeks of 2024-25, all of 2025-26, and on. A game
without substitutions gets a report row and no stints.

## The tables

`build.py stints` writes four, under `{out}/league={league}/season={season}/`,
and `build.py context` a fifth:

| table | one row per | |
| --- | --- | --- |
| `stints.parquet` | stint and side with the ball | the possession table |
| `games.parquet` | game | how far to trust it |
| `players.parquet` | player | ESPN athlete id to name |
| `shots.parquet` | field goal or free throw in a stint | shooter, value, made, type and where from, for expected points; `(game_id, stint, offense_side)` joins it to its row |
| `context.parquet` | game | the teams, their conferences and whether ESPN files them in D1 today, neutral site, conference game -- from ESPN's game summary |

A **stint** is a run of possessions with the same ten players on the floor. It
gives one row per side that had the ball in it: that side's five
(`offense_players`) against the other's (`defense_players`), with

- `possessions`, `points`, `empty` (possessions with no points)
- how each possession ended, which add up to `possessions`: `ended_fg2`,
  `ended_fg3`, `ended_ft`, `ended_turnover`, `ended_miss` (a missed shot the
  defense got), `ended_other` (mostly the end of a period)
- `fga2`, `fgm2`, `fga3`, `fgm3`, `fta`, `ftm`, `oreb`, `tov`
- `start_seconds`, `end_seconds` since the tip, and `start_margin`, the
  offense's lead when the stint started, for garbage time

Rows add: a lineup's season, or a player's on-floor possessions, is a
group-by. See `ten_minute_version/stints.py` for the schemas.

**Teams from outside Division I** play as one generic team: their side of a
row is the single id `"generic"`, against the other side's five, and the game
row's `generic_side` says which side that was. Most of their players have no
ESPN id, so they can't be followed one by one. They're told apart by their
team's schedule in the summaries: a D1 team plays ~30 of the season's games,
a team from outside at most 13, and never a conference game (`opponents.py`).
In 2025-26 that's 535 mens and 395 womens games, and it keeps 5,795 mens and
5,478 womens possessions the table used to leave out, most of them in
November and December.

ESPN's own Division I flag (`build.py context`) agrees on all but 95 of
149,110 games since 2013-14, and those are mostly the flag's mistakes: it
files each team under the conference it's in now, so Lindenwood and Utah
Tech were D1 years before they were. A game without a summary falls back
to the players: whether they turn up in only a few of the season's games.

## How a game becomes possessions

| module | what it does |
| --- | --- |
| `plays` | what a play is, and reading endgame's parquet |
| `events` | what kind of thing each play is, from ESPN's category codes |
| `lineups` | who was on the floor at every play |
| `possessions` | the game cut into possessions, each credited to a floor |
| `stints` | possessions grouped into the tables above |
| `names` | ids for the plays that only name their player |
| `opponents` | which side, if either, is a team from outside Division I |
| `context` | who played each game and where, from the game summaries endgame stores |
| `estimate`, `probabilities`, `evaluate` | lineups where the feed has no substitutions |
| `shot_model` | the chance a shot goes in, from where and how it was taken and who took it |
| `ratings` | player ratings fitted to the stint table, and how they're scored |
| `box_scores`, `priors` | box-score lines from the summaries, and what to shrink ratings toward |
| `estimated` | possession rows for games with no substitutions, from estimated lineups |
| `carryover` | last season's ratings scored on this season's early games |

- **Starters** aren't in the feed. A player whose first appearance is anything
  but subbing in started. Substitutions -- including the ones at the start of
  each period -- are replayed over them.
- **Repairs**, for the substitutions the feed drops: a player who does
  something while on the bench was on the floor, and a side that comes out of
  a run of substitutions with other than five is fixed from what its players
  do next.
- **Who has the ball** comes from the plays only one side can make: its shots,
  free throws, turnovers and offensive rebounds, and the other side's
  defensive rebounds, steals and blocks.
- **Credit**: a possession goes to the floor at its last play, except free
  throws, which go to the floor at the foul.
- **Points** come from the scoring plays, never ESPN's running score, which is
  sometimes stale. Technical free throws are no one's possession; their points
  are in the game report.

## The checks

What `build.py` prints for the two seasons with substitutions: 2024 (2024-25,
from 2025-02-15, when the feed started logging them) and 2025 (2025-26, all
of it). Seasons are named for the year they start in, as in endgame.

| | mens 2024 | mens 2025 | womens 2024 | womens 2025 |
| --- | --- | --- | --- | --- |
| games with substitutions | 1,482 | 6,270 | 1,415 | 6,010 |
| possessions five on five (or against a generic team) | 99.82% | 99.91% | 99.94% | 99.85% |
| games against a generic team | 0 | 535 | 0 | 395 |
| points counted = ESPN's final | 95.2% | 99.7% | 99.9% | 99.8% |
| starters found | 99.2% | 99.1% | 98.5% | 98.4% |
| repairs per game | 0.74 | 0.88 | 0.75 | 0.90 |
| possessions per side per game | 67.8 | 69.3 | 70.0 | 71.7 |

The mens 2024 points miss is the feed's first four days: 70 of the 71 games
that don't add up are 2025-02-15 to 02-18, where ESPN's score moves on baskets
the play-by-play never logged. Possessions per side agree with the totals on
ESPN's own matchup page -- median difference 0, within one on 77% of 5,554
mens team-games since 2025-02-19 -- which are themselves an estimate.

## Ratings

`build.py ratings` fits regularised adjusted plus-minus: each stint row's
points per possession as a baseline, a home edge, the offense's five ratings
and minus the defense's five, by ridge regression weighted by possessions.
Defense ratings are points kept off the board, and everything is per 100
possessions. The home edge applies at both ends and not at a neutral site
(`context`). A generic team is one more player.

**What they're judged on**: predicting games they weren't fitted to. Five folds
of games are held out in turn; each held-out game is predicted from ratings
fitted on the rest, with its real lineups and possessions, and the error in its
home margin is what counts. Two baselines get the same treatment: the baseline
and home edge alone, and a team model (each team's offense and defense, and
nothing about who's on the floor). The t statistic compares each with the best
players model on the same games.

**The target** is what the ratings are fitted to; every model is scored on real
points. `shot_model` is the-archer's shot model (a cubic in distance for jump
shots, league rates for layups, dunks, tips and free throws) plus each
shooter's own skill, shrunk by how many shots he took. Expected points for
every shot turn out to throw away real signal: a layup gets one league chance,
so finishing and rim protection go with it. Threes and free throws are where
the defense has least say:

| 2025-26 margin RMSE (t vs best) | mens | womens |
| --- | --- | --- |
| baseline and home edge | 17.89 (28.9) | 21.66 (33.5) |
| teams | 12.78 (3.2) | 13.88 (3.7) |
| players, points | 12.47 (2.1) | 13.35 (0.5) |
| players, 3s and FTs half actual, half expected | **12.41** | **13.33** |
| players, 3s expected | 12.64 (8.7) | 13.55 (8.4) |
| players, 3s and FTs expected | 12.70 (9.9) | 13.61 (9.4) |
| players, every shot expected (shooter's skill) | 14.54 (19.4) | 15.90 (20.2) |
| players, every shot expected (league chance) | 14.90 (21.3) | 17.00 (25.0) |

Players beat teams in both leagues, so who's on the floor carries what a
team's season doesn't. Taking half the luck out of threes and free throws is
the best target in both, though only just beyond points as they happened. The
best penalty is 300 possessions in both leagues, and the home edge comes to
3.0 points a game for mens, 2.7 for womens. `ratings.parquet` is the best
target fitted on every game with the team columns `priors` found best
(player penalty 1,000, team penalty 30): `offense` and `defense` there are a
player's rating against his team's plus a fifth of his team's, so a lineup's
five add up to what the fit predicts; `offense_vs_team` and
`defense_vs_team` are the first part alone.

## Priors

A ridge fit pulls every player toward zero by how little he played; a prior
says where to pull him instead (`priors`, `make priors`). On the same held-out
games:

| 2025-26 margin RMSE (t vs no prior) | mens | womens |
| --- | --- | --- |
| no prior | 12.409 | 13.332 |
| box score (a statistical plus-minus from per-40 rates) | 12.408 (-0.1) | 13.293 (-2.8) |
| last season, its real lineups only | 12.410 (0.3) | 13.312 (-3.4) |
| last season, with its estimated possessions | 12.390 (-2.5) | 13.253 (-7.7) |
| **team**: team columns alongside the players | **12.007 (-8.0)** | **12.843 (-8.7)** |
| team + last season with its estimated possessions | 11.997 (-8.5) | 12.809 (-9.9) |
| team + box score | 12.022 (-7.8) | 12.867 (-8.3) |

Pulling a player toward his team is by far the best prior: it's what a team
model knows and the players model doesn't, and the two together beat either.
The others add little on top, within a season: the games they'd say something
about are already in the fit.

**Looking forward, players add less.** Random folds let a held-out game lean
on the games around it. Fitting on everything before 2025-02-15 (2026-02-15)
and predicting the games after:

| margin RMSE | 2024-25, estimated lineups before, real after | 2025-26, real lineups throughout |
| --- | --- | --- |
| teams | 10.781 | 11.039 |
| players | 11.015 | 11.225 |
| players + teams | 10.766 | 11.031 |

(mens, the best target and penalties for each.) Within a season, a team's own
results already say most of what its players will. And estimated lineups do
as well here as real ones: the gap between players + teams and teams is the
same size.

## Before the substitutions: estimated lineups

Every game before 2025-02-15 has players' actions but no substitutions, so its
lineups are estimated (`estimate`) and turned into chances of being on the
floor (`probabilities`), scored by hiding the substitutions of games whose
lineups are known (`evaluate`, `make evaluate`):

- **names first**: before 2022-23 shots, free throws, steals and assists carry
  no player id, only a name; `names` recovers ~88% of them from the ids the
  same players carry on rebounds and fouls, which restores the old feed to
  today's density;
- **a persistent lineup**: the five only changes when someone off it acts.
  The leaver is whoever's next action is furthest off, foul trouble first;
  the swap is backdated a third of the way to the leaver's last action and
  snapped to a TV timeout; the starters come back at halftime;
- **probabilities** from a logistic regression on what the estimate saw,
  fitted on known games and scaled to five a side (`releases/`).

On January 2026, with the probabilities fitted on February-April 2025:

| | mens | womens | mens, old feed |
| --- | --- | --- | --- |
| nearest action: players right / all five | 85.5% / 40.2% | 85.0% / 38.6% | 85.1% / 39.1% |
| estimate | 86.0% / 43.9% | 85.7% / 42.2% | 85.7% / 42.9% |
| estimate with box-score starters and minutes | 86.8% / 47.3% | 86.5% / 45.9% | 86.5% / 46.4% |
| brier: probabilities / hard estimate | 0.102 / 0.149 | 0.104 / 0.150 | 0.105 / 0.153 |

All five right less than half the time is why a rating should take the
probabilities, which are calibrated (a 0.43 chance is on the floor 41% of the
time, 0.70 is 68%), rather than the hard guesses.

`build.py estimated` does that for a season's games without substitutions
(`estimated`): each game's floors are estimated with its own box score's
starters and minutes, and every possession becomes a row crediting each
player by his chance of being on the floor at its credit play -- weights that
add up to five a side, scaled down where more than five acted between dead
balls. 2024-25 gets 641,057 mens possessions in 4,634 games and 621,932
womens in 4,347, which the ratings read alongside its stint rows.

[endgame]: https://github.com/NathanDeMaria/EndGame

## Across seasons: what the estimated lineups are worth

Where player ratings should earn their keep is the next season, when a team's
minutes have turned over and its rating carries over badly
(`carryover`, `make carryover`). Each 2025-26 game from 2025-11-17 to
12-31 is predicted with its real lineups from 2024-25's ratings: each player
on the floor at their rating from last season, a player it didn't rate
counted as a newcomer. How much of each term to use is fitted on 2025-11-03
to 11-16.

| margin RMSE (t vs team carryover) | mens | womens |
| --- | --- | --- |
| home edge only | 19.98 (18.6) | 25.01 (16.2) |
| last season's teams | 15.03 | 18.24 |
| last season's players, real lineups only | 19.74 (17.1) | 23.46 (15.0) |
| last season's players, with estimated possessions | 15.12 (0.4) | 16.68 (-4.4) |
| teams + players, real lineups only | 14.97 (-1.3) | 18.23 (-0.1) |
| **teams + players, with estimated possessions** | **13.91 (-7.8)** | **15.62 (-10.4)** |

This is what the estimated lineups are for. Rated from 2024-25's real lineups
alone -- the six weeks from 2025-02-15 -- last season's players add nothing to
its teams. Rated from every game, with the estimated possessions of the first
three and a half months, they're as good as the teams on their own, and with
them they take 1.1 points (mens) and 2.6 points (womens) off the early-season
error.

**Every season** (`make carryovers`, after `make seasons`): the same test for
each season from 2014-15 on, from the one before, every game of both seasons
on estimated lineups until 2025. Weights are fitted on each season's first
two weeks (2020-21 started on Nov 25) and scored to Dec 31:

| season | mens games | teams | teams + players | t | womens games | teams | teams + players | t |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 2014-15 | 1,229 | 13.48 | 12.78 | -6.2 | 315 | 15.98 | 16.12 | 1.5 |
| 2015-16 | 1,356 | 13.61 | 12.82 | -6.8 | 364 | 15.64 | 15.18 | -1.8 |
| 2016-17 | 1,527 | 13.49 | 12.86 | -4.7 | 410 | 15.27 | 15.14 | -0.7 |
| 2017-18 | 1,561 | 14.38 | 13.42 | -6.9 | 441 | 16.05 | 15.91 | -1.4 |
| 2018-19 | 1,373 | 14.20 | 13.60 | -3.7 | 420 | 15.93 | 15.66 | -2.4 |
| 2019-20 | 1,388 | 15.07 | 13.77 | -8.1 | 1,323 | 18.47 | 17.88 | -3.9 |
| 2020-21 | 623 | 14.24 | 13.72 | -3.8 | 534 | 15.82 | 15.49 | -1.1 |
| 2021-22 | 1,321 | 16.87 | 14.31 | -10.8 | 1,287 | 18.34 | 15.61 | -8.3 |
| 2022-23 | 1,669 | 14.60 | 13.28 | -8.3 | 1,582 | 16.34 | 14.92 | -6.8 |
| 2023-24 | 1,668 | 14.85 | 13.89 | -7.8 | 1,561 | 16.87 | 15.21 | -7.7 |
| 2024-25 | 1,805 | 14.53 | 13.45 | -7.2 | 1,622 | 17.10 | 15.26 | -8.1 |
| 2025-26 | 1,909 | 15.03 | 13.91 | -7.8 | 1,747 | 18.25 | 15.62 | -10.4 |
| all | 17,429 | 14.58 | 13.50 | -23.9 | 11,606 | 17.18 | 15.69 | -18.9 |

Last season's players beat its teams in every mens season, by 0.5 to 2.6
points, and by most the season after COVID, when a team's 2020-21 said least
about it. In womens they do from 2019-20, the first season with plays for
every game; before it, with plays for only a quarter of games, last season's
players are rated from too few of them to add much.
