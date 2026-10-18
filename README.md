# ten-minute-version

Player ratings for college basketball, from who was on the floor.

The name is the ten: every possession is five players against five, and the
point of this package is to split what happened on it between them.

The first step, and all that's here so far, is the possession table: each game
broken into lineup-vs-lineup stints, and each side's possessions in a stint
counted by how they ended.

```bash
make plays                                         # endgame's processed plays, locally
make summaries                                     # endgame's game summaries, 2024 and 2025
make stints ARGS="--league mens --season 2025"     # -> ./stints/league=mens/season=2025/
make context ARGS="--league mens --season 2025"    # who played and where, next to them
```

## What the data allows

The plays are [endgame][endgame]'s processed NCAABB play-by-play, ESPN's feed.
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
| `context.parquet` | game | the teams, their conferences and whether they're D1, neutral site, conference game -- from ESPN's game summary |

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
ESPN id, so they can't be followed one by one; the ones who do turn up in only
the few games their team plays against D1, which is how they're told apart
from a D1 team with a player ESPN hasn't linked (`opponents.py`). In 2025-26
that's 535 mens and 395 womens games, and it keeps 5,795 mens and 5,478 womens
possessions the table used to leave out, most of them in November and
December. ESPN's own Division I flag in the summaries (`build.py context`)
agrees with every one: no side called generic is a D1 team, and no outside
team was kept as players.

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

[endgame]: https://github.com/NathanDeMaria/EndGame
