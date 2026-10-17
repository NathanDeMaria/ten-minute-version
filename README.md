# ten-minute-version

Player ratings for college basketball, from who was on the floor.

The name is the ten: every possession is five players against five, and the
point of this package is to split what happened on it between them.

The first step, and all that's here so far, is the possession table: each game
broken into lineup-vs-lineup stints, and each side's possessions in a stint
counted by how they ended.

```bash
make plays                                         # endgame's processed plays, locally
make stints ARGS="--league mens --season 2025"     # -> ./stints/league=mens/season=2025/
```

## What the data allows

The plays are [endgame][endgame]'s processed NCAABB play-by-play, ESPN's feed.
**ESPN only logs substitutions from mid-February 2025** -- every game before
2025-02-15 has none, in ESPN's own API as well as in what endgame stored -- so
lineups exist for the last weeks of 2024-25, all of 2025-26, and on. A game
without substitutions gets a report row and no stints.

## The tables

`build.py stints` writes three, under `{out}/league={league}/season={season}/`:

| table | one row per | |
| --- | --- | --- |
| `stints.parquet` | stint and side with the ball | the possession table |
| `games.parquet` | game | how far to trust it |
| `players.parquet` | player | ESPN athlete id to name |

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

## How a game becomes possessions

| module | what it does |
| --- | --- |
| `plays` | what a play is, and reading endgame's parquet |
| `events` | what kind of thing each play is, from ESPN's category codes |
| `lineups` | who was on the floor at every play |
| `possessions` | the game cut into possessions, each credited to a floor |
| `stints` | possessions grouped into the tables above |

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

What `build.py` prints, on the two windows built so far: season 2024
(2024-25) from 2025-02-15 to the end, and season 2025 (2025-26) for January
2026. Seasons are named for the year they start in, as in endgame.

| | mens 2024 | mens 2025 | womens 2024 | womens 2025 |
| --- | --- | --- | --- | --- |
| games | 1,466 | 1,556 | 1,405 | 1,543 |
| possessions five on five | 99.82% | 99.92% | 99.94% | 99.95% |
| points counted = ESPN's final | 95.5% | 99.8% | 99.9% | 99.9% |
| starters found | 99.2% | 99.0% | 98.5% | 99.0% |
| repairs per game | 0.70 | 0.78 | 0.72 | 0.72 |
| possessions per side per game | 67.8 | 68.3 | 70.0 | 71.0 |

The mens 2024 points miss is the feed's first four days: 65 of the 66 games
that don't add up are 2025-02-15 to 02-18, where ESPN's score moves on baskets
the play-by-play never logged. Possessions per side agree with the totals on
ESPN's own matchup page -- median difference 0, within one on 77% of 5,554
mens team-games since 2025-02-19 -- which are themselves an estimate.

[endgame]: https://github.com/NathanDeMaria/EndGame
