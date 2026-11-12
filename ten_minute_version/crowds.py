"""
How big a crowd each team plays its home games in front of, for a model
that gives the home side an edge to scale it by: Cassandra's Glicko.

A team's reading is a running average of the log of the attendance at its
home games -- the summaries' `gameInfo.attendance`, neutral sites left out --
weighing each game half as much as the one `HALF_LIFE` home games after it
and carried from season to season. It's filed as the log of that crowd over
the league's typical one (`TYPICAL`), so 0 is an ordinary building and 0.69
one with twice the people, into `{league}_crowds.csv`: a row for each team
as each season opens, and one the day after each of its home games, each
the reading as of that day. A team has no reading (`crowd` empty) until
`MIN_GAMES` of its home games have one.

An attendance of 0 is a game ESPN has no number for, one or two in a
hundred, and is skipped. Except in 2020-21 (`EMPTY_SEASONS`), when over half
are 0 and it means what it says: that season is read from its own crowds
alone -- every team opening it at `FLOOR` people, a 0 counted as that many,
and the average moving faster (`EMPTY_HALF_LIFE`) -- and left out of the
reading a team carries into the next one.

Measured in Cassandra's `glicko_players` at its 2026-11-12 fits, on the file
this writes, the home edge taking 15 rating points per unit of it (and the
men's slope on a team's tier at 0, which this stands in for): Brier over
every game 2010-2026 against the fit as published is 0.00012 better mens and
0.00010 womens, each in eleven of the twelve seasons 2014-2025.

Home sides won 2.5 points of win probability less than the model gave them
in 2020-21, in both leagues. Read like any other season -- its zeros
skipped, its crowds averaged in with the rest -- the gain is about half
that.

What didn't add to it: the crowd as a share of the building's biggest
(ESPN files no capacity), the visitor's own home crowd, and any shape but a
line in the log -- a cap, a knee, the raw count.
"""

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.csv

from .box_scores import game_date
from .context import parse_context

# About the league's median home crowd: what a reading of 0 is.
TYPICAL = {"mens": 2500.0, "womens": 900.0}
HALF_LIFE = 15.0
MIN_GAMES = 5

# The seasons (by the year they start) played to limited crowds.
EMPTY_SEASONS = frozenset({2020})
FLOOR = 200.0
EMPTY_HALF_LIFE = 2.0

CROWD_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("team_id", pa.string(), nullable=False),
        pa.field("team", pa.string()),
        # the reading holds from this day until the team's next row
        pa.field("as_of", pa.date32(), nullable=False),
        # log of the running crowd over the league's typical one; empty
        # until enough home games have an attendance
        pa.field("crowd", pa.float32()),
        # the running average's two sums and how many games are in it: what
        # the next game's reading is built from
        pa.field("total", pa.float64(), nullable=False),
        pa.field("weight", pa.float64(), nullable=False),
        pa.field("games", pa.int32(), nullable=False),
    ]
)


@dataclass(frozen=True)
class HomeGame:
    """A game at its home side's own building."""

    game_id: str
    day: date
    team_id: str
    team: str | None
    # None where the summary has no number
    attendance: int | None


@dataclass(frozen=True)
class Reading:
    """A running average of log attendance: `total` over `weight`."""

    total: float = 0.0
    weight: float = 0.0
    games: int = 0
    # the team's name as last filed, for a row written before it plays again
    team: str | None = None

    def after(self, log_crowd: float, half_life: float) -> Reading:
        keep = 0.5 ** (1 / half_life)
        return Reading(
            self.total * keep + log_crowd,
            self.weight * keep + 1.0,
            self.games + 1,
            self.team,
        )

    @property
    def log_crowd(self) -> float:
        return self.total / self.weight


def home_game(game_id: str, summary: Mapping[str, Any]) -> HomeGame | None:
    """A summary's game as its home side's; None at a neutral site, or
    without a header to say who was home and when."""
    context = parse_context(game_id, summary)
    day = game_date(summary)
    if context is None or day is None or context.neutral_site:
        return None
    attendance = (summary.get("gameInfo") or {}).get("attendance")
    return HomeGame(
        game_id,
        day,
        context.home.team_id,
        context.home.name,
        attendance if isinstance(attendance, int) else None,
    )


def season_open(season: int) -> date:
    """A day before any game of a season, after every game of the last."""
    return date(season, 8, 1)


def carried(rows: Iterable[Mapping], league: str, season: int) -> dict[str, Reading]:
    """Each team's reading as `season` opens: its last before it, the
    seasons played to limited crowds passed over."""
    last: dict[str, Mapping] = {}
    for row in sorted(rows, key=lambda r: r["as_of"]):
        if (
            row["league"] == league
            and row["season"] < season
            and row["season"] not in EMPTY_SEASONS
        ):
            last[row["team_id"]] = row
    return {
        team: Reading(row["total"], row["weight"], row["games"], row["team"])
        for team, row in last.items()
    }


def season_rows(
    league: str,
    season: int,
    games: Iterable[HomeGame],
    opening: Mapping[str, Reading],
    cut: date | None = None,
) -> list[dict]:
    """A season's rows: each team's reading as it opens, from `opening`, and
    after each of its home games in `games` (those before `cut`, given one)."""
    empty = season in EMPTY_SEASONS
    by_team: dict[str, list[HomeGame]] = defaultdict(list)
    for game in games:
        if cut is None or game.day < cut:
            by_team[game.team_id].append(game)
    names = {team: reading.team for team, reading in opening.items()}
    names |= {g.team_id: g.team for team in by_team.values() for g in team}
    rows = []

    def row(team: str, as_of: date, reading: Reading) -> dict:
        known = empty or reading.games >= MIN_GAMES
        return {
            "league": league,
            "season": season,
            "team_id": team,
            "team": names.get(team),
            "as_of": as_of,
            "crowd": (reading.log_crowd - math.log(TYPICAL[league]) if known else None),
            "total": reading.total,
            "weight": reading.weight,
            "games": reading.games,
        }

    for team in sorted(set(opening) | set(by_team)):
        # A season of limited crowds starts everyone at the floor.
        reading = (
            Reading().after(math.log(FLOOR), EMPTY_HALF_LIFE)
            if empty
            else opening.get(team, Reading())
        )
        if reading.weight:
            rows.append(row(team, season_open(season), reading))
        for game in sorted(by_team.get(team, []), key=lambda g: (g.day, g.game_id)):
            if empty:
                crowd = max(float(game.attendance or 0), FLOOR)
                reading = reading.after(math.log(crowd), EMPTY_HALF_LIFE)
            elif game.attendance:
                reading = reading.after(math.log(game.attendance), HALF_LIFE)
            else:
                continue
            rows.append(row(team, game.day + timedelta(days=1), reading))
    return rows


def with_season(
    known: list[dict],
    league: str,
    season: int,
    games: Iterable[HomeGame],
    cut: date | None = None,
) -> list[dict]:
    """The file's rows with `league`'s `season` rebuilt from `games`, on the
    readings the seasons before it left."""
    kept = [r for r in known if not (r["league"] == league and r["season"] == season)]
    built = season_rows(league, season, games, carried(kept, league, season), cut)
    return sorted(
        kept + built,
        key=lambda r: (r["league"], r["season"], r["team_id"], r["as_of"]),
    )


def read_crowds(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pyarrow.csv.read_csv(
        path, convert_options=pyarrow.csv.ConvertOptions(column_types=CROWD_SCHEMA)
    ).to_pylist()


def write_crowds(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pyarrow.csv.write_csv(pa.Table.from_pylist(rows, CROWD_SCHEMA), path)
