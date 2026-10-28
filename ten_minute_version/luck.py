"""
Each game's shooting luck, for a model that learns from results to discount:
Cassandra's Glicko, which otherwise takes a final margin as all signal.

A game's luck is each side's points on threes and free throws less what the
shot model expected of its shooters (`ratings.Targets`: the-archer's chance
from where and how each shot was taken, plus the shooter's own skill), the
home side's less the away side's. Twos are left alone: a layup is one league
chance, so finishing and rim protection would go with it, and the defense
has least say over threes and free throws -- the same split the ratings'
target ("half on 3s and FTs") makes.

A game's luck is as known soon after it: the shot model and the shooters'
skills are fitted on the games before the first cut after it -- the first
weekly fit whose training games include it, in the backfill (`build.luck`);
the morning after, live (`build.live`). It doesn't change after that, so a
replay sees what a morning would have.

Measured on 2014-2025, each team's luck so far this season predicts the
next game's residual of Cassandra's `glicko_players` against it (-0.0011
logit per point mens, -0.0016 womens, t -6): a team that's been lucky is
overrated. Taking part of each game's luck out of the margin Glicko learns
from, its Brier on 2014+ games: 0.3 of it -0.00027 mens, -0.00035 womens;
0.4 -0.00028, -0.00040; all of it +0.0008, +0.0004 -- the shot model's
expectation misses some of what's real, as the ratings' targets found.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv

from .context import GameContext
from .ratings import Rows, Targets

LUCK_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("home_team_id", pa.string(), nullable=False),
        pa.field("home", pa.string()),
        pa.field("away_team_id", pa.string(), nullable=False),
        pa.field("away", pa.string()),
        # points, the home side's luck less the away side's
        pa.field("luck", pa.float32(), nullable=False),
        # the cut it was fitted at: on the games before it
        pa.field("as_of", pa.date32(), nullable=False),
    ]
)

# The kinds of shot whose luck is counted: threes and free throws.
LUCKY = (3, 1)


def game_luck(rows: Rows, fitted: Targets, games: Iterable[str]) -> dict[str, float]:
    """Each of `games`' luck in points, home side less away, from `fitted`'s
    shot model and skills."""
    per_row = sum(fitted.actual[v] - fitted.shooter[v] for v in LUCKY)
    points = per_row * rows.possessions * np.where(rows.offense_home, 1.0, -1.0)
    wanted = set(games)
    found: dict[str, float] = defaultdict(float)
    for game_id, value in zip(rows.game_id, points):
        if game_id in wanted:
            found[game_id] += float(value)
    return dict(found)


def luck_rows(
    league: str,
    season: int,
    as_of: date,
    luck: Mapping[str, float],
    contexts: Mapping[str, GameContext],
    day: Mapping[str, date],
) -> list[dict]:
    """The file's rows for `luck`'s games; a game without a context has no
    teams to file it under, and is left out."""
    return [
        {
            "league": league,
            "season": season,
            "game_id": game_id,
            "date": day[game_id],
            "home_team_id": contexts[game_id].home.team_id,
            "home": contexts[game_id].home.name,
            "away_team_id": contexts[game_id].away.team_id,
            "away": contexts[game_id].away.name,
            "luck": value,
            "as_of": as_of,
        }
        for game_id, value in sorted(luck.items())
        if game_id in contexts
    ]


def merged_luck(existing: list[dict], found: list[dict]) -> list[dict]:
    """The file's rows with `found`'s games added: a game already there
    keeps its luck, as first known."""
    have = {(r["league"], r["game_id"]) for r in existing}
    added = [r for r in found if (r["league"], r["game_id"]) not in have]
    return sorted(
        existing + added,
        key=lambda r: (r["league"], r["season"], r["date"], r["game_id"]),
    )


def read_luck(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pyarrow.csv.read_csv(
        path, convert_options=pyarrow.csv.ConvertOptions(column_types=LUCK_SCHEMA)
    ).to_pylist()


def write_luck(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pyarrow.csv.write_csv(pa.Table.from_pylist(rows, LUCK_SCHEMA), path)
