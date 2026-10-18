"""
What a play is to this package, and where plays come from.

The plays are endgame's processed NCAABB play-by-play: one parquet object
per league-day, under

    {root}/league={league}/season={season}/date={day}/data.parquet

where `root` is endgame's `processed/basketball_plays` prefix -- in the
bucket, or a local copy of it (`make plays`). Only the columns below are
read. The layout is the contract with endgame; nothing here imports it.
"""

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Iterator

import pyarrow.dataset as ds

# The columns this package reads, out of the ~25 endgame writes.
_COLUMNS = [
    "game_id",
    "play_number",
    "period",
    "clock_seconds",
    "side",
    "play_category_id",
    "play_type",
    "text",
    "athlete_id",
    "athlete_name",
    "participant_ids",
    "home_score",
    "away_score",
    "shot_x",
    "shot_y",
]


@dataclass(frozen=True, slots=True)
class Play:
    """
    One row of the play-by-play, trimmed to what lineups and possessions
    need. Fields mean what endgame's `BASKETBALL_PLAY_SCHEMA` says they do.
    """

    period: int
    # Seconds left in the period.
    clock_seconds: int
    # "home", "away", or None for a play that belongs to neither side.
    side: str | None
    category: str | None
    play_type: str | None
    text: str
    athlete_id: str | None
    athlete_name: str | None
    # Everyone ESPN tied to the play. On a made shot, the shooter and then
    # the assister; otherwise at most the one player.
    participant_ids: tuple[str, ...]
    # ESPN's score after the play. Read only for the final score: it's
    # stale often enough that points are counted from the scoring plays.
    home_score: int | None
    away_score: int | None
    # Where a shot was taken from, in feet: x across the court (0 to 50, the
    # hoop at 25) and y out from the shooting team's baseline. None for
    # anything but a shot, and for the odd shot ESPN didn't place.
    x: float | None = None
    y: float | None = None


@dataclass(frozen=True, slots=True)
class Game:
    league: str
    season: int
    date: date
    game_id: str
    plays: tuple[Play, ...]


def read_games(root: str | Path, league: str, season: int) -> Iterator[Game]:
    """
    Every game of a season, a day at a time and in date order.

    A day at a time so a season -- a few million plays -- never has to be
    in memory as Python objects at once.
    """
    season_dir = Path(root) / f"league={league}" / f"season={season}"
    day_dirs = sorted(season_dir.glob("date=*"))
    for day_dir in day_dirs:
        day = date.fromisoformat(day_dir.name.removeprefix("date="))
        table = ds.dataset(day_dir, format="parquet").to_table(columns=_COLUMNS)
        table = table.sort_by([("game_id", "ascending"), ("play_number", "ascending")])
        yield from _games_from_rows(table.to_pylist(), league, season, day)


def _games_from_rows(
    rows: list[dict], league: str, season: int, day: date
) -> Iterator[Game]:
    plays: list[Play] = []
    game_id = None
    for row in rows:
        if row["game_id"] != game_id:
            if game_id is not None:
                yield Game(league, season, day, game_id, tuple(plays))
            game_id, plays = row["game_id"], []
        plays.append(_play(row))
    if game_id is not None:
        yield Game(league, season, day, game_id, tuple(plays))


def _play(row: dict) -> Play:
    return Play(
        period=row["period"] or 0,
        clock_seconds=row["clock_seconds"] or 0,
        side=row["side"],
        category=row["play_category_id"],
        play_type=row["play_type"],
        text=row["text"] or "",
        athlete_id=row["athlete_id"],
        athlete_name=row["athlete_name"],
        participant_ids=tuple(row["participant_ids"] or ()),
        home_score=row["home_score"],
        away_score=row["away_score"],
        x=row["shot_x"],
        y=row["shot_y"],
    )
