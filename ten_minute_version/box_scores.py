"""
Each player's box score line in each game, from the game summaries
`context` reads -- for the box-score prior on the ratings (`priors`) -- and
each player's minutes for each team over a season, for the preseason
projection (`preseason`).
"""

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pyarrow as pa

from .context import read_summaries

# The summary's stat columns this keeps, and the fields they fill; a pair
# like "4-16" fills two.
_COLUMNS = {
    "minutes": ("minutes",),
    "points": ("points",),
    "fieldGoalsMade-fieldGoalsAttempted": ("fgm", "fga"),
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted": ("tpm", "tpa"),
    "freeThrowsMade-freeThrowsAttempted": ("ftm", "fta"),
    "offensiveRebounds": ("oreb",),
    "defensiveRebounds": ("dreb",),
    "assists": ("ast",),
    "steals": ("stl",),
    "blocks": ("blk",),
    "turnovers": ("tov",),
    "fouls": ("pf",),
}
STATS = tuple(name for names in _COLUMNS.values() for name in names)


@dataclass(frozen=True)
class BoxLine:
    game_id: str
    team_id: str
    athlete_id: str
    starter: bool
    position: str | None
    stats: Mapping[str, float]


def read_box_lines(root: str | Path, league: str, season: int) -> Iterator[BoxLine]:
    for game_id, summary in read_summaries(root, league, season):
        yield from parse_box_lines(game_id, summary)


def parse_box_lines(game_id: str, summary: Mapping[str, Any]) -> Iterator[BoxLine]:
    """Every player with stats; one marked didNotPlay, or with no id, has none."""
    for team in (summary.get("boxscore") or {}).get("players", []):
        team_id = str((team.get("team") or {}).get("id") or "")
        for group in team.get("statistics", []):
            keys = group.get("keys", [])
            for entry in group.get("athletes", []):
                athlete = entry.get("athlete") or {}
                if (
                    entry.get("didNotPlay")
                    or not entry.get("stats")
                    or not athlete.get("id")
                ):
                    continue
                raw = dict(zip(keys, entry["stats"]))
                stats: dict[str, float] = {}
                for key, names in _COLUMNS.items():
                    parts = (raw.get(key) or "").split("-")
                    for name, part in zip(names, parts + [""] * len(names)):
                        stats[name] = _number(part)
                yield BoxLine(
                    game_id=game_id,
                    team_id=team_id,
                    athlete_id=str(athlete["id"]),
                    starter=bool(entry.get("starter")),
                    position=(athlete.get("position") or {}).get("abbreviation"),
                    stats=stats,
                )


MINUTES_SCHEMA = pa.schema(
    [
        pa.field("athlete_id", pa.string(), nullable=False),
        pa.field("team_id", pa.string(), nullable=False),
        pa.field("minutes", pa.float64(), nullable=False),
        pa.field("games", pa.int32(), nullable=False),
        pa.field("starts", pa.int32(), nullable=False),
        # his share of the team's minutes, times five: a team's add up to 5
        pa.field("share", pa.float64(), nullable=False),
        # the first game he played for the team; None if no summary dates it
        pa.field("first_date", pa.date32()),
    ]
)


def season_minutes(root: str | Path, league: str, season: int) -> pa.Table:
    """
    Each player's minutes for each team he played for in a season, from the
    box scores: one row per player and team (`MINUTES_SCHEMA`), for the
    preseason projection (`preseason`).
    """
    totals: dict[tuple[str, str], list] = {}
    team_minutes: dict[str, float] = {}
    for game_id, summary in read_summaries(root, league, season):
        day = _date(summary)
        for line in parse_box_lines(game_id, summary):
            minutes = max(line.stats["minutes"], 0.0)
            seen = totals.setdefault((line.athlete_id, line.team_id), [0.0, 0, 0, day])
            seen[0] += minutes
            seen[1] += 1
            seen[2] += line.starter
            if day is not None and (seen[3] is None or day < seen[3]):
                seen[3] = day
            team_minutes[line.team_id] = team_minutes.get(line.team_id, 0.0) + minutes
    rows = [
        {
            "athlete_id": athlete,
            "team_id": team,
            "minutes": minutes,
            "games": games,
            "starts": starts,
            "share": 5 * minutes / team_minutes[team] if team_minutes[team] else 0.0,
            "first_date": first,
        }
        for (athlete, team), (minutes, games, starts, first) in sorted(totals.items())
    ]
    return pa.Table.from_pylist(rows, schema=MINUTES_SCHEMA)


def _date(summary: Mapping[str, Any]) -> date | None:
    competitions = (summary.get("header") or {}).get("competitions") or [{}]
    text = competitions[0].get("date") or ""
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def _number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return 0.0
