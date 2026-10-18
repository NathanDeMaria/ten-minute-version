"""
Each player's box score line in each game, from the game summaries
`context` reads -- for the box-score prior on the ratings (`priors`).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

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


def _number(text: str) -> float:
    try:
        return float(text)
    except ValueError:
        return 0.0
