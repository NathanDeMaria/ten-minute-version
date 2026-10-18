"""
Who played each game and where, from ESPN's game summaries.

endgame stores every NCAABB game's summary whole, one gzipped JSON object a
game, under

    {root}/{league}/{season}/{game_id}.json.gz

where `root` is endgame's `summaries/ncaabb` prefix or a local copy of it
(`make summaries`). The play-by-play only says home and away; the summary's
header says which teams those were, whether the game was at a neutral site,
whether it was a conference game, and each team's conference -- including
whether that conference is in Division I. Only the header is read.

The conference is the one ESPN files the team under now, not the one it was
in that season: Lindenwood shows in a D1 conference in 2011-12, when it was
in Division II. So `division_one` is right for the current season and not
always before it; `opponents` says which teams were from outside D1 by
their schedules instead.
"""

import gzip
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping

import pyarrow as pa

# ESPN's group id for NCAA Division I, the parent of every D1 conference;
# and for NCAA basketball, the parent of the group for teams outside D1.
_DIVISION_ONE = "50"
_NCAA = "52"


@dataclass(frozen=True)
class Team:
    team_id: str
    name: str | None
    # ESPN's conference id and slug (or the conference's division, for
    # one that had them), and whether it's in Division I.
    conference_id: str | None
    conference: str | None
    division_one: bool


@dataclass(frozen=True)
class GameContext:
    game_id: str
    neutral_site: bool
    conference_game: bool
    home: Team
    away: Team

    def team(self, side: str) -> Team:
        return self.home if side == "home" else self.away


CONTEXT_SCHEMA = pa.schema(
    [
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("neutral_site", pa.bool_(), nullable=False),
        pa.field("conference_game", pa.bool_(), nullable=False),
        *[
            pa.field(f"{side}_{name}", kind, nullable=nullable)
            for side in ("home", "away")
            for name, kind, nullable in (
                ("team_id", pa.string(), False),
                ("team", pa.string(), True),
                ("conference_id", pa.string(), True),
                ("conference", pa.string(), True),
                ("division_one", pa.bool_(), False),
            )
        ],
    ]
)


def read_summaries(
    root: str | Path, league: str, season: int
) -> Iterator[tuple[str, dict]]:
    """Every stored summary of a season, as (game id, summary), in id order."""
    for path in sorted((Path(root) / league / str(season)).glob("*.json.gz")):
        with gzip.open(path, "rt") as file:
            yield path.name.removesuffix(".json.gz"), json.load(file)


def read_context(root: str | Path, league: str, season: int) -> dict[str, GameContext]:
    """Every stored game of a season, by game id. A summary without a usable
    header -- no home and away team -- is skipped."""
    found: dict[str, GameContext] = {}
    for game_id, summary in read_summaries(root, league, season):
        context = parse_context(game_id, summary)
        if context is not None:
            found[context.game_id] = context
    return found


def parse_context(game_id: str, summary: Mapping[str, Any]) -> GameContext | None:
    competitions = (summary.get("header") or {}).get("competitions") or [{}]
    competition = competitions[0]
    teams = {
        competitor.get("homeAway"): _team(competitor.get("team") or {})
        for competitor in competition.get("competitors", [])
    }
    home, away = teams.get("home"), teams.get("away")
    if home is None or away is None:
        return None
    return GameContext(
        game_id=game_id,
        neutral_site=bool(competition.get("neutralSite")),
        conference_game=bool(competition.get("conferenceCompetition")),
        home=home,
        away=away,
    )


def context_table(contexts: Mapping[str, GameContext]) -> pa.Table:
    rows = []
    for context in contexts.values():
        row: dict[str, object] = {
            "game_id": context.game_id,
            "neutral_site": context.neutral_site,
            "conference_game": context.conference_game,
        }
        for side in ("home", "away"):
            team = context.team(side)
            row[f"{side}_team_id"] = team.team_id
            row[f"{side}_team"] = team.name
            row[f"{side}_conference_id"] = team.conference_id
            row[f"{side}_conference"] = team.conference
            row[f"{side}_division_one"] = team.division_one
        rows.append(row)
    return pa.Table.from_pylist(rows, schema=CONTEXT_SCHEMA)


def _team(team: Mapping[str, Any]) -> Team | None:
    if not team.get("id"):
        return None
    groups = team.get("groups") or {}
    parent = str((groups.get("parent") or {}).get("id"))
    # A conference with divisions (the MAC to 2019-20, the OVC to 2016-17,
    # the pods of 2020-21) files its teams under the division, whose parent
    # is the conference rather than D1.
    division = groups.get("isConference") is False and parent not in ("None", _NCAA)
    return Team(
        team_id=str(team["id"]),
        name=team.get("displayName"),
        conference_id=str(groups["id"]) if groups.get("id") else None,
        conference=groups.get("slug"),
        division_one=parent == _DIVISION_ONE or division,
    )
