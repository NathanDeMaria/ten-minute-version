"""
The possession table: a game as lineup-vs-lineup stints, each side's
possessions in a stint counted by how they ended.

A stint is a run of possessions with the same ten players on the floor. It
gives one row per side that had the ball in it: that side's five on offense
against the other five on defense. Rows are additive -- a lineup's season
is the sum of its rows -- so anything coarser is a group-by away.

A team from outside Division I plays as one generic team (`opponents`):
its side of every row is the single id `GENERIC`, against the other side's
five, so its games stay in the table even though its players can't be
followed.

Possessions whose floor isn't five against five (or five against the
generic team) are left out of the table and counted in the game's report
row instead, along with everything else that says how far to trust the
game: whether it had substitutions at all, whether the starters came out at
five, how many repairs the lineups needed, and whether the points counted
match ESPN's final.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import NamedTuple

import pyarrow as pa

from .events import Kind, classify
from .lineups import SIDES, Floor, on_floor
from .names import resolve_names
from .opponents import GENERIC, Opponents
from .plays import Game
from .possessions import Possession, Shot, possessions

STINT_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        # 1-based, in game order. A stint split by a possession that wasn't
        # five on five comes back as two stints, even with the same ten.
        pa.field("stint", pa.int16(), nullable=False),
        # The period the stint started in, and seconds since the tip at its
        # first possession and its last.
        pa.field("period", pa.int8(), nullable=False),
        pa.field("start_seconds", pa.int16(), nullable=False),
        pa.field("end_seconds", pa.int16(), nullable=False),
        # "home" or "away". Neutral sites aren't known here.
        pa.field("offense_side", pa.string(), nullable=False),
        # Five ESPN athlete ids each, sorted -- or just `GENERIC` ("generic")
        # for a team from outside Division I (`opponents`).
        pa.field("offense_players", pa.list_(pa.string()), nullable=False),
        pa.field("defense_players", pa.list_(pa.string()), nullable=False),
        # The offense's lead when the stint started, from the points counted.
        pa.field("start_margin", pa.int16(), nullable=False),
        pa.field("possessions", pa.int16(), nullable=False),
        pa.field("points", pa.int16(), nullable=False),
        # Possessions with no points.
        pa.field("empty", pa.int16(), nullable=False),
        # How each possession ended. These six add up to `possessions`.
        pa.field("ended_fg2", pa.int16(), nullable=False),
        pa.field("ended_fg3", pa.int16(), nullable=False),
        pa.field("ended_ft", pa.int16(), nullable=False),
        pa.field("ended_turnover", pa.int16(), nullable=False),
        pa.field("ended_miss", pa.int16(), nullable=False),
        pa.field("ended_other", pa.int16(), nullable=False),
        pa.field("fga2", pa.int16(), nullable=False),
        pa.field("fgm2", pa.int16(), nullable=False),
        pa.field("fga3", pa.int16(), nullable=False),
        pa.field("fgm3", pa.int16(), nullable=False),
        pa.field("fta", pa.int16(), nullable=False),
        pa.field("ftm", pa.int16(), nullable=False),
        pa.field("oreb", pa.int16(), nullable=False),
        pa.field("tov", pa.int16(), nullable=False),
    ]
)

GAME_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        # False for every game before mid-February 2025, when ESPN started
        # logging substitutions, and a few a week since. Such a game has no
        # lineups and no rows in the stint table.
        pa.field("has_substitutions", pa.bool_(), nullable=False),
        # "home" or "away" for the side played as one generic team, if either.
        pa.field("generic_side", pa.string()),
        # The starters came out at five, on every side that's followed.
        pa.field("starters_found", pa.bool_(), nullable=False),
        pa.field("repairs", pa.int16(), nullable=False),
        pa.field("possessions", pa.int16(), nullable=False),
        # Possessions left out of the stint table because a side wasn't five.
        pa.field("possessions_dropped", pa.int16(), nullable=False),
        # Counted from the scoring plays, technical free throws included...
        pa.field("home_points", pa.int16(), nullable=False),
        pa.field("away_points", pa.int16(), nullable=False),
        # ...of which these were technical free throws, in no possession.
        pa.field("home_technical_points", pa.int16(), nullable=False),
        pa.field("away_technical_points", pa.int16(), nullable=False),
        # ESPN's final, from the last play that carries a score.
        pa.field("home_final", pa.int16()),
        pa.field("away_final", pa.int16()),
    ]
)

PLAYER_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("athlete_id", pa.string(), nullable=False),
        # The last spelling seen in the season.
        pa.field("athlete_name", pa.string()),
    ]
)

# Every field goal and free throw in the stint table's possessions, for
# expected points: (game_id, stint, offense_side) joins a shot to its row.
SHOT_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("stint", pa.int16(), nullable=False),
        pa.field("offense_side", pa.string(), nullable=False),
        pa.field("shooter", pa.string()),
        # 2, 3, or 1 for a free throw
        pa.field("value", pa.int8(), nullable=False),
        pa.field("made", pa.bool_(), nullable=False),
        # JumpShot, LayUpShot, DunkShot, TipShot; a free throw's own type
        pa.field("shot_type", pa.string()),
        # Feet: x across the court, the hoop at 25; y out from the baseline.
        pa.field("x", pa.float32()),
        pa.field("y", pa.float32()),
    ]
)

_ENDINGS = ("fg2", "fg3", "ft", "turnover", "miss", "other")
_COUNTS = ("fga2", "fgm2", "fga3", "fgm3", "fta", "ftm", "oreb", "tov")


@dataclass
class GameStints:
    rows: list[dict]
    report: dict
    shots: list[dict]


class SeasonTables(NamedTuple):
    stints: pa.Table
    games: pa.Table
    players: pa.Table
    shots: pa.Table


def game_stints(game: Game, opponents: Opponents | None = None) -> GameStints:
    """
    A game's stint rows and its report row. `opponents` says which side of
    the season's games is generic; without it no side is.
    """
    game = resolve_names(game)
    kinds = [classify(play) for play in game.plays]
    report = _report(game)
    # Said for every game, stints or not: the estimated lineups play the
    # same side as generic, and it's checked against ESPN's own D1 flag.
    generic = opponents.generic_side(game) if opponents is not None else None
    report["generic_side"] = generic
    if not any(kind in (Kind.SUB_IN, Kind.SUB_OUT) for kind in kinds):
        return GameStints([], report, [])
    floors, lineups = on_floor(game.plays, kinds)
    found, score = possessions(game, kinds, floors)
    rows, dropped, shots = _rows(game, found, generic)
    starters = dict(zip(SIDES, lineups.starters))
    report.update(
        has_substitutions=True,
        starters_found=all(starters[side] == 5 for side in SIDES if side != generic),
        repairs=lineups.repairs,
        possessions=len(found),
        possessions_dropped=dropped,
        home_points=score.points["home"],
        away_points=score.points["away"],
        home_technical_points=score.technical_points["home"],
        away_technical_points=score.technical_points["away"],
    )
    return GameStints(rows, report, shots)


def season_tables(
    games: Iterable[Game], opponents: Opponents | None = None
) -> SeasonTables:
    """
    The stint, game, player and shot tables for a run of games, with
    `opponents` as for `game_stints`.
    """
    rows: list[dict] = []
    reports: list[dict] = []
    shots: list[dict] = []
    names: dict[tuple[str, int, str], str | None] = {}
    for game in games:
        built = game_stints(game, opponents)
        rows += built.rows
        reports.append(built.report)
        shots += built.shots
        for play in game.plays:
            if play.athlete_id:
                names[(game.league, game.season, play.athlete_id)] = play.athlete_name
    players = [
        {"league": league, "season": season, "athlete_id": athlete, "athlete_name": n}
        for (league, season, athlete), n in sorted(names.items())
    ]
    return SeasonTables(
        pa.Table.from_pylist(rows, schema=STINT_SCHEMA),
        pa.Table.from_pylist(reports, schema=GAME_SCHEMA),
        pa.Table.from_pylist(players, schema=PLAYER_SCHEMA),
        pa.Table.from_pylist(shots, schema=SHOT_SCHEMA),
    )


_GENERIC_SIDE = frozenset({GENERIC})


def _rows(
    game: Game, found: list[Possession], generic: str | None
) -> tuple[list[dict], int, list[dict]]:
    rows: dict[tuple[int, str], dict] = {}
    shots: list[dict] = []
    stint = 0
    previous = None
    stint_start: Possession | None = None
    dropped = 0
    for possession in found:
        floor = _played(possession.floor, generic)
        if any(len(side) != 5 and side != _GENERIC_SIDE for side in floor):
            dropped += 1
            previous = None
            continue
        # A generic side never changes, so a stint is the other side's five.
        if floor != previous:
            stint += 1
            previous = floor
            stint_start = possession
        assert stint_start is not None
        key = (stint, possession.offense)
        if key not in rows:
            rows[key] = _new_row(game, stint, possession, stint_start, floor)
        _add(rows[key], possession)
        shots += [
            shot_row(game, stint, possession.offense, shot) for shot in possession.shots
        ]
    # A stint ends at its last possession, whichever side had it, so both of
    # its rows say the same.
    ends: dict[int, int] = {}
    for (stint, _), row in rows.items():
        ends[stint] = max(ends.get(stint, 0), row["end_seconds"])
    for (stint, _), row in rows.items():
        row["end_seconds"] = ends[stint]
    return list(rows.values()), dropped, shots


def shot_row(game: Game, stint: int, offense: str, shot: Shot) -> dict:
    return {
        "league": game.league,
        "season": game.season,
        "game_id": game.game_id,
        "stint": stint,
        "offense_side": offense,
        "shooter": shot.shooter,
        "value": shot.value,
        "made": shot.made,
        "shot_type": shot.shot_type,
        "x": shot.x,
        "y": shot.y,
    }


def _played(floor: Floor, generic: str | None) -> Floor:
    """The floor as the table has it: a generic side as `GENERIC` alone."""
    home, away = floor
    if generic == "home":
        return _GENERIC_SIDE, away
    if generic == "away":
        return home, _GENERIC_SIDE
    return floor


def _new_row(
    game: Game, stint: int, possession: Possession, start: Possession, floor: Floor
) -> dict:
    home, away = floor
    offense, defense = (home, away) if possession.offense == "home" else (away, home)
    margin = start.start_margin
    if start.offense != possession.offense:
        margin = -margin
    return {
        "league": game.league,
        "season": game.season,
        "date": game.date,
        "game_id": game.game_id,
        "stint": stint,
        "period": start.period,
        "start_seconds": start.start_seconds,
        "end_seconds": possession.end_seconds,
        "offense_side": possession.offense,
        "offense_players": sorted(offense),
        "defense_players": sorted(defense),
        "start_margin": margin,
        "possessions": 0,
        "points": 0,
        "empty": 0,
        **{f"ended_{ending}": 0 for ending in _ENDINGS},
        **{count: 0 for count in _COUNTS},
    }


def _add(row: dict, possession: Possession) -> None:
    row["possessions"] += 1
    row["points"] += possession.points
    row["empty"] += possession.points == 0
    row[f"ended_{possession.ended}"] += 1
    for count in _COUNTS:
        row[count] += getattr(possession, count)
    row["end_seconds"] = max(row["end_seconds"], possession.end_seconds)


def _report(game: Game) -> dict:
    final = next(
        (
            (play.home_score, play.away_score)
            for play in reversed(game.plays)
            if play.home_score is not None and play.away_score is not None
        ),
        (None, None),
    )
    return {
        "league": game.league,
        "season": game.season,
        "date": game.date,
        "game_id": game.game_id,
        "has_substitutions": False,
        "generic_side": None,
        "starters_found": False,
        "repairs": 0,
        "possessions": 0,
        "possessions_dropped": 0,
        "home_points": 0,
        "away_points": 0,
        "home_technical_points": 0,
        "away_technical_points": 0,
        "home_final": final[0],
        "away_final": final[1],
    }
