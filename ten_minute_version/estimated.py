"""
Possession rows for the games ESPN logged no substitutions for -- every game
before 2025-02-15 -- from estimated lineups.

Each game's floors are estimated (`estimate`) with its own box score's
starters and minutes (`box_scores`), turned into each player's chance of
being on the floor at every play (`probabilities`, the box-score release),
and each possession is credited to every player by his chance at its credit
play -- the play `possessions` gives its floor from. A row is one
possession, its players weighted by those chances (five a side), in the
shape `ratings.Rows` reads: `stint` numbers the game's possessions, so the
shot table joins the same way the stint table's does.

A team from outside Division I plays as the generic team here too
(`opponents`). A game with no summary, so no box score, or whose box score
doesn't name five starters a side, is left out.
"""

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import pyarrow as pa

from .box_scores import BoxLine
from .context import GameContext
from .estimate import (
    BoxScore,
    Rules,
    SeasonContext,
    estimate_floors,
    nearest_action_floors,
)
from .events import Kind, classify
from .lineups import SIDES
from .names import resolve_names
from .opponents import GENERIC, Opponents
from .plays import Game
from .possessions import possessions
from .probabilities import OnFloorModel
from .stints import SHOT_SCHEMA, shot_row

ESTIMATED_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        # The possession's number in its game, from 1.
        pa.field("stint", pa.int16(), nullable=False),
        pa.field("offense_side", pa.string(), nullable=False),
        # Everyone with a chance of being on the floor, and that chance.
        pa.field("offense_players", pa.list_(pa.string()), nullable=False),
        pa.field("offense_weights", pa.list_(pa.float32()), nullable=False),
        pa.field("defense_players", pa.list_(pa.string()), nullable=False),
        pa.field("defense_weights", pa.list_(pa.float32()), nullable=False),
        pa.field("possessions", pa.int16(), nullable=False),
        pa.field("points", pa.int16(), nullable=False),
    ]
)

_SUBSTITUTIONS = (Kind.SUB_IN, Kind.SUB_OUT)


def box_score(lines: Sequence[BoxLine], context: GameContext) -> BoxScore | None:
    """
    What the estimator reads from a game's box score: each side's starters,
    and each player's share of the game on the floor. None unless both
    sides name five starters.
    """
    side_of = {context.home.team_id: "home", context.away.team_id: "away"}
    starters: dict[str, set[str]] = {side: set() for side in SIDES}
    minutes: dict[str, dict[str, float]] = {side: {} for side in SIDES}
    for line in lines:
        side = side_of.get(line.team_id)
        if side is None:
            continue
        if line.starter:
            starters[side].add(line.athlete_id)
        minutes[side][line.athlete_id] = line.stats["minutes"]
    if any(len(starters[side]) != 5 for side in SIDES):
        return None
    # A side's minutes add to five times the game's length, overtime and all.
    shares = {}
    for side in SIDES:
        length = sum(minutes[side].values()) / 5 or 1.0
        shares[side] = {p: m / length for p, m in minutes[side].items()}
    return BoxScore({side: frozenset(starters[side]) for side in SIDES}, shares)


def estimated_tables(
    games: Iterable[Game],
    box_lines: Iterable[BoxLine],
    contexts: Mapping[str, GameContext],
    model: OnFloorModel,
    season: SeasonContext,
    opponents: Opponents,
    rules: Rules = Rules(),
) -> tuple[pa.Table, pa.Table]:
    """The estimated possession rows and their shots, for every game without
    substitutions that has a usable box score."""
    lines_of: dict[str, list[BoxLine]] = defaultdict(list)
    for line in box_lines:
        lines_of[line.game_id].append(line)
    rows: list[dict] = []
    shots: list[dict] = []
    for game in games:
        context = contexts.get(game.game_id)
        if context is None:
            continue
        game = resolve_names(game)
        kinds = [classify(play) for play in game.plays]
        if any(kind in _SUBSTITUTIONS for kind in kinds):
            continue
        box = box_score(lines_of.get(game.game_id, []), context)
        if box is None:
            continue
        floors = estimate_floors(game, kinds, season, box, rules)
        chances = model.chances(
            game, kinds, floors, nearest_action_floors(game, kinds), box.minutes
        )
        generic = opponents.generic_side(game)
        found, _ = possessions(game, kinds, floors)
        for number, possession in enumerate(found, start=1):
            home, away = chances[possession.credit_play]
            on = {"home": _five(home), "away": _five(away)}
            if generic is not None:
                on[generic] = {GENERIC: 1.0}
            offense = on[possession.offense]
            defense = on["away" if possession.offense == "home" else "home"]
            rows.append(
                {
                    "league": game.league,
                    "season": game.season,
                    "game_id": game.game_id,
                    "stint": number,
                    "offense_side": possession.offense,
                    "offense_players": list(offense),
                    "offense_weights": list(offense.values()),
                    "defense_players": list(defense),
                    "defense_weights": list(defense.values()),
                    "possessions": 1,
                    "points": possession.points,
                }
            )
            shots += [
                shot_row(game, number, possession.offense, shot)
                for shot in possession.shots
            ]
    return (
        pa.Table.from_pylist(rows, schema=ESTIMATED_SCHEMA),
        pa.Table.from_pylist(shots, schema=SHOT_SCHEMA),
    )


def _five(chances: Mapping[str, float]) -> dict[str, float]:
    """
    A side's chances, scaled down to five where they add up to more: more
    than five players acted in the stretch between dead balls (a
    substitution the dead balls didn't catch), and each is pinned at 1.
    """
    total = sum(chances.values())
    if total <= 5:
        return dict(chances)
    return {player: chance * 5 / total for player, chance in chances.items()}
