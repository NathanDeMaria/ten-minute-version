"""
How good the estimated lineups are, on games where the real ones are known.

Every game since 2025-02-15 has its substitutions, so its real floor is known
(`lineups.on_floor`). Delete the substitutions and estimate the floor from
what's left, and the gap between the two is the estimate's error -- measured
the same way for every rule, and every change to one.

Only games whose real floor is clean are used: starters found, and no
repairs, so that the truth being scored against is the feed's own.

`old_feed` strips ids the way the feed before 2022-23 lacks them -- shots,
free throws, steals and assists name their player only in the text -- before
`names` puts them back, so the old seasons' error can be estimated too.
"""

from collections import Counter
from dataclasses import dataclass, replace
from typing import Callable, Iterable, Sequence

import numpy as np

from .estimate import (
    BoxScore,
    Rules,
    SeasonContext,
    estimate_floors,
    nearest_action_floors,
    season_context,
)
from .events import Kind, classify
from .lineups import SIDES, Floor, on_floor
from .names import resolve_names
from .plays import Game
from .possessions import Possession, possessions
from .probabilities import Chances, OnFloorModel, features, fit, labels

_SUBSTITUTIONS = (Kind.SUB_IN, Kind.SUB_OUT)
# The categories whose plays carry no id before 2022-23: shots, free throws
# and steals.
_UNNAMED_IN_THE_OLD_FEED = {"1001", "1002", "1003", "1004", "1005", "1006", "1022"}


@dataclass
class Known:
    """A game whose real lineups are known, with its substitutions taken out."""

    game: Game
    kinds: list[Kind]
    # The real floor at every remaining play
    floors: list[Floor]
    truth: list[Possession]
    # What a box score would have said: the real starters, and each
    # player's share of the possessions.
    box: BoxScore


def known_games(games: Iterable[Game], old_feed: bool = False) -> list[Known]:
    known = []
    for game in games:
        game = resolve_names(game)
        kinds = [classify(play) for play in game.plays]
        if not any(kind in _SUBSTITUTIONS for kind in kinds):
            continue
        floors, report = on_floor(game.plays, kinds)
        if not report.starters_found or report.repairs:
            continue
        keep = [i for i, kind in enumerate(kinds) if kind not in _SUBSTITUTIONS]
        hidden = replace(game, plays=tuple(game.plays[i] for i in keep))
        if old_feed:
            hidden = resolve_names(_as_old_feed(hidden))
        hidden_kinds = [classify(play) for play in hidden.plays]
        hidden_floors = [floors[i] for i in keep]
        truth, _ = possessions(hidden, hidden_kinds, hidden_floors)
        known.append(
            Known(
                hidden,
                hidden_kinds,
                hidden_floors,
                truth,
                _box_score(hidden_floors[0], truth),
            )
        )
    return known


@dataclass
class Accuracy:
    players_right: float
    all_five_right: float


def accuracy(
    known: Sequence[Known], floors_of: Callable[[Known], list[Floor]]
) -> Accuracy:
    """Share of the players on the floor found, and of sides found whole."""
    hits = sides = whole = 0
    for game in known:
        estimated, _ = possessions(game.game, game.kinds, floors_of(game))
        for real, guess in zip(game.truth, estimated):
            for s in range(2):
                hit = len(real.floor[s] & guess.floor[s])
                hits += hit
                sides += 1
                whole += hit == 5
    return Accuracy(hits / max(sides, 1) / 5, whole / max(sides, 1))


@dataclass
class Calibration:
    # Squared error of the chances, over every player who appeared for a side
    # at every possession -- and of the hard estimate's 0s and 1s on the same
    # rows, for scale.
    brier: float
    hard_brier: float
    # (mean chance, share really on, player-possessions) by decile of chance
    deciles: list[tuple[float, float, int]]


def calibration(
    known: Sequence[Known],
    chances_of: Callable[[Known], list[Chances]],
    floors_of: Callable[[Known], list[Floor]],
) -> Calibration:
    p_all, y_all, hard_all = [], [], []
    for game in known:
        chances = chances_of(game)
        floors = floors_of(game)
        for possession in game.truth:
            at = chances[possession.credit_play]
            guess = floors[possession.credit_play]
            for s in range(2):
                for player, p in at[s].items():
                    p_all.append(p)
                    y_all.append(player in possession.floor[s])
                    hard_all.append(player in guess[s])
    p = np.asarray(p_all)
    y = np.asarray(y_all, dtype=float)
    hard = np.asarray(hard_all, dtype=float)
    edges = np.quantile(p, np.linspace(0, 1, 11))
    deciles = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        inside = (p >= lo) & (p <= hi)
        if inside.any():
            deciles.append(
                (float(p[inside].mean()), float(y[inside].mean()), int(inside.sum()))
            )
    return Calibration(
        float(np.mean((p - y) ** 2)), float(np.mean((hard - y) ** 2)), deciles
    )


def estimate_season(
    known: Sequence[Known], rules: Rules = Rules(), box: bool = False
) -> tuple[SeasonContext, list[list[Floor]]]:
    """
    Every game's estimate. Without `box`, the way a season without box
    scores gets one: roles from the season's plays, and no minutes -- minutes
    estimated from a first pass of the estimate only reinforce its mistakes
    (a point of all-five-right, measured). With `box`, each game's own
    starters and minutes.
    """
    context = season_context(k.game for k in known)
    return context, [
        estimate_floors(k.game, k.kinds, context, k.box if box else None, rules)
        for k in known
    ]


def fit_on_floor(
    league: str, known: Sequence[Known], rules: Rules = Rules(), box: bool = False
) -> OnFloorModel:
    context, estimates = estimate_season(known, rules, box)
    rows, ys = [], []
    for game, estimate in zip(known, estimates):
        frame = features(
            game.game,
            game.kinds,
            estimate,
            nearest_action_floors(game.game, game.kinds),
            _minutes(game, context, box),
        )
        rows.append(frame.matrix)
        ys.append(labels(frame, game.floors))
    return fit(league, rows, ys, with_box_scores=box)


def chances_for(
    model: OnFloorModel,
    known: Sequence[Known],
    rules: Rules = Rules(),
    box: bool = False,
) -> dict[str, list[Chances]]:
    context, estimates = estimate_season(known, rules, box)
    return {
        game.game.game_id: model.chances(
            game.game,
            game.kinds,
            estimate,
            nearest_action_floors(game.game, game.kinds),
            _minutes(game, context, box),
        )
        for game, estimate in zip(known, estimates)
    }


def _minutes(game: Known, context: SeasonContext, box: bool) -> dict:
    if box:
        return {side: game.box.minutes[side] for side in SIDES}
    return {side: context.minutes for side in SIDES}


def _box_score(first: Floor, truth: list[Possession]) -> BoxScore:
    on = {side: Counter() for side in SIDES}
    for possession in truth:
        for side, five in zip(SIDES, possession.floor):
            on[side].update(five)
    n = max(len(truth), 1)
    return BoxScore(
        starters={"home": first[0], "away": first[1]},
        minutes={side: {p: c / n for p, c in on[side].items()} for side in SIDES},
    )


def _as_old_feed(game: Game) -> Game:
    plays = tuple(
        replace(play, athlete_id=None, athlete_name=None, participant_ids=())
        if play.category in _UNNAMED_IN_THE_OLD_FEED
        else play
        for play in game.plays
    )
    return replace(game, plays=plays)
