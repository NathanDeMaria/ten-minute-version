"""
The chance each player was on the floor, for games without substitutions.

A hard estimate (`estimate`) has all five right at fewer than half of the
possessions, and a rating fit on hard guesses would read every wrong one as
a fact. This turns the estimate into probabilities instead: a logistic
regression on what the estimate saw, fitted on the games whose lineups are
known, then scaled so each side's probabilities add up to five. A player who
did something in the stretch of play is on it, at 1.

Fitted with plain Newton steps on a handful of features -- no solver to
install, and the fit is a few coefficients in `releases/`.
"""

import json
import math
from bisect import bisect_left, bisect_right
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .estimate import Windows, _actions
from .events import Kind
from .lineups import SIDES, Floor
from .plays import Game

FEATURES = (
    "intercept",
    "in_estimate",
    "in_nearest_action",
    "log_seconds_since_action",
    "log_seconds_to_action",
    "no_action_since",
    "no_action_to_come",
    "minutes_share",
    "foul_trouble",
)

_RELEASES = Path(__file__).parent / "releases"

# Each side's chances at one play: player -> probability on the floor.
Chances = tuple[Mapping[str, float], Mapping[str, float]]


@dataclass(frozen=True)
class OnFloorModel:
    league: str
    coefficients: tuple[float, ...]
    # Fitted with each game's box-score minutes rather than season minutes
    # estimated from the plays. The two read `minutes_share` differently,
    # so each is its own release.
    with_box_scores: bool = False

    @staticmethod
    def release_path(league: str, with_box_scores: bool) -> Path:
        suffix = "-box" if with_box_scores else ""
        return _RELEASES / f"on-floor-{league}{suffix}.json"

    def chances(
        self,
        game: Game,
        kinds: Sequence[Kind],
        estimate: Sequence[Floor],
        nearest: Sequence[Floor],
        minutes: Mapping[str, Mapping[str, float]],
    ) -> list[Chances]:
        """Each player's chance of being on the floor, at every play."""
        frame = features(game, kinds, estimate, nearest, minutes)
        p = _sigmoid(frame.matrix @ np.asarray(self.coefficients))
        return frame.to_chances(p)

    def save(self, path: Path | None = None, **about) -> Path:
        """Write the fit, with whatever `about` says of how it was made."""
        path = path or self.release_path(self.league, self.with_box_scores)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "league": self.league,
                    "with_box_scores": self.with_box_scores,
                    "features": FEATURES,
                    "coefficients": dict(zip(FEATURES, self.coefficients)),
                    **about,
                },
                indent=2,
            )
            + "\n"
        )
        return path

    @classmethod
    def load(
        cls, league: str, with_box_scores: bool = False, path: Path | None = None
    ) -> OnFloorModel:
        path = path or cls.release_path(league, with_box_scores)
        raw = json.loads(path.read_text())
        if tuple(raw["features"]) != FEATURES:
            raise ValueError(f"{path} was fitted on other features: {raw['features']}")
        coefficients = tuple(raw["coefficients"][name] for name in FEATURES)
        return cls(raw["league"], coefficients, raw["with_box_scores"])


def fit(
    league: str,
    rows: Sequence[np.ndarray],
    labels: Sequence[np.ndarray],
    with_box_scores: bool = False,
) -> OnFloorModel:
    """
    Logistic regression by Newton's method, with a little ridge so a feature
    that never varies in the training games can't run off.
    """
    x = np.concatenate(rows)
    y = np.concatenate(labels)
    beta = np.zeros(x.shape[1])
    for _ in range(25):
        p = _sigmoid(x @ beta)
        gradient = x.T @ (y - p) - 1e-3 * beta
        hessian = (x * (p * (1 - p))[:, None]).T @ x + 1e-3 * np.eye(len(beta))
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.abs(step).max() < 1e-8:
            break
    return OnFloorModel(league, tuple(float(b) for b in beta), with_box_scores)


@dataclass
class Frame:
    """One row per play-window, side and player who appears for that side."""

    matrix: np.ndarray
    # Per row: window, side index, player, and whether he acted in the window
    window: np.ndarray
    side: np.ndarray
    players: list[str]
    acted: np.ndarray
    of_play: list[int]
    count: int

    def to_chances(self, p: np.ndarray) -> list[Chances]:
        """Scale each side of each window to five, with actors pinned at 1."""
        p = np.where(self.acted, 1.0, p)
        per_window: list[list[dict[str, float]]] = [[{}, {}] for _ in range(self.count)]
        order = np.lexsort((self.side, self.window))
        start = 0
        while start < len(order):
            end = start
            w, s = self.window[order[start]], self.side[order[start]]
            while end < len(order) and (
                self.window[order[end]] == w and self.side[order[end]] == s
            ):
                end += 1
            group = order[start:end]
            scaled = _to_five(p[group], self.acted[group])
            per_window[w][s] = {
                self.players[i]: float(q) for i, q in zip(group, scaled)
            }
            start = end
        return [(per_window[w][0], per_window[w][1]) for w in self.of_play]


def features(
    game: Game,
    kinds: Sequence[Kind],
    estimate: Sequence[Floor],
    nearest: Sequence[Floor],
    minutes: Mapping[str, Mapping[str, float]],
) -> Frame:
    windows = Windows(game, kinds)
    first_play = {}
    for i, w in enumerate(windows.of_play):
        first_play.setdefault(w, i)
    rows, window_ix, side_ix, players, acted = [], [], [], [], []
    for s, side in enumerate(SIDES):
        acts = _actions(game, kinds, windows, side)
        for w in range(windows.count):
            i = first_play.get(w)
            if i is None:
                continue
            now = windows.start[w]
            here = acts.by_window.get(w, set())
            for player, times in acts.times.items():
                before = bisect_left(times, now)
                after = bisect_right(times, now)
                since = now - times[before - 1] if before else None
                until = times[after] - now if after < len(times) else None
                fouls = [period for t, period in acts.fouls.get(player, ()) if t <= now]
                rows.append(
                    (
                        1.0,
                        float(player in estimate[i][s]),
                        float(player in nearest[i][s]),
                        math.log1p(since) if since is not None else 0.0,
                        math.log1p(until) if until is not None else 0.0,
                        float(since is None),
                        float(until is None),
                        minutes[side].get(player, 0.0),
                        float(_foul_trouble(game.league, windows.period[w], fouls)),
                    )
                )
                window_ix.append(w)
                side_ix.append(s)
                players.append(player)
                acted.append(player in here)
    return Frame(
        np.asarray(rows, dtype=float).reshape(-1, len(FEATURES)),
        np.asarray(window_ix, dtype=int),
        np.asarray(side_ix, dtype=int),
        players,
        np.asarray(acted, dtype=bool),
        windows.of_play,
        windows.count,
    )


def labels(frame: Frame, floors: Sequence[Floor]) -> np.ndarray:
    """1 where the player really was on the floor, for fitting."""
    first_play = {}
    for i, w in enumerate(frame.of_play):
        first_play.setdefault(w, i)
    return np.asarray(
        [
            float(player in floors[first_play[w]][s])
            for w, s, player in zip(frame.window, frame.side, frame.players)
        ]
    )


def _foul_trouble(league: str, period: int, fouls: list[int]) -> bool:
    first_half = 1 if league == "mens" else 2
    early = sum(1 for p in fouls if p <= first_half)
    return (period <= first_half and early >= 2) or len(fouls) >= 4


def _to_five(p: np.ndarray, pinned: np.ndarray) -> np.ndarray:
    """
    Scale a side's chances to sum to five, none above 1. Players pinned at 1
    stay there; the rest share what's left, in proportion, re-spreading
    whatever a cap at 1 frees up.
    """
    out = p.astype(float).copy()
    fixed = pinned.copy()
    for _ in range(len(out)):
        room = 5.0 - out[fixed].sum()
        free = ~fixed
        total = out[free].sum()
        if room <= 0 or total <= 0:
            out[free] = 0.0 if room <= 0 else out[free]
            break
        out[free] *= room / total
        over = free & (out > 1.0)
        if not over.any():
            break
        out[over] = 1.0
        fixed = fixed | over
    return out


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
