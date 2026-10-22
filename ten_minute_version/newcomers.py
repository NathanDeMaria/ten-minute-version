"""
What a newcomer is worth before he plays, and how sure that is: a normal
prior on each side whose mean and width both go with his recruiting rank.

`recruiting.RecruitModel` regresses newcomers' first-season ratings on their
rank, but those ratings are shrunk -- toward the box score, by penalty 1000
-- and a newcomer's first season is a single noisy season, so they're
shrunk hard: a top-5 recruit's average comes out ~3 points per 100 under
what his lineups say. And a regression says nothing about width.

So instead each season is fitted with every newcomer (never rated before)
at a nearly flat prior and no box score, and the posterior covariance
(`posterior`) says how noisy each newcomer's estimate is: an estimate is
his talent plus known normal noise, and the talent's distribution by rank
can be deconvolved from it (`NewcomerEstimate`, `NewcomerPrior.learn`):

- ranked recruits (this season's class or last's): mean a + b log(rank),
  sd s exp(d score) with score = (101 - rank) / 100, a, b, s per side and
  the shape d shared by both. Width goes with the rank score rather than
  log rank so that no one rank -- the #1 pick, at log rank 0 -- gets a free
  width of its own;
- unranked newcomers (walk-ons, transfers from outside Division I, players
  ESPN didn't rank): one mean and sd per side.

What's well measured is the mean. The width is not, on defense especially:
a newcomer's defensive noise variance nearly equals the variance of the
estimates themselves, so a few percent off in the noise moves the width a
lot (scaled by the split-half calibration's 0.91, mens defense goes from
~1 per 100 to ~2). The shape is near enough normal: a skew-normal (which,
plus normal noise, stays skew-normal) and a free-form distribution fit no
better than a normal for mens; womens has a mild right tail (a few
freshmen far above the rest).
"""

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pyarrow as pa

from .recruiting import Recruit, features

_RELEASES = Path(__file__).parent / "releases"
SIDES = ("offense", "defense")
# An estimate noisier than this (per 100, squared) says next to nothing,
# and a few hundred such players only slow the fit.
MAX_NOISE = 200.0


@dataclass(frozen=True)
class NewcomerEstimate:
    """A newcomer's first season, nearly unshrunk: per 100, with its noise."""

    season: int
    athlete_id: str
    rank: int | None  # his recruiting rank, if ranked this season
    possessions: float
    offense: float
    defense: float
    var_offense: float
    var_defense: float
    cov: float


NEWCOMER_SCHEMA = pa.schema(
    [
        pa.field("season", pa.int16(), nullable=False),
        pa.field("athlete_id", pa.string(), nullable=False),
        pa.field("rank", pa.int16()),
        pa.field("possessions", pa.float32(), nullable=False),
        pa.field("offense", pa.float32(), nullable=False),
        pa.field("defense", pa.float32(), nullable=False),
        pa.field("var_offense", pa.float32(), nullable=False),
        pa.field("var_defense", pa.float32(), nullable=False),
        pa.field("cov", pa.float32(), nullable=False),
    ]
)


def nelder_mead(
    f: Callable[[np.ndarray], float],
    start: Sequence[float],
    step: float = 0.5,
    iterations: int = 2000,
) -> tuple[np.ndarray, float]:
    """The minimum of `f` near `start`, by Nelder-Mead (numpy only)."""
    n = len(start)
    points = [np.array(start, float)] + [
        np.array(start, float) + step * np.eye(n)[i] for i in range(n)
    ]
    values = [f(p) for p in points]
    for _ in range(iterations):
        order = np.argsort(values)
        points, values = [points[i] for i in order], [values[i] for i in order]
        centre = np.mean(points[:-1], axis=0)
        reflected = centre + (centre - points[-1])
        fr = f(reflected)
        if fr < values[0]:
            expanded = centre + 2 * (centre - points[-1])
            fe = f(expanded)
            points[-1], values[-1] = (expanded, fe) if fe < fr else (reflected, fr)
        elif fr < values[-2]:
            points[-1], values[-1] = reflected, fr
        else:
            contracted = centre + 0.5 * (points[-1] - centre)
            fc = f(contracted)
            if fc < values[-1]:
                points[-1], values[-1] = contracted, fc
            else:
                points = [points[0] + 0.5 * (p - points[0]) for p in points]
                values = [f(p) for p in points]
    best = int(np.argmin(values))
    return points[best], values[best]


def normal_eb(x: np.ndarray, noise: np.ndarray) -> tuple[float, float]:
    """(mean, sd) of the talent behind estimates x, each with normal noise of
    variance `noise`: x ~ N(mean, sd^2 + noise), by maximum likelihood."""

    def nll(p: np.ndarray) -> float:
        s2 = np.exp(2 * p[1]) + noise
        return float(0.5 * np.sum(np.log(s2) + (x - p[0]) ** 2 / s2))

    (mean, log_sd), _ = nelder_mead(nll, [float(np.mean(x)), 0.5])
    return float(mean), float(np.exp(log_sd))


@dataclass(frozen=True)
class SidePrior:
    """One side's newcomer prior, per 100: ranked mean a + b log(rank) and sd
    exp(log_sd + shape score); unranked, one mean and sd."""

    a: float
    b: float
    log_sd: float
    unranked_mean: float
    unranked_sd: float


@dataclass(frozen=True)
class NewcomerPrior:
    league: str
    offense: SidePrior
    defense: SidePrior
    shape: float  # d: log sd per unit of rank score, both sides

    @staticmethod
    def release_path(league: str) -> Path:
        return _RELEASES / f"newcomer-prior-{league}.json"

    def side(
        self, side: str, recruit: Recruit | None, season: int
    ) -> tuple[float, float]:
        """(mean, variance) per possession of a newcomer's `side` in `season`."""
        p = self.offense if side == "offense" else self.defense
        if recruit is None or features(recruit, season)[0] == 0:
            return p.unranked_mean / 100, (p.unranked_sd / 100) ** 2
        score = (101 - recruit.rank) / 100
        mean = p.a + p.b * math.log(recruit.rank)
        sd = math.exp(p.log_sd + self.shape * score)
        return mean / 100, (sd / 100) ** 2

    @classmethod
    def learn(
        cls,
        league: str,
        found: Sequence[NewcomerEstimate],
        noise_scale: tuple[float, float] = (1.0, 1.0),
    ) -> "NewcomerPrior":
        """Maximum likelihood on newcomers' nearly unshrunk estimates, each
        side's noise scaled by `noise_scale` (offense, defense)."""
        kept = [n for n in found if max(n.var_offense, n.var_defense) < MAX_NOISE]
        ranked = [n for n in kept if n.rank is not None]
        rank = np.array([n.rank or 0 for n in ranked], float)
        log_rank, score = np.log(rank), (101 - rank) / 100
        x = {s: np.array([getattr(n, s) for n in ranked]) for s in SIDES}
        noise = {
            s: k * np.array([getattr(n, f"var_{s}") for n in ranked])
            for s, k in zip(SIDES, noise_scale)
        }

        def nll(p: np.ndarray) -> float:
            total = 0.0
            for s, (a, b, log_sd) in zip(SIDES, (p[0:3], p[3:6])):
                s2 = np.exp(2 * (log_sd + p[6] * score)) + noise[s]
                total += 0.5 * np.sum(np.log(s2) + (x[s] - a - b * log_rank) ** 2 / s2)
            return float(total)

        p, _ = nelder_mead(nll, [4.0, -0.7, 0.5, 4.0, -0.7, 0.0, 0.5], 0.3, 6000)
        p, _ = nelder_mead(nll, list(p), 0.1, 6000)
        unranked = [n for n in kept if n.rank is None]
        sides = []
        for i, (s, k) in enumerate(zip(SIDES, noise_scale)):
            mean, sd = normal_eb(
                np.array([getattr(n, s) for n in unranked]),
                k * np.array([getattr(n, f"var_{s}") for n in unranked]),
            )
            a, b, log_sd = p[3 * i : 3 * i + 3]
            sides.append(SidePrior(float(a), float(b), float(log_sd), mean, sd))
        return cls(league, sides[0], sides[1], float(p[6]))

    def save(self, path: Path | None = None, **about) -> Path:
        path = path or self.release_path(self.league)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "league": self.league,
                    "offense": asdict(self.offense),
                    "defense": asdict(self.defense),
                    "shape": self.shape,
                    **about,
                },
                indent=2,
            )
            + "\n"
        )
        return path

    @classmethod
    def load(cls, league: str, path: Path | None = None) -> "NewcomerPrior":
        raw = json.loads((path or cls.release_path(league)).read_text())
        return cls(
            raw["league"],
            SidePrior(**raw["offense"]),
            SidePrior(**raw["defense"]),
            raw["shape"],
        )
