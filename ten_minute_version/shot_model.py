"""
The chance a shot goes in, for expected points.

**Where and how it was taken** is the-archer's model
(github.com/NathanDeMaria/the-archer), ported to numpy so this package keeps
no fitting library:

- a jump shot's chance is a logistic regression on a cubic in its distance
  from the hoop, with makes from beyond 35 feet counted as misses -- heaves,
  and shots placed at the wrong end, would otherwise make deep shots look
  good;
- a layup, dunk or tip is that type's league make rate;
- a free throw is the league free-throw rate.

**Who took it** is what the-archer leaves out, and what ratings can't: a
shooter who's better than the model's average keeps the credit for it. Each
shooter gets a shift on the log-odds of each kind of shot (jumpers, shots at
the rim, free throws), shrunk toward none in proportion to how few shots of
that kind he took:

    skill = sum(made - p) / (sum(p * (1 - p)) + k)

-- one Newton step from no skill on a ridge-penalised logistic shift, with
the penalty `k` chosen by how well skills from half a season's games
predict the other half's makes (`choose_shrinkage`).
"""

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

# The hoop, in the shot table's feet: x across the court, y from the baseline.
_HOOP = (25.0, 0.0)
# Makes from further out than this are counted as misses when fitting.
_DEEPEST_MAKE = 35.0
_AT_THE_RIM = ("LayUpShot", "DunkShot", "TipShot")

# The kinds of shot a shooter's skill is kept for.
Group = str
JUMPER: Group = "jumper"
RIM: Group = "rim"
FREE_THROW: Group = "ft"


@dataclass(frozen=True)
class Shots:
    """A table of shots as arrays, for fitting and predicting in bulk."""

    shooter: np.ndarray  # object: athlete id, or None
    value: np.ndarray  # int: 2, 3, or 1 for a free throw
    made: np.ndarray  # bool
    shot_type: np.ndarray  # object
    x: np.ndarray  # float, nan where unplaced
    y: np.ndarray

    @classmethod
    def from_rows(cls, rows: Sequence[Mapping]) -> "Shots":
        def floats(name: str) -> np.ndarray:
            return np.array(
                [np.nan if r[name] is None else r[name] for r in rows], float
            )

        return cls(
            shooter=np.array([r["shooter"] for r in rows], object),
            value=np.array([r["value"] for r in rows], int),
            made=np.array([r["made"] for r in rows], bool),
            shot_type=np.array([r["shot_type"] for r in rows], object),
            x=floats("x"),
            y=floats("y"),
        )

    def take(self, mask: np.ndarray) -> "Shots":
        return Shots(
            *(getattr(self, name)[mask] for name in Shots.__dataclass_fields__)
        )

    @property
    def group(self) -> np.ndarray:
        rim = np.isin(self.shot_type, _AT_THE_RIM)
        return np.where(self.value == 1, FREE_THROW, np.where(rim, RIM, JUMPER))

    @property
    def distance(self) -> np.ndarray:
        return np.hypot(self.x - _HOOP[0], self.y - _HOOP[1])


@dataclass(frozen=True)
class ShotModel:
    # The jump-shot logistic: coefficients on (1, z, z^2, z^3) for z the
    # distance standardised by these two.
    jumper: tuple[float, float, float, float]
    distance_mean: float
    distance_sd: float
    # Make rates: per rim shot type, free throws, and jump shots by value for
    # the ones with no location.
    rates: Mapping[str, float]
    free_throw: float
    unplaced: Mapping[int, float] = field(default_factory=dict)

    def chance(self, shots: Shots) -> np.ndarray:
        """Each shot's league-average chance of going in."""
        p = np.full(len(shots.value), np.nan)
        group = shots.group
        jumper = group == JUMPER
        d = shots.distance
        placed = jumper & np.isfinite(d)
        p[placed] = _sigmoid(self._features(d[placed]) @ np.array(self.jumper))
        for value, rate in self.unplaced.items():
            p[jumper & ~placed & (shots.value == value)] = rate
        for shot_type, rate in self.rates.items():
            p[shots.shot_type == shot_type] = rate
        p[group == FREE_THROW] = self.free_throw
        # Anything left -- a rim type with no rate yet -- gets the jump rate
        # for its value.
        missing = np.isnan(p)
        p[missing] = [self.unplaced.get(int(v), 0.4) for v in shots.value[missing]]
        return p

    def _features(self, distance: np.ndarray) -> np.ndarray:
        z = (distance - self.distance_mean) / self.distance_sd
        return np.stack([np.ones_like(z), z, z**2, z**3], axis=1)


def fit_shot_model(shots: Shots) -> ShotModel:
    group = shots.group
    jumper = group == JUMPER
    d = shots.distance
    placed = jumper & np.isfinite(d)
    distance = d[placed]
    made = shots.made[placed] & (distance <= _DEEPEST_MAKE)
    mean, sd = float(distance.mean()), float(distance.std())
    model = ShotModel((0.0, 0.0, 0.0, 0.0), mean, sd, {}, 0.0)
    x = model._features(distance)
    beta = _newton(x, made.astype(float))
    rates = {
        shot_type: float(shots.made[shots.shot_type == shot_type].mean())
        for shot_type in _AT_THE_RIM
        if (shots.shot_type == shot_type).any()
    }
    unplaced = {
        value: float(shots.made[jumper & (shots.value == value)].mean())
        for value in (2, 3)
        if (jumper & (shots.value == value)).any()
    }
    free_throws = group == FREE_THROW
    return ShotModel(
        jumper=(float(beta[0]), float(beta[1]), float(beta[2]), float(beta[3])),
        distance_mean=mean,
        distance_sd=sd,
        rates=rates,
        free_throw=float(shots.made[free_throws].mean()) if free_throws.any() else 0.7,
        unplaced=unplaced,
    )


# Each shooter's log-odds shift, by (shooter, group).
Skills = Mapping[tuple[str, Group], float]


def shooter_skills(
    shots: Shots, chance: np.ndarray, shrinkage: Mapping[Group, float]
) -> dict[tuple[str, Group], float]:
    residual: dict[tuple[str, Group], float] = {}
    information: dict[tuple[str, Group], float] = {}
    for shooter, group, made, p in zip(shots.shooter, shots.group, shots.made, chance):
        if shooter is None:
            continue
        key = (shooter, group)
        residual[key] = residual.get(key, 0.0) + (made - p)
        information[key] = information.get(key, 0.0) + p * (1 - p)
    return {
        key: residual[key] / (information[key] + shrinkage[key[1]]) for key in residual
    }


def with_skill(shots: Shots, chance: np.ndarray, skills: Skills) -> np.ndarray:
    """Each shot's chance, shifted by its shooter's skill at that kind of shot."""
    shift = np.array(
        [
            skills.get((shooter, group), 0.0) if shooter is not None else 0.0
            for shooter, group in zip(shots.shooter, shots.group)
        ]
    )
    p = np.clip(chance, 1e-6, 1 - 1e-6)
    return _sigmoid(np.log(p / (1 - p)) + shift)


def choose_shrinkage(
    shots: Shots,
    chance: np.ndarray,
    first_half: np.ndarray,
    candidates: Sequence[float] = (2, 5, 10, 25, 50, 100, 200, 400, 800),
) -> dict[Group, float]:
    """
    For each kind of shot, the shrinkage whose skills from `first_half` of
    the shots best predict the rest's makes (log likelihood).
    """
    chosen: dict[Group, float] = {}
    group = shots.group
    for kind in (JUMPER, RIM, FREE_THROW):
        train = first_half & (group == kind)
        test = ~first_half & (group == kind)
        best = None
        for k in candidates:
            skills = shooter_skills(shots.take(train), chance[train], {kind: k})
            p = with_skill(shots.take(test), chance[test], skills)
            made = shots.made[test]
            loglik = float(np.sum(np.where(made, np.log(p), np.log(1 - p))))
            if best is None or loglik > best[1]:
                best = (k, loglik)
        assert best is not None
        chosen[kind] = float(best[0])
    return chosen


def _newton(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta = np.zeros(x.shape[1])
    for _ in range(50):
        p = _sigmoid(x @ beta)
        gradient = x.T @ (y - p) - 1e-6 * beta
        hessian = (x * (p * (1 - p))[:, None]).T @ x + 1e-6 * np.eye(len(beta))
        step = np.linalg.solve(hessian, gradient)
        beta += step
        if np.abs(step).max() < 1e-10:
            break
    return beta


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))
