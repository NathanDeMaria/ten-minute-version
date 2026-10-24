"""
How sure a rating is: the ridge fit's posterior covariance.

`ratings.solve` minimises sum(weight * (target - Xb)^2) + sum(penalty * (b -
prior)^2). Read as a model -- each possession's points per possession with
noise of variance sigma^2, each rating drawn from a normal around its prior
with variance sigma^2 / penalty -- that's the posterior mode, and the
posterior covariance is

    sigma^2 (X'WX + diag(penalty))^-1.

With ~11,000 columns that's a dense inverse of a 1 GB matrix: seconds to a
minute, so once a season, not once a fit. sigma^2 is the fit's residual
variance per possession (~0.94^2 for either league).

A player's reported rating (`ratings.ratings_table`) is his column plus a
fifth of his team's, so its variance takes in the team column's and their
covariance (`Reported`).

**Is it right?** `split_half` fits two halves of a season's games apart and
says how far each player's two ratings should differ if the model is right,
to set against how far they do. The halves' targets and box-score priors
must each come from their own games only: the shot model's shooter skills
and the box score both carry the other half's results otherwise, and the
halves agree more than they should.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .ratings import Design, Rows

SIGMA = 0.94  # points per possession, the residual sd either league fits to


def gram(design: Design, weight: np.ndarray, chunk: int = 30_000_000) -> np.ndarray:
    """X'WX, dense: every pair of a row's entries, summed by bincount."""
    columns = design.columns
    starts = np.searchsorted(design.row, np.arange(design.rows))
    counts = np.bincount(design.row, minlength=design.rows)
    out = np.zeros(columns * columns)
    for k in np.unique(counts):
        if k == 0:
            continue
        rows = np.nonzero(counts == k)[0]
        for part in np.array_split(rows, max(1, len(rows) * k * k // chunk + 1)):
            at = starts[part][:, None] + np.arange(k)
            cols, vals = design.col[at], design.val[at]
            flat = (cols[:, :, None] * columns + cols[:, None, :]).ravel()
            products = vals[:, :, None] * vals[:, None, :] * weight[part][:, None, None]
            out += np.bincount(flat, weights=products.ravel(), minlength=columns**2)
    return out.reshape(columns, columns)


def precision(normal: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    """X'WX + diag(penalty), with a column nothing touches pinned (as `solve`
    pins it to its prior), so the inverse exists."""
    out = normal + np.diag(penalty)
    free = np.diag(out) > 0
    out[~free, ~free] = 1.0
    return out


def covariance(
    design: Design, weight: np.ndarray, penalty: np.ndarray, sigma: float = SIGMA
) -> np.ndarray:
    """The posterior covariance of every column, per possession squared."""
    return sigma**2 * np.linalg.inv(precision(gram(design, weight), penalty))


def residual_variance(
    design: Design, target: np.ndarray, weight: np.ndarray, b: np.ndarray
) -> float:
    """sigma^2: the weighted squared residual per row (a row's points per
    possession has variance sigma^2 / possessions)."""
    r = target - design.times(b)
    return float(np.sum(weight * r**2) / max(int(np.sum(weight > 0)), 1))


def main_teams(rows: Rows, design: Design) -> np.ndarray:
    """Each player's team -- the one he was on the floor for most, as
    `ratings_table` picks it -- as its offense column, or -1 if the design
    has no column for it."""
    on: dict[str, dict[str, float]] = {}
    for i in range(len(rows)):
        n = rows.possessions[i]
        for five, weights, team in (
            (rows.offense[i], rows.offense_weights[i], rows.offense_team[i]),
            (rows.defense[i], rows.defense_weights[i], rows.defense_team[i]),
        ):
            if not team:
                continue
            for p, w in zip(five, weights):
                by_team = on.setdefault(p, {})
                by_team[team] = by_team.get(team, 0.0) + w * n
    team_at = {t: i for i, t in enumerate(design.teams)}
    out = np.full(len(design.players), -1)
    for i, p in enumerate(design.players):
        teams = on.get(p)
        if not teams:
            continue
        team = max(teams, key=lambda t: (teams[t], t))
        if team in team_at:
            out[i] = design.team_column(team_at[team], "offense")
    return out


@dataclass(frozen=True)
class Reported:
    """Each player's reported offense and defense as combinations of the
    design's columns: his own, plus a fifth of his team's."""

    offense: np.ndarray  # player's offense column
    defense: np.ndarray
    team: np.ndarray  # team offense column (defense is the next), or -1

    @classmethod
    def build(cls, rows: Rows, design: Design) -> "Reported":
        offense = np.array(
            [design.player_column(i, "offense") for i in range(len(design.players))],
            dtype=int,
        )
        return cls(offense, offense + 1, main_teams(rows, design))

    @property
    def _teamed(self) -> np.ndarray:
        return self.team >= 0

    def side_rows(self, m: np.ndarray, side: str) -> np.ndarray:
        """C @ m for C one side's combinations (or the total's, side
        "total"): m's rows, or entries, gathered per player."""
        out = np.zeros((len(self.offense),) + m.shape[1:])
        t = self._teamed
        for s, own, team in (
            ("offense", self.offense, self.team),
            ("defense", self.defense, self.team + 1),
        ):
            if side not in (s, "total"):
                continue
            out += m[own]
            out[t] += m[team[t]] / 5
        return out

    def means(self, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(offense, defense) per possession."""
        return self.side_rows(b, "offense"), self.side_rows(b, "defense")

    def variances(self, cov: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(offense, defense, their covariance) per possession squared."""
        t = self._teamed

        def pair(a_own, a_team, b_own, b_team) -> np.ndarray:
            out = cov[a_own, b_own].copy()
            out[t] += (
                cov[a_own[t], b_team[t]] / 5
                + cov[a_team[t], b_own[t]] / 5
                + cov[a_team[t], b_team[t]] / 25
            )
            return out

        o, d = (self.offense, self.team), (self.defense, self.team + 1)
        return pair(*o, *o), pair(*d, *d), pair(*o, *d)


@dataclass(frozen=True)
class SplitHalf:
    """Per player: how far apart his two halves' ratings are, and how far the
    model says they should be (variance), per 100 possessions."""

    difference: np.ndarray
    predicted: np.ndarray
    possessions: np.ndarray


def split_half(
    design: Design,
    reported: Reported,
    weight: np.ndarray,
    half_a: np.ndarray,
    targets: Mapping[str, np.ndarray],
    priors: Mapping[str, np.ndarray],
    penalty: np.ndarray,
    side: str = "total",
    sigma: float = SIGMA,
) -> SplitHalf:
    """
    Fit each half (`half_a` rows, the rest) to its own `targets` and
    `priors` ("A", "B"), and compare. With a common prior mean the two fits'
    difference has covariance

        sigma^2 [ D P D + H_A^-1 G_A H_A^-1 + H_B^-1 G_B H_B^-1 ],

    G the halves' X'WX, H = G + P, D = H_B^-1 - H_A^-1: each half's noise,
    and how differently the two shrink toward the prior. Each half's own
    box-score prior adds noise this leaves out, so players who barely
    played -- most of whose rating is the prior -- differ more than it says.
    """
    out: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for name, mask in (("A", half_a), ("B", ~half_a)):
        w = weight * mask
        g = gram(design, w)
        inverse = np.linalg.inv(precision(g, penalty))
        rhs = design.transpose_times(w * targets[name]) + penalty * priors[name]
        out[name] = (g, inverse, inverse @ rhs)
    (g_a, inv_a, b_a), (g_b, inv_b, b_b) = out["A"], out["B"]
    u_a, u_b = reported.side_rows(inv_a, side), reported.side_rows(inv_b, side)
    noise = np.einsum("ij,ij->i", u_a @ g_a, u_a) + np.einsum(
        "ij,ij->i", u_b @ g_b, u_b
    )
    v = u_b - u_a
    shrink = np.einsum("ij,ij->i", v * penalty, v)
    difference = 100 * (reported.side_rows(b_a, side) - reported.side_rows(b_b, side))
    possessions = np.diag(g_a)[reported.offense] + np.diag(g_b)[reported.offense]
    return SplitHalf(difference, 1e4 * sigma**2 * (noise + shrink), possessions)
