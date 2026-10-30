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

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.linalg import get_lapack_funcs

from .ratings import Design, Rows

SIGMA = 0.94  # points per possession, the residual sd either league fits to


def gram(design: Design, weight: np.ndarray) -> np.ndarray:
    """X'WX, dense, from the sparse design."""
    x = design.matrix
    return (x.T @ sparse.diags(weight) @ x).toarray()


def precision(normal: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    """X'WX + diag(penalty), with a column nothing touches pinned (as `solve`
    pins it to its prior), so the inverse exists."""
    out = normal + np.diag(penalty)
    free = np.diag(out) > 0
    out[~free, ~free] = 1.0
    return out


def spd_inverse(a: np.ndarray) -> np.ndarray:
    """The inverse of a symmetric positive definite matrix, by its Cholesky
    factor (LAPACK potrf/potri): a third of a general inverse's work."""
    potrf, potri = get_lapack_funcs(("potrf", "potri"), (a,))
    factor, info = potrf(a, lower=True, clean=True, overwrite_a=True)
    if info != 0:
        raise np.linalg.LinAlgError(f"not positive definite (potrf {info})")
    inverse, info = potri(factor, lower=True, overwrite_c=True)
    if info != 0:
        raise np.linalg.LinAlgError(f"singular (potri {info})")
    # potri fills the lower triangle only
    return np.tril(inverse) + np.tril(inverse, -1).T


def variances(
    design: Design, weight: np.ndarray, penalty: np.ndarray, sigma: float = SIGMA
) -> np.ndarray:
    """The posterior variance of every column -- `covariance`'s diagonal --
    without the rest of it: the column norms of the inverse Cholesky factor,
    two thirds of the work."""
    a = precision(gram(design, weight), penalty)
    potrf, trtri = get_lapack_funcs(("potrf", "trtri"), (a,))
    factor, info = potrf(a, lower=True, clean=True, overwrite_a=True)
    if info != 0:
        raise np.linalg.LinAlgError(f"not positive definite (potrf {info})")
    inverse, info = trtri(factor, lower=True, overwrite_c=True)
    if info != 0:
        raise np.linalg.LinAlgError(f"singular (trtri {info})")
    # A^-1 = L^-T L^-1, so its diagonal is each column of L^-1 squared, summed
    return sigma**2 * np.einsum("ij,ij->j", inverse, inverse)


def factor(design: Design, weight: np.ndarray, penalty: np.ndarray) -> np.ndarray:
    """The lower Cholesky factor L of X'WX + diag(penalty): a linear mix c
    of the columns has posterior variance sigma^2 |L^-1 c|^2, a triangular
    solve per mix rather than the whole inverse."""
    (potrf,) = get_lapack_funcs(("potrf",), (np.empty(0),))
    out, info = potrf(precision(gram(design, weight), penalty), lower=True, clean=True)
    if info != 0:
        raise np.linalg.LinAlgError(f"not positive definite (potrf {info})")
    return out


def covariance(
    design: Design, weight: np.ndarray, penalty: np.ndarray, sigma: float = SIGMA
) -> np.ndarray:
    """The posterior covariance of every column, per possession squared."""
    return sigma**2 * spd_inverse(precision(gram(design, weight), penalty))


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
    # every player entry of the design, with the team it's on the floor for
    # (teams coded once per row: string arrays are slow to index and sort)
    names = sorted({t for t in (*rows.offense_team, *rows.defense_team) if t})
    code = {t: i for i, t in enumerate(names)}
    offense = np.array([code.get(t, -1) for t in rows.offense_team])
    defense = np.array([code.get(t, -1) for t in rows.defense_team])
    entry = design.col < 2 * len(design.players)
    row, col = design.row[entry], design.col[entry]
    player, defending = col // 2, col % 2 == 1
    team = np.where(defending, defense[row], offense[row])
    on = team >= 0
    player, team_code = player[on], team[on]
    weight = (np.abs(design.val[entry]) * rows.possessions[row])[on]
    # each (player, team)'s possessions, then each player's most -- ties to
    # the later team id, as max over (possessions, team) has it
    pair = player * len(names) + team_code
    pairs, at = np.unique(pair, return_inverse=True)
    total = np.bincount(at, weights=weight)
    by_player, by_team = pairs // len(names), pairs % len(names)
    order = np.lexsort((by_team, total, by_player))
    last = np.r_[by_player[order][1:] != by_player[order][:-1], True]
    team_at = {t: i for i, t in enumerate(design.teams)}
    out = np.full(len(design.players), -1)
    for p, t in zip(by_player[order][last], by_team[order][last]):
        j = team_at.get(names[t])
        if j is not None:
            out[p] = design.team_column(j, "offense")
    return out


@dataclass(frozen=True)
class Reported:
    """Each player's reported offense and defense as combinations of the
    design's columns: his own, plus a fifth of his team's."""

    offense: np.ndarray  # player's offense column
    defense: np.ndarray
    team: np.ndarray  # team offense column (defense is the next), or -1

    @classmethod
    def build(cls, rows: Rows, design: Design) -> Reported:
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
