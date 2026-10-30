"""
How well last season's ratings predict this season's first weeks: the test
of whether the estimated lineups (`estimated`) are worth anything.

Within a season, player ratings add little to team ratings looking forward
(`priors`). Across seasons is where they should: a team's rating carries
over badly when half its minutes leave, and its players' don't. So each of
this season's early games is predicted from last season -- its real
lineups, with each player on the floor at last season's rating, and a
player last season didn't rate counted as a newcomer -- in a few ways:

- **team carryover**: last season's team ratings, nothing about players;
- **players**: last season's player ratings, fitted from its real lineups
  alone (only from 2025-02-15 in 2024-25), or with its estimated
  possessions as well (every game);
- **both**: team carryover and players together.

How much of each to use is fitted on the first two weeks' games (a
least-squares weight per term), and scored on the weeks after, by the
error in their home margins.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from .ratings import Rows

# (offense, defense) per possession
Ratings = Mapping[str, tuple[float, float]]


@dataclass(frozen=True)
class GameTerms:
    """Per game, home minus away, possessions times each term."""

    games: np.ndarray
    actual: np.ndarray
    terms: Mapping[str, np.ndarray]


def game_terms(
    rows: Rows,
    games: Sequence[str],
    teams: Ratings,
    players: Mapping[str, Ratings],
) -> GameTerms:
    """
    Each game's home margin, and what each of last season's ratings says it
    should be: `home` (the home edge's share), `balance` (possessions home
    less away, for the baseline), `team`, and for each set of `players`
    ratings its rated players' sum and its newcomers' count.
    """
    at = {g: i for i, g in enumerate(games)}
    n_games = len(games)
    terms = {name: np.zeros(n_games) for name in ("home", "balance", "team")}
    for name in players:
        terms[name] = np.zeros(n_games)
        terms[f"{name}, newcomers"] = np.zeros(n_games)
    actual = np.zeros(n_games)
    for i in range(len(rows)):
        g = at.get(rows.game_id[i])
        if g is None:
            continue
        weight = (1.0 if rows.offense_home[i] else -1.0) * rows.possessions[i]
        actual[g] += (1.0 if rows.offense_home[i] else -1.0) * rows.points[i]
        terms["home"][g] += weight * rows.home[i]
        terms["balance"][g] += weight
        offense_team, defense_team = (
            teams.get(rows.offense_team[i]),
            teams.get(rows.defense_team[i]),
        )
        terms["team"][g] += weight * (
            (offense_team[0] if offense_team else 0.0)
            - (defense_team[1] if defense_team else 0.0)
        )
        for name, rated in players.items():
            known = 0.0
            new = 0.0
            for p, w in zip(rows.offense[i], rows.offense_weights[i]):
                if p in rated:
                    known += w * rated[p][0]
                else:
                    new += w
            for p, w in zip(rows.defense[i], rows.defense_weights[i]):
                if p in rated:
                    known -= w * rated[p][1]
                else:
                    new -= w
            terms[name][g] += weight * known
            terms[f"{name}, newcomers"][g] += weight * new
    return GameTerms(np.asarray(games, object), actual, terms)


@dataclass(frozen=True)
class Result:
    name: str
    rmse: float
    # positive when worse than team carryover, on the same games
    t_vs_team: float
    weights: Mapping[str, float]
    errors: np.ndarray


def compare(
    terms: GameTerms,
    day: Mapping[str, date],
    tune_until: date,
    variants: Mapping[str, Sequence[str]],
) -> list[Result]:
    """Each variant's weights fitted on the games through `tune_until`,
    scored on the rest."""
    tune = np.array([day[g] <= tune_until for g in terms.games])
    fitted: dict[str, tuple[np.ndarray, dict[str, float]]] = {}
    for name, columns in variants.items():
        x = np.stack([terms.terms[c] for c in columns], axis=1)
        beta = np.linalg.lstsq(x[tune], terms.actual[tune], rcond=None)[0]
        fitted[name] = (
            x[~tune] @ beta - terms.actual[~tune],
            dict(zip(columns, map(float, beta))),
        )
    reference = fitted["team carryover"][0]
    results = []
    for name, (errors, weights) in fitted.items():
        gap = errors**2 - reference**2
        sd = gap.std(ddof=1)
        t = float(gap.mean() / (sd / np.sqrt(len(gap)))) if sd > 0 else 0.0
        results.append(
            Result(name, float(np.sqrt(np.mean(errors**2))), t, weights, errors)
        )
    return sorted(results, key=lambda r: r.rmse)


def variants(player_sets: Sequence[str]) -> dict[str, list[str]]:
    out = {
        "home edge only": ["home", "balance"],
        "team carryover": ["home", "balance", "team"],
    }
    for name in player_sets:
        out[f"players, {name}"] = ["home", "balance", name, f"{name}, newcomers"]
        out[f"team + players, {name}"] = [
            "home",
            "balance",
            "team",
            name,
            f"{name}, newcomers",
        ]
    return out
