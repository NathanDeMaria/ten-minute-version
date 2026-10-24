"""
Ratings that carry from season to season, each with how sure it is.

`season_ratings` rates each season alone: every player starts at his box
score with the same prior width (penalty 1000), whether he's a fifth-year
starter or a freshman. Here each player's prior is what's known of him
going in, mean and variance (`chained_priors`):

- **a returner** -- rated within the last `gap` seasons -- starts at his
  last season's posterior (`posterior.Reported`), plus the average
  development of players with as many seasons behind them
  (`development`), his variance widened by `drift_sd` per season since;
- **a newcomer** starts at what his recruiting rank says
  (`newcomers.NewcomerPrior`), mean and width;
- either is then combined, by precision, with the box-score prior from the
  games so far (`priors.BoxModel`, at the width penalty 1000 gives it), if
  he has a box score yet.

At a season's end, every game fitted, each player's posterior (the dense
covariance, `posterior.covariance`) is carried to the next (`carry`).

Tested week by week (`build.py chain`): each week's games predicted from a
fit on the games before it, against the same with every season alone. In
2014-2025 the chain is better at every week of the first ten (mens margin
rmse 13.17 -> 12.61, womens 14.57 -> 13.78, weeks 1-10), and the
per-player variances themselves -- against every prior at one width
(`flat_sd`) -- earn most in the first two weeks, and the more of a game's
minutes are newcomers', the more.

One season is a single level in this: a player who improves through his
freshman season is rated at his average, and a narrow prior holds him near
where he started. Tighter priors (less drift, narrower newcomers) help the
first weeks and hurt the next preseason, which reads stale end-of-season
ratings -- the sign that within-season change is missing.
"""

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

import numpy as np
import pyarrow as pa

from .posterior import SIGMA, Reported
from .ratings import Design, penalties

# (mean, variance) per possession of a newcomer's offense and defense
NewcomerSides = tuple[tuple[float, float], tuple[float, float]]
# a newcomer's prior by athlete id and season
Newcomer = Callable[[str, int], NewcomerSides]


@dataclass(frozen=True)
class ChainSettings:
    """Per side, per 100 possessions where they're widths."""

    box_sd: float = 2.97  # the box-score prior: what penalty 1000 means at SIGMA
    drift_sd: float = 1.5  # added per season since a player was last rated
    gap: int = 2  # seasons a player can miss and still carry his rating
    team_penalty: float = 30.0
    development: bool = True
    flat_sd: float = 0.0  # >0: every carried and newcomer prior at this width instead
    sigma: float = SIGMA


@dataclass(frozen=True)
class PlayerState:
    """A player's posterior at the end of the last season he played, per
    possession."""

    season: int
    offense: float
    defense: float
    var_offense: float
    var_defense: float
    seasons: int  # seasons he's had possessions in


def development(
    tables: Mapping[int, Iterable[Mapping]], min_possessions: int = 300
) -> dict[int, tuple[float, float]]:
    """
    How much a returner's (offense, defense) changes from one season to the
    next, per possession, by how many seasons he'd been rated before (1, 2,
    3 or more): from `season_ratings.parquet` rows by season, players with
    `min_possessions` both seasons.
    """
    by_season = {s: {r["athlete_id"]: r for r in rows} for s, rows in tables.items()}
    seasons = sorted(by_season)
    changes: dict[int, list[tuple[float, float]]] = {1: [], 2: [], 3: []}
    for s in seasons:
        if s - 1 not in by_season:
            continue
        for p, r in by_season[s].items():
            q = by_season[s - 1].get(p)
            if q is None:
                continue
            if (
                min(r["offense_possessions"], q["offense_possessions"])
                < min_possessions
            ):
                continue
            n = sum(1 for t in seasons if t < s and p in by_season[t])
            changes[min(n, 3)].append(
                (
                    (r["offense"] - q["offense"]) / 100,
                    (r["defense"] - q["defense"]) / 100,
                )
            )
    return {
        n: (
            float(np.mean([c[0] for c in found])),
            float(np.mean([c[1] for c in found])),
        )
        if found
        else (0.0, 0.0)
        for n, found in changes.items()
    }


def baseline_priors(
    design: Design, box: np.ndarray, settings: ChainSettings = ChainSettings()
) -> tuple[np.ndarray, np.ndarray]:
    """`season_ratings`' priors: every player at his box score, penalty 1000."""
    return box.copy(), penalties(design, player=1000.0, team=settings.team_penalty)


def chained_priors(
    design: Design,
    season: int,
    states: Mapping[str, PlayerState],
    newcomer: Newcomer,
    box: np.ndarray,
    has_box: np.ndarray,
    settings: ChainSettings = ChainSettings(),
    grown: Mapping[int, tuple[float, float]] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """(prior mean, penalty) for every column of `design` in `season`: see
    the module's docstring. `box` is the box-score prior from the games so
    far and `has_box` which players have one; `grown` is `development`."""
    players = len(design.players)
    pen = penalties(design, player=1000.0, team=settings.team_penalty)
    mean = np.zeros(design.columns)
    var = np.zeros(2 * players)
    box_var = (settings.box_sd / 100) ** 2
    drift = (settings.drift_sd / 100) ** 2
    for i, p in enumerate(design.players):
        o, d = 2 * i, 2 * i + 1
        state = states.get(p)
        if state is not None and season - state.season <= settings.gap:
            gap = season - state.season
            step = (grown or {}).get(min(state.seasons, 3), (0.0, 0.0))
            if not settings.development:
                step = (0.0, 0.0)
            mean[o], mean[d] = state.offense + step[0], state.defense + step[1]
            var[o] = state.var_offense + drift * gap
            var[d] = state.var_defense + drift * gap
        else:
            (mean[o], var[o]), (mean[d], var[d]) = newcomer(p, season)
        if settings.flat_sd > 0:
            var[o] = var[d] = (settings.flat_sd / 100) ** 2
        if has_box[i]:
            for c in (o, d):
                weight = 1 / var[c] + 1 / box_var
                mean[c] = (mean[c] / var[c] + box[c] / box_var) / weight
                var[c] = 1 / weight
    pen[: 2 * players] = settings.sigma**2 / var
    return mean, pen


def carry(
    design: Design,
    reported: Reported,
    b: np.ndarray,
    cov: np.ndarray,
    possessions: np.ndarray,
    season: int,
    states: Mapping[str, PlayerState],
) -> dict[str, PlayerState]:
    """`states` with every player who had possessions in `season` (by
    column, `Design.column_squares`) at his posterior from it."""
    out = dict(states)
    offense, defense = reported.means(b)
    var_offense, var_defense, _ = reported.variances(cov)
    for i, p in enumerate(design.players):
        if possessions[2 * i] <= 0 and possessions[2 * i + 1] <= 0:
            continue
        before = states.get(p)
        out[p] = PlayerState(
            season,
            float(offense[i]),
            float(defense[i]),
            float(var_offense[i]),
            float(var_defense[i]),
            (before.seasons if before else 0) + 1,
        )
    return out


CHAINED_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("athlete_id", pa.string(), nullable=False),
        pa.field("athlete_name", pa.string()),
        pa.field("team_id", pa.string()),
        pa.field("possessions", pa.int32(), nullable=False),
        # per 100 possessions, as `ratings.RATINGS_SCHEMA` reports them
        # (his column plus a fifth of his team's), and how sure each is
        pa.field("offense", pa.float32(), nullable=False),
        pa.field("defense", pa.float32(), nullable=False),
        pa.field("total", pa.float32(), nullable=False),
        pa.field("offense_sd", pa.float32(), nullable=False),
        pa.field("defense_sd", pa.float32(), nullable=False),
        pa.field("total_sd", pa.float32(), nullable=False),
    ]
)


def chained_table(
    league: str,
    season: int,
    design: Design,
    reported: Reported,
    b: np.ndarray,
    cov: np.ndarray,
    possessions: np.ndarray,
    names: Mapping[str, str | None],
) -> pa.Table:
    """Every player with possessions in the season, at his posterior: mean
    and sd per 100."""
    offense, defense = reported.means(b)
    var_offense, var_defense, both = reported.variances(cov)
    first_team = 2 * len(design.players)
    out = []
    for i, p in enumerate(design.players):
        if possessions[2 * i] <= 0 and possessions[2 * i + 1] <= 0:
            continue
        team = reported.team[i]
        out.append(
            {
                "league": league,
                "season": season,
                "athlete_id": p,
                "athlete_name": names.get(p),
                "team_id": design.teams[(team - first_team) // 2]
                if team >= 0
                else None,
                "possessions": int(possessions[2 * i]),
                "offense": 100 * offense[i],
                "defense": 100 * defense[i],
                "total": 100 * (offense[i] + defense[i]),
                "offense_sd": 100 * np.sqrt(var_offense[i]),
                "defense_sd": 100 * np.sqrt(var_defense[i]),
                "total_sd": 100
                * np.sqrt(var_offense[i] + var_defense[i] + 2 * both[i]),
            }
        )
    return pa.Table.from_pylist(out, schema=CHAINED_SCHEMA)
