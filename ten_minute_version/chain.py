"""
Ratings that carry from season to season, each with how sure it is.

`season_ratings` rates each season alone: every player starts at his box
score with the same prior width (penalty 1000), whether he's a fifth-year
starter or a freshman. Here each player's prior is what's known of him
going in, mean and variance (`chained_priors`):

- **a returner** -- rated within the last `gap` seasons -- starts at his
  last season's posterior for his own columns, plus the average
  development of players with as many seasons behind them
  (`development`), his variance widened per season since: by
  `week_drift_sd` in the weekly fits before a season's end, by `drift_sd`
  in the end's fit, the one carried;
- **a newcomer** starts at what his recruiting rank says
  (`newcomers.NewcomerPrior`), mean and width;
- either is then combined, by precision, with the box-score prior from the
  games so far (`priors.BoxModel`, at the width penalty 1000 gives it), if
  he has a box score yet.

At a season's end, every game fitted, each player's posterior (the dense
covariance, `posterior.covariance`) is carried to the next (`carry`) --
and so is what it leaves besides its players (`SeasonEnd`):

- **the home edge**, which before a season's first game nothing else
  sets: at zero, every week-0 prediction had no home edge (mens home
  margins came out 6.7 points under, on average);
- **each team's column**, its next season's prior mean (`program` of it),
  at a tighter width than a season alone gives it (`team_penalty` 300, ~5
  per 100 per side, against 30's ~17). Restarted at zero, a team's
  strength carried only in its returners' fifths, so last season's strong
  programs came out under all season; and nearly free, a team's first
  games swung it too far (weeks 1-2 too extreme, slope +0.24 of the
  error on the prediction). All of it predicts best (0.75 and 1.25 both
  worse).

A returner carries his own columns, not his rating as reported: that has a
fifth of his team's column in it, which the team's carried column counts
again. Carried twice, a strong program's returners came in too strong and
a weak one's too weak: carrying the own columns, mens margin rmse goes
14.36 -> 14.10 at week 0 and 12.055 -> 12.021 over weeks 1-10, womens
16.44 -> 16.01 and 13.148 -> 13.102. (Taking the old team's fifth out for
transfers alone does nothing: it's the players who stay.)

The two drifts differ because the two fits do different jobs. Week 0's
predictions are the carried end-of-season fit's alone, and want it loose
(drift 1.5): at 0.75 there, mens week-0 rmse goes 14.10 -> 14.13 and womens
16.01 -> 16.10. Each later week's fit, a few games on top of that, wants its
returners held tighter: anything that does -- less drift, a larger sigma,
narrower newcomers -- helps weeks 1-10 and, applied to the end's fit too,
hurts week 0. Split, week 0 is untouched, and weekly drift 0.5, 0.75 and
1.0 take mens weeks 1-10 from 12.021 to 12.005, 12.007 and 12.010 (t
-11 to -13), womens 0.75, 1.0 and 1.25 from 13.102 to 13.089, 13.092 and
13.096 (t -8 to -10), most of it in weeks 1-2. Nothing else moved off its
setting, both leagues: box_sd 2.5 and 3.5, gap 1 and 3, team_penalty 150
and 600, sigma 0.8 and 1.1 at every fit, and newcomer variances halved or
doubled are all flat or worse over weeks 1-10.

Tested week by week (`build.py chain`): each week's games predicted from a
fit on the games before it, against the same with every season alone. In
2014-2025 the chain is better at every week of the first ten (mens margin
rmse 13.17 -> 12.10, womens 14.57 -> 13.18, weeks 1-10; 12.60 and 13.74
of that without the teams), and the per-player variances themselves --
against every prior at one width (`flat_sd`) -- earn most in the first two
weeks, and the more of a game's minutes are newcomers', the more.

**Within a season** players change too, but not where it matters: by
class, against returners, ranked newcomers improve no faster (-0.2 +- 0.3
per 100 per 100 days, both sides together) and unranked ones a little
(+0.8 +- 0.2 mens, +1.1 +- 0.2 womens) -- a few tenths of a point a game.
Letting older games count less (an exponential forgetting factor, as a
random-walk Kalman filter would) did nothing at a 120-day memory and hurt
at 60.
"""

from collections.abc import Callable, Collection, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pyarrow as pa

from .divisions import is_generic
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
    drift_sd: float = 1.5  # added per season since a player was last rated, at the end
    week_drift_sd: float = 0.75  # the same, in the weekly fits before it
    gap: int = 2  # seasons a player can miss and still carry his rating
    team_penalty: float = 300.0  # a team's column: ~5 per 100 per side
    development: bool = True
    flat_sd: float = 0.0  # >0: every carried and newcomer prior at this width instead
    program: float = 1.0  # how much of a team's column its next season starts at
    sigma: float = SIGMA
    # >0: a team from outside Division I's column at this width instead of
    # `team_penalty`'s -- such teams vary far more around the generic player
    # than a program does around its last season. Weeks 0-10 of 2014-2025,
    # margin rmse: 0 (team_penalty's ~5) 12.284 mens / 13.481 womens; 10
    # 12.239 / 13.428 (t -5.8, -5.4); 15, 20 and 30 worse. Division I's own
    # games gain too (11.592 -> 11.567 mens): an outside team's results land
    # on its column instead of on the players who played it. Wider helps
    # those a little more and the outside teams' own games a lot less.
    outside_sd: float = 10.0
    # A generic player's (`divisions`) prior width each season, on top of
    # its last season's posterior however long ago (or the newcomer prior,
    # before it's played): its division's level is learned again each
    # season, not held. Held to a player's drift, it stays within ~1 per 100
    # of last season's and the season's shift lands on its teams' columns
    # (outside teams' columns averaged -2 to -14 at the season's end); and a
    # generic gone longer than `gap` came back as a newcomer, an average
    # Division I player -- womens' outside teams are scarce before 2019.
    # Weeks 0-10 of 2014-2025, margin rmse, one generic player: 12.239 mens,
    # 13.428 womens; a generic per division at 20, relearned every season:
    # 12.153 / 13.227 (t -7.5, -8.3); at 20 but lapsing after `gap` as a
    # player does: 12.167 / 13.357. Division I's own games gain too (11.567
    # -> 11.558 mens, 12.675 -> 12.666 womens).
    generic_sd: float = 20.0


@dataclass(frozen=True)
class SeasonEnd:
    """What a season's end leaves the next besides its players, per
    possession: the baseline, the home edge, and each team's (offense,
    defense) columns."""

    baseline: float = 0.0
    home: float = 0.0
    teams: Mapping[str, tuple[float, float]] = field(default_factory=dict)

    @classmethod
    def of(cls, design: Design, b: np.ndarray) -> SeasonEnd:
        return cls(
            float(b[design.baseline]),
            float(b[design.home]),
            {
                t: (
                    float(b[design.team_column(j, "offense")]),
                    float(b[design.team_column(j, "defense")]),
                )
                for j, t in enumerate(design.teams)
            },
        )


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


def baseline_priors(design: Design, box: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """`season_ratings`' priors: every player at his box score, penalty 1000,
    and each team at penalty 30."""
    return box.copy(), penalties(design, player=1000.0, team=30.0)


def chained_priors(
    design: Design,
    season: int,
    states: Mapping[str, PlayerState],
    newcomer: Newcomer,
    box: np.ndarray,
    has_box: np.ndarray,
    settings: ChainSettings = ChainSettings(),
    grown: Mapping[int, tuple[float, float]] | None = None,
    last: SeasonEnd = SeasonEnd(),
    end: bool = False,
    outside: Collection[str] = frozenset(),
) -> tuple[np.ndarray, np.ndarray]:
    """(prior mean, penalty) for every column of `design` in `season`: see
    the module's docstring. `box` is the box-score prior from the games so
    far and `has_box` which players have one; `grown` is `development`;
    `last` the last season's end. Its baseline and home edge are
    unpenalised, so they're their prior only until a game touches them --
    but before the first game, without them, every prediction is at no home
    edge. A team starts at `program` of its last column. `end` is the
    season-end fit, the one carried: its returners drift by `drift_sd`,
    the weekly fits' by `week_drift_sd`. A team in `outside` (from outside
    Division I) has its column at `outside_sd`, if set."""
    players = len(design.players)
    pen = penalties(design, player=1000.0, team=settings.team_penalty)
    mean = np.zeros(design.columns)
    mean[design.baseline], mean[design.home] = last.baseline, last.home
    for j, t in enumerate(design.teams):
        if t in last.teams:
            o, d = last.teams[t]
            mean[design.team_column(j, "offense")] = settings.program * o
            mean[design.team_column(j, "defense")] = settings.program * d
        if settings.outside_sd > 0 and t in outside:
            for side in ("offense", "defense"):
                pen[design.team_column(j, side)] = (
                    settings.sigma**2 / (settings.outside_sd / 100) ** 2
                )
    var = np.zeros(2 * players)
    box_var = (settings.box_sd / 100) ** 2
    drift = ((settings.drift_sd if end else settings.week_drift_sd) / 100) ** 2
    for i, p in enumerate(design.players):
        o, d = 2 * i, 2 * i + 1
        state = states.get(p)
        generic = is_generic(p)
        # a generic player is a division: it carries however long since it
        # last played, and doesn't grow up
        if state is not None and (generic or season - state.season <= settings.gap):
            gap = season - state.season
            step = (grown or {}).get(min(state.seasons, 3), (0.0, 0.0))
            if not settings.development or generic:
                step = (0.0, 0.0)
            mean[o], mean[d] = state.offense + step[0], state.defense + step[1]
            var[o] = state.var_offense + drift * gap
            var[d] = state.var_defense + drift * gap
        else:
            (mean[o], var[o]), (mean[d], var[d]) = newcomer(p, season)
        if generic:
            var[o] += (settings.generic_sd / 100) ** 2
            var[d] += (settings.generic_sd / 100) ** 2
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
    b: np.ndarray,
    square: np.ndarray,
    possessions: np.ndarray,
    season: int,
    states: Mapping[str, PlayerState],
) -> dict[str, PlayerState]:
    """`states` with every player who had possessions in `season` (by
    column, `Design.column_squares`) at his own columns' posterior from it:
    without his team's fifth, which the team's carried column has.
    `square` is every column's posterior variance (`posterior.variances`,
    or the covariance's diagonal)."""
    out = dict(states)
    columns = 2 * len(design.players)
    offense, defense = b[0:columns:2], b[1:columns:2]
    var_offense, var_defense = square[0:columns:2], square[1:columns:2]
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
