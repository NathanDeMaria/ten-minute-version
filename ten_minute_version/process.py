"""
How much of a game's margin its box score says won't carry over: for
Cassandra's Glicko, which takes the margin as evidence of strength and
already takes part of the shooting luck out of it (`luck`, `flow`).

A margin mixes what a team will do again with what it won't. Shot selection
and the possession battle repeat from game to game; shooting percentages,
the defense's above all, mostly don't (odd-vs-even-game reliability of a
team-season's average, mens 2022-25: 3PA rate 0.88, offensive rebound rate
0.76, blocks 0.74, assists per make 0.72, eFG 0.58, opponents' eFG 0.47).

Each side of a game gets ten rates (`side_stats`), each in standard
deviations of its league's team-games, and a score: the coefficients times
those, fitted on how many points a game's rates put a team over or under
the team ratings' expectation in its next five games, with the game's own
margin over expectation beside them (`fit_process`). The reading is the home
side's score less the away side's, halved, in points of next-game
expectation, scaled to points of margin by how far a point of margin moves
that expectation (`MARGIN_MOVES`): the points of the home margin the box
score says won't repeat. The model reading it takes a share of that off the
margin, as it does luck.

The fit rolls: a season is read with coefficients fitted on every season
before it and none of its own (`fit_seasons`, shipped in `releases/` by
`build.process_fits`), so a replay of an old season sees what was knowable
then. The coefficients drift as the luck file's other readings fill in --
the shooting discount was about nothing in 2014-17 and is larger since --
and an expanding window follows that without a window length to choose
(two, three and four seasons did no better). A season reads nothing until
the fit has `MIN_ROWS` team-games under it.

Only box scores whose players' lines add up to the final score count, for
the fit and the readings: a tenth of mens games in 2018-22 and most womens
games before 2019 are missing players, and their rates are from part of a
game (unfiltered, they put the shooting discount at four times its size).

Measured in Cassandra's glicko_players replay at its 2026-11-16 fit, each
season read with its own rolling fit, Brier against none per season:

    share      0.05      0.10      0.20
    mens       -3.6e-5   -6.1e-5   -7.9e-5   2016-25, 8/10 seasons better
    womens     -5.0e-5   -2.7e-5   +21.9e-5  2018-25, 7/8 at 0.05
"""

import json
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import numpy as np

from .box_scores import BoxLine
from .context import GameContext

# The ten rates, each from one side's view of a game: its shooting and its
# opponents', offensive rebounding, getting to the line and keeping the
# opponent off it, ball movement, rim protection, shot selection, and the
# turnovers on each side.
STATS = (
    "efg",
    "opp_efg",
    "orb",
    "ftr",
    "opp_ftr",
    "ast",
    "blk",
    "tpar",
    "tov",
    "opp_tov",
)

# Box columns summed to a side's totals (`box_scores.BoxLine.stats`).
_TOTALS = (
    "points",
    "fgm",
    "fga",
    "tpm",
    "tpa",
    "fta",
    "oreb",
    "dreb",
    "ast",
    "blk",
    "tov",
)

# Points of next-game expectation one point of margin moves, in Cassandra's
# mens glicko_players replay at its 2026-11-16 fit: the Glicko rating's
# change over a game per point of surprise (1.17 rating points) at that
# fit's 2.49 points per 100 rating points.
MARGIN_MOVES = 0.029

# Team-games a fit needs under it before a season is read with it.
MIN_ROWS = 15_000

# Games ahead a game's rates are scored against.
AHEAD = 5

_RELEASES = Path(__file__).parent / "releases"


@dataclass(frozen=True)
class ProcessFit:
    """The coefficients a season is read with, fitted on the seasons before."""

    season: int
    # standard deviation of each rate over the training team-games
    sd: Mapping[str, float]
    # points over expectation per game, in a team's next `AHEAD`, per sd
    coef: Mapping[str, float]
    # team-games it was fitted on
    rows: int

    def score(self, rates: Mapping[str, float]) -> float:
        return sum(self.coef[s] * rates[s] / self.sd[s] for s in STATS)


def release_path(league: str) -> Path:
    return _RELEASES / f"process-{league}.json"


def load_fits(league: str) -> dict[int, ProcessFit]:
    path = release_path(league)
    if not path.exists():
        return {}
    return {
        int(season): ProcessFit(int(season), fit["sd"], fit["coef"], fit["rows"])
        for season, fit in json.loads(path.read_text()).items()
    }


def write_fits(league: str, fits: Mapping[int, ProcessFit]) -> Path:
    path = release_path(league)
    path.write_text(
        json.dumps(
            {
                str(s): {"sd": f.sd, "coef": f.coef, "rows": f.rows}
                for s, f in sorted(fits.items())
            },
            indent=1,
        )
        + "\n"
    )
    return path


def fit_for(fits: Mapping[int, ProcessFit], season: int) -> ProcessFit | None:
    """The season's own fit; for a season not fitted yet, the latest one
    before it -- every season it was fitted on is still before this one."""
    known = [s for s in fits if s <= season]
    return fits[max(known)] if known else None


def team_totals(lines: Iterable[BoxLine]) -> dict[tuple[str, str], dict[str, float]]:
    """Each (game, team)'s box totals, summed over its players' lines."""
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: dict.fromkeys(_TOTALS, 0.0)
    )
    for line in lines:
        team = totals[line.game_id, line.team_id]
        for column in _TOTALS:
            team[column] += line.stats.get(column, 0.0)
    return dict(totals)


def _possessions(t: Mapping[str, float]) -> float:
    return t["fga"] - t["oreb"] + t["tov"] + 0.475 * t["fta"]


def side_stats(
    own: Mapping[str, float], opp: Mapping[str, float]
) -> dict[str, float] | None:
    """`STATS` for one side of a game; None when a rate has nothing under it
    (a side with no shots, say), as a game with a broken box score does."""
    parts = {
        "efg": (own["fgm"] + 0.5 * own["tpm"], own["fga"]),
        "opp_efg": (opp["fgm"] + 0.5 * opp["tpm"], opp["fga"]),
        "orb": (own["oreb"], own["oreb"] + opp["dreb"]),
        "ftr": (own["fta"], own["fga"]),
        "opp_ftr": (opp["fta"], opp["fga"]),
        "ast": (own["ast"], own["fgm"]),
        "blk": (own["blk"], opp["fga"] - opp["tpa"]),
        "tpar": (own["tpa"], own["fga"]),
        "tov": (own["tov"], _possessions(own)),
        "opp_tov": (opp["tov"], _possessions(opp)),
    }
    if any(under <= 0 for _, under in parts.values()):
        return None
    return {s: over / under for s, (over, under) in parts.items()}


def complete_games(
    lines: Iterable[BoxLine],
    contexts: Mapping[str, GameContext],
    finals: Mapping[str, tuple[int, int]],
) -> dict[str, tuple[dict[str, float], dict[str, float]]]:
    """Each game's (home, away) rates, for the games whose players' lines add
    up to the final score (`finals`, home and away) and whose rates all
    have something under them."""
    totals = team_totals(lines)
    found = {}
    for game_id, context in contexts.items():
        home = totals.get((game_id, context.home.team_id))
        away = totals.get((game_id, context.away.team_id))
        final = finals.get(game_id)
        if home is None or away is None or final is None:
            continue
        if (home["points"], away["points"]) != (float(final[0]), float(final[1])):
            continue
        h, a = side_stats(home, away), side_stats(away, home)
        if h is not None and a is not None:
            found[game_id] = (h, a)
    return found


def box_process(
    lines: Iterable[BoxLine],
    contexts: Mapping[str, GameContext],
    finals: Mapping[str, tuple[int, int]],
    fit: ProcessFit | None,
) -> dict[str, float]:
    """Each complete game's reading (`complete_games`): points of the home
    margin its box score says won't carry over. None without a fit."""
    if fit is None:
        return {}
    return {
        game_id: (fit.score(a) - fit.score(h)) / 2 / MARGIN_MOVES
        for game_id, (h, a) in complete_games(lines, contexts, finals).items()
    }


@dataclass(frozen=True)
class Side:
    """One team's view of a complete game, for the fit."""

    season: int
    day: date
    game_id: str
    team_id: str
    # its margin less what the team ratings expected of it
    over: float
    rates: Mapping[str, float]


@dataclass(frozen=True)
class Rating:
    """What the team ratings say of a team for games from `as_of` on."""

    as_of: date
    net: float
    pace: float
    home_edge: float


def expected_margin(mine: Rating, theirs: Rating, home: bool, neutral: bool) -> float:
    """The team ratings' margin for a side: the net gap at the pair's pace,
    plus the home edge at a home site."""
    edge = 0.0 if neutral else (mine.home_edge if home else -mine.home_edge)
    return (mine.net - theirs.net) * (mine.pace + theirs.pace) / 200 + edge


def latest(ratings: list[Rating], day: date) -> Rating | None:
    """The row in force for a game on `day`: the last one dated on or before
    the day before it (`ratings` sorted by `as_of`)."""
    found = None
    for row in ratings:
        if row.as_of <= day - timedelta(days=1):
            found = row
        else:
            break
    return found


def season_sides(
    season: int,
    lines: Iterable[BoxLine],
    contexts: Mapping[str, GameContext],
    finals: Mapping[str, tuple[int, int]],
    day: Mapping[str, date],
    ratings: Mapping[str, list[Rating]],
) -> list[Side]:
    """Both sides of each complete game the team ratings cover."""
    sides = []
    for game_id, (h, a) in complete_games(lines, contexts, finals).items():
        context, final = contexts[game_id], finals[game_id]
        if game_id not in day:
            continue
        rh = latest(ratings.get(context.home.team_id, []), day[game_id])
        ra = latest(ratings.get(context.away.team_id, []), day[game_id])
        if rh is None or ra is None:
            continue
        margin = final[0] - final[1]
        expected = expected_margin(rh, ra, True, context.neutral_site)
        sides.append(
            Side(
                season,
                day[game_id],
                game_id,
                context.home.team_id,
                margin - expected,
                h,
            )
        )
        sides.append(
            Side(
                season,
                day[game_id],
                game_id,
                context.away.team_id,
                expected - margin,
                a,
            )
        )
    return sides


def with_ahead(sides: list[Side]) -> list[tuple[Side, float]]:
    """Each side with its team's mean `over` in its next `AHEAD` games of the
    season; a side without that many after it is left out."""
    by_team: dict[tuple[int, str], list[Side]] = defaultdict(list)
    for side in sides:
        by_team[side.season, side.team_id].append(side)
    found = []
    for games in by_team.values():
        games.sort(key=lambda s: (s.day, s.game_id))
        for i, side in enumerate(games[: len(games) - AHEAD]):
            found.append(
                (side, float(np.mean([g.over for g in games[i + 1 : i + 1 + AHEAD]])))
            )
    return found


def fit_process(season: int, rows: list[tuple[Side, float]]) -> ProcessFit:
    """Least squares of the next games' mean `over` on the game's own `over`
    and its rates in sds, each rate less its season's mean."""
    seasons = sorted({side.season for side, _ in rows})
    x = np.array([[side.rates[s] for s in STATS] for side, _ in rows])
    which = np.array([seasons.index(side.season) for side, _ in rows])
    means = np.array([x[which == i].mean(axis=0) for i in range(len(seasons))])
    x = x - means[which]
    sd = x.std(axis=0)
    sd[sd == 0] = 1.0  # a rate that never varies says nothing
    design = np.column_stack(
        [np.ones(len(rows)), [side.over for side, _ in rows], x / sd]
    )
    b, *_ = np.linalg.lstsq(design, np.array([ahead for _, ahead in rows]), rcond=None)
    return ProcessFit(
        season,
        {s: float(v) for s, v in zip(STATS, sd)},
        {s: float(v) for s, v in zip(STATS, b[2:])},
        len(rows),
    )


def fit_seasons(
    rows: list[tuple[Side, float]], seasons: Iterable[int]
) -> dict[int, ProcessFit]:
    """Each of `seasons`' fit on every season before it, where that's at
    least `MIN_ROWS` team-games."""
    fits = {}
    for season in seasons:
        before = [r for r in rows if r[0].season < season]
        if len(before) >= MIN_ROWS:
            fits[season] = fit_process(season, before)
    return fits
