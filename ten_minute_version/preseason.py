"""
How good a team is before it plays: its roster, how much each player is
likely to play, and what each is worth.

Everything else here rates players from games already played. A preseason
rating has only what's known before the first: who's on the roster, last
season's ratings (`season_ratings`), recruiting ranks (`recruiting`), and
how much each player played last season. So a team's projected offense and
defense are its players' ratings, each weighted by a projected share of its
minutes (a team's shares add up to five, as a lineup's players do):

- a player last season rated is at that rating, wherever he played it;
- a newcomer is at what `recruiting.RecruitModel` says of his rank, or of
  being unranked;
- a team from outside Division I, which plays as one generic team, is at
  last season's generic rating.

`MinutesModel` projects each player's share from what's known before the
season: his share last season on the same team, or on another (a
transfer), whether and how well he was rated, his recruiting rank, whether
he's new, and how many seasons he's played. It's a least-squares fit,
floored at zero and scaled so each team's shares add up to five.

**The roster** is everyone who plays for the team in the season --
published before it, as rosters are, though it also leaves out players who
never play (hurt, redshirting) and takes in ones who join late. A stricter
version counts only players who've played by the end of the tuning window
(the season's first two weeks).
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from .divisions import is_generic
from .recruiting import Recruit, RecruitModel
from .recruiting import features as recruit_features

# (offense, defense) per possession
Ratings = Mapping[str, tuple[float, float]]
FEATURES = (
    "same_team",
    "other_team",
    "rated",
    "rating",
    "ranked",
    "rank_score",
    "newcomer",
    "one_season",
    "two_seasons",
    "three_or_more",
)


@dataclass(frozen=True)
class Spell:
    """A player's season on one team (a `minutes.parquet` row)."""

    athlete_id: str
    team_id: str
    share: float
    first_date: date | None


@dataclass(frozen=True)
class History:
    """What's known of every player before a season: last season's minutes
    and ratings, recruits, and every season he's played in before."""

    last_spells: Mapping[str, Sequence[Spell]]
    last_ratings: Ratings
    recruits: Mapping[str, Recruit]
    played: Mapping[str, set[int]]


def features(
    athlete_id: str, team_id: str, season: int, history: History
) -> list[float]:
    last = history.last_spells.get(athlete_id, ())
    same = sum(s.share for s in last if s.team_id == team_id)
    other = max((s.share for s in last if s.team_id != team_id), default=0.0)
    rating = history.last_ratings.get(athlete_id)
    ranked, score = recruit_features(history.recruits.get(athlete_id), season)
    seasons = len([s for s in history.played.get(athlete_id, ()) if s < season])
    return [
        same,
        other,
        float(rating is not None),
        100 * sum(rating) if rating is not None else 0.0,
        ranked,
        score,
        float(not last),
        float(seasons == 1),
        float(seasons == 2),
        float(seasons >= 3),
    ]


@dataclass(frozen=True)
class MinutesModel:
    """A player's projected share of his team's minutes: intercept, then one
    coefficient per `FEATURES`."""

    coefficients: tuple[float, ...]

    @classmethod
    def learn(cls, x: np.ndarray, shares: np.ndarray) -> "MinutesModel":
        design = np.hstack([np.ones((len(x), 1)), x])
        return cls(tuple(map(float, np.linalg.lstsq(design, shares, rcond=None)[0])))

    def project(
        self, roster: Sequence[tuple[str, str]], x: np.ndarray
    ) -> dict[tuple[str, str], float]:
        """Each (player, team)'s share, floored at zero and scaled so a
        team's add up to five."""
        raw = np.clip(
            np.hstack([np.ones((len(x), 1)), x]) @ np.array(self.coefficients), 0, None
        )
        totals: dict[str, float] = {}
        for (_, team), value in zip(roster, raw):
            totals[team] = totals.get(team, 0.0) + value
        return {
            (player, team): 5 * value / totals[team]
            for (player, team), value in zip(roster, raw)
            if totals[team] > 0
        }


def project_teams(
    shares: Mapping[tuple[str, str], float],
    season: int,
    history: History,
    recruit_model: RecruitModel,
) -> dict[str, tuple[float, float]]:
    """Each team's projected (offense, defense) per possession: its players'
    ratings weighted by their shares."""
    teams: dict[str, list[float]] = {}
    for (player, team), share in shares.items():
        rating = history.last_ratings.get(player)
        if rating is None:
            rating = recruit_model.rating(history.recruits.get(player), season)
        at = teams.setdefault(team, [0.0, 0.0])
        at[0] += share * rating[0]
        at[1] += share * rating[1]
    return {team: (o, d) for team, (o, d) in teams.items()}


def roster(spells: Iterable[Spell], by: date | None = None) -> list[tuple[str, str]]:
    """(player, team) for everyone who played for a team -- by `by`, if given."""
    return sorted(
        (s.athlete_id, s.team_id)
        for s in spells
        if by is None or (s.first_date is not None and s.first_date <= by)
    )


def team_term(
    rows,
    games: Sequence[str],
    ratings: Ratings,
    generic: tuple[float, float],
) -> np.ndarray:
    """
    Each game's home margin as `ratings` of its teams say it should be (home
    less away, possessions times offense less the other side's defense), in
    `carryover.game_terms`' form; a generic side (`opponents`) is at
    `generic`, and a team `ratings` doesn't have at 0.
    """
    at = {g: i for i, g in enumerate(games)}
    out = np.zeros(len(games))

    def side(team: str, five: Sequence[str], k: int) -> float:
        if len(five) == 1 and is_generic(five[0]):
            return generic[k]
        rated = ratings.get(team)
        return rated[k] if rated is not None else 0.0

    for i in range(len(rows)):
        g = at.get(rows.game_id[i])
        if g is None:
            continue
        weight = (1.0 if rows.offense_home[i] else -1.0) * rows.possessions[i]
        out[g] += weight * (
            side(rows.offense_team[i], rows.offense[i], 0)
            - side(rows.defense_team[i], rows.defense[i], 1)
        )
    return out
