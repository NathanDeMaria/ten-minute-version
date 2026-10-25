"""
Today's team ratings, in season: what `build.py live` adds to the file
Cassandra reads (`team_ratings`), every morning before Cassandra publishes.

The backfill (`team_ratings`) refits every season from 2013 to get to one;
a morning has minutes. So what carries into the season -- every player's
posterior at the end of the last one, the teams' columns and home edge
(`chain.SeasonEnd`), development, the minutes model and what it reads of
last season, last season's paces -- is fitted once, before the season, and
frozen in a `LiveKit` (`build.py live_kit`). A morning is then one fit: the
season's games before today, on the kit's priors.

A team that has played is its players at their shares so far, as in the
backfill. One that hasn't is its roster -- ESPN's published one, and
anyone who has played for it -- at the minutes model's projection, rated
at their priors alone: a design with no rows (`prior_design`), its means
the chain's priors.
"""

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date

import numpy as np

from .chain import PlayerState, SeasonEnd
from .preseason import History, MinutesModel
from .ratings import Design


@dataclass(frozen=True)
class LiveKit:
    """What carries into `season`, fitted on every season before it."""

    league: str
    season: int
    states: dict[str, PlayerState]
    last_end: SeasonEnd
    # `chain.development`: (offense, defense) growth by seasons played
    grown: dict[int, tuple[float, float]]
    minutes: MinutesModel
    # the minutes model's view of `season`: last season's minutes and
    # ratings, the recruits, who played when
    history: History
    # last season's possessions per game, by team
    pace: dict[str, float]
    # every team's name by ESPN id, as of last season: so a morning before a
    # team's first game needs only this season's summaries
    names: dict[str, str | None]
    # the teams ESPN filed in Division I last season: whose rosters a
    # morning fetches until they've played
    division_one: list[str]


def prior_design(players: Iterable[str], teams: Iterable[str]) -> Design:
    """A design with columns for `players` and `teams` and no rows: what a
    team that hasn't played is rated in."""
    return Design(
        np.empty(0, int),
        np.empty(0, int),
        np.empty(0),
        0,
        sorted(set(players)),
        sorted(set(teams)),
    )


def merged(existing: list[dict], found: list[dict]) -> list[dict]:
    """The file's rows with `found` in: any row already there for the same
    league, season and day is replaced, so a morning can be rerun."""
    days = {(r["league"], r["season"], r["as_of"]) for r in found}
    kept = [r for r in existing if (r["league"], r["season"], r["as_of"]) not in days]
    return sorted(
        kept + found, key=lambda r: (r["league"], r["season"], r["as_of"], r["team_id"])
    )


def as_of(today: str | None) -> date:
    """The day a morning's rows stand for: `today`, ISO, or the date now."""
    return date.fromisoformat(today) if today else date.today()
