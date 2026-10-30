"""
Which side of a game is a team from outside Division I, played as one
generic team rather than as its players.

ESPN's schedule is Division I's, and early each season a D1 team plays a
few teams from outside it -- NAIA, Division II and III. Most of their
players have no ESPN record, so their plays carry no athlete id and their
side of the floor can't be followed.

**By the team's schedule**, where the game has a summary (`context`). The
summaries hold every game a D1 team played, so they say how many of the
season's games each team was in: a D1 team ~30, a team from outside at most
13 (Virginia Lynchburg's men in 2024-25). A side is generic when its team
played under 60% as many of the season's games as the other side's team,
and no conference game. The conference game is for 2020-21, when COVID cut
some D1 seasons to a handful of games (Duke's women played four) that were
still in their conference.

ESPN's own Division I flag can't be the rule: a summary files each team
under the conference it's in now, not then, so teams that have since moved
up -- Lindenwood, Utah Tech, Queens, Mercyhurst -- show as D1 in seasons
they came as Division II opponents; and Texas A&M-Corpus Christi's women
have no conference at all in 2022-23. Over 2013-14 to 2025-26 (149,110
games, 9,680 generic sides) the rule and ESPN's flag disagree on 95 games:
about half where the flag is wrong, a third in 2020-21 (Carver College's
19 games kept as players, Chicago State's and Howard's men played as
generic), and 14 where ESPN calls a game against a team from outside a
conference game. Before 2013-14 ESPN does that often enough to keep ~150 a
season as players, but those seasons have no plays.

**By the players' appearances**, where it has none -- what the plays alone
can say. A side is generic when its typical player, the median over the
players it used in the game, appeared in under 40% as many of the season's
games as the other side's, or when it has no identified player and the
other side does. A play with no id isn't enough: D1 teams have players ESPN
hasn't linked too (all 26 mens sides with over 5% of their shots
unidentified in 2024-25 were D1). It's second because it counts games with
plays, and those are missing for some teams far more than others: it plays
40-51 D1 sides a season as generic in 2013-14 to 2015-16, and before
2019-20, when only a quarter of womens games have plays, two in five of
them.

Both are ratios rather than counts of games so they hold on part of a
season: on opening day, when everyone has played once, no side is generic,
and an outside team's possessions are left out as they were before.

A generic side is `GENERIC` in the tables; the fits play it as its
division's generic player (`divisions`).
"""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from .context import GameContext
from .events import other
from .lineups import SIDES
from .names import resolve_names
from .plays import Game

# The player id that stands in for a generic side's five.
GENERIC = "generic"

# A side is generic below this share of the other side's team's games...
TEAM_CUTOFF = 0.6
# ...or, without a summary, of the other side's typical player's.
CUTOFF = 0.4


@dataclass(frozen=True)
class Schedule:
    """A team's games in a season, from its summaries."""

    games: int = 0
    conference_games: int = 0


class Opponents:
    """
    Which side of each of a season's games is generic: by its team's
    schedule (`contexts`, the season's summaries) where the game has a
    summary, by its players' `appearances` where it doesn't.
    """

    def __init__(
        self,
        appearances: Mapping[str, int],
        contexts: Mapping[str, GameContext] | None = None,
    ):
        self.appearances = appearances
        self.contexts = contexts or {}
        self.schedules = schedules(self.contexts.values())

    def generic_side(self, game: Game) -> str | None:
        """`game` should have its names resolved, as `appearances` counted it."""
        context = self.contexts.get(game.game_id)
        if context is None:
            return generic_side(game, self.appearances)
        return outside_side(context, self.schedules)


def schedules(contexts: Iterable[GameContext]) -> dict[str, Schedule]:
    """Each team's games and conference games in `contexts`."""
    games: Counter[str] = Counter()
    conference: Counter[str] = Counter()
    for context in contexts:
        for side in SIDES:
            team = context.team(side).team_id
            games[team] += 1
            conference[team] += context.conference_game
    return {team: Schedule(games[team], conference[team]) for team in games}


def outside_side(context: GameContext, schedules: Mapping[str, Schedule]) -> str | None:
    """The side of a game whose team's schedule says it's from outside D1."""
    for side in SIDES:
        mine = schedules.get(context.team(side).team_id, Schedule())
        theirs = schedules.get(context.team(other(side)).team_id, Schedule())
        if not mine.conference_games and mine.games < TEAM_CUTOFF * theirs.games:
            return side
    return None


def appearances(games: Iterable[Game]) -> Counter[str]:
    """How many of `games` each player appears in, for `generic_side`."""
    seen: Counter[str] = Counter()
    for game in games:
        players = _players(resolve_names(game))
        seen.update(players["home"] | players["away"])
    return seen


def generic_side(game: Game, appearances: Mapping[str, int]) -> str | None:
    """
    The side of `game` its players' appearances say is from outside D1, if
    either. `game` should have its names resolved (`resolve_names`), as
    `appearances` counted it.
    """
    players = _players(game)
    typical = {
        side: _median([appearances.get(p, 0) for p in players[side]]) for side in SIDES
    }
    for side in SIDES:
        theirs = typical[other(side)]
        if not players[other(side)]:
            continue
        if not players[side] or typical[side] < CUTOFF * theirs:
            return side
    return None


def _players(game: Game) -> dict[str, set[str]]:
    players: dict[str, set[str]] = {side: set() for side in SIDES}
    for play in game.plays:
        if play.side in players and play.athlete_id:
            players[play.side].add(play.athlete_id)
    return players


def _median(values: list[int]) -> float:
    ordered = sorted(values)
    return ordered[len(ordered) // 2] if ordered else 0
