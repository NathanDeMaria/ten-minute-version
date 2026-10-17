"""
Which side of a game is a team from outside Division I, played as one
generic team rather than as its players.

ESPN's schedule is Division I's, and early each season a D1 team plays a
few teams from outside it -- NAIA, Division II and III. Most of their
players have no ESPN record, so their plays carry no athlete id and their
side of the floor can't be followed. Those that do have records turn up in
only the handful of games their team plays against D1.

That second fact is what tells them apart. A play with no id isn't enough:
D1 teams have players ESPN hasn't linked too -- all 26 mens sides with over
5% of their shots unidentified in 2024-25 were D1, as were UCLA's women in
four 2025-26 games -- and treating those teams as generic would throw away
the players who are known. But a D1 team's players appear all season. So a side is generic when
its typical player -- the median over the players it used in the game --
has appeared in under 40% as many of the season's games as the other side's
typical player, or when it has no identified player and the other side
does.

Over 2025-26 that marks 535 mens and 395 womens sides, every one a team
outside D1, and no D1 team. The nearest outside team sits at 0.375 (College
of Biblical Studies, who play enough D1 games, and whose players are all
known, to stay real players); the nearest D1 team at 0.47 (Bellarmine's
women).

It's a ratio rather than a count of games so it holds on part of a season:
on opening day, when everyone has played once, no side is generic, and an
outside team's possessions are left out as they were before.
"""

from collections import Counter
from typing import Iterable, Mapping

from .events import other
from .lineups import SIDES
from .names import resolve_names
from .plays import Game

# The player id that stands in for a generic side's five.
GENERIC = "generic"

# A side is generic below this share of the other side's typical games.
CUTOFF = 0.4


def appearances(games: Iterable[Game]) -> Counter[str]:
    """How many of `games` each player appears in, for `generic_side`."""
    seen: Counter[str] = Counter()
    for game in games:
        players = _players(resolve_names(game))
        seen.update(players["home"] | players["away"])
    return seen


def generic_side(game: Game, appearances: Mapping[str, int]) -> str | None:
    """
    The side of `game` to play as one generic team, if either. `game` should
    have its names resolved (`resolve_names`), as `appearances` counted it.
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
