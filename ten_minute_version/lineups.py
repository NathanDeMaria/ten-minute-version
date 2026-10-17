"""
Who was on the floor for every play of a game.

ESPN logs each substitution as its own play -- "X subbing out for Purdue",
"Y subbing in for Purdue" -- including the changes at the start of every
period, which come in at the period's opening clock. What it doesn't log is
the starting five. So the starters are inferred, the substitutions are
replayed over them, and where the replay can't be right the lineup is
repaired from what the players do next.

**Starters.** A player whose first appearance in the game is anything but
subbing in -- a shot, a rebound, a foul, subbing *out* -- was on the floor
from the tip. Over ~6,000 games from 2025 and 2026 this finds exactly five
a side in 98.5-99% of them.

**Repairs.** The feed drops a substitution now and then: an "out" with no
"in", or a player who scores without ever having checked in. Two rules:

- a player who does something while the replay has him on the bench was
  on the floor, and goes on it;
- a side that ends a run of substitutions with other than five gets fixed
  by looking ahead. With too many, the ones whose next appearance is
  subbing in come off first (they were already off), then the ones who
  never appear again, then whoever's next appearance is furthest away.
  With too few, the bench players whose next appearance is anything but
  subbing in come on, nearest first.

Neither rule guesses about a player the feed never mentions again, which
is what keeps a short side short rather than invented.
"""

from bisect import bisect_left
from dataclasses import dataclass
from typing import Literal, Sequence

from .events import Kind
from .plays import Play

SIDES = ("home", "away")

# The two sides on the floor, home first.
Floor = tuple[frozenset[str], frozenset[str]]

How = Literal["in", "out", "act"]

# Plays that put their player on the floor. A technical doesn't -- a player
# on the bench can be given one -- and neither does anything with no player.
_NOT_ON_THE_FLOOR = {Kind.SUB_IN, Kind.SUB_OUT, Kind.TECHNICAL}


@dataclass(frozen=True)
class LineupReport:
    # Both sides' starters came out at exactly five.
    starters_found: bool
    # Players put on or taken off the floor by a repair rule rather than a
    # logged substitution.
    repairs: int
    # How many starters each side came out with, home first: for a game
    # whose one side is left untracked, whether the other's were found.
    starters: tuple[int, int]


def on_floor(
    plays: Sequence[Play], kinds: Sequence[Kind]
) -> tuple[list[Floor], LineupReport]:
    """
    The floor at every play: after the play for a substitution, and at the
    moment of the play for anything else.
    """
    return _Replay(plays, kinds).run()


def actors(play: Play, kind: Kind) -> list[str]:
    """
    The players a play shows were on the floor for `play.side`: whoever it's
    about, and on a made shot the assister too (ESPN's second participant,
    always a teammate of the shooter).
    """
    if play.side is None or kind in _NOT_ON_THE_FLOOR:
        return []
    players = [play.athlete_id] if play.athlete_id else []
    if kind in (Kind.MADE_2, Kind.MADE_3):
        players += [p for p in play.participant_ids[1:] if p not in players]
    return players


class _Replay:
    def __init__(self, plays: Sequence[Play], kinds: Sequence[Kind]) -> None:
        self._plays = plays
        self._kinds = kinds
        # side -> player -> their appearances, in order, as (play index, how)
        self._appearances: dict[str, dict[str, list[tuple[int, How]]]] = {
            side: {} for side in SIDES
        }
        for i, (play, kind) in enumerate(zip(plays, kinds)):
            if play.side not in SIDES:
                continue
            if kind in (Kind.SUB_IN, Kind.SUB_OUT) and play.athlete_id:
                how: How = "in" if kind == Kind.SUB_IN else "out"
                self._note(play.side, play.athlete_id, i, how)
            for player in actors(play, kind):
                self._note(play.side, player, i, "act")
        self._indexes = {
            side: {
                player: [i for i, _ in seen]
                for player, seen in self._appearances[side].items()
            }
            for side in SIDES
        }
        self._repairs = 0

    def _note(self, side: str, player: str, i: int, how: How) -> None:
        self._appearances[side].setdefault(player, []).append((i, how))

    def run(self) -> tuple[list[Floor], LineupReport]:
        on = {
            side: {
                player
                for player, seen in self._appearances[side].items()
                if seen[0][1] != "in"
            }
            for side in SIDES
        }
        starting = {side: set(on[side]) for side in SIDES}
        starters_found = all(len(on[side]) == 5 for side in SIDES)
        # A side is checked before the first play that isn't a substitution,
        # and again after every run of substitutions -- never in the middle
        # of one, where it's normal to be a player short or over.
        unchecked = set(SIDES)
        floors: list[Floor] = []
        for i, (play, kind) in enumerate(zip(self._plays, self._kinds)):
            if kind in (Kind.SUB_IN, Kind.SUB_OUT):
                if play.side in SIDES and play.athlete_id:
                    if kind == Kind.SUB_IN:
                        on[play.side].add(play.athlete_id)
                    else:
                        on[play.side].discard(play.athlete_id)
                    unchecked.add(play.side)
                floors.append(_floor(on))
                continue
            for side in sorted(unchecked):
                if len(on[side]) != 5:
                    self._fix(on[side], side, i, keep=set())
            unchecked.clear()
            if play.side in SIDES:
                for player in actors(play, kind):
                    if player not in on[play.side]:
                        on[play.side].add(player)
                        self._repairs += 1
                        if len(on[play.side]) > 5:
                            self._fix(on[play.side], play.side, i, keep={player})
            floors.append(_floor(on))
        starters = (len(starting["home"]), len(starting["away"]))
        return floors, LineupReport(starters_found, self._repairs, starters)

    def _next(self, side: str, player: str, i: int) -> tuple[int, How] | None:
        """A player's first appearance at or after play `i`."""
        indexes = self._indexes[side].get(player, [])
        at = bisect_left(indexes, i)
        if at == len(indexes):
            return None
        return self._appearances[side][player][at]

    def _fix(self, on: set[str], side: str, i: int, keep: set[str]) -> None:
        if len(on) > 5:

            def drop_order(player: str) -> tuple[int, int]:
                upcoming = self._next(side, player, i)
                if upcoming is None:
                    return (1, 0)
                index, how = upcoming
                return (0, 0) if how == "in" else (2, -index)

            for player in sorted(on - keep, key=drop_order)[: len(on) - 5]:
                on.discard(player)
                self._repairs += 1
        elif len(on) < 5:
            candidates = []
            for player in self._appearances[side]:
                if player in on:
                    continue
                upcoming = self._next(side, player, i)
                if upcoming is not None and upcoming[1] != "in":
                    candidates.append((upcoming[0], player))
            for _, player in sorted(candidates)[: 5 - len(on)]:
                on.add(player)
                self._repairs += 1


def _floor(on: dict[str, set[str]]) -> Floor:
    return frozenset(on["home"]), frozenset(on["away"])
