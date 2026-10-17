"""
A game cut into possessions, each credited to the ten players on the floor.

**Who has the ball** comes from the plays that only one side can make (see
`events.offense`). A possession is a run of those plays for the same side
within a period; it ends when the other side makes one, or the period ends.
Dead-ball rebounds, fouls, timeouts and jump balls say nothing about the
ball and are skipped.

**Credit.** A possession goes to the lineup on the floor at its last play,
except that free throws go to the lineup on the floor at the foul that led
to them -- substitutions between free throws are routine, and the players
coming in had nothing to do with the foul.

**Points** are counted from the scoring plays (2, 3 or 1), never from the
score ESPN stamps on each play, which is sometimes stale and occasionally
runs backwards.

**Technical free throws** don't belong to anyone's possession: they're
taken by whichever side was awarded them, usually while the other side has
the ball, and treating them as a possession would add two empty ones. A free
throw is technical when its side was awarded a technical since the last
play that wasn't a free throw, and hasn't been fouled since. Their points
are kept apart (`technical_points`), so the game's total still adds up.
"""

from dataclasses import dataclass, field
from typing import Literal, Sequence

from .events import FREE_THROWS, POINTS, SHOTS, Kind, offense, other
from .lineups import Floor
from .plays import Game

Ended = Literal["fg2", "fg3", "ft", "turnover", "miss", "other"]

# Seconds in each period: mens play two 20-minute halves, womens four
# 10-minute quarters, and every overtime is five minutes.
_REGULATION = {"mens": (2, 1200), "womens": (4, 600)}
_OVERTIME = 300


@dataclass
class Possession:
    offense: str
    period: int
    # Seconds since the tip, at the possession's first play and its last.
    start_seconds: int
    end_seconds: int
    # The offense's lead when the possession started.
    start_margin: int
    floor: Floor
    # The play whose floor `floor` is: the possession's last, or the foul a
    # final free throw came from.
    credit_play: int = 0
    points: int = 0
    fga2: int = 0
    fgm2: int = 0
    fga3: int = 0
    fgm3: int = 0
    fta: int = 0
    ftm: int = 0
    oreb: int = 0
    tov: int = 0
    ended: Ended = "other"
    # The kind of the last play that counted towards the possession, and of
    # the last made field goal in it, for `ended`.
    _last: Kind | None = field(default=None, repr=False)
    _last_made: Kind | None = field(default=None, repr=False)

    def count(self, kind: Kind) -> None:
        self.points += POINTS.get(kind, 0)
        if kind in (Kind.MADE_2, Kind.MISSED_2):
            self.fga2 += 1
            self.fgm2 += kind == Kind.MADE_2
        elif kind in (Kind.MADE_3, Kind.MISSED_3):
            self.fga3 += 1
            self.fgm3 += kind == Kind.MADE_3
        elif kind in FREE_THROWS:
            self.fta += 1
            self.ftm += kind == Kind.MADE_FT
        elif kind == Kind.OFFENSIVE_REBOUND:
            self.oreb += 1
        elif kind == Kind.TURNOVER:
            self.tov += 1
        else:
            return
        self._last = kind
        if kind in (Kind.MADE_2, Kind.MADE_3):
            self._last_made = kind

    def close(self) -> None:
        """
        How the possession ended. A made field goal wins over the free
        throw that can follow it (the and-one); otherwise it's the last
        play that counted.
        """
        if self._last_made is not None:
            self.ended = "fg3" if self._last_made == Kind.MADE_3 else "fg2"
        elif self._last in FREE_THROWS:
            self.ended = "ft"
        elif self._last == Kind.TURNOVER:
            self.ended = "turnover"
        elif self._last in SHOTS:
            self.ended = "miss"
        else:
            self.ended = "other"


@dataclass
class Score:
    """Points per side, as counted from the plays."""

    points: dict[str, int] = field(default_factory=lambda: {"home": 0, "away": 0})
    technical_points: dict[str, int] = field(
        default_factory=lambda: {"home": 0, "away": 0}
    )


def possessions(
    game: Game, kinds: Sequence[Kind], floors: Sequence[Floor]
) -> tuple[list[Possession], Score]:
    score = Score()
    found: list[Possession] = []
    current: Possession | None = None
    # The floor at the last foul, for the free throws it leads to, and the
    # side holding technical free throws -- both forgotten at the next play
    # that isn't a free throw.
    foul_floor: Floor | None = None
    foul_play = 0
    technical_side: str | None = None

    for i, (play, kind, floor) in enumerate(zip(game.plays, kinds, floors)):
        if kind == Kind.PERIOD_END:
            _finish(current, found)
            current = None
            continue
        if kind in (Kind.FOUL, Kind.TECHNICAL):
            foul_floor, foul_play = floor, i
            if kind == Kind.TECHNICAL and play.side is not None:
                technical_side = other(play.side)
            elif (
                kind == Kind.FOUL
                and technical_side is not None
                and play.side == other(technical_side)
            ):
                # A personal foul against the side holding technical free
                # throws gives it regular ones too, and the free throws
                # coming could be either.
                technical_side = None
            continue
        side = offense(kind, play.side)
        if side is None:
            continue
        if kind in FREE_THROWS and side == technical_side:
            points = POINTS.get(kind, 0)
            score.points[side] += points
            score.technical_points[side] += points
            continue
        if kind not in FREE_THROWS:
            foul_floor = None
            technical_side = None
        seconds = _elapsed(game.league, play.period, play.clock_seconds)
        if current is None or current.offense != side:
            _finish(current, found)
            current = Possession(
                offense=side,
                period=play.period,
                start_seconds=seconds,
                end_seconds=seconds,
                start_margin=score.points[side] - score.points[other(side)],
                floor=floor,
            )
        current.count(kind)
        score.points[side] += POINTS.get(kind, 0)
        current.end_seconds = seconds
        if kind in FREE_THROWS and foul_floor:
            current.floor, current.credit_play = foul_floor, foul_play
        else:
            current.floor, current.credit_play = floor, i
    _finish(current, found)
    return found, score


def _finish(possession: Possession | None, found: list[Possession]) -> None:
    if possession is not None:
        possession.close()
        found.append(possession)


def _elapsed(league: str, period: int, clock_seconds: int) -> int:
    periods, length = _REGULATION[league]
    if period <= periods:
        return (period - 1) * length + (length - clock_seconds)
    regulation = periods * length
    return regulation + (period - periods - 1) * _OVERTIME + (_OVERTIME - clock_seconds)
