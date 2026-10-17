"""
Lineups for games without substitutions, estimated from what players do.

Every game before 2025-02-15 has no substitutions in the feed. What it does
have is every player's actions -- shots, rebounds, fouls, turnovers, with the
ids `names` recovers -- and the dead balls, which are the only moments a
substitution can happen. The estimate is a **persistent** lineup: each side's
five only changes when it has to, because someone not in it does something.

When that happens, the rules choose who leaves and when:

- **who**: the player in the five whose next action is furthest off, with a
  player in foul trouble going first. With a box score, high-minute players
  are kept on too. The leaver is also nudged towards resembling the arrival
  (three-point share, rebounds and blocks, assists -- a stand-in for
  position), which measured as neither help nor harm;
- **when**: a third of the way from the leaver's last action to the arrival's
  first (the median of real substitutions), snapped to a TV timeout if one
  falls in between -- teams make their biggest changes at those;
- **halftime**: the starting five comes back out (85% of second halves open
  with exactly the starters, and 97% with at least four of them).

Measured by hiding the substitutions of games where the lineups are known
(`evaluate`, January 2026): 86.0% of mens players on the floor right at each
possession and all five right at 43.9%, against 85.5% and 40.2% for the
nearest-action rule this replaces; womens 85.7% and 42.2%. Each game's
starters and minutes, from a box score, take mens to 86.8% and 47.3%. What's
left is mostly *when* a swap happened, not who -- which is why the rating
should take `probabilities` rather than these guesses.
"""

import math
from bisect import bisect_left, bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from functools import partial
from typing import Iterable, Mapping, Sequence

from .events import Kind, classify
from .lineups import SIDES, Floor, actors
from .plays import Game
from .possessions import _elapsed

# The plays a substitution can follow: the dead balls. Measured where real
# substitutions happen, these cover ~85% of them; most of the rest come
# after a rebound or a made basket, which can't be told from a live ball.
DEAD_BALLS = {
    Kind.FOUL,
    Kind.TECHNICAL,
    Kind.MADE_FT,
    Kind.MISSED_FT,
    Kind.TURNOVER,
    Kind.STEAL,
    Kind.DEAD_BALL_REBOUND,
    Kind.TIMEOUT,
    Kind.TV_TIMEOUT,
    Kind.PERIOD_END,
}

# A player whose next action is nowhere in the rest of the game.
_NEVER = 10_000.0


@dataclass(frozen=True)
class BoxScore:
    """What a box score adds for one game."""

    # Each side's starting five.
    starters: Mapping[str, frozenset[str]]
    # Each player's share of the game on the floor, 0 to 1.
    minutes: Mapping[str, Mapping[str, float]]


@dataclass(frozen=True)
class SeasonContext:
    """What a season says about each player, for the rules that need it."""

    # (three-point share of shots, rebounds and blocks per action, assists per
    # action) -- a stand-in for position.
    roles: Mapping[str, tuple[float, float, float]] = field(default_factory=dict)
    # Share of his side's possessions on the floor, in the games he played.
    minutes: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Rules:
    """
    The weights, in seconds of "next action" each rule is worth. Set by
    hand, not fitted: see `evaluate` for what each one adds.
    """

    # Fraction of the way back to place a swap; None swaps at the forced play.
    backdate: float | None = 0.36
    tv_timeouts: bool = True
    halftime_starters: bool = True
    minutes_weight: float = 1200.0
    position_weight: float = 400.0
    foul_weight: float = 300.0


def estimate_floors(
    game: Game,
    kinds: Sequence[Kind],
    context: SeasonContext = SeasonContext(),
    box: BoxScore | None = None,
    rules: Rules = Rules(),
) -> list[Floor]:
    """The estimated floor at every play, in the shape `lineups.on_floor` gives."""
    windows = Windows(game, kinds)
    by_side = {
        side: _Side(game, kinds, windows, side, context, box, rules).run()
        for side in SIDES
    }
    return [(by_side["home"][w], by_side["away"][w]) for w in windows.of_play]


def nearest_action_floors(game: Game, kinds: Sequence[Kind]) -> list[Floor]:
    """
    The rule `estimate_floors` replaced, kept as the baseline: in each
    stretch between dead balls, whoever acted, then whoever's nearest action
    is closest.
    """
    windows = Windows(game, kinds)
    fives = {}
    for side in SIDES:
        acts = _actions(game, kinds, windows, side)
        side_fives = []
        for w in range(windows.count):
            forced = sorted(acts.by_window.get(w, ()))
            mid = windows.mid[w]

            def distance(player: str) -> float:
                times = acts.times[player]
                at = bisect_left(times, mid)
                before = mid - times[at - 1] if at else math.inf
                after = times[at] - mid if at < len(times) else math.inf
                return min(before, after)

            rest = sorted((p for p in acts.times if p not in forced), key=distance)
            side_fives.append(frozenset(forced[:5] + rest[: max(0, 5 - len(forced))]))
        fives[side] = side_fives
    return [(fives["home"][w], fives["away"][w]) for w in windows.of_play]


def season_context(
    games: Iterable[Game], minutes: Mapping[str, float] | None = None
) -> SeasonContext:
    """
    Roles from a season's actions, and season minutes if something --
    box scores -- says what they were.
    """
    counts: dict[str, Counter] = defaultdict(Counter)
    for game in games:
        for play in game.plays:
            kind = classify(play)
            for n, player in enumerate(actors(play, kind)):
                count = counts[player]
                count["actions"] += 1
                if n == 1:
                    count["assists"] += 1
                elif kind in (Kind.MADE_3, Kind.MISSED_3):
                    count["threes"] += 1
                    count["shots"] += 1
                elif kind in (Kind.MADE_2, Kind.MISSED_2):
                    count["shots"] += 1
                elif kind in (
                    Kind.OFFENSIVE_REBOUND,
                    Kind.DEFENSIVE_REBOUND,
                    Kind.BLOCK,
                ):
                    count["bigs"] += 1
    roles = {
        player: (
            count["threes"] / max(count["shots"], 1),
            3 * count["bigs"] / count["actions"],
            3 * count["assists"] / count["actions"],
        )
        for player, count in counts.items()
    }
    return SeasonContext(roles, dict(minutes or {}))


class Windows:
    """
    A game cut at its dead balls. Nothing can change between two dead balls,
    so a lineup is one per window.
    """

    def __init__(self, game: Game, kinds: Sequence[Kind]) -> None:
        self.of_play: list[int] = []
        window = 0
        for kind in kinds:
            self.of_play.append(window)
            if kind in DEAD_BALLS:
                window += 1
        self.count = window + 1
        self.seconds = [
            _elapsed(game.league, play.period, play.clock_seconds)
            for play in game.plays
        ]
        self.start = [math.nan] * self.count
        self.end = [math.nan] * self.count
        self.period = [0] * self.count
        self.after_tv_timeout = [False] * self.count
        for i, w in enumerate(self.of_play):
            if math.isnan(self.start[w]):
                self.start[w] = self.seconds[i]
                self.period[w] = game.plays[i].period
            self.end[w] = self.seconds[i]
            if kinds[i] == Kind.TV_TIMEOUT and w + 1 < self.count:
                self.after_tv_timeout[w + 1] = True
        for w in range(self.count):
            # A window with no plays of its own (a game ending on a dead
            # ball) takes the one before it.
            if math.isnan(self.start[w]):
                previous = w - 1 if w else 0
                self.start[w] = self.end[w] = self.end[previous] if w else 0.0
                self.period[w] = self.period[previous] if w else 1
        self.mid = [(a + b) / 2 for a, b in zip(self.start, self.end)]


@dataclass
class _Actions:
    times: dict[str, list[float]]
    by_window: dict[int, set[str]]
    fouls: dict[str, list[tuple[float, int]]]


def _actions(
    game: Game, kinds: Sequence[Kind], windows: Windows, side: str
) -> _Actions:
    times: dict[str, list[float]] = defaultdict(list)
    by_window: dict[int, set[str]] = defaultdict(set)
    fouls: dict[str, list[tuple[float, int]]] = defaultdict(list)
    for i, (play, kind) in enumerate(zip(game.plays, kinds)):
        if play.side != side:
            continue
        for player in actors(play, kind):
            times[player].append(windows.seconds[i])
            by_window[windows.of_play[i]].add(player)
        if kind == Kind.FOUL and play.athlete_id:
            fouls[play.athlete_id].append((windows.seconds[i], play.period))
    return _Actions(dict(times), dict(by_window), dict(fouls))


class _Side:
    def __init__(
        self,
        game: Game,
        kinds: Sequence[Kind],
        windows: Windows,
        side: str,
        context: SeasonContext,
        box: BoxScore | None,
        rules: Rules,
    ) -> None:
        self._league = game.league
        self._windows = windows
        self._acts = _actions(game, kinds, windows, side)
        self._context = context
        self._minutes = box.minutes[side] if box else context.minutes
        self._box_starters = box.starters[side] if box else None
        self._rules = rules

    def run(self) -> list[frozenset[str]]:
        windows, rules = self._windows, self._rules
        starters = self._starters()
        five = set(starters)
        fives: list[set[str]] = []
        halftime = 2 if self._league == "mens" else 3
        for w in range(windows.count):
            if (
                rules.halftime_starters
                and w
                and windows.period[w] == halftime
                and windows.period[w - 1] != halftime
            ):
                five = set(starters)
            required: set[str] = self._acts.by_window.get(w) or set()
            while len(five) > 5:
                # More than five left over from a window where six acted
                # (a substitution the dead balls didn't catch)
                spare = [p for p in five if p not in required] or list(five)
                five.discard(
                    max(spare, key=partial(self._leave_order, arrival=None, w=w))
                )
            for arrival in sorted(required - five):
                if len(five) < 5:
                    five.add(arrival)
                    continue
                candidates = [p for p in five if p not in required]
                if not candidates:
                    five.add(arrival)
                    continue
                leaver = max(candidates, key=lambda p: self._leave_order(p, arrival, w))
                five.discard(leaver)
                five.add(arrival)
                self._backdate(fives, leaver, arrival, w)
            fives.append(set(five))
        return [frozenset(f) for f in fives]

    def _starters(self) -> set[str]:
        if self._box_starters is not None:
            return set(self._box_starters)
        first = {p: times[0] for p, times in self._acts.times.items()}
        weight = self._rules.minutes_weight / 10

        def order(player: str) -> float:
            return first[player] - weight * self._minutes.get(player, 0.0)

        return set(sorted(first, key=order)[:5])

    def _leave_order(self, player: str, arrival: str | None, w: int) -> float:
        """Higher is likelier to be the one who left."""
        rules = self._rules
        now = self._windows.start[w]
        times = self._acts.times.get(player, [])
        at = bisect_right(times, now)
        order = times[at] - now if at < len(times) else _NEVER
        order -= rules.minutes_weight * self._minutes.get(player, 0.0)
        if arrival is not None and rules.position_weight:
            mine, theirs = (
                self._context.roles.get(player),
                self._context.roles.get(arrival),
            )
            if mine and theirs:
                order -= rules.position_weight * math.dist(mine, theirs)
        if rules.foul_weight and self._in_foul_trouble(player, now, w):
            order += rules.foul_weight
        return order

    def _in_foul_trouble(self, player: str, now: float, w: int) -> bool:
        """Two fouls in the first half, or four in the game."""
        fouls = [
            period for when, period in self._acts.fouls.get(player, ()) if when <= now
        ]
        first_half = 1 if self._league == "mens" else 2
        period = self._windows.period[w]
        early = sum(1 for p in fouls if p <= first_half)
        return (period <= first_half and early >= 2) or len(fouls) >= 4

    def _backdate(
        self, fives: list[set[str]], leaver: str, arrival: str, w: int
    ) -> None:
        rules, windows = self._rules, self._windows
        if rules.backdate is None:
            return
        times = self._acts.times.get(leaver, [])
        now = windows.start[w]
        at = bisect_left(times, now)
        last = times[at - 1] if at else 0.0
        swap_at = last + rules.backdate * (now - last)
        if rules.tv_timeouts:
            timeouts = [
                windows.start[x]
                for x in range(w, -1, -1)
                if windows.start[x] > last
                and windows.after_tv_timeout[x]
                and windows.period[x] == windows.period[w]
            ]
            if timeouts:
                swap_at = min(timeouts, key=lambda t: abs(t - swap_at))
        for x in range(w - 1, -1, -1):
            if windows.start[x] < swap_at or windows.period[x] != windows.period[w]:
                break
            if leaver in fives[x] and arrival not in fives[x]:
                fives[x].discard(leaver)
                fives[x].add(arrival)
