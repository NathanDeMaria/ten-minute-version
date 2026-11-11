"""
How each game went on the way to its final score, for a model that learns
from margins to discount -- beside the shooting luck `luck` files, and in
the same file.

A final margin says more than the game did in three ways this measures,
each in points toward the home side:

`jumpers`
    Two-point jump shots that went in beyond the league's rate: two points
    for each make less the rate times the attempts, the home side's less the
    away's. A jumper inside the arc is as much the bounce as a three is. A
    layup isn't -- finishing and rim protection are the teams' own -- and
    taking the same off shots at the rim costs a model that reads it.

`over_lead`
    The margin at the end of regulation less twice the home side's average
    lead through it. A game that grew evenly to its final margin averages
    half of it, so this is 0; a side that led by more for most of the night
    and let it close leaves it negative. The average lead is the steadier
    reading of the two, in men's games: a first half's margin predicts what
    a rating misses next, and a second half's doesn't.

`closing`
    How far the margin moved over the last five minutes of regulation: the
    fouling, the free throws and the threes of a game already decided.

Measured in Cassandra's `glicko_players` at its 2026-11-08 fits, on the file
this writes, taking a share of each off the margin Glicko learns from: Brier
over every game 2010-2026, against the same file with none taken.

                                       mens        womens
    jumpers at 0.4                  -0.00013     -0.00013
    over_lead at 0.25               -0.00013
    closing at 0.3                               -0.00005
    jumpers 0.4 + over_lead 0.25    -0.00028
    jumpers 0.4 + closing 0.3                    -0.00018

Each pair beats the fit it started from in all twelve seasons 2014-2025.
`over_lead` adds 0.00001 to the women's pair and `closing` nothing to the
men's. All of the jumpers is too much, as all of the shooting luck is, and
a team's own make rate in place of the league's reads the same (the two
agree at 0.98), so the league's it is.

The margin's path is ESPN's running score, each side's held at the most it
has shown (it goes stale, and comes back). Counting the scoring plays
instead loses one game in eight, to baskets the feed sends with no category
and no text. The running score has its own bad games -- the old feed
credits whole ones to one side -- so a flow carries the score it ended on
(`points`), for whoever knows the real final to check it against.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from .events import POINTS, Kind, classify
from .plays import Game

# The columns this adds to the luck file.
FLOW = ("jumpers", "over_lead", "closing")

REGULATION = 2400.0
_OVERTIME = 300.0
_CLOSING = 300.0
_AT_THE_RIM = ("LayUpShot", "DunkShot", "TipShot")

# About the league's make rate on two-point jumpers, and how many attempts
# of it are mixed into a season's own: an opening day has too few to say.
PRIOR_RATE = {"mens": 0.40, "womens": 0.36}
_PRIOR_ATTEMPTS = 2000.0


@dataclass(frozen=True)
class GameFlow:
    """One game on the way to its score; margins are the home side's."""

    game_id: str
    # at the end of regulation
    regulation: float
    # averaged over regulation's clock
    average_lead: float
    # with five minutes of regulation left
    before_closing: float
    # two-point jump shots: (made, attempted)
    home_jumpers: tuple[int, int]
    away_jumpers: tuple[int, int]
    # the score the plays end on: (home, away)
    points: tuple[int, int]

    @property
    def over_lead(self) -> float:
        return self.regulation - 2 * self.average_lead

    @property
    def closing(self) -> float:
        return self.regulation - self.before_closing


def game_flow(game: Game) -> GameFlow | None:
    """`game`'s flow; None if nobody scores in it, or it has no first period
    to tell how long its periods are."""
    opening = [p.clock_seconds for p in game.plays if p.period == 1]
    if not opening:
        return None
    # Halves of twenty minutes, or quarters of ten (women's, since 2015-16).
    length = 1200.0 if max(opening) > 620 else 600.0
    periods = int(REGULATION / length)
    score = {"home": 0, "away": 0}
    jumpers = {"home": [0, 0], "away": [0, 0]}
    # The margin is a step: 0 from the tip, then each play's until the next.
    times, margins = [0.0], [0.0]
    for play in game.plays:
        before = dict(score)
        score["home"] = max(score["home"], play.home_score or 0)
        score["away"] = max(score["away"], play.away_score or 0)
        kind = classify(play)
        if (
            play.side in score
            and kind in (Kind.MADE_2, Kind.MISSED_2)
            and play.play_type not in _AT_THE_RIM
        ):
            # A basket the feed sent with no category and no text reads as a
            # miss: the score going up two with it says it went in.
            went_in = kind is Kind.MADE_2 or (
                play.category is None
                and score[play.side] - before[play.side] == POINTS[Kind.MADE_2]
            )
            jumpers[play.side][0] += went_in
            jumpers[play.side][1] += 1
        if play.period > periods:
            continue
        elapsed = (play.period - 1) * length + (length - play.clock_seconds)
        times.append(float(np.clip(elapsed, 0.0, REGULATION)))
        margins.append(float(score["home"] - score["away"]))
    if score["home"] + score["away"] == 0:
        return None
    # A play logged out of order would run the clock backwards, so each time
    # is no earlier than the one before.
    at = np.maximum.accumulate(np.array(times))
    step = np.array(margins)
    held = np.diff(np.append(at, REGULATION))
    closing = int(np.flatnonzero(at <= REGULATION - _CLOSING)[-1])
    return GameFlow(
        game_id=game.game_id,
        regulation=float(step[-1]),
        average_lead=float((step * held).sum() / REGULATION),
        before_closing=float(step[closing]),
        home_jumpers=(jumpers["home"][0], jumpers["home"][1]),
        away_jumpers=(jumpers["away"][0], jumpers["away"][1]),
        points=(score["home"], score["away"]),
    )


def jumper_rate(flows: Iterable[GameFlow], league: str) -> float:
    """The make rate on `flows`' two-point jumpers, with the prior's
    attempts mixed in."""
    made = attempts = 0
    for flow in flows:
        made += flow.home_jumpers[0] + flow.away_jumpers[0]
        attempts += flow.home_jumpers[1] + flow.away_jumpers[1]
    prior = PRIOR_RATE[league]
    return (made + prior * _PRIOR_ATTEMPTS) / (attempts + _PRIOR_ATTEMPTS)


def flow_values(
    flows: Mapping[str, GameFlow], league: str
) -> dict[str, dict[str, float]]:
    """Each of `FLOW`'s columns, by game: every game of `flows` in points
    toward its home side, the jumpers against `flows`' own make rate."""
    rate = jumper_rate(flows.values(), league)

    def beyond(jumpers: tuple[int, int]) -> float:
        return 2 * (jumpers[0] - rate * jumpers[1])

    return {
        "jumpers": {
            g: beyond(f.home_jumpers) - beyond(f.away_jumpers) for g, f in flows.items()
        },
        "over_lead": {g: f.over_lead for g, f in flows.items()},
        "closing": {g: f.closing for g, f in flows.items()},
    }
