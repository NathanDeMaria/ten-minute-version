"""
What kind of thing a play is, as far as lineups and possessions care.

ESPN's `play_category_id` is the source: a short list of codes that groups
plays by meaning. Every code that occurs in ~830 games between March 2025
and January 2026:

    1001 made FT        1007 foul (personal or technical)   1015 substitution
    1002 made two       1008 timeout                        1017/1018/1020 period end
    1003 made three     1009 rebound (offensive, defensive, 1019 game end
    1004 missed FT           dead ball)                     1021 block
    1005 missed two     1010 turnover                       1022 steal
    1006 missed three   1011 coach's challenge
                        1012 jump ball

A few hundred shots and free throws in those games came with no code at
all, so a play without one is read from its type and text instead -- see
`_uncoded`.
"""

from enum import Enum, auto

from .plays import Play


class Kind(Enum):
    MADE_2 = auto()
    MADE_3 = auto()
    MISSED_2 = auto()
    MISSED_3 = auto()
    MADE_FT = auto()
    MISSED_FT = auto()
    OFFENSIVE_REBOUND = auto()
    DEFENSIVE_REBOUND = auto()
    # A rebound credited to a team while the ball is dead -- after a missed
    # first free throw, or a miss that went out of bounds. It says nothing
    # dependable about who has the ball next.
    DEAD_BALL_REBOUND = auto()
    TURNOVER = auto()
    STEAL = auto()
    BLOCK = auto()
    FOUL = auto()
    TECHNICAL = auto()
    SUB_IN = auto()
    SUB_OUT = auto()
    PERIOD_END = auto()
    OTHER = auto()


SHOTS = {Kind.MADE_2, Kind.MADE_3, Kind.MISSED_2, Kind.MISSED_3}
FREE_THROWS = {Kind.MADE_FT, Kind.MISSED_FT}
POINTS = {Kind.MADE_2: 2, Kind.MADE_3: 3, Kind.MADE_FT: 1}

_BY_CATEGORY = {
    "1001": Kind.MADE_FT,
    "1002": Kind.MADE_2,
    "1003": Kind.MADE_3,
    "1004": Kind.MISSED_FT,
    "1005": Kind.MISSED_2,
    "1006": Kind.MISSED_3,
    "1010": Kind.TURNOVER,
    "1017": Kind.PERIOD_END,
    "1018": Kind.PERIOD_END,
    "1019": Kind.PERIOD_END,
    "1020": Kind.PERIOD_END,
    "1021": Kind.BLOCK,
    "1022": Kind.STEAL,
}

_REBOUNDS = {
    "Offensive Rebound": Kind.OFFENSIVE_REBOUND,
    "Defensive Rebound": Kind.DEFENSIVE_REBOUND,
    "Dead Ball Rebound": Kind.DEAD_BALL_REBOUND,
}

_SHOT_TYPES = {"JumpShot", "LayUpShot", "DunkShot", "TipShot", "Three Point Jump Shot"}


def classify(play: Play) -> Kind:
    category = play.category
    if category in _BY_CATEGORY:
        return _BY_CATEGORY[category]
    if category == "1009":
        return _REBOUNDS.get(play.play_type or "", Kind.OTHER)
    if category == "1007":
        return Kind.TECHNICAL if play.play_type == "Technical Foul" else Kind.FOUL
    if category == "1015":
        return _substitution(play)
    if category is None:
        return _uncoded(play)
    return Kind.OTHER


def offense(kind: Kind, side: str | None) -> str | None:
    """
    The side with the ball, as far as one play says -- or None when the
    play doesn't say.

    A shot, a free throw, a turnover and an offensive rebound are the
    offense's own plays. A defensive rebound is the new offense's first. A
    steal or a block is the defense's play, so the ball was the other
    side's. Fouls say nothing: an offensive foul looks the same as any other
    in this feed (the turnover logged with it is what moves the ball).
    """
    if side is None:
        return None
    if kind in SHOTS or kind in FREE_THROWS:
        return side
    if kind in (Kind.TURNOVER, Kind.OFFENSIVE_REBOUND, Kind.DEFENSIVE_REBOUND):
        return side
    if kind in (Kind.STEAL, Kind.BLOCK):
        return other(side)
    return None


def other(side: str) -> str:
    return "away" if side == "home" else "home"


def _substitution(play: Play) -> Kind:
    if "subbing in" in play.text:
        return Kind.SUB_IN
    if "subbing out" in play.text:
        return Kind.SUB_OUT
    return Kind.OTHER


def _uncoded(play: Play) -> Kind:
    """
    A shot or free throw ESPN sent without a category. The two text styles
    in the feed -- "X makes 6-foot jumper", "X made Three Point Jumper." --
    both put made/missed and the shot's value in words.
    """
    text = play.text.lower()
    made = " made " in f" {text} " or " makes " in f" {text} "
    if play.play_type == "MadeFreeThrow" or "free throw" in text:
        return Kind.MADE_FT if made else Kind.MISSED_FT
    if play.play_type in _SHOT_TYPES:
        three = "three point" in text
        if made:
            return Kind.MADE_3 if three else Kind.MADE_2
        return Kind.MISSED_3 if three else Kind.MISSED_2
    return Kind.OTHER
