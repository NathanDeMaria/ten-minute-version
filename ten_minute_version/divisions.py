"""
Which generic player a team from outside Division I plays as: one for each
division it can come from, not one for all of them.

A generic side (`opponents`) is played as one player, `GENERIC`, its team
column on top. As one player it's the average of everyone from outside --
and they aren't one level: at the 2025-26 season's end, per 100
possessions a side, mens Division II teams are -28 against Division I,
NAIA -41, Division III -48 and the rest (NCCAA, USCAA, branch campuses)
-52. With one generic player, each team's column carries the difference,
at a width (`ChainSettings.outside_sd`) meant for how one team differs from
its peers; and a team new to the table starts at the average of all of
them, a Division III side as strong as a Division II one.

So a generic side plays as its division's player (`generic_id`), by the
team's division in call-it-what-you-want's classifications: Division II,
Division III, NAIA, or "other" -- a team it doesn't place, or places in
Division I (a school that's since moved up). Its season there is the one
it ends in, a year on from this package's: 2025-26 is 2025 here and 2026
there.

The stint tables keep `GENERIC`; the relabelling is at `Rows.build`, so a
table built before any of this reads the same as one built after.
"""

from collections.abc import Callable

from call_it_what_you_want import default_classifications

from .opponents import GENERIC

# call-it-what-you-want's league for each of this package's
_LEAGUE = {"mens": "ncaambb", "womens": "ncaawbb"}
# its divisions outside Division I, by the suffix of their generic player
_DIVISIONS = {
    "NCAA Division II": "d2",
    "NCAA Division III": "d3",
    "NAIA": "naia",
}
OTHER = "other"


def generic_id(division: str) -> str:
    """The generic player for a division's suffix ("d2", ..., "other")."""
    return f"{GENERIC}:{division}"


# Every division's generic player.
GENERICS = tuple(generic_id(d) for d in (*_DIVISIONS.values(), OTHER))


def is_generic(player: str) -> bool:
    """Whether `player` stands in for a generic side: `GENERIC` itself, as
    the stint tables have it, or a division's."""
    return player == GENERIC or player.startswith(f"{GENERIC}:")


def division_generics(league: str, season: int) -> Callable[[str], str]:
    """Each team's generic player in `season`: its division's."""
    classes = default_classifications("ncaa")
    found: dict[str, str] = {}

    def generic(team: str) -> str:
        if team not in found:
            c = classes.classification_in(team, season + 1, _LEAGUE[league])
            division = _DIVISIONS.get(c.division, OTHER) if c is not None else OTHER
            found[team] = generic_id(division)
        return found[team]

    return generic
