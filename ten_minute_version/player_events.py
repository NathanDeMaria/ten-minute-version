"""
Each player's events in each game, from the plays: what he did with the
ball and without it, to split a possession's credit by who did what rather
than equally among the five on the floor.

**Offense.** A field goal is the shooter's, by where it was taken: at the
rim (a layup, dunk or tip), a two from anywhere else (mid), or a three. A
make is assisted when ESPN names a second participant, and that player gets
the assist, by the zone of the shot he set up. A turnover and an offensive
rebound are the player's ESPN names; a team turnover or rebound is no one's.

**Free-throw trips** stand in for fouls drawn, which the feed never records
("Foul on X." names only who committed it): a trip is a run of free throws
by one player with nothing between them but substitutions and timeouts. The
2025-26 feed says "1 of 2"; the feed before 2022-23 doesn't, so the run is
what's counted in every season.

**Defense.** Defensive rebounds, blocks, steals and personal fouls
committed. Steals carry no athlete id in any season, and shots, free throws
and assists none before 2022-23: `names.resolve_names` gives each the id its
player carries on his other plays in the game.

**Possessions on the floor** come from the season's stint rows (and the
estimated rows of its games without substitutions, each player at his
chance of being on the floor), so a rate per possession is a count over
them. A player on a generic side (`opponents`) isn't on the floor as
himself, and a game the estimated rows dropped (its points don't add up)
has no one on the floor: their events are kept, their possessions are 0 --
count a rate over rows with possessions. That's 5% of 2025-26's events
(generic sides), 13% of 2018-19's (a further 631 dropped games).

**Against the box score** (`box_agreement`, games `box_ok`), every season
2013-14 to 2026-27: since 2022-23 every count is within 2% of the box's
total (assists 0.98-0.99, steals 0.97-1.00), and 99%+ of player-games match
it exactly. Before, a player who never does anything that carries his id in
a game can't be named, so 3-5% of field goals and free throws and 5-11% of
assists go to no one; turnovers, fouls and rebounds, which always carry an
id, still match (0.99-1.00). Scale a player's counts to his box score where
the share matters.
"""

from collections import Counter
from collections.abc import Iterable, Mapping

import pyarrow as pa

from .context import GameContext
from .divisions import is_generic
from .events import FREE_THROWS, Kind, classify, other
from .names import resolve_names
from .plays import Game, Play

ZONES = ("rim", "mid", "three")
_AT_THE_RIM = frozenset({"LayUpShot", "DunkShot", "TipShot"})

OFFENSE = (
    *(f"made_{zone}_unassisted" for zone in ZONES),
    *(f"made_{zone}_assisted" for zone in ZONES),
    *(f"missed_{zone}" for zone in ZONES),
    *(f"assists_{zone}" for zone in ZONES),
    "ft_made",
    "ft_missed",
    "ft_trips",
    "turnovers",
    "offensive_rebounds",
)
DEFENSE = ("defensive_rebounds", "blocks", "steals", "fouls")
COUNTS = (*OFFENSE, *DEFENSE)

# Plays a free-throw trip carries on through.
_BETWEEN_FREE_THROWS = frozenset(
    {Kind.SUB_IN, Kind.SUB_OUT, Kind.TIMEOUT, Kind.TV_TIMEOUT, Kind.OTHER}
)

EVENT_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("side", pa.string(), nullable=False),
        # From the game's summary; None for a game without one.
        pa.field("team_id", pa.string()),
        pa.field("athlete_id", pa.string(), nullable=False),
        # His side's possessions with him on the floor, offense and defense,
        # from the stint (and estimated) rows: fractional where estimated.
        pa.field("offense_possessions", pa.float32(), nullable=False),
        pa.field("defense_possessions", pa.float32(), nullable=False),
        *(pa.field(count, pa.int16(), nullable=False) for count in COUNTS),
        # Whether the game's plays agree with its box score (`mark_box`);
        # None for a game without one.
        pa.field("box_ok", pa.bool_()),
    ]
)

# Counts every season's feed ties to a player by id, so a game whose plays
# are whole has about as many as its box score.
_WHOLE = {"turnovers": "tov", "fouls": "pf", "defensive_rebounds": "dreb"}


def zone(play: Play, kind: Kind) -> str:
    """Where a field goal was taken: "rim", "mid" or "three"."""
    if kind in (Kind.MADE_3, Kind.MISSED_3):
        return "three"
    return "rim" if play.play_type in _AT_THE_RIM else "mid"


def game_events(game: Game) -> dict[tuple[str, str], Counter[str]]:
    """Each (side, athlete id)'s counts in `game`, names resolved."""
    game = resolve_names(game)
    found: dict[tuple[str, str], Counter[str]] = {}

    def add(side: str, player: str | None, count: str) -> None:
        if player:
            found.setdefault((side, player), Counter())[count] += 1

    # the shooter of the free throw a trip is on, and his side
    trip: tuple[str, str] | None = None
    for play in game.plays:
        kind = classify(play)
        side = play.side
        if side is None:
            continue
        player = play.athlete_id
        if kind in FREE_THROWS:
            if player and trip != (side, player):
                add(side, player, "ft_trips")
            trip = (side, player) if player else None
            add(side, player, "ft_made" if kind == Kind.MADE_FT else "ft_missed")
            continue
        if kind not in _BETWEEN_FREE_THROWS:
            trip = None
        if kind in (Kind.MADE_2, Kind.MADE_3):
            where = zone(play, kind)
            assister = next(
                (p for p in play.participant_ids[1:] if p and p != player), None
            )
            add(
                side, player, f"made_{where}_{'assisted' if assister else 'unassisted'}"
            )
            add(side, assister, f"assists_{where}")
        elif kind in (Kind.MISSED_2, Kind.MISSED_3):
            add(side, player, f"missed_{zone(play, kind)}")
        elif kind == Kind.TURNOVER:
            add(side, player, "turnovers")
        elif kind == Kind.OFFENSIVE_REBOUND:
            add(side, player, "offensive_rebounds")
        elif kind == Kind.DEFENSIVE_REBOUND:
            add(side, player, "defensive_rebounds")
        elif kind == Kind.BLOCK:
            add(side, player, "blocks")
        elif kind == Kind.STEAL:
            add(side, player, "steals")
        elif kind == Kind.FOUL:
            add(side, player, "fouls")
    return found


def floor_possessions(
    rows: Iterable[Mapping],
) -> dict[tuple[str, str], tuple[str, float, float]]:
    """
    Each (game id, athlete id)'s side and possessions on the floor
    (offense, defense), from stint or estimated rows: each player listed at
    his weight (1 where the row has none) times the row's possessions. A
    generic side's stand-in isn't anyone.
    """
    found: dict[tuple[str, str], list] = {}
    for row in rows:
        possessions = float(row["possessions"])
        offense_side = row["offense_side"]
        for players, weights, at, side in (
            (row["offense_players"], row.get("offense_weights"), 1, offense_side),
            (
                row["defense_players"],
                row.get("defense_weights"),
                2,
                other(offense_side),
            ),
        ):
            for i, player in enumerate(players):
                if is_generic(player):
                    continue
                weight = 1.0 if weights is None else float(weights[i])
                seen = found.setdefault((row["game_id"], player), [side, 0.0, 0.0])
                seen[at] += weight * possessions
    return {key: (side, o, d) for key, (side, o, d) in found.items()}


def event_rows(
    game: Game,
    possessions: Mapping[tuple[str, str], tuple[str, float, float]],
    context: GameContext | None = None,
) -> list[dict]:
    """`game`'s rows of the event table: one per player with an event or a
    possession on the floor (`floor_possessions`), with his team from
    `context` if given."""
    counts = game_events(game)
    players = dict(counts)
    for (g, player), (side, _, _) in possessions.items():
        if g == game.game_id and (side, player) not in players:
            players[side, player] = Counter()
    out = []
    for side, player in sorted(players):
        _, offense, defense = possessions.get((game.game_id, player), (side, 0.0, 0.0))
        out.append(
            {
                "league": game.league,
                "season": game.season,
                "game_id": game.game_id,
                "side": side,
                "team_id": context.team(side).team_id if context else None,
                "athlete_id": player,
                "offense_possessions": offense,
                "defense_possessions": defense,
                **{count: players[side, player][count] for count in COUNTS},
            }
        )
    return out


# What each box-score stat is in the event table's counts.
_BOX = {
    "fgm": [f"made_{z}_{a}" for z in ZONES for a in ("unassisted", "assisted")],
    "fga": [f"made_{z}_{a}" for z in ZONES for a in ("unassisted", "assisted")]
    + [f"missed_{z}" for z in ZONES],
    "tpm": ["made_three_unassisted", "made_three_assisted"],
    "tpa": ["made_three_unassisted", "made_three_assisted", "missed_three"],
    "ftm": ["ft_made"],
    "fta": ["ft_made", "ft_missed"],
    "oreb": ["offensive_rebounds"],
    "dreb": ["defensive_rebounds"],
    "ast": [f"assists_{z}" for z in ZONES],
    "stl": ["steals"],
    "blk": ["blocks"],
    "tov": ["turnovers"],
    "pf": ["fouls"],
}


def box_agreement(
    rows: Iterable[Mapping], box: Mapping[tuple[str, str], Mapping[str, float]]
) -> dict[str, tuple[float, float]]:
    """
    Per box-score stat, over the player-games in both the event rows and
    `box` ((game id, athlete id) -> stats): the share whose count matches
    the box exactly, and the events' total over the box's.
    """
    found = {stat: [0, 0, 0.0, 0.0] for stat in _BOX}
    for row in rows:
        line = box.get((row["game_id"], row["athlete_id"]))
        if line is None:
            continue
        for stat, counts in _BOX.items():
            mine = sum(row[c] for c in counts)
            seen = found[stat]
            seen[0] += 1
            seen[1] += mine == line[stat]
            seen[2] += mine
            seen[3] += line[stat]
    return {
        stat: (exact / n if n else 0.0, mine / box_ if box_ else 0.0)
        for stat, (n, exact, mine, box_) in found.items()
    }


def mark_box(
    rows: list[dict],
    box: Mapping[tuple[str, str], Mapping[str, float]],
    tolerance: float = 0.2,
) -> None:
    """
    Set each row's `box_ok`: whether its game's turnovers, fouls and
    defensive rebounds -- tied to a player by id in every season's feed --
    add up to within `tolerance` of the box score's, over the players in
    both. The feed before 2022-23 has games whose plays loop (4% of
    2014-15's, events up to twice the box) or stop early (6.5% of
    2017-18's); other seasons, 1% or fewer. The estimated rows drop the
    same games by their points. None where the game has no box score.
    """
    totals: dict[str, list[float]] = {}
    for row in rows:
        line = box.get((row["game_id"], row["athlete_id"]))
        if line is None:
            continue
        seen = totals.setdefault(row["game_id"], [0.0, 0.0])
        seen[0] += sum(row[count] for count in _WHOLE)
        seen[1] += sum(line[stat] for stat in _WHOLE.values())
    for row in rows:
        found = totals.get(row["game_id"])
        if found is None or found[1] == 0:
            row["box_ok"] = None
        else:
            row["box_ok"] = abs(found[0] / found[1] - 1) <= tolerance
