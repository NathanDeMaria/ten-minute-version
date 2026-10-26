"""
Who is likely to play a team's next game, from its games so far: the
shares a team's row is rated at, reweighted for who has been missing.

Nobody publishes college basketball availability before tip -- ESPN's
summaries carry no injuries for it (the NBA's do), and the NCAA's reports
are a pilot for the 2026 tournaments alone -- so the box scores are what
there is. Two things in them say a player may not play:

- **he missed the last game(s).** The chance he misses the next one rises
  with the streak: about 55% after one, 70% after two, 95% after eight;
- **his last game was far below his usual minutes**, which is how an injury
  in the game looks: under a quarter of his usual share, 30-35% miss the
  next one, against 6% of everyone else.

Measured on 2014-2025 (`OUT_CHANCE`, a cell's rate from every season).
A player's expected share is then his *recent* share -- each game's weight
halving every `HALF_LIFE` games back, a miss counting as none -- in place
of his season's, times his chance of playing, the team's rescaled to the
same total. Scored against Cassandra's `glicko_players` on held-out
seasons, what that does to a team's net (`available` beside `net` in the
row) is worth Brier -0.00020 mens and -0.00022 womens (t -3.9), at about
0.2 logit a point: not folded into `net`, whose weight in Cassandra is the
players' margin's, but carried apart for a weight of its own. In
Cassandra at weight 30 on this daily backfill, the replay agrees: -0.00024
mens, -0.00033 womens. Knowing who actually played would be worth -0.0008
-- the rest is absences nobody could see coming.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np

from .team_ratings import Floor, Shares

# Each game's weight in a player's recent share halves this many games back.
HALF_LIFE = 3.0

# The chance a player misses a team's next game, by what its games so far
# say of him (`cell`): each league's rate over 2014-2025.
OUT_CHANCE: Mapping[str, Mapping[str, float]] = {
    "mens": {
        "first game": 0.577,
        "not yet played": 0.791,
        "missed 1": 0.569,
        "missed 2": 0.711,
        "missed 3": 0.783,
        "missed 4-7": 0.855,
        "missed 8+": 0.946,
        "played, no usual": 0.181,
        "played, last at <25": 0.349,
        "played, last at 25-50": 0.257,
        "played, last at 50-75": 0.147,
        "played, last at >=75": 0.061,
    },
    "womens": {
        "first game": 0.409,
        "not yet played": 0.741,
        "missed 1": 0.532,
        "missed 2": 0.663,
        "missed 3": 0.743,
        "missed 4-7": 0.834,
        "missed 8+": 0.948,
        "played, no usual": 0.146,
        "played, last at <25": 0.302,
        "played, last at 25-50": 0.243,
        "played, last at 50-75": 0.141,
        "played, last at >=75": 0.061,
    },
}


def game_shares(floor: Floor) -> dict[tuple[str, str], dict[str, float]]:
    """(game, team) -> each player's share of the team's floor time in that
    game, a team's adding up to five, as `Floor.shares` does over many."""
    rows = floor.rows
    team_possessions: dict[tuple[str, str], float] = defaultdict(float)
    for g, offense, defense, n in zip(
        rows.game_id, rows.offense_team, rows.defense_team, rows.possessions
    ):
        for team in (offense, defense):
            if team:
                team_possessions[(g, team)] += n
    on: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    games = rows.game_id[floor.entry_row]
    weights = floor.entry_weight * rows.possessions[floor.entry_row]
    for g, k, weight in zip(games, floor.entry_pair, weights):
        team, player = floor.pairs[k]
        if team and weight > 0:
            on[(g, team)][player] += weight
    return {
        key: {p: v / team_possessions[key] for p, v in players.items()}
        for key, players in on.items()
    }


def histories(
    floor: Floor,
    day: Mapping[str, date],
    before: date,
    shares: Mapping[tuple[str, str], Shares] | None = None,
) -> dict[str, list[Shares]]:
    """Each team's games before `before`, in date order, as each game's
    shares (`game_shares`)."""
    shares = shares if shares is not None else game_shares(floor)
    found: dict[str, list[tuple[date, str, Shares]]] = defaultdict(list)
    for g in set(floor.home_team) | set(floor.away_team):
        if day[g] >= before:
            continue
        for team in (floor.home_team.get(g), floor.away_team.get(g)):
            if team:
                found[team].append((day[g], g, shares.get((g, team), {})))
    return {team: [s for _, _, s in sorted(games)] for team, games in found.items()}


def cell(player: str, games: Sequence[Shares]) -> str:
    """What a team's games so far say of `player` -- the key to `OUT_CHANCE`."""
    if not games:
        return "first game"
    past = [g.get(player, 0.0) for g in games]
    played = [x for x in past if x > 0]
    if not played:
        return "not yet played"
    streak = len(past) - 1 - max(i for i, x in enumerate(past) if x > 0)
    if streak:
        return f"missed {streak if streak <= 3 else '4-7' if streak <= 7 else '8+'}"
    if len(played) < 2:
        return "played, no usual"
    ratio = played[-1] / float(np.mean(played[:-1]))
    for top, name in ((0.25, "<25"), (0.5, "25-50"), (0.75, "50-75")):
        if ratio < top:
            return f"played, last at {name}"
    return "played, last at >=75"


def recent_share(player: str, games: Sequence[Shares]) -> float | None:
    """His share over a team's games so far, each game's weight halving every
    `HALF_LIFE` games back, a game he missed counting as none."""
    if not games:
        return None
    past = np.array([g.get(player, 0.0) for g in games])
    weights = 0.5 ** (np.arange(len(past))[::-1] / HALF_LIFE)
    return float(np.average(past, weights=weights))


def available_shares(league: str, expected: Shares, games: Sequence[Shares]) -> Shares:
    """`expected`, each player reweighted to his recent share and his chance
    of playing, rescaled to the same total: the shares a team's next game is
    likeliest to be played at."""
    chances = OUT_CHANCE[league]
    weighted = {}
    for player, share in expected.items():
        recent = recent_share(player, games)
        tilt = recent / share if recent is not None and share > 0 else 1.0
        weighted[player] = share * tilt * (1 - chances[cell(player, games)])
    total, kept = sum(expected.values()), sum(weighted.values())
    if kept <= 0:
        return dict(expected)
    return {p: s * total / kept for p, s in weighted.items()}
