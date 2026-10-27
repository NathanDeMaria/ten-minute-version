"""
Who is likely to play a team's next game, from its games so far, and what
the team is worth over the ways it might line up.

Nobody publishes college basketball availability before tip -- ESPN's
summaries carry no injuries for it (the NBA's do), and the NCAA's reports
are a pilot for the 2026 tournaments alone -- so the box scores are what
there is. Two things in them say a player may not play:

- **he missed the last game(s).** The chance he misses the next one rises
  with the streak: about 55% after one, 70% after two, 95% after eight;
- **his last game was far below his usual minutes**, which is how an injury
  in the game looks: under a quarter of his usual share, 30-35% miss the
  next one, against 6% of everyone else.

Each league's rates over 2014-2025 are `OUT_CHANCE`.

A team's row then carries, beside `net`, its **expected** net
(`expected_net`): the team healthy -- each player at his recent share,
each game's weight halving every `HALF_LIFE` games back, a miss counting as
none (`healthy_shares`) -- averaged over who might be out, each player
independently at his chance of playing, each scenario's lineup its
available players at their shares rescaled to the team's total.

Scored against Cassandra's `glicko_players` on held-out seasons (2014+),
the gap between the expected net and the row's net is worth Brier -0.00020
mens and -0.00021 womens (t -3.8). Knowing who actually played would be
worth -0.0008 -- the rest is absences nobody could see coming.
"""

import itertools
from collections import defaultdict
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np

from .team_ratings import Floor, Shares

# Each game's weight in a player's recent share halves this many games back.
HALF_LIFE = 4.0

# A player at least this likely to miss is one whose absence `expected_net`
# enumerates, at most `MAX_EXACT` of them; the rest it takes to first order.
RISKY = 0.2
MAX_EXACT = 8

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


def healthy_shares(expected: Shares, games: Sequence[Shares]) -> Shares:
    """The team with everyone available: each player at his recent share
    (`recent_share`) in place of his season's, rescaled to the same total --
    a rotation as it has lately been, not as it was in November."""
    tilted = {}
    for player, share in expected.items():
        recent = recent_share(player, games)
        tilted[player] = recent if recent is not None and share > 0 else share
    total, kept = sum(expected.values()), sum(tilted.values())
    if kept <= 0:
        return dict(expected)
    return {p: s * total / kept for p, s in tilted.items()}


def chances_in(
    league: str, players: Sequence[str], games: Sequence[Shares]
) -> dict[str, float]:
    """Each player's chance of playing the team's next game (`OUT_CHANCE`)."""
    out = OUT_CHANCE[league]
    return {p: 1 - out[cell(p, games)] for p in players}


def expected_net(
    shares: Shares, values: Mapping[str, float], chances: Mapping[str, float]
) -> float:
    """The team's players' net averaged over who might be out.

    A scenario is a set of players available; its lineup is them at their
    `shares`, rescaled to the team's total, and its net is each one's share
    times his `values` (offense plus defense; 0 for a player without one).
    Each player is available independently at his `chances`.

    Every player has some chance of missing -- 6% even after a full game --
    so every scenario can't be counted. The ones whose chance of missing is
    at least `RISKY` (the `MAX_EXACT` likeliest) are enumerated exactly;
    the rest are taken to first order, each one's absence on its own, on
    top of each scenario of the risky ones: two of them missing together is
    a fraction of a percent. Against simulating every player at once, this
    agrees to a correlation of 0.98-0.99 on 2014-2025's team-games.
    """
    players = list(shares)
    s = np.array([shares[p] for p in players])
    v = np.array([values.get(p, 0.0) for p in players])
    risk = np.array([1 - chances.get(p, 1.0) for p in players])
    total = s.sum()
    order = np.argsort(-risk, kind="stable")[:MAX_EXACT]
    risky = order[risk[order] >= RISKY]
    safe = np.setdiff1d(np.arange(len(players)), risky)
    found, weight = 0.0, 0.0
    for pattern in itertools.product((True, False), repeat=len(risky)):
        available = np.ones(len(players), bool)
        available[risky] = pattern
        p = float(np.prod(np.where(pattern, 1 - risk[risky], risk[risky])))
        kept = s * available
        on = kept.sum()
        if p <= 0 or on <= 0:
            continue
        value = (kept * v).sum()
        net = value * total / on
        for k in safe:
            rest = on - s[k]
            if rest > 0:
                net += risk[k] * (
                    (value - s[k] * v[k]) * total / rest - value * total / on
                )
        found += p * net
        weight += p
    return found / weight if weight else float((s * v).sum())
