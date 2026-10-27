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
(`expected_net`): the team healthy -- each player at his recent share, each
game's weight halving every `HALF_LIFE` games back, a blowout's at
`BLOWOUT_WEIGHT`, a miss counting as none (`healthy_shares`) -- averaged
over who might be out, each player independently at his chance of playing.
In each scenario the absent players' minutes go to whoever has played when
they sat (`TeamGames.substitutes`, shrunk toward their teammates' shares),
nobody above `CAP` of the floor (`lineup`).

Scored against Cassandra's `glicko_players` on held-out seasons (2014+),
the gap between the expected net and the row's net is worth Brier -0.00020
mens and -0.00021 womens (t -3.8). Knowing who actually played would be
worth -0.0008 -- the rest is absences nobody could see coming.
"""

import itertools
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np

from .team_ratings import Floor, Shares

# Each game's weight in a player's recent share halves this many games back.
HALF_LIFE = 4.0

# A player at least this likely to miss is one whose absence `expected_net`
# enumerates, at most `MAX_EXACT` of them; the rest it takes to first order.
RISKY = 0.2
MAX_EXACT = 8

# Nobody plays more than this share of a team's possessions in a lineup:
# a game's 99th percentile is 0.975.
CAP = 0.95

# Possessions of an absent player's minutes split in proportion to his
# teammates' shares, before his own on/off possessions say who takes them.
# Against what the others played in the games rotation players missed,
# 2014-2025: RMSE 0.175 mens / 0.182 womens, against 0.180 / 0.187 split in
# proportion (capped) and 0.182 / 0.190 uncapped.
SUB_PRIOR = 1000.0

# A game won or lost by more than this is a blowout, and its minutes count
# at `BLOWOUT_WEIGHT` in a player's recent share: whose minutes the bench
# played. Predicting the next close game's shares, RMSE 0.173 against 0.174
# mens, 0.180 against 0.181 womens.
BLOWOUT = 20.0
BLOWOUT_WEIGHT = 0.3

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


@dataclass(frozen=True)
class TeamGames:
    """A team's games before some day, in date order: each game's shares,
    whether it was a blowout, and who was on the floor with whom."""

    shares: list[Shares]
    blowout: list[bool]
    roster: list[str]
    # possessions teammate q was on the floor with p on ([p, q]), and with p
    # off: who plays when p sits
    with_on: np.ndarray
    with_off: np.ndarray

    def substitutes(self, players: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        """For `players` (in that order): how each one's minutes split among
        the others when he sits, from his on and off possessions so far --
        each teammate's rise in floor time with him off -- and how many off
        possessions that rests on."""
        n = len(players)
        gains, evidence = np.zeros((n, n)), np.zeros(n)
        at = {p: i for i, p in enumerate(self.roster)}
        index = np.array([at.get(p, -1) for p in players])
        known = index >= 0
        if not known.any():
            return gains, evidence
        k = index[known]
        on = self.with_on[np.ix_(k, k)]
        off = self.with_off[np.ix_(k, k)]
        on_possessions = np.maximum(np.diag(self.with_on)[k], 1e-9)
        off_possessions = self.with_off[k].sum(axis=1) / 5
        rise = np.clip(
            off / np.maximum(off_possessions, 1e-9)[:, None]
            - on / on_possessions[:, None],
            0,
            None,
        )
        np.fill_diagonal(rise, 0.0)
        gains[np.ix_(known, known)] = rise
        evidence[known] = off_possessions
        return gains, evidence


@dataclass(frozen=True)
class SeasonGames:
    """A season's games team by team, for `before` to slice at any day."""

    games: Mapping[str, list[str]]
    days: Mapping[str, list[date]]
    shares: Mapping[str, list[Shares]]
    blowout: Mapping[str, list[bool]]
    roster: Mapping[str, list[str]]
    # each team's on/off counts summed through each of its games
    with_on: Mapping[str, np.ndarray]
    with_off: Mapping[str, np.ndarray]

    @classmethod
    def build(cls, floor: Floor, day: Mapping[str, date]) -> "SeasonGames":
        rows = floor.rows
        per_game = game_shares(floor)
        margin: dict[str, float] = defaultdict(float)
        lines: dict[tuple[str, str], list[tuple[float, list[str], list[float]]]] = (
            defaultdict(list)
        )
        for i in range(len(rows)):
            g = rows.game_id[i]
            margin[g] += rows.points[i] * (1 if rows.offense_home[i] else -1)
            for team, five, weights in (
                (rows.offense_team[i], rows.offense[i], rows.offense_weights[i]),
                (rows.defense_team[i], rows.defense[i], rows.defense_weights[i]),
            ):
                if team:
                    lines[(team, g)].append((rows.possessions[i], five, weights))
        by_team: dict[str, list[str]] = defaultdict(list)
        for team, g in lines:
            by_team[team].append(g)
        games, days, shares, blowout, roster, with_on, with_off = (
            {},
            {},
            {},
            {},
            {},
            {},
            {},
        )
        for team, gs in by_team.items():
            gs.sort(key=lambda g: (day[g], g))
            players = sorted(
                {p for g in gs for _, five, _ in lines[(team, g)] for p in five}
            )
            at = {p: i for i, p in enumerate(players)}
            on_sum = np.zeros((len(gs), len(players), len(players)))
            off_sum = np.zeros_like(on_sum)
            for k, g in enumerate(gs):
                x = np.zeros((len(lines[(team, g)]), len(players)))
                possessions = np.zeros(len(x))
                for r, (n, five, weights) in enumerate(lines[(team, g)]):
                    possessions[r] = n
                    for p, w in zip(five, weights):
                        x[r, at[p]] += w
                x = x.clip(0, 1)
                on = (x * possessions[:, None]).T @ x
                off = ((1 - x) * possessions[:, None]).T @ x
                on_sum[k] = on + (on_sum[k - 1] if k else 0)
                off_sum[k] = off + (off_sum[k - 1] if k else 0)
            games[team] = gs
            days[team] = [day[g] for g in gs]
            shares[team] = [per_game.get((g, team), {}) for g in gs]
            blowout[team] = [abs(margin[g]) > BLOWOUT for g in gs]
            roster[team] = players
            with_on[team] = on_sum
            with_off[team] = off_sum
        return cls(games, days, shares, blowout, roster, with_on, with_off)

    def before(self, team: str, day: date) -> TeamGames:
        """`team`'s games before `day`."""
        if team not in self.games:
            return TeamGames([], [], [], np.zeros((0, 0)), np.zeros((0, 0)))
        k = bisect_left(self.days[team], day)
        n = len(self.roster[team])
        return TeamGames(
            self.shares[team][:k],
            self.blowout[team][:k],
            self.roster[team],
            self.with_on[team][k - 1] if k else np.zeros((n, n)),
            self.with_off[team][k - 1] if k else np.zeros((n, n)),
        )


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


def recent_share(player: str, games: TeamGames) -> float | None:
    """His share over a team's games so far, each game's weight halving every
    `HALF_LIFE` games back and a blowout's at `BLOWOUT_WEIGHT` -- whose
    minutes the walk-ons played -- a game he missed counting as none."""
    if not games.shares:
        return None
    past = np.array([g.get(player, 0.0) for g in games.shares])
    weights = 0.5 ** (np.arange(len(past))[::-1] / HALF_LIFE)
    weights = weights * np.where(games.blowout, BLOWOUT_WEIGHT, 1.0)
    return float(np.average(past, weights=weights))


def healthy_shares(expected: Shares, games: TeamGames) -> Shares:
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
    league: str, players: Sequence[str], games: TeamGames
) -> dict[str, float]:
    """Each player's chance of playing the team's next game (`OUT_CHANCE`)."""
    out = OUT_CHANCE[league]
    return {p: 1 - out[cell(p, games.shares)] for p in players}


def capped(shares: np.ndarray, total: float, cap: float = CAP) -> np.ndarray:
    """`shares` rescaled to `total`, nobody above `cap`: anyone the scaling
    takes past it is held there and the rest share what he can't play."""
    shares = np.asarray(shares, float)
    held = np.zeros(len(shares), bool)
    out = shares
    for _ in range(len(shares) + 1):
        free = shares * ~held
        if free.sum() <= 0:
            break
        out = np.where(held, cap, free * (total - cap * held.sum()) / free.sum())
        over = (out > cap) & ~held
        if not over.any():
            break
        held |= over
    return out


def lineup(
    shares: np.ndarray,
    absent: np.ndarray,
    gains: np.ndarray,
    evidence: np.ndarray,
) -> np.ndarray:
    """The team's shares with the `absent` out: each one's share split among
    those left by who has played when he sat (`TeamGames.substitutes`),
    `SUB_PRIOR` possessions of it in proportion to their shares instead,
    then capped (`capped`) at the team's total."""
    keep = ~absent
    left = shares * keep
    if left.sum() <= 0:
        return left
    proportional = left / left.sum()
    out = left.copy()
    for p in np.flatnonzero(absent):
        rise = gains[p] * keep
        split = rise / rise.sum() if rise.sum() > 0 else proportional
        weight = (evidence[p] * split + SUB_PRIOR * proportional) / (
            evidence[p] + SUB_PRIOR
        )
        out = out + shares[p] * weight
    return capped(out, shares.sum()) * keep


def expected_net(
    shares: Shares,
    values: Mapping[str, float],
    chances: Mapping[str, float],
    games: TeamGames | None = None,
) -> float:
    """The team's players' net averaged over who might be out.

    A scenario is a set of players available, each independently at his
    `chances`; its lineup is theirs (`lineup`): the absent players' shares
    split among those left by who has played when they sat, nobody above
    `CAP`. Its net is each one's share times his `values` (offense plus
    defense; 0 for a player without one).

    Every player has some chance of missing -- 6% even after a full game --
    so every scenario can't be counted. The ones whose chance of missing is
    at least `RISKY` (the `MAX_EXACT` likeliest) are enumerated exactly; the
    rest are taken to first order, each one's absence on its own, on top of
    each scenario of the risky ones: two of them missing together is a
    fraction of a percent. Against simulating every player at once, this
    agrees to a correlation of 0.98-0.99 on 2014-2025's team-games.
    """
    players = list(shares)
    s = np.array([shares[p] for p in players])
    v = np.array([values.get(p, 0.0) for p in players])
    risk = np.array([1 - chances.get(p, 1.0) for p in players])
    if games is not None:
        gains, evidence = games.substitutes(players)
    else:
        gains, evidence = np.zeros((len(s), len(s))), np.zeros(len(s))
    order = np.argsort(-risk, kind="stable")[:MAX_EXACT]
    risky = order[risk[order] >= RISKY]
    safe = np.setdiff1d(np.arange(len(players)), risky)
    found, weight = 0.0, 0.0
    for pattern in itertools.product((False, True), repeat=len(risky)):
        absent = np.zeros(len(players), bool)
        absent[risky] = pattern
        p = float(np.prod(np.where(pattern, risk[risky], 1 - risk[risky])))
        if p <= 0 or s[~absent].sum() <= 0:
            continue
        base = float(lineup(s, absent, gains, evidence) @ v)
        net = base
        for k in safe:
            if not risk[k] or s[~absent].sum() - s[k] <= 0:
                continue
            without = absent.copy()
            without[k] = True
            net += risk[k] * (float(lineup(s, without, gains, evidence) @ v) - base)
        found += p * net
        weight += p
    return found / weight if weight else float(s @ v)
