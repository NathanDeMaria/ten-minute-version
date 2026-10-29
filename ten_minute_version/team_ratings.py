"""
Each team's rating as of a date, for a model that rates teams rather than
players to read: Cassandra's mens and womens predictors.

A chained fit (`chain`) says what every player is worth, and a game's
prediction from it is its stints': who was on the floor, for how many
possessions. Before the game neither is known. So a team is its players at
their expected shares of its floor time (`Floor.shares`) -- adding up to
five, as a lineup does -- plus its team column, and a game is played at its
teams' possessions per game so far (`Floor.pace`):

- **shares**: before a team's first game, `preseason.MinutesModel`'s
  projection over its roster; after, the share of its possessions each
  player has been on the floor for so far. A side the rows call generic
  (`opponents`) is the generic player at weight 1, as they have it;
- **pace**: a team's possessions per game so far, averaged with two
  games at last season's (`shrunk_pace`).

A game's home margin is then the two teams' paces averaged, times the
difference in their nets (offense plus defense, per possession), plus the
home edge at both ends at a home site.

Measured on 2014-2025's first ten weeks against the same with each game's
real lineups and possessions, the expected version costs little: mens
margin rmse 12.143 -> 12.234, womens 13.193 -> 13.291, and almost all of
that is the pace (at the game's real possessions, 12.137 and 13.203). And
it beats Cassandra's team-level Glicko on the same games: Brier 0.1573 ->
0.1526 mens, 0.1395 -> 0.1349 womens, a normal on the margin.
"""

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import numpy as np
import pyarrow as pa
from scipy.linalg import solve_triangular

from .opponents import GENERIC
from .posterior import SIGMA
from .ratings import Design, Rows

# player -> share of his team's floor time, a team's adding up to five
Shares = Mapping[str, float]


@dataclass(frozen=True)
class Floor:
    """Who was on the floor for whom in a season's rows, ready to count
    over any subset of its games."""

    rows: Rows
    entry_row: np.ndarray  # one entry per (row, side, listed player)
    entry_pair: np.ndarray  # its (team, player) pair
    entry_weight: np.ndarray
    pairs: list[tuple[str, str]]
    teams: list[str]
    offense_team: np.ndarray  # each row's index into `teams`, -1 for none
    defense_team: np.ndarray
    home_team: Mapping[str, str]  # game -> team, and away
    away_team: Mapping[str, str]

    @classmethod
    def build(cls, rows: Rows) -> "Floor":
        row, pair, weight = [], [], []
        pair_at: dict[tuple[str, str], int] = {}
        for i in range(len(rows)):
            for five, weights, team in (
                (rows.offense[i], rows.offense_weights[i], rows.offense_team[i]),
                (rows.defense[i], rows.defense_weights[i], rows.defense_team[i]),
            ):
                for p, w in zip(five, weights):
                    row.append(i)
                    pair.append(pair_at.setdefault((team, p), len(pair_at)))
                    weight.append(w)
        teams = sorted({t for t in (*rows.offense_team, *rows.defense_team) if t})
        team_at = {t: j for j, t in enumerate(teams)}
        home: dict[str, str] = {}
        away: dict[str, str] = {}
        for g, offense_home, team in zip(
            rows.game_id, rows.offense_home, rows.offense_team
        ):
            (home if offense_home else away)[g] = team
        return cls(
            rows,
            np.array(row, int),
            np.array(pair, int),
            np.array(weight, float),
            list(pair_at),
            teams,
            np.array([team_at.get(t, -1) for t in rows.offense_team]),
            np.array([team_at.get(t, -1) for t in rows.defense_team]),
            home,
            away,
        )

    def _team_sum(self, values: np.ndarray, side: np.ndarray) -> np.ndarray:
        keep = side >= 0
        return np.bincount(side[keep], weights=values[keep], minlength=len(self.teams))

    def shares(self, played: np.ndarray) -> dict[str, dict[str, float]]:
        """Each team's players' shares of its possessions (both ends) in the
        `played` rows."""
        weight = self.rows.possessions * played
        on_floor = self._team_sum(weight, self.offense_team) + self._team_sum(
            weight, self.defense_team
        )
        on = np.bincount(
            self.entry_pair,
            weights=self.entry_weight * weight[self.entry_row],
            minlength=len(self.pairs),
        )
        team_at = {t: j for j, t in enumerate(self.teams)}
        out: dict[str, dict[str, float]] = defaultdict(dict)
        for k in np.flatnonzero(on):
            team, player = self.pairs[k]
            if team:
                out[team][player] = float(on[k] / on_floor[team_at[team]])
        return dict(out)

    def games(self, played: np.ndarray) -> dict[str, int]:
        """How many of the `played` rows' games each team has played."""
        seen: dict[str, set[str]] = defaultdict(set)
        for g in set(self.rows.game_id[played.astype(bool)]):
            for team in (self.home_team.get(g), self.away_team.get(g)):
                if team:
                    seen[team].add(g)
        return {t: len(gs) for t, gs in seen.items()}

    def pace(self, played: np.ndarray) -> dict[str, float]:
        """Each team's possessions per game, on offense, in the `played` rows."""
        possessions = self._team_sum(self.rows.possessions * played, self.offense_team)
        return {
            t: float(possessions[self.teams.index(t)] / n)
            for t, n in self.games(played).items()
            if n
        }


# Games' worth of a team's prior pace (last season's, else the league's)
# its own games so far are averaged with: one partly recorded game against
# a school outside Division I can say 13 possessions.
PACE_PRIOR_GAMES = 2.0


def shrunk_pace(
    pace: Mapping[str, float],
    games: Mapping[str, int],
    prior: Mapping[str, float],
    typical: float,
    teams: list[str],
) -> dict[str, float]:
    """Each of `teams`' pace so far, averaged with its prior's
    `PACE_PRIOR_GAMES` games."""
    out = {}
    for team in teams:
        n = games.get(team, 0) if team in pace else 0
        before = prior.get(team, typical)
        out[team] = (n * pace.get(team, 0.0) + PACE_PRIOR_GAMES * before) / (
            n + PACE_PRIOR_GAMES
        )
    return out


def team_shares(
    observed: Mapping[str, Shares],
    projected: Mapping[str, Shares],
    team: str,
) -> Shares:
    """A team's shares: as it's played so far, else as projected, else (a
    team from outside Division I nobody projected) the generic player's."""
    return observed.get(team) or projected.get(team) or {GENERIC: 1.0}


def rated_shares(at: Mapping[str, int], shares: Shares) -> Shares:
    """`shares` as a design whose players are `at` can rate them: a team none of whose players it
    has is the generic player's. A team from outside Division I is projected
    from its box scores before it plays, but its players were never rated --
    every possession against it is the generic side's -- and summed over
    nobody, it came out an average Division I team: 0 where it's -34 per
    100 mens, -22 womens, in every opener against one."""
    if any(p in at for p in shares):
        return shares
    return {GENERIC: 1.0}


def team_sides(
    design: Design, b: np.ndarray, team: str, shares: Shares
) -> tuple[float, float]:
    """A team's (offense, defense) per possession under `b`: its players'
    columns at their shares (`rated_shares`), plus its team column. A
    player `design` doesn't have counts at 0."""
    at = {p: i for i, p in enumerate(design.players)}
    shares = rated_shares(at, shares)
    offense = sum(s * b[2 * at[p]] for p, s in shares.items() if p in at)
    defense = sum(s * b[2 * at[p] + 1] for p, s in shares.items() if p in at)
    if team in design.teams:
        j = design.teams.index(team)
        offense += b[design.team_column(j, "offense")]
        defense += b[design.team_column(j, "defense")]
    return float(offense), float(defense)


def net_sds(
    design: Design,
    lower: np.ndarray,
    sides: Mapping[str, Shares],
    sigma: float = SIGMA,
) -> dict[str, float]:
    """Each team's posterior sd of its net per possession at its shares in
    `sides` -- the same columns `team_sides` adds up -- from the fit's
    Cholesky factor `lower` (`posterior.factor`). Through the full
    covariance, not each column's alone: a team's players and its team
    column are fitted from the same possessions, and their errors lean
    against each other."""
    at = {p: i for i, p in enumerate(design.players)}
    teams = list(sides)
    mix = np.zeros((design.columns, len(teams)))
    for j, team in enumerate(teams):
        for p, s in rated_shares(at, sides[team]).items():
            if p in at:
                mix[2 * at[p], j] += s
                mix[2 * at[p] + 1, j] += s
        if team in design.teams:
            k = design.teams.index(team)
            mix[design.team_column(k, "offense"), j] += 1.0
            mix[design.team_column(k, "defense"), j] += 1.0
    if not teams:
        return {}
    z = solve_triangular(lower, mix, lower=True, check_finite=False)
    return dict(zip(teams, (sigma * np.sqrt((z**2).sum(axis=0))).tolist()))


def home_margin(
    home: tuple[float, float, float],
    away: tuple[float, float, float],
    edge: float,
    neutral: bool,
) -> float:
    """A game's expected home margin from each team's (offense, defense,
    pace): both paces averaged, times the difference in nets, plus the home
    edge at both ends unless neutral."""
    pace = (home[2] + away[2]) / 2
    net = (home[0] + home[1]) - (away[0] + away[1])
    return pace * (net + (0.0 if neutral else 2 * edge))


TEAM_RATING_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        # for games on or after this date, until the next row's; fitted on
        # the games before it
        pa.field("as_of", pa.date32(), nullable=False),
        pa.field("team_id", pa.string(), nullable=False),
        pa.field("team", pa.string()),
        # per 100 possessions; net is offense plus defense
        pa.field("offense", pa.float32(), nullable=False),
        pa.field("defense", pa.float32(), nullable=False),
        pa.field("net", pa.float32(), nullable=False),
        # possessions per game, each team's
        pa.field("pace", pa.float32(), nullable=False),
        pa.field("games", pa.int16(), nullable=False),
        # the league's, per 100, at each end
        pa.field("home_edge", pa.float32(), nullable=False),
        # net per 100 averaged over who might miss the team's next game
        # (`availability.expected_net`); `net` where nothing says otherwise
        pa.field("available", pa.float32()),
        # per 100, how sure the fit is of `net`: its posterior sd at the
        # row's shares (`net_sds`); null where not worked out
        pa.field("net_sd", pa.float32()),
    ]
)


def team_rating_rows(
    league: str,
    season: int,
    as_of: date,
    design: Design,
    b: np.ndarray,
    teams: list[str],
    observed: Mapping[str, Shares],
    projected: Mapping[str, Shares],
    pace: Mapping[str, float],
    games: Mapping[str, int],
    names: Mapping[str, str | None],
    available: Mapping[str, float] | None = None,
    sds: Mapping[str, float] | None = None,
) -> list[dict]:
    """One `TEAM_RATING_SCHEMA` row per team in `teams`, `pace` already
    filled in from last season's where a team has none of its own,
    `available` its net per possession over who might miss its next game,
    where known (`availability.expected_net`), and `sds` its net's posterior
    sd per possession, where known (`net_sds`)."""
    edge = float(b[design.home]) * 100
    out = []
    for team in teams:
        offense, defense = team_sides(
            design, b, team, team_shares(observed, projected, team)
        )
        ready = (available or {}).get(team, offense + defense)
        sd = (sds or {}).get(team)
        out.append(
            {
                "league": league,
                "season": season,
                "as_of": as_of,
                "team_id": team,
                "team": names.get(team),
                "offense": offense * 100,
                "defense": defense * 100,
                "net": (offense + defense) * 100,
                "pace": pace[team],
                "games": games.get(team, 0),
                "home_edge": edge,
                "available": ready * 100,
                "net_sd": None if sd is None else sd * 100,
            }
        )
    return out
