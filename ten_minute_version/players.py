"""
Every player of a season, for reading rather than predicting: the file
behind invisible-string's Players page, `{league}_players_{season}.csv`.

A player who has played is his chained rating (`chain.chained_table`) and his
box-score minutes for his main team. Minutes rather than possessions say how
much he played, because in a season with estimated lineups possessions are
fractional weights, and read low.

A player on a roster who hasn't played is his prior, at the minutes the
preseason projection gives him (`preseason.MinutesModel`) and marked
`projected`: all of a table before the season, a redshirt or a late arrival
after it. His own columns have seen nothing, so they're independent of
whatever his team's have seen.

Ratings are reported as everywhere else, his column plus a fifth of his
team's; `vs_team` is his column alone.

`play_chance` is his chance of playing the team's next game
(`availability.OUT_CHANCE`, from his missed games and his last game's minutes),
the chance the team's `available` averages over. Nobody reports college
injuries, so it's an estimate from the box scores, and only a morning's: a
finished season has no next game.
"""

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv

from .availability import OUT_CHANCE, SeasonGames, cell
from .divisions import is_generic
from .posterior import SIGMA
from .ratings import Design

# Where each league's table starts: seasons with lineups or estimated lineups
# for most games. Play-by-play reaches about a quarter of womens games before
# 2019-20.
FIRST_SEASON = {"mens": 2013, "womens": 2019}

PLAYER_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        # the day a morning's file stands for; none on a finished season
        pa.field("as_of", pa.date32()),
        pa.field("athlete_id", pa.string(), nullable=False),
        pa.field("player", pa.string()),
        pa.field("team_id", pa.string(), nullable=False),
        pa.field("team", pa.string()),
        # hasn't played: a prior, and a projected share
        pa.field("projected", pa.bool_(), nullable=False),
        # per 100 possessions
        pa.field("offense", pa.float32(), nullable=False),
        pa.field("defense", pa.float32(), nullable=False),
        pa.field("total", pa.float32(), nullable=False),
        pa.field("offense_sd", pa.float32(), nullable=False),
        pa.field("defense_sd", pa.float32(), nullable=False),
        pa.field("total_sd", pa.float32(), nullable=False),
        pa.field("vs_team", pa.float32(), nullable=False),
        # empty where the box scores don't have him
        pa.field("games", pa.int32()),
        pa.field("starts", pa.int32()),
        pa.field("minutes", pa.float32()),
        # of the time his team was on the floor: 1.0 played every minute
        pa.field("share", pa.float32()),
        # of playing the team's next game
        pa.field("play_chance", pa.float32()),
    ]
)


@dataclass(frozen=True)
class TeamSides:
    """A team's two columns and how sure they are, per possession."""

    offense: float
    defense: float
    var_offense: float
    var_defense: float
    both: float


def fitted_team_sides(
    design: Design, b: np.ndarray, cov: np.ndarray
) -> dict[str, TeamSides]:
    """Every team in a fit, from its posterior covariance."""
    out = {}
    for k, team in enumerate(design.teams):
        o, d = design.team_column(k, "offense"), design.team_column(k, "defense")
        out[team] = TeamSides(
            float(b[o]),
            float(b[d]),
            float(cov[o, o]),
            float(cov[d, d]),
            float(cov[o, d]),
        )
    return out


def prior_team_sides(
    design: Design, mean: np.ndarray, penalty: np.ndarray, sigma: float = SIGMA
) -> dict[str, TeamSides]:
    """Every team in a design with no rows, whose columns are independent."""
    out = {}
    for k, team in enumerate(design.teams):
        o, d = design.team_column(k, "offense"), design.team_column(k, "defense")
        out[team] = TeamSides(
            float(mean[o]),
            float(mean[d]),
            float(sigma**2 / penalty[o]),
            float(sigma**2 / penalty[d]),
            0.0,
        )
    return out


def play_chances(
    league: str,
    pairs: Iterable[tuple[str, str]],
    games: SeasonGames | None,
    as_of: date,
) -> dict[tuple[str, str], float]:
    """Each (player, team)'s chance of playing the team's next game after
    `as_of`, from its games before it; `games` is None before anyone's played."""
    out = OUT_CHANCE[league]
    found: dict[str, list] = {}
    chances = {}
    for player, team in pairs:
        if team not in found:
            found[team] = games.before(team, as_of).shares if games else []
        chances[(player, team)] = 1 - out[cell(player, found[team])]
    return chances


def played_rows(
    chained: Iterable[Mapping],
    minutes: Iterable[Mapping],
    teams: Mapping[str, str | None],
    as_of: date | None = None,
    chances: Mapping[tuple[str, str], float] | None = None,
) -> list[dict]:
    """One row per rated player, from his `chain.CHAINED_SCHEMA` row and his
    `box_scores.MINUTES_SCHEMA` row for the same team."""
    box = {(m["athlete_id"], m["team_id"]): m for m in minutes}
    out = []
    for r in chained:
        if r["team_id"] is None or is_generic(r["athlete_id"]):
            continue
        m = box.get((r["athlete_id"], r["team_id"]), {})
        out.append(
            {
                "league": r["league"],
                "season": r["season"],
                "as_of": as_of,
                "athlete_id": r["athlete_id"],
                "player": r["athlete_name"],
                "team_id": r["team_id"],
                "team": teams.get(r["team_id"]),
                "projected": False,
                "offense": r["offense"],
                "defense": r["defense"],
                "total": r["total"],
                "offense_sd": r["offense_sd"],
                "defense_sd": r["defense_sd"],
                "total_sd": r["total_sd"],
                "vs_team": r["offense_vs_team"] + r["defense_vs_team"],
                "games": m.get("games"),
                "starts": m.get("starts"),
                "minutes": m.get("minutes"),
                "share": m.get("share"),
                "play_chance": (chances or {}).get((r["athlete_id"], r["team_id"])),
            }
        )
    return out


def waiting_rows(
    league: str,
    season: int,
    as_of: date | None,
    roster: Sequence[tuple[str, str]],
    shares: Mapping[str, Mapping[str, float]],
    design: Design,
    mean: np.ndarray,
    penalty: np.ndarray,
    sides: Mapping[str, TeamSides],
    names: Mapping[str, str | None],
    teams: Mapping[str, str | None],
    sigma: float = SIGMA,
    chances: Mapping[tuple[str, str], float] | None = None,
) -> list[dict]:
    """One row per (player, team) in `roster`, at the priors of `design`, a
    design with no rows (`live.prior_design`) and his team at `sides`, by
    team: what it's fitted to so far, if it has played."""
    at = {p: i for i, p in enumerate(design.players)}
    out = []
    for player, team in roster:
        i = at[player]
        o, d = 2 * i, 2 * i + 1
        own_o, own_d = float(mean[o]), float(mean[d])
        var_o, var_d = sigma**2 / penalty[o], sigma**2 / penalty[d]
        side = sides.get(team)
        offense, defense = own_o, own_d
        both = 0.0
        if side is not None:
            offense += side.offense / 5
            defense += side.defense / 5
            var_o += side.var_offense / 25
            var_d += side.var_defense / 25
            both = side.both / 25
        out.append(
            {
                "league": league,
                "season": season,
                "as_of": as_of,
                "athlete_id": player,
                "player": names.get(player),
                "team_id": team,
                "team": teams.get(team),
                "projected": True,
                "offense": 100 * offense,
                "defense": 100 * defense,
                "total": 100 * (offense + defense),
                "offense_sd": 100 * np.sqrt(var_o),
                "defense_sd": 100 * np.sqrt(var_d),
                "total_sd": 100 * np.sqrt(var_o + var_d + 2 * both),
                "vs_team": 100 * (own_o + own_d),
                "games": 0,
                "starts": 0,
                "minutes": 0.0,
                "share": shares.get(team, {}).get(player, 0.0),
                "play_chance": (chances or {}).get((player, team)),
            }
        )
    return out


def players_path(out: str | Path, league: str, season: int) -> Path:
    return Path(out) / f"{league}_players_{season}.csv"


def write_players(path: Path, rows: Iterable[dict]) -> None:
    ordered = sorted(rows, key=lambda r: (-r["total"], r["athlete_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    pyarrow.csv.write_csv(pa.Table.from_pylist(ordered, schema=PLAYER_SCHEMA), path)
