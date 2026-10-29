from datetime import date

import numpy as np
import pyarrow as pa
import pytest

from .context import GameContext, Team
from .opponents import GENERIC
from .posterior import covariance, factor
from .ratings import Design, Rows, penalties
from .team_ratings import (
    TEAM_RATING_SCHEMA,
    Floor,
    home_margin,
    net_sds,
    shrunk_pace,
    team_rating_rows,
    team_shares,
    team_sides,
)

HOME = ["h1", "h2", "h3", "h4", "h5"]
BENCH = ["h1", "h2", "h3", "h4", "h6"]
AWAY = ["a1", "a2", "a3", "a4", "a5"]


def _rows() -> Rows:
    """Two games: H hosts A (h5 plays 6 of 10 possessions a side, h6 the
    other 4), then A hosts a generic side G."""
    keys = ("game_id", "stint", "offense_side", "offense_players", "defense_players")
    stints = [
        dict(zip(keys, ("g1", 1, "home", HOME, AWAY)))
        | {"possessions": 6, "points": 7},
        dict(zip(keys, ("g1", 1, "away", AWAY, HOME)))
        | {"possessions": 6, "points": 6},
        dict(zip(keys, ("g1", 2, "home", BENCH, AWAY)))
        | {"possessions": 4, "points": 4},
        dict(zip(keys, ("g1", 2, "away", AWAY, BENCH)))
        | {"possessions": 4, "points": 5},
        dict(zip(keys, ("g2", 1, "home", AWAY, [GENERIC])))
        | {"possessions": 8, "points": 9},
        dict(zip(keys, ("g2", 1, "away", [GENERIC], AWAY)))
        | {"possessions": 8, "points": 7},
    ]
    team = lambda t: Team(t, t, None, None, True)  # noqa: E731
    contexts = {
        "g1": GameContext("g1", False, False, team("H"), team("A")),
        "g2": GameContext("g2", False, False, team("A"), team("G")),
    }
    return Rows.build(pa.Table.from_pylist(stints), contexts)


def test_shares_are_each_players_part_of_his_teams_possessions():
    rows = _rows()
    floor = Floor.build(rows)
    shares = floor.shares(np.ones(len(rows), bool))
    assert shares["H"]["h1"] == pytest.approx(1.0)
    assert shares["H"]["h5"] == pytest.approx(0.6)
    assert shares["H"]["h6"] == pytest.approx(0.4)
    assert sum(shares["H"].values()) == pytest.approx(5.0)
    assert sum(shares["A"].values()) == pytest.approx(5.0)
    assert shares["G"] == {GENERIC: pytest.approx(1.0)}


def test_only_the_played_rows_count():
    rows = _rows()
    floor = Floor.build(rows)
    first = np.array([g == "g1" for g in rows.game_id])
    assert floor.games(first) == {"H": 1, "A": 1}
    assert floor.games(np.ones(len(rows), bool)) == {"H": 1, "A": 2, "G": 1}
    assert floor.pace(first) == {"H": pytest.approx(10.0), "A": pytest.approx(10.0)}
    assert floor.pace(np.ones(len(rows), bool))["A"] == pytest.approx(9.0)
    assert "G" not in floor.shares(first)


def test_a_team_is_as_played_else_as_projected_else_generic():
    played = {"H": {"h1": 5.0}}
    projected = {"H": {"h2": 5.0}, "A": {"a1": 5.0}}
    assert team_shares(played, projected, "H") == {"h1": 5.0}
    assert team_shares(played, projected, "A") == {"a1": 5.0}
    assert team_shares(played, projected, "G") == {GENERIC: 1.0}


def test_a_team_is_its_players_at_their_shares_plus_its_column():
    rows = _rows()
    design = Design.build(rows, teams=True)
    b = np.zeros(design.columns)
    h1 = design.players.index("h1")
    b[2 * h1], b[2 * h1 + 1] = 0.02, 0.01
    b[design.team_column(design.teams.index("H"), "offense")] = 0.05
    offense, defense = team_sides(design, b, "H", {"h1": 2.0, "nobody": 3.0})
    assert offense == pytest.approx(0.09) and defense == pytest.approx(0.02)


def test_the_home_margin_is_pace_times_the_net_gap_plus_the_edge_at_both_ends():
    home, away = (0.10, 0.02, 70.0), (0.05, 0.01, 66.0)
    assert home_margin(home, away, 0.01, neutral=True) == pytest.approx(68 * 0.06)
    assert home_margin(home, away, 0.01, neutral=False) == pytest.approx(68 * 0.08)


def test_rows_fit_the_schema():
    rows = _rows()
    design = Design.build(rows, teams=True)
    b = np.zeros(design.columns)
    floor = Floor.build(rows)
    every = np.ones(len(rows), bool)
    found = team_rating_rows(
        "mens",
        2025,
        date(2024, 11, 4),
        design,
        b,
        floor.teams,
        floor.shares(every),
        {},
        floor.pace(every),
        floor.games(every),
        {"H": "Home U"},
    )
    table = pa.Table.from_pylist(found, TEAM_RATING_SCHEMA)
    assert table.num_rows == 3 and table.column("team").to_pylist() == [
        None,
        None,
        "Home U",
    ]


def test_pace_is_averaged_with_two_games_at_the_prior():
    pace = shrunk_pace({"H": 60.0}, {"H": 2}, {"H": 70.0}, 68.0, ["H", "A"])
    assert pace == {"H": pytest.approx(65.0), "A": pytest.approx(68.0)}


def test_available_is_the_expected_net_where_given():
    rows = _rows()
    design = Design.build(rows, teams=True)
    b = np.zeros(design.columns)
    at = {p: i for i, p in enumerate(design.players)}
    b[2 * at["h5"]] = 0.05  # h5 is H's good one
    floor = Floor.build(rows)
    every = np.ones(len(rows), bool)
    found = team_rating_rows(
        "mens",
        2025,
        date(2024, 11, 4),
        design,
        b,
        ["H", "A"],
        floor.shares(every),
        {},
        floor.pace(every),
        floor.games(every),
        {},
        {"H": 0.01},
    )
    home, away = found
    assert home["net"] == pytest.approx(0.6 * 5)
    assert home["available"] == pytest.approx(1.0)
    assert away["available"] == pytest.approx(away["net"])


def test_a_nets_sd_is_its_columns_mix_through_the_full_covariance():
    rows = _rows()
    design = Design.build(rows, teams=True)
    w = rows.possessions.astype(float)
    pen = penalties(design, player=10.0, team=3.0)
    shares = {"h1": 1.0, "h5": 0.6, "h6": 0.4}
    found = net_sds(design, factor(design, w, pen), {"H": shares})
    at = {p: i for i, p in enumerate(design.players)}
    mix = np.zeros(design.columns)
    for p, s in shares.items():
        mix[2 * at[p]] += s
        mix[2 * at[p] + 1] += s
    j = design.teams.index("H")
    mix[design.team_column(j, "offense")] += 1
    mix[design.team_column(j, "defense")] += 1
    assert found["H"] == pytest.approx(np.sqrt(mix @ covariance(design, w, pen) @ mix))
    # fewer games, less sure
    first = w * np.array([g == "g1" for g in rows.game_id])
    none = net_sds(design, factor(design, first * 0, pen), {"H": shares})
    assert none["H"] > found["H"]
    assert net_sds(design, factor(design, w, pen), {}) == {}
