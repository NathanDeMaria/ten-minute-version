from datetime import date

import numpy as np
import pyarrow.csv

from .live import prior_design
from .players import (
    PLAYER_SCHEMA,
    TeamSides,
    fitted_team_sides,
    played_rows,
    players_path,
    prior_team_sides,
    waiting_rows,
    write_players,
)


def _chained(athlete: str, team: str | None = "T", total: float = 3.0) -> dict:
    return {
        "league": "mens",
        "season": 2025,
        "athlete_id": athlete,
        "athlete_name": athlete.upper(),
        "team_id": team,
        "offense": total - 1,
        "defense": 1.0,
        "total": total,
        "offense_sd": 1.0,
        "defense_sd": 1.0,
        "total_sd": 1.4,
        "offense_vs_team": 0.5,
        "defense_vs_team": 0.25,
    }


def test_a_player_who_played_takes_his_minutes_from_the_team_he_is_rated_on():
    minutes = [
        {
            "athlete_id": "a",
            "team_id": "OLD",
            "games": 3,
            "starts": 0,
            "minutes": 9.0,
            "share": 0.1,
        },
        {
            "athlete_id": "a",
            "team_id": "T",
            "games": 20,
            "starts": 18,
            "minutes": 600.0,
            "share": 0.75,
        },
    ]
    [row] = played_rows([_chained("a")], minutes, {"T": "Team"})
    assert row["team"] == "Team" and row["player"] == "A"
    assert row["games"] == 20 and row["share"] == 0.75
    assert row["vs_team"] == 0.75 and row["projected"] is False


def test_nobody_without_a_team_or_generic_gets_a_row():
    rows = played_rows([_chained("a", None), _chained("generic")], [], {})
    assert rows == []


def test_a_player_the_box_scores_miss_has_no_share_rather_than_none_of_it():
    [row] = played_rows([_chained("a")], [], {})
    assert row["share"] is None and row["games"] is None


def test_a_waiting_player_is_his_prior_plus_a_fifth_of_his_team():
    design = prior_design(["p", "q"], [])
    mean = np.array([0.03, 0.01, 0.0, 0.0, 0.0, 0.0][: design.columns])
    penalty = np.full(design.columns, 4.0)
    side = TeamSides(0.05, 0.10, 0.25, 0.5, 0.1)
    [row] = waiting_rows(
        "mens",
        2026,
        date(2026, 11, 5),
        [("p", "T")],
        {"T": {"p": 0.4}},
        design,
        mean,
        penalty,
        {"T": side},
        {"p": "Pat"},
        {"T": "Team"},
        sigma=1.0,
    )
    assert np.isclose(row["offense"], 100 * (0.03 + 0.01))
    assert np.isclose(row["defense"], 100 * (0.01 + 0.02))
    assert np.isclose(row["vs_team"], 100 * 0.04)
    assert np.isclose(row["offense_sd"], 100 * np.sqrt(0.25 + 0.25 / 25))
    assert np.isclose(
        row["total_sd"],
        100 * np.sqrt(0.25 + 0.25 / 25 + 0.25 + 0.5 / 25 + 2 * 0.1 / 25),
    )
    assert row["projected"] is True and row["share"] == 0.4 and row["games"] == 0


def test_a_team_without_a_column_leaves_a_waiting_player_his_own():
    design = prior_design(["p"], [])
    mean = np.zeros(design.columns)
    mean[0] = 0.02
    [row] = waiting_rows(
        "mens",
        2026,
        None,
        [("p", "T")],
        {},
        design,
        mean,
        np.full(design.columns, 1.0),
        {},
        {},
        {},
        sigma=1.0,
    )
    assert np.isclose(row["offense"], 2.0) and row["share"] == 0.0


def test_team_sides_read_a_fit_and_a_prior_alike():
    design = prior_design(["p"], ["T"])
    n = design.columns
    o = design.team_column(0, "offense")
    d = design.team_column(0, "defense")
    mean = np.zeros(n)
    mean[o], mean[d] = 0.01, 0.02
    penalty = np.full(n, 4.0)
    prior = prior_team_sides(design, mean, penalty, sigma=1.0)["T"]
    fitted = fitted_team_sides(design, mean, np.diag(1 / penalty))["T"]
    assert prior == fitted == TeamSides(0.01, 0.02, 0.25, 0.25, 0.0)


def test_the_file_reads_back_best_first(tmp_path):
    rows = played_rows(
        [_chained("low", total=1.0), _chained("high", total=9.0)], [], {"T": "Team"}
    )
    path = players_path(tmp_path, "mens", 2025)
    write_players(path, rows)
    assert path.name == "mens_players_2025.csv"
    table = pyarrow.csv.read_csv(
        path, convert_options=pyarrow.csv.ConvertOptions(column_types=PLAYER_SCHEMA)
    ).to_pylist()
    assert [r["athlete_id"] for r in table] == ["high", "low"]
