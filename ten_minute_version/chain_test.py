from dataclasses import replace

import numpy as np
import pyarrow as pa
import pytest

from .chain import (
    ChainSettings,
    PlayerState,
    SeasonEnd,
    carry,
    chained_priors,
    chained_table,
    development,
)
from .context import GameContext, Team
from .posterior import Reported
from .ratings import Design, Rows


def _design() -> tuple[Rows, Design]:
    keys = ("game_id", "stint", "offense_side", "offense_players", "defense_players")
    home, away = ["old", "new", "h3", "h4", "h5"], ["a1", "a2", "a3", "a4", "a5"]
    table = pa.Table.from_pylist(
        [
            dict(zip(keys, ("g1", 1, "home", home, away)))
            | {"possessions": 10, "points": 11},
            dict(zip(keys, ("g1", 1, "away", away, home)))
            | {"possessions": 10, "points": 9},
        ]
    )
    team = lambda t: Team(t, t, None, None, True)  # noqa: E731
    rows = Rows.build(
        table, {"g1": GameContext("g1", False, False, team("H"), team("A"))}
    )
    return rows, Design.build(rows, teams=True)


def _newcomer(p: str, season: int):
    return (0.03, 0.04**2), (0.02, 0.02**2)


SETTINGS = ChainSettings(
    box_sd=3.0, drift_sd=2.0, week_drift_sd=1.0, gap=2, team_penalty=30.0, sigma=1.0
)


def test_a_returner_starts_at_his_last_posterior_grown_and_widened():
    rows, design = _design()
    states = {"old": PlayerState(2023, 0.02, 0.01, 0.01**2, 0.015**2, 2)}
    grown = {1: (0.0, 0.0), 2: (0.004, 0.002), 3: (0.0, 0.0)}
    no_box = np.zeros(len(design.players), bool)
    mean, pen = chained_priors(
        design,
        2024,
        states,
        _newcomer,
        np.zeros(design.columns),
        no_box,
        SETTINGS,
        grown,
    )
    o = design.player_column(design.players.index("old"), "offense")
    assert np.isclose(mean[o], 0.024) and np.isclose(mean[o + 1], 0.012)
    assert np.isclose(pen[o], 1 / (0.01**2 + 0.01**2))
    n = design.player_column(design.players.index("new"), "offense")
    assert np.isclose(mean[n], 0.03) and np.isclose(pen[n], 1 / 0.04**2)
    t = design.team_column(0, "offense")
    assert pen[t] == 30.0 and pen[design.baseline] == 0.0


def test_a_generic_player_is_relearned_each_season_and_doesnt_grow():
    rows, design = _design()
    # "old" standing in for a division's generic player
    renamed = Design(
        design.row,
        design.col,
        design.val,
        design.rows,
        ["generic:d2" if p == "old" else p for p in design.players],
        design.teams,
    )
    states = {"generic:d2": PlayerState(2023, -0.3, -0.2, 0.001**2, 0.001**2, 9)}
    grown = {3: (0.004, 0.002)}
    no_box = np.zeros(len(design.players), bool)
    settings = replace(SETTINGS, generic_sd=5.0)
    mean, pen = chained_priors(
        renamed,
        2024,
        states,
        _newcomer,
        np.zeros(design.columns),
        no_box,
        settings,
        grown,
    )
    o = renamed.player_column(renamed.players.index("generic:d2"), "offense")
    assert np.isclose(mean[o], -0.3) and np.isclose(mean[o + 1], -0.2)
    assert np.isclose(pen[o], 1 / (0.001**2 + 0.01**2 + 0.05**2))
    # seasons since it last played don't make it a newcomer
    mean, pen = chained_priors(
        renamed, 2030, states, _newcomer, np.zeros(design.columns), no_box, settings
    )
    assert np.isclose(mean[o], -0.3)
    assert np.isclose(pen[o], 1 / (0.001**2 + 7 * 0.01**2 + 0.05**2))


def test_the_season_end_fit_drifts_by_drift_sd_and_the_weeks_by_week_drift_sd():
    rows, design = _design()
    states = {"old": PlayerState(2023, 0.02, 0.01, 0.01**2, 0.015**2, 2)}
    no_box = np.zeros(len(design.players), bool)
    settings = ChainSettings(drift_sd=2.0, week_drift_sd=1.0, sigma=1.0)
    o = design.player_column(design.players.index("old"), "offense")
    for end, drift in ((False, 0.01), (True, 0.02)):
        _, pen = chained_priors(
            design,
            2024,
            states,
            _newcomer,
            np.zeros(design.columns),
            no_box,
            settings,
            end=end,
        )
        assert np.isclose(pen[o], 1 / (0.01**2 + drift**2))


def test_a_player_away_longer_than_the_gap_starts_as_a_newcomer():
    rows, design = _design()
    states = {"old": PlayerState(2020, 0.05, 0.05, 0.0001, 0.0001, 3)}
    no_box = np.zeros(len(design.players), bool)
    mean, _ = chained_priors(
        design, 2024, states, _newcomer, np.zeros(design.columns), no_box, SETTINGS
    )
    assert np.isclose(
        mean[design.player_column(design.players.index("old"), "offense")], 0.03
    )


def test_the_box_score_joins_by_precision():
    rows, design = _design()
    box = np.zeros(design.columns)
    i = design.players.index("new")
    box[design.player_column(i, "offense")] = 0.06
    has = np.zeros(len(design.players), bool)
    has[i] = True
    mean, pen = chained_priors(design, 2024, {}, _newcomer, box, has, SETTINGS)
    o = design.player_column(i, "offense")
    w_new, w_box = 1 / 0.04**2, 1 / 0.03**2
    assert np.isclose(mean[o], (0.03 * w_new + 0.06 * w_box) / (w_new + w_box))
    assert np.isclose(pen[o], w_new + w_box)


def test_flat_puts_every_prior_at_one_width():
    rows, design = _design()
    no_box = np.zeros(len(design.players), bool)
    flat = ChainSettings(flat_sd=2.0, sigma=1.0)
    _, pen = chained_priors(
        design, 2024, {}, _newcomer, np.zeros(design.columns), no_box, flat
    )
    assert np.allclose(pen[: 2 * len(design.players)], 1 / 0.02**2)


def test_carry_moves_who_played_and_counts_their_seasons():
    rows, design = _design()
    b = np.zeros(design.columns)
    cov = np.eye(design.columns) * 1e-4
    possessions = design.column_squares(rows.possessions)
    states = {
        "old": PlayerState(2023, 0.0, 0.0, 1.0, 1.0, 2),
        "gone": PlayerState(2023, 0.1, 0.1, 1.0, 1.0, 1),
    }
    out = carry(design, b, np.diagonal(cov), possessions, 2024, states)
    assert out["old"].season == 2024 and out["old"].seasons == 3
    assert out["new"].seasons == 1
    assert out["gone"] == states["gone"]
    assert np.isclose(out["old"].var_offense, 1e-4)


def test_carry_leaves_the_teams_fifth_to_the_teams_column():
    rows, design = _design()
    b = np.zeros(design.columns)
    i = design.players.index("old")
    b[design.player_column(i, "offense")] = 0.02
    for j in range(len(design.teams)):
        b[design.team_column(j, "offense")] = 0.05
    cov = np.eye(design.columns) * 1e-4
    possessions = design.column_squares(rows.possessions)
    out = carry(design, b, np.diagonal(cov), possessions, 2024, {})
    assert np.isclose(out["old"].offense, 0.02)


def test_development_is_the_average_change_by_seasons_before():
    def row(p, o, d):
        return {"athlete_id": p, "offense": o, "defense": d, "offense_possessions": 500}

    tables = {
        2022: [row("a", 1.0, 0.0)],
        2023: [row("a", 2.0, 1.0), row("b", 0.0, 0.0)],
        2024: [row("a", 2.5, 1.0), row("b", 1.0, 2.0)],
    }
    grown = development(tables)
    assert np.allclose(grown[1], ((1.0 + 1.0) / 2 / 100, (1.0 + 2.0) / 2 / 100))
    assert np.allclose(grown[2], (0.005, 0.0))


def test_the_chained_table_reports_sds_per_100():
    rows, design = _design()
    reported = Reported.build(rows, design)
    cov = np.eye(design.columns) * 1e-4
    possessions = design.column_squares(rows.possessions)
    table = chained_table(
        "mens", 2024, design, reported, np.zeros(design.columns), cov, possessions, {}
    ).to_pylist()
    row = next(r for r in table if r["athlete_id"] == "new")
    assert row["team_id"] == "H"
    assert np.isclose(row["offense_sd"], 100 * np.sqrt(1e-4 * (1 + 1 / 25)), rtol=1e-5)


def test_the_home_edge_and_some_of_each_team_carry_from_the_last_season():
    rows, design = _design()
    no_box = np.zeros(len(design.players), bool)
    b = np.zeros(design.columns)
    b[design.baseline], b[design.home] = 1.0, 0.03
    b[design.team_column(design.teams.index("H"), "offense")] = 0.04
    last = SeasonEnd.of(design, b)
    assert last.teams["H"] == (0.04, 0.0)
    mean, pen = chained_priors(
        design,
        2024,
        {},
        _newcomer,
        np.zeros(design.columns),
        no_box,
        ChainSettings(program=0.5, sigma=1.0),
        last=last,
    )
    assert mean[design.baseline] == 1.0 and mean[design.home] == 0.03
    assert pen[design.baseline] == 0.0 and pen[design.home] == 0.0
    assert np.isclose(
        mean[design.team_column(design.teams.index("H"), "offense")], 0.02
    )


def test_an_outside_teams_column_is_as_wide_as_outside_sd():
    _, design = _design()
    box = np.zeros(design.columns)
    has = np.zeros(len(design.players), bool)
    team = design.teams[0]
    j = design.team_column(0, "offense")
    wide = ChainSettings(outside_sd=15.0)
    _, plain = chained_priors(design, 2024, {}, _newcomer, box, has, wide)
    _, out = chained_priors(
        design, 2024, {}, _newcomer, box, has, wide, outside=frozenset({team})
    )
    assert plain[j] == pytest.approx(wide.team_penalty)
    assert out[j] == pytest.approx(wide.sigma**2 / 0.15**2)
    # 0 is off
    flat = replace(SETTINGS, outside_sd=0.0)
    _, off = chained_priors(
        design, 2024, {}, _newcomer, box, has, flat, outside=frozenset({team})
    )
    assert off[j] == pytest.approx(flat.team_penalty)
