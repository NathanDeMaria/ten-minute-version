import numpy as np
import pyarrow as pa

from .context import GameContext, Team
from .posterior import Reported, covariance, gram, main_teams, split_half
from .ratings import Design, Rows, penalties


def _team(team_id: str) -> Team:
    return Team(team_id, team_id, None, None, True)


def _rows(n_games: int = 6, seed: int = 0) -> Rows:
    """A home team of eight and an away team of seven, in random fives."""
    rng = np.random.default_rng(seed)
    home = [f"h{i}" for i in range(8)]
    away = [f"a{i}" for i in range(7)]
    keys = ("game_id", "stint", "offense_side", "offense_players", "defense_players")
    out = []
    for g in range(n_games):
        for stint in range(12):
            h = list(rng.choice(home, 5, replace=False))
            a = list(rng.choice(away, 5, replace=False))
            n = int(rng.integers(2, 8))
            for side, offense, defense in (("home", h, a), ("away", a, h)):
                row = dict(zip(keys, (f"g{g}", stint, side, offense, defense)))
                out.append(row | {"possessions": n, "points": int(rng.poisson(n))})
    contexts = {
        f"g{g}": GameContext(f"g{g}", False, False, _team("H"), _team("A"))
        for g in range(n_games)
    }
    return Rows.build(pa.Table.from_pylist(out), contexts)


def _dense(design: Design) -> np.ndarray:
    x = np.zeros((design.rows, design.columns))
    np.add.at(x, (design.row, design.col), design.val)
    return x


def test_the_gram_is_xtwx():
    rows = _rows()
    design = Design.build(rows, teams=True)
    x, w = _dense(design), rows.possessions
    assert np.allclose(gram(design, w), x.T @ (w[:, None] * x))


def test_the_covariance_is_the_inverse_precision():
    rows = _rows()
    design = Design.build(rows, teams=True)
    x, w = _dense(design), rows.possessions
    pen = penalties(design, player=50.0, team=5.0)
    pen[design.baseline] = pen[design.home] = 1e-6
    expected = 0.9**2 * np.linalg.inv(x.T @ (w[:, None] * x) + np.diag(pen))
    assert np.allclose(covariance(design, w, pen, sigma=0.9), expected)


def test_a_reported_rating_carries_a_fifth_of_its_teams_variance():
    rows = _rows()
    design = Design.build(rows, teams=True)
    reported = Reported.build(rows, design)
    rng = np.random.default_rng(1)
    a = rng.normal(size=(design.columns, design.columns))
    cov = a @ a.T
    var_o, var_d, both = reported.variances(cov)
    i = design.players.index("h3")
    team = reported.team[i]
    c_o = np.zeros(design.columns)
    c_d = np.zeros(design.columns)
    c_o[design.player_column(i, "offense")], c_o[team] = 1.0, 0.2
    c_d[design.player_column(i, "defense")], c_d[team + 1] = 1.0, 0.2
    assert np.isclose(var_o[i], c_o @ cov @ c_o)
    assert np.isclose(var_d[i], c_d @ cov @ c_d)
    assert np.isclose(both[i], c_o @ cov @ c_d)


def test_a_players_team_is_the_one_he_played_most_for():
    rows = _rows()
    design = Design.build(rows, teams=True)
    teams = main_teams(rows, design)
    home = design.team_column(design.teams.index("H"), "offense")
    assert teams[design.players.index("h0")] == home


def test_split_halves_differ_as_much_as_the_model_says():
    """Simulated from the model itself -- ratings from their prior, points
    per possession with noise sigma^2 / possessions -- the halves' squared
    differences average what `split_half` predicts."""
    rows = _rows(n_games=8, seed=2)
    design = Design.build(rows, teams=True)
    reported = Reported.build(rows, design)
    sigma, lam = 1.0, 40.0
    pen = penalties(design, player=lam, team=10.0)
    pen[design.baseline] = pen[design.home] = 1.0
    half_a = np.array([int(g[1:]) % 2 == 0 for g in rows.game_id])
    prior = np.zeros(design.columns)
    rng = np.random.default_rng(5)
    squared = []
    for _ in range(300):
        theta = rng.normal(0, sigma / np.sqrt(pen))
        y = design.times(theta) + rng.normal(0, sigma / np.sqrt(rows.possessions))
        found = split_half(
            design,
            reported,
            rows.possessions,
            half_a,
            {"A": y, "B": y},
            {"A": prior, "B": prior},
            pen,
            "total",
            sigma,
        )
        squared.append(found.difference**2)
    ratio = np.mean(squared) / np.mean(found.predicted)
    assert 0.85 < ratio < 1.15
