from datetime import date

import numpy as np
import pyarrow as pa

from .carryover import GameTerms, compare, game_terms, variants
from .context import GameContext, Team
from .ratings import Rows

HOME = ["h1", "h2", "h3", "h4", "h5"]
AWAY = ["a1", "a2", "a3", "a4", "a5"]


def _rows() -> Rows:
    keys = (
        "game_id",
        "stint",
        "offense_side",
        "offense_players",
        "defense_players",
        "possessions",
        "points",
    )
    table = pa.Table.from_pylist(
        [
            dict(zip(keys, ("g1", 1, "home", HOME, AWAY, 10, 12))),
            dict(zip(keys, ("g1", 1, "away", AWAY, HOME, 10, 9))),
        ]
    )
    home, away = Team("H", "H", None, None, True), Team("A", "A", None, None, True)
    return Rows.build(table, {"g1": GameContext("g1", False, False, home, away)})


def test_a_games_terms_are_home_minus_away():
    teams = {"H": (0.05, 0.0), "A": (0.0, 0.02)}
    # h1 rated, everyone else new
    players = {"last": {"h1": (0.1, 0.0)}}
    terms = game_terms(_rows(), ["g1"], teams, players)
    assert terms.actual[0] == 3
    assert terms.terms["home"][0] == 20  # +1 for home's 10, -1 * -1 for away's 10
    # home offense: 10 * (0.05 - 0.02); away offense: -10 * (0.0 - 0.0)
    assert np.isclose(terms.terms["team"][0], 0.3)
    # home offense has h1's 0.1; away offense faces him on defense (rated 0)
    assert np.isclose(terms.terms["last"][0], 1.0)
    # home offense: 4 new less 5 new on defense = -1; away offense: 5 new less 4 = 1, times -1
    assert terms.terms["last, newcomers"][0] == -20


def test_weights_are_fitted_on_the_first_games_and_scored_on_the_rest():
    rng = np.random.default_rng(0)
    n = 400
    team = rng.normal(0, 10, n)
    players = rng.normal(0, 10, n)
    actual = 0.5 * team + 0.8 * players + rng.normal(0, 1, n)
    games = np.array([f"g{i}" for i in range(n)], object)
    terms = GameTerms(
        games,
        actual,
        {
            "home": np.zeros(n),
            "balance": np.ones(n),
            "team": team,
            "last": players,
            "last, newcomers": np.zeros(n),
        },
    )
    day = {
        g: date(2025, 11, 10) if i < 100 else date(2025, 12, 1)
        for i, g in enumerate(games)
    }
    results = {
        r.name: r for r in compare(terms, day, date(2025, 11, 16), variants(["last"]))
    }
    both = results["team + players, last"]
    assert both.rmse < results["team carryover"].rmse
    assert both.t_vs_team < 0
    assert abs(both.weights["last"] - 0.8) < 0.1
    assert len(both.errors) == 300
