from datetime import date

import numpy as np
import pyarrow as pa

from .context import GameContext, Team
from .preseason import (
    History,
    MinutesModel,
    Spell,
    features,
    project_teams,
    roster,
    team_term,
)
from .ratings import Rows
from .recruiting import Recruit, RecruitModel

NO_RECRUITS = RecruitModel("mens", (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))


def _history(**kwargs) -> History:
    return History(
        kwargs.get("last_spells", {}),
        kwargs.get("last_ratings", {}),
        kwargs.get("recruits", {}),
        kwargs.get("played", {}),
    )


def test_a_returner_a_transfer_and_a_freshman():
    history = _history(
        last_spells={
            "stay": [Spell("stay", "A", 1.2, None)],
            "move": [Spell("move", "B", 0.9, None)],
        },
        last_ratings={"stay": (0.02, 0.01), "move": (0.01, 0.0)},
        recruits={"fresh": Recruit("fresh", 2024, 11, 95)},
        played={"stay": {2022, 2023}, "move": {2023}},
    )
    stay = features("stay", "A", 2024, history)
    move = features("move", "A", 2024, history)
    fresh = features("fresh", "A", 2024, history)
    # same team, other team, rated, rating per 100, ranked, rank score,
    # newcomer, then one/two/three-plus seasons played
    assert stay == [1.2, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert move[:2] == [0.0, 0.9] and move[7] == 1.0
    assert fresh == [0.0, 0.0, 0.0, 0.0, 1.0, 0.9, 1.0, 0.0, 0.0, 0.0]


def test_projected_shares_add_up_to_five_a_team():
    model = MinutesModel((0.1, 1.0))
    players = [("a", "A"), ("b", "A"), ("c", "B")]
    shares = model.project(players, np.array([[1.0], [-5.0], [0.2]]))
    assert np.isclose(shares[("a", "A")], 5.0)
    assert shares[("b", "A")] == 0.0
    assert np.isclose(shares[("c", "B")], 5.0)


def test_a_team_is_its_players_by_their_shares():
    history = _history(last_ratings={"a": (0.02, 0.01)})
    recruits = RecruitModel("mens", (0.001, 0.0, 0.0), (0.0, 0.0, 0.0))
    teams = project_teams({("a", "A"): 3.0, ("new", "A"): 2.0}, 2024, history, recruits)
    assert np.allclose(teams["A"], (3 * 0.02 + 2 * 0.001, 3 * 0.01))


def test_the_early_roster_is_who_has_played_by_then():
    spells = [
        Spell("a", "A", 1.0, date(2024, 11, 4)),
        Spell("b", "A", 1.0, date(2024, 12, 20)),
        Spell("c", "A", 1.0, None),
    ]
    assert roster(spells) == [("a", "A"), ("b", "A"), ("c", "A")]
    assert roster(spells, date(2024, 11, 17)) == [("a", "A")]


def test_a_generic_side_is_at_the_generic_rating():
    keys = (
        "game_id",
        "stint",
        "offense_side",
        "offense_players",
        "defense_players",
        "possessions",
        "points",
    )
    five = ["p1", "p2", "p3", "p4", "p5"]
    table = pa.Table.from_pylist(
        [
            dict(zip(keys, ("g1", 1, "home", five, ["generic"], 10, 15))),
            dict(zip(keys, ("g1", 1, "away", ["generic"], five, 10, 5))),
        ]
    )
    home, away = Team("H", "H", None, None, True), Team("X", "X", None, None, False)
    rows = Rows.build(table, {"g1": GameContext("g1", False, False, home, away)})
    term = team_term(rows, ["g1"], {"H": (0.1, 0.05)}, generic=(-0.2, -0.1))
    # home: 10 x (0.1 - -0.1); away, against: 10 x (-0.2 - 0.05)
    assert np.isclose(term[0], 10 * 0.2 - 10 * -0.25)
