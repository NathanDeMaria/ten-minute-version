from datetime import date

import numpy as np
import pytest

from .context import GameContext, Team
from .luck import game_luck, luck_rows, merged_luck, read_luck, write_luck
from .ratings import Targets
from .team_ratings_test import _rows

DAYS = {"g1": date(2025, 11, 4), "g2": date(2025, 11, 6)}


def _targets(n: int, three: list[float], made_three: list[float]) -> Targets:
    """Per possession: the threes as expected of their shooters and as they
    went in; twos and free throws exactly as expected."""
    zero = np.zeros(n)
    return Targets(
        actual={2: zero, 3: np.array(made_three), 1: zero},
        expected={2: zero, 3: np.array(three), 1: zero},
        shooter={2: zero, 3: np.array(three), 1: zero},
    )


def test_a_games_luck_is_the_home_sides_less_the_aways_in_points():
    rows = _rows()
    # g1's rows: home 6 poss, away 6, home 4, away 4; g2's: home 8, away 8.
    # The home side hits 0.5 per possession more than its shooters' chance in
    # its first stint, the away side 0.25 more in its second.
    made = [0.5, 0.0, 0.0, 0.25, 0.0, 0.0]
    luck = game_luck(rows, _targets(len(rows), [0.0] * 6, made), ["g1", "g2"])
    assert luck["g1"] == pytest.approx(0.5 * 6 - 0.25 * 4)
    assert luck["g2"] == pytest.approx(0.0)
    assert set(game_luck(rows, _targets(len(rows), [0.0] * 6, made), ["g2"])) == {"g2"}


def test_twos_are_not_luck():
    rows = _rows()
    n = len(rows)
    zero = np.zeros(n)
    fitted = Targets(
        actual={2: np.ones(n), 3: zero, 1: zero},
        expected={2: zero, 3: zero, 1: zero},
        shooter={2: zero, 3: zero, 1: zero},
    )
    assert game_luck(rows, fitted, ["g1"])["g1"] == pytest.approx(0.0)


def _contexts() -> dict[str, GameContext]:
    team = lambda t: Team(t, f"{t} U", None, None, True)  # noqa: E731
    return {"g1": GameContext("g1", False, False, team("H"), team("A"))}


def test_a_game_without_a_context_is_left_out():
    rows = luck_rows(
        "mens", 2025, date(2025, 11, 7), {"g1": 2.0, "g2": 1.0}, _contexts(), DAYS
    )
    assert [r["game_id"] for r in rows] == ["g1"]
    assert rows[0]["home"] == "H U" and rows[0]["away_team_id"] == "A"
    assert rows[0]["luck"] == 2.0 and rows[0]["date"] == DAYS["g1"]


def test_a_games_luck_stays_as_first_known(tmp_path):
    first = luck_rows("mens", 2025, date(2025, 11, 5), {"g1": 2.0}, _contexts(), DAYS)
    later = luck_rows("mens", 2025, date(2025, 11, 7), {"g1": 9.0}, _contexts(), DAYS)
    path = tmp_path / "mens_luck.csv"
    write_luck(path, merged_luck([], first))
    write_luck(path, merged_luck(read_luck(path), later))
    (row,) = read_luck(path)
    assert row["luck"] == pytest.approx(2.0)
    assert row["as_of"] == date(2025, 11, 5)
    assert read_luck(tmp_path / "nothing.csv") == []
