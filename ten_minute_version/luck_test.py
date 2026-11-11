from datetime import date

import numpy as np
import pytest

from .box_scores import BoxLine
from .context import GameContext, Team
from .flow import GameFlow
from .luck import (
    box_luck,
    centered,
    game_luck,
    luck_rows,
    merged_luck,
    read_luck,
    with_flow,
    with_season,
    write_luck,
)
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


def test_a_home_sites_luck_is_less_the_home_sides_average():
    team = lambda t: Team(t, t, None, None, True)  # noqa: E731
    contexts = {
        "g1": GameContext("g1", False, False, team("H"), team("A")),
        "g2": GameContext("g2", False, False, team("A"), team("H")),
        "g3": GameContext("g3", True, False, team("H"), team("A")),
    }
    luck = {"g1": 3.0, "g2": 1.0, "g3": 5.0, "g4": 9.0}
    # home sites average 2; the neutral site's stays; g4 has no context
    assert centered(luck, ["g1", "g3", "g4"], contexts) == {"g1": 1.0, "g3": 5.0}


def _line(game_id: str, team: str, tpm: float, tpa: float, ftm: float, fta: float):
    stats = {"tpm": tpm, "tpa": tpa, "ftm": ftm, "fta": fta}
    return BoxLine(game_id, team, f"{team}-{tpm}", True, None, stats)


def test_box_luck_is_threes_and_free_throws_against_the_lines_own_rates():
    lines = [
        _line("g1", "H", 3000, 6000, 0, 0),
        _line("g1", "H", 0, 0, 7000, 10000),  # a second player: a team adds up
        _line("g1", "A", 1000, 4000, 7000, 10000),
        _line("g2", "H", 1, 2, 0, 0),  # the other side has no line
    ]
    luck = box_luck(lines, _contexts() | {"g2": _contexts()["g1"]})
    three = (4001 + 0.34 * 2000) / (10002 + 2000)
    assert set(luck) == {"g1"}
    # the free throws are even; the threes are 3 points each
    assert luck["g1"] == pytest.approx(
        3 * ((3000 - three * 6000) - (1000 - three * 4000))
    )


def test_flow_fills_what_a_row_lacks_and_stays_as_first_known():
    rows = [{"game_id": "g1", "jumpers": None}, {"game_id": "g2", "jumpers": 1.0}]
    values = {"jumpers": {"g1": 3.0, "g2": 9.0}, "closing": {"g2": 2.0}}
    filled = with_flow(rows, values)
    assert filled == [
        {"game_id": "g1", "jumpers": 3.0},
        {"game_id": "g2", "jumpers": 1.0, "closing": 2.0},
    ]
    assert with_flow(rows, values, replace=True)[1]["jumpers"] == 9.0
    assert rows[0]["jumpers"] is None


def _season(known: list[dict], **kw) -> list[dict]:
    team = lambda t: Team(t, f"{t} U", None, None, True)  # noqa: E731
    contexts = {
        "g1": GameContext("g1", True, False, team("H"), team("A")),
        "g2": GameContext("g2", True, False, team("H"), team("A")),
    }
    flows = {"g1": GameFlow("g1", 10.0, 4.0, 7.0, (5, 10), (5, 10), (70, 60))}
    lines = [
        _line(g, t, tpm, 10, 10, 10)
        for g, t, tpm in (
            ("g1", "H", 6),
            ("g1", "A", 2),
            ("g2", "H", 2),
            ("g2", "A", 5),
        )
    ]
    return with_season(known, "mens", 2025, flows, lines, contexts, DAYS, **kw)


def test_a_season_adds_flow_to_the_shots_luck_and_box_luck_where_there_is_none(
    tmp_path,
):
    shots = luck_rows("mens", 2025, date(2025, 11, 5), {"g1": 2.0}, _contexts(), DAYS)
    other = [{**shots[0], "league": "womens"}]
    rows = _season(shots + other, as_of=date(2025, 11, 8))
    by_game = {(r["league"], r["game_id"]): r for r in rows}
    first, second = by_game["mens", "g1"], by_game["mens", "g2"]
    # the shot model's luck stays, with the game's flow beside it
    assert (first["luck"], first["source"], first["as_of"]) == (
        2.0,
        "shots",
        date(2025, 11, 5),
    )
    assert first["over_lead"] == pytest.approx(2.0)
    assert first["closing"] == pytest.approx(3.0)
    assert first["jumpers"] == pytest.approx(0.0)
    # the game without any gets its box score's: three threes short, at 3
    assert second["source"] == "box" and second["as_of"] == date(2025, 11, 8)
    assert second["luck"] == pytest.approx(-9.0)
    assert second.get("jumpers") is None
    # another league's game of the same id is not this season's to fill
    assert by_game["womens", "g1"].get("over_lead") is None
    path = tmp_path / "mens_luck.csv"
    write_luck(path, rows)
    assert {r["game_id"]: r["source"] for r in read_luck(path)} == {
        "g1": "shots",
        "g2": "box",
    }


def test_a_rebuilt_season_keeps_only_the_shots_luck():
    first = _season([], as_of=None)
    # with no shots luck both games are the box score's, a day after each
    assert [(r["source"], r["as_of"]) for r in first] == [
        ("box", date(2025, 11, 5)),
        ("box", date(2025, 11, 7)),
    ]
    stale = [{**r, "luck": 99.0, "over_lead": 99.0} for r in first]
    again = _season(stale, replace=True)
    assert [r["luck"] for r in again] == pytest.approx([r["luck"] for r in first])
    assert again[0]["over_lead"] == pytest.approx(2.0)
    # and without `replace`, what was filed stays
    assert _season(stale)[0]["luck"] == 99.0
