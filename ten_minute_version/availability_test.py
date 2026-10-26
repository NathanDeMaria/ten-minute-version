from datetime import date

import pytest

from .availability import (
    OUT_CHANCE,
    available_shares,
    cell,
    game_shares,
    histories,
    recent_share,
)
from .team_ratings import Floor
from .team_ratings_test import _rows

FIVE = {"s1": 1.0, "s2": 1.0, "s3": 1.0, "s4": 1.0, "s5": 1.0}


def test_a_games_shares_are_its_own():
    shares = game_shares(Floor.build(_rows()))
    assert shares[("g1", "H")]["h5"] == pytest.approx(0.6)
    assert shares[("g1", "H")]["h6"] == pytest.approx(0.4)
    assert sum(shares[("g1", "A")].values()) == pytest.approx(5.0)


def test_a_history_is_the_games_before_the_day_in_order():
    floor = Floor.build(_rows())
    day = {"g1": date(2025, 11, 4), "g2": date(2025, 11, 6)}
    found = histories(floor, day, date(2025, 11, 6))
    assert set(found) == {"H", "A"}
    assert len(found["A"]) == 1  # g2, on the day, is not before it
    assert len(histories(floor, day, date(2025, 11, 7))["A"]) == 2


def test_the_cell_reads_the_streak_and_the_last_games_drop():
    played = {"p": 1.0}
    missed: dict[str, float] = {}
    hurt = {"p": 0.2}
    assert cell("p", []) == "first game"
    assert cell("p", [missed]) == "not yet played"
    assert cell("p", [played]) == "played, no usual"
    assert cell("p", [played, played]) == "played, last at >=75"
    assert cell("p", [played, hurt]) == "played, last at <25"
    assert cell("p", [played, missed]) == "missed 1"
    assert cell("p", [played, *[missed] * 5]) == "missed 4-7"
    assert cell("p", [played, *[missed] * 9]) == "missed 8+"
    assert set(OUT_CHANCE["mens"]) == set(OUT_CHANCE["womens"])


def test_recent_games_count_most():
    assert recent_share("p", []) is None
    # a starter who has missed the last two: well below his season's 2/3
    games = [{"p": 1.0}] * 4 + [{}, {}]
    recent = recent_share("p", games)
    assert recent is not None and recent < 2 / 3


def test_a_player_who_has_been_missing_gives_his_share_to_the_rest():
    games = [FIVE] * 5 + [{p: 1.25 for p in ("s1", "s2", "s3", "s4")}] * 2
    likely = available_shares("mens", FIVE, games)
    assert sum(likely.values()) == pytest.approx(5.0)
    assert likely["s5"] < 0.2
    assert likely["s1"] > 1.0


def test_with_nothing_to_go_on_the_shares_stand():
    assert available_shares("womens", FIVE, []) == pytest.approx(FIVE)
    assert available_shares("womens", FIVE, [FIVE] * 4) == pytest.approx(FIVE)
