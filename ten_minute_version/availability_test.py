from datetime import date

import pytest

from .availability import (
    OUT_CHANCE,
    cell,
    chances_in,
    expected_net,
    game_shares,
    healthy_shares,
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


def test_a_lineup_with_nobody_at_risk_is_the_healthy_net():
    values = {"s1": 0.1, "s2": 0.0, "s3": 0.0, "s4": 0.0, "s5": -0.1}
    sure = {p: 1.0 for p in FIVE}
    assert expected_net(FIVE, values, sure) == pytest.approx(0.0)


def test_a_star_who_might_sit_is_worth_his_chance_of_playing():
    # s1 (+0.1) is a coin flip; without him the other four play five shares
    values = {"s1": 0.1, "s2": 0.0, "s3": 0.0, "s4": 0.0, "s5": 0.0}
    chances = {p: 1.0 for p in FIVE} | {"s1": 0.5}
    assert expected_net(FIVE, values, chances) == pytest.approx(0.5 * 0.1)


def test_small_risks_are_counted_one_at_a_time():
    # nobody risky enough to enumerate: each 10% absence on its own
    values = {"s1": 0.1, "s2": 0.0, "s3": 0.0, "s4": 0.0, "s5": 0.0}
    chances = {p: 0.9 for p in FIVE}
    without_s1 = 0.0
    without_other = 0.1 * 5 / 4  # s1's share rescaled up
    want = (
        0.1 + 0.1 * (without_s1 - 0.1) + 3 * 0.1 * 0 + 4 * 0.1 * (without_other - 0.1)
    )
    assert expected_net(FIVE, values, chances) == pytest.approx(want)


def test_healthy_shares_follow_the_recent_rotation():
    games = [FIVE] * 5 + [{p: 1.25 for p in ("s1", "s2", "s3", "s4")}] * 2
    healthy = healthy_shares(FIVE, games)
    assert sum(healthy.values()) == pytest.approx(5.0)
    assert healthy["s5"] < 1.0 < healthy["s1"]
    assert healthy_shares(FIVE, []) == pytest.approx(FIVE)


def test_chances_come_from_the_cells():
    games = [FIVE, {p: 1.25 for p in ("s1", "s2", "s3", "s4")}]
    chances = chances_in("mens", ["s1", "s5"], games)
    assert chances["s5"] == pytest.approx(1 - OUT_CHANCE["mens"]["missed 1"])
    assert chances["s1"] == pytest.approx(
        1 - OUT_CHANCE["mens"]["played, last at >=75"]
    )
