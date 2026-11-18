from datetime import date

import numpy as np
import pytest

from .availability import (
    BLOWOUT_WEIGHT,
    CAP,
    OUT_CHANCE,
    ROTATION,
    SeasonGames,
    TeamGames,
    capped,
    cell,
    chances_in,
    expected_net,
    game_shares,
    healthy_shares,
    lineup,
    out_chance,
    recent_share,
    role,
    situation,
)
from .team_ratings import Floor, Shares
from .team_ratings_test import _rows

FIVE = {"s1": 1.0, "s2": 1.0, "s3": 1.0, "s4": 1.0, "s5": 1.0}
DAYS = {"g1": date(2025, 11, 4), "g2": date(2025, 11, 6)}


def _games(shares: list[Shares], blowout: list[bool] | None = None) -> TeamGames:
    return TeamGames(
        shares, blowout or [False] * len(shares), [], np.zeros((0, 0)), np.zeros((0, 0))
    )


def test_a_games_shares_are_its_own():
    shares = game_shares(Floor.build(_rows()))
    assert shares[("g1", "H")]["h5"] == pytest.approx(0.6)
    assert shares[("g1", "H")]["h6"] == pytest.approx(0.4)
    assert sum(shares[("g1", "A")].values()) == pytest.approx(5.0)


def test_a_season_is_sliced_at_the_day():
    season = SeasonGames.build(Floor.build(_rows()), DAYS)
    assert len(season.before("A", date(2025, 11, 6)).shares) == 1  # g2 is on the day
    assert len(season.before("A", date(2025, 11, 7)).shares) == 2
    assert season.before("Nobody", date(2025, 11, 7)).shares == []


def test_who_plays_when_he_sits_is_his_substitute():
    # in g1, h6 comes on for every possession h5 sits
    games = SeasonGames.build(Floor.build(_rows()), DAYS).before("H", date(2025, 11, 5))
    players = ["h1", "h5", "h6"]
    gains, evidence = games.substitutes(players)
    assert gains[1, 2] == pytest.approx(1.0)  # h5 off: h6 on, always
    assert gains[1, 0] == pytest.approx(0.0)  # h1 plays either way
    assert evidence[1] == pytest.approx(8.0)  # 4 possessions a side


def test_the_situation_reads_the_streak_and_the_last_games_drop():
    played = {"p": 1.0}
    missed: dict[str, float] = {}
    hurt = {"p": 0.2}
    assert situation("p", _games([])) == "first game"
    assert situation("p", _games([missed])) == "not yet played"
    assert situation("p", _games([played])) == "played, no usual"
    assert situation("p", _games([played, played])) == "played, last at >=75"
    assert situation("p", _games([played, hurt])) == "played, last at <25"
    assert situation("p", _games([played, missed])) == "missed 1"
    assert situation("p", _games([played, *[missed] * 5])) == "missed 4-7"
    assert situation("p", _games([played, *[missed] * 9])) == "missed 8+"


def test_short_minutes_in_a_blowout_read_as_rest():
    hurt = {"p": 0.2}
    games = _games([{"p": 1.0}, hurt], [False, True])
    assert situation("p", games) == "played, last at <25 after a blowout"
    # a full game in a blowout is still a full game
    games = _games([{"p": 1.0}, {"p": 1.0}], [False, True])
    assert situation("p", games) == "played, last at >=75"


def test_the_role_is_the_recent_share():
    assert role("p", _games([])) is None
    assert role("p", _games([{"p": 0.05}] * 3)) == "deep bench"
    assert role("p", _games([{"p": 0.2}] * 3)) == "bench"
    assert role("p", _games([{"p": 0.4}] * 3)) == "fringe"
    assert role("p", _games([{"p": 0.8}] * 3)) == ROTATION
    assert role("q", _games([{"p": 0.8}] * 3)) == "deep bench"  # hasn't played


def test_every_league_has_the_same_cells_and_a_pooled_rate_for_each():
    assert set(OUT_CHANCE["mens"]) == set(OUT_CHANCE["womens"])
    for league in OUT_CHANCE.values():
        for rates in league.values():
            assert "any" in rates
            assert all(0 < r < 1 for r in rates.values())


def test_after_a_full_game_a_regular_is_far_likelier_to_play_than_a_walk_on():
    regular = out_chance("mens", "p", _games([{"p": 0.8}] * 3))
    walk_on = out_chance("mens", "p", _games([{"p": 0.05}] * 3))
    rates = OUT_CHANCE["mens"]["played, last at >=75"]
    assert regular == rates[ROTATION]
    assert walk_on == rates["deep bench"]
    assert regular < rates["any"] < walk_on
    assert cell("p", _games([{"p": 0.8}] * 3)) == ("played, last at >=75", ROTATION)


def test_a_thin_cell_takes_its_situations_pooled_rate():
    # a starter out four straight still reads as rotation, and too few
    # rotation players miss four to have their own rate
    games = _games([{"p": 1.2}] * 30 + [{}] * 4)
    assert situation("p", games) == "missed 4-7"
    assert role("p", games) == ROTATION
    assert ROTATION not in OUT_CHANCE["mens"]["missed 4-7"]
    assert out_chance("mens", "p", games) == OUT_CHANCE["mens"]["missed 4-7"]["any"]
    assert (
        out_chance("mens", "p", _games([])) == OUT_CHANCE["mens"]["first game"]["any"]
    )


def test_recent_games_count_most_and_blowouts_least():
    assert recent_share("p", _games([])) is None
    # a starter who has missed the last two: well below his season's 2/3
    recent = recent_share("p", _games([{"p": 1.0}] * 4 + [{}, {}]))
    assert recent is not None and recent < 2 / 3
    # sat the second half of a blowout: it barely counts
    damped = recent_share("p", _games([{"p": 0.8}, {"p": 0.4}], [False, True]))
    weights = np.array([0.5 ** (1 / 4), BLOWOUT_WEIGHT])
    assert damped == pytest.approx(np.average([0.8, 0.4], weights=weights))


def test_healthy_shares_follow_the_recent_rotation():
    games = _games([FIVE] * 5 + [{p: 1.25 for p in ("s1", "s2", "s3", "s4")}] * 2)
    healthy = healthy_shares(FIVE, games)
    assert sum(healthy.values()) == pytest.approx(5.0)
    assert healthy["s5"] < 1.0 < healthy["s1"]
    assert healthy_shares(FIVE, _games([])) == pytest.approx(FIVE)


def test_chances_come_from_the_cells():
    games = _games([FIVE, {p: 1.25 for p in ("s1", "s2", "s3", "s4")}])
    chances = chances_in("mens", ["s1", "s5"], games)
    assert chances["s5"] == pytest.approx(1 - out_chance("mens", "s5", games))
    assert chances["s5"] == pytest.approx(1 - OUT_CHANCE["mens"]["missed 1"]["fringe"])
    assert chances["s1"] == pytest.approx(
        1 - OUT_CHANCE["mens"]["played, last at >=75"][ROTATION]
    )


def test_nobody_plays_more_than_the_cap():
    out = capped(np.array([0.9, 0.9, 0.5, 0.5]), 3.5)
    assert out.max() == pytest.approx(CAP)
    assert out.sum() == pytest.approx(3.5)
    assert out[2] == pytest.approx(out[3])


def test_an_absent_players_minutes_go_to_his_substitute():
    shares = np.array([0.8, 0.5, 0.5])  # a starter and two backups
    absent = np.array([True, False, False])
    gains = np.zeros((3, 3))
    gains[0, 1] = 1.0  # the first backup is who plays when he sits
    sure = lineup(shares, absent, gains, np.array([1e9, 0.0, 0.0]))
    # 1.3 would be past the cap: held at it, the rest to the other
    assert sure == pytest.approx([0.0, CAP, 1.8 - CAP])
    # with nothing seen, in proportion to the shares left
    unseen = lineup(shares, absent, gains, np.zeros(3))
    assert unseen == pytest.approx([0.0, 0.9, 0.9])


def test_a_lineup_with_nobody_at_risk_is_the_healthy_net():
    values = {"s1": 0.1, "s2": 0.0, "s3": 0.0, "s4": 0.0, "s5": -0.1}
    sure = {p: 1.0 for p in FIVE}
    assert expected_net(FIVE, values, sure) == pytest.approx(0.0)


def test_a_star_who_might_sit_is_worth_his_chance_of_playing():
    # s1 (+0.1) is a coin flip; without him nobody's better than 0
    shares = {"s1": 0.8, "s2": 0.8, "s3": 0.8, "s4": 0.8, "s5": 0.8, "b": 0.9}
    values = {"s1": 0.1}
    chances = {p: 1.0 for p in shares} | {"s1": 0.5}
    assert expected_net(shares, values, chances) == pytest.approx(0.5 * 0.8 * 0.1)


def test_small_risks_are_counted_one_at_a_time():
    # nobody risky enough to enumerate: each 10% absence on its own
    shares = {"s1": 0.7, "s2": 0.7, "s3": 0.7, "s4": 0.7, "s5": 0.7, "b": 0.5}
    values = {"s1": 0.1}
    chances = {p: 0.9 for p in shares}
    healthy = 0.07
    others_out = []
    for k in shares:
        if k == "s1":
            others_out.append(0.0)
            continue
        left = {p: s for p, s in shares.items() if p != k}
        others_out.append(0.1 * 0.7 * 4.0 / sum(left.values()))
    want = healthy + sum(0.1 * (n - healthy) for n in others_out)
    assert expected_net(shares, values, chances) == pytest.approx(want)
