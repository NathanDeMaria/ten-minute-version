import pytest

from .flow import GameFlow, flow_values, game_flow, jumper_rate
from .testing import free_throw, game, made_2, made_3, missed_2, period_end, play


def _jumper(side: str, made: bool = True, **kw):
    """A two-point jump shot, as the feed codes one."""
    return play("jumper", side, "1002" if made else "1005", "JumpShot", "p", **kw)


# A first play with twenty minutes on the clock: the game is in halves.
TIP = play("Jump Ball", period=1, clock=1200)


def test_a_games_margins_are_read_off_its_running_score():
    # Home leads by 2 from the 10:00 mark of the first half to the 10:00
    # mark of the second, then by 6 to the last minute, and wins by 4.
    flow = game_flow(
        game(
            TIP,
            _jumper("home", period=1, clock=600, score=(2, 0)),
            period_end(period=1, clock=0, score=(2, 0)),
            made_3("home", "h1", period=2, clock=600, score=(5, 0)),
            free_throw("home", "h1", period=2, clock=600, score=(6, 0)),
            made_2("away", "a1", period=2, clock=60, score=(6, 2)),
            period_end(period=2, clock=0, score=(6, 2)),
        )
    )
    assert flow is not None
    assert flow.points == (6, 2)
    assert flow.regulation == 4
    # 600s at 0, 1200s at 2, 540s at 6 and 60s at 4, over 2400
    assert flow.average_lead == pytest.approx((1200 * 2 + 540 * 6 + 60 * 4) / 2400)
    # with five minutes left it was 6
    assert flow.before_closing == 6
    assert flow.closing == -2
    assert flow.over_lead == pytest.approx(4 - 2 * flow.average_lead)


def test_only_two_point_jumpers_are_counted_as_jumpers():
    flow = game_flow(
        game(
            _jumper("home", score=(2, 0)),
            _jumper("home", made=False, score=(2, 0)),
            made_2("home", "h1", score=(4, 0)),  # a layup
            made_3("home", "h1", score=(7, 0)),
            missed_2("away", "a1", score=(7, 0)),  # a jumper, the helper's
            free_throw("away", "a1", score=(7, 1)),
        )
    )
    assert flow is not None
    assert flow.home_jumpers == (1, 2)
    assert flow.away_jumpers == (0, 1)


def test_an_uncoded_jumper_went_in_if_the_score_went_up_two():
    blank = lambda score: play("", "home", None, "JumpShot", score=score)  # noqa: E731
    flow = game_flow(game(blank((2, 0)), blank((2, 0))))
    assert flow is not None
    assert flow.home_jumpers == (1, 2)


def test_a_stale_score_is_held_at_the_most_it_has_shown():
    flow = game_flow(
        game(
            _jumper("home", period=1, clock=1000, score=(2, 0)),
            made_2("away", "a1", period=1, clock=900, score=(0, 2)),
            period_end(period=2, clock=0, score=(2, 2)),
        )
    )
    assert flow is not None
    # tied from the away basket on, not away by two
    assert flow.average_lead == pytest.approx(100 * 2 / 2400)


def test_overtime_is_outside_regulations_margin():
    flow = game_flow(
        game(
            TIP,
            _jumper("home", period=1, clock=600, score=(2, 0)),
            made_2("away", "a1", period=2, clock=10, score=(2, 2)),
            _jumper("home", period=3, clock=100, score=(4, 2)),
        )
    )
    assert flow is not None
    assert flow.regulation == 0
    assert flow.points == (4, 2)
    assert flow.home_jumpers == (2, 2)


def test_quarters_are_ten_minutes():
    # No play of the first period shows more than ten minutes left.
    flow = game_flow(
        game(
            _jumper("home", period=1, clock=600, score=(2, 0)),
            made_2("away", "a1", period=4, clock=0, score=(2, 2)),
            league="womens",
        )
    )
    assert flow is not None
    # ahead by two from the tip to the horn
    assert flow.average_lead == pytest.approx(2.0)
    assert flow.before_closing == 2


def test_a_game_nobody_scores_in_has_no_flow():
    assert game_flow(game(missed_2("home", "h1"))) is None
    assert game_flow(game(play("tip", period=2))) is None


def _flow(game_id: str, home: tuple[int, int], away: tuple[int, int]) -> GameFlow:
    return GameFlow(game_id, 10.0, 4.0, 7.0, home, away, (70, 60))


def test_jumpers_are_luck_against_the_games_own_rate():
    flows = {
        "g1": _flow("g1", (6000, 10000), (4000, 10000)),
        "g2": _flow("g2", (0, 0), (0, 0)),
    }
    # the prior's 2,000 attempts at 0.40 against 20,000 at 0.50
    rate = jumper_rate(flows.values(), "mens")
    assert rate == pytest.approx((10000 + 800) / 22000)
    values = flow_values(flows, "mens")
    # the attempts are even, so the rate drops out of home less away
    assert values["jumpers"]["g1"] == pytest.approx(2 * 2000)
    assert values["jumpers"]["g2"] == 0.0
    assert values["over_lead"]["g1"] == pytest.approx(2.0)
    assert values["closing"]["g1"] == pytest.approx(3.0)


def test_an_opening_days_rate_is_mostly_the_prior():
    assert jumper_rate([_flow("g1", (10, 10), (10, 10))], "womens") == pytest.approx(
        (20 + 0.36 * 2000) / 2020
    )
