from .events import classify
from .lineups import on_floor
from .possessions import _elapsed, possessions
from .testing import (
    defensive_rebound,
    foul,
    free_throw,
    game,
    made_2,
    made_3,
    missed_2,
    offensive_rebound,
    period_end,
    starters_act,
    steal,
    sub_in,
    sub_out,
    technical,
    turnover,
)


def _possessions(*plays, league="mens"):
    played = game(*starters_act(), *plays, league=league)
    kinds = [classify(p) for p in played.plays]
    floors, _ = on_floor(played.plays, kinds)
    return possessions(played, kinds, floors)


def test_the_ball_changes_hands_when_the_other_side_makes_a_play():
    found, score = _possessions(
        made_2("home", "h1"),
        missed_2("away", "a1"),
        offensive_rebound("away", "a2"),
        missed_2("away", "a2"),
        defensive_rebound("home", "h3"),
        turnover("home", "h3"),
        steal("away"),
        made_3("away", "a4"),
    )
    assert [(p.offense, p.points, p.ended) for p in found] == [
        ("home", 2, "fg2"),
        ("away", 0, "miss"),
        ("home", 0, "turnover"),
        ("away", 3, "fg3"),
    ]
    assert found[1].oreb == 1 and found[1].fga2 == 2
    assert score.points == {"home": 2, "away": 3}


def test_an_and_one_is_one_possession_that_ended_in_a_basket():
    (only,), _ = _possessions(
        made_2("home", "h1"), foul("away", "a1"), free_throw("home", "h1")
    )
    assert (only.points, only.ended, only.fta, only.ftm) == (3, "fg2", 1, 1)


def test_a_possession_ending_at_the_line_ended_ft_made_or_missed():
    found, _ = _possessions(
        foul("away", "a1"),
        free_throw("home", "h1"),
        free_throw("home", "h1", made=False),
        defensive_rebound("away", "a2"),
    )
    assert (found[0].points, found[0].ended) == (1, "ft")


def test_free_throws_go_to_the_floor_at_the_foul():
    found, _ = _possessions(
        missed_2("home", "h1"),
        foul("away", "a1"),
        sub_out("home", "h1"),
        sub_in("home", "h6"),
        free_throw("home", "h6"),
        free_throw("home", "h6"),
    )
    home, _ = found[0].floor
    # h6 shot them, but h1 was on the floor when the foul was drawn
    assert "h1" in home and "h6" not in home


def test_technical_free_throws_are_nobodys_possession():
    found, score = _possessions(
        missed_2("home", "h1"),
        offensive_rebound("home", "h2"),
        technical("home", "h3"),
        free_throw("away", "a1"),
        free_throw("away", "a1"),
        missed_2("home", "h2"),
    )
    assert [p.offense for p in found] == ["home"]
    assert found[0].fga2 == 2
    assert score.technical_points == {"home": 0, "away": 2}
    assert score.points == {"home": 0, "away": 2}


def test_free_throws_after_a_personal_foul_too_are_not_technical():
    found, score = _possessions(
        missed_2("home", "h1"),
        defensive_rebound("away", "a1"),
        technical("home", "h3"),
        foul("home", "h3"),
        free_throw("away", "a1"),
        free_throw("away", "a1"),
    )
    assert [p.offense for p in found] == ["home", "away"]
    assert found[1].points == 2
    assert score.technical_points == {"home": 0, "away": 0}


def test_the_end_of_a_period_ends_the_possession():
    found, _ = _possessions(
        missed_2("home", "h1"),
        offensive_rebound("home", "h2"),
        period_end(),
        missed_2("home", "h2", period=2),
    )
    assert [(p.offense, p.ended, p.period) for p in found] == [
        ("home", "other", 1),
        ("home", "miss", 2),
    ]


def test_the_start_margin_is_the_offenses_lead():
    found, _ = _possessions(
        made_3("home", "h1"), made_2("away", "a1"), missed_2("home", "h1")
    )
    assert [p.start_margin for p in found] == [0, -3, 1]


def test_seconds_count_from_the_tip_through_overtime():
    assert _elapsed("mens", 1, 1200) == 0
    assert _elapsed("mens", 2, 600) == 1800
    assert _elapsed("mens", 3, 0) == 2700
    assert _elapsed("womens", 4, 0) == 2400
    assert _elapsed("womens", 5, 60) == 2640
