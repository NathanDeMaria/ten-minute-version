from .events import classify
from .lineups import on_floor
from .testing import (
    AWAY,
    HOME,
    made_2,
    missed_2,
    play,
    starters_act,
    sub_in,
    sub_out,
    technical,
)


def _floors(*plays):
    kinds = [classify(p) for p in plays]
    return on_floor(plays, kinds)


def test_the_starters_are_whoever_appears_first_other_than_by_subbing_in():
    floors, report = _floors(
        *starters_act(), sub_in("home", "h6"), sub_out("home", "h1")
    )
    assert floors[9] == (frozenset(HOME), frozenset(AWAY))
    assert report.starters_found


def test_a_substitution_swaps_one_player_for_another():
    floors, report = _floors(
        *starters_act(),
        sub_out("home", "h1"),
        sub_in("home", "h6"),
        missed_2("home", "h6"),
    )
    assert floors[-1][0] == frozenset({"h2", "h3", "h4", "h5", "h6"})
    assert report.repairs == 0


def test_a_player_who_plays_while_benched_goes_on_and_the_next_to_check_in_comes_off():
    """
    h6 subs in and back out, then takes a shot: the feed dropped his second
    "subbing in" and somebody's "subbing out". h3 checking in later is what
    says h3 was the one who'd gone off.
    """
    floors, report = _floors(
        *starters_act(),
        sub_out("home", "h1"),
        sub_in("home", "h6"),
        missed_2("home", "h6"),
        sub_out("home", "h6"),
        sub_in("home", "h1"),
        missed_2("home", "h6"),  # index 15
        sub_in("home", "h3"),
    )
    assert floors[15][0] == frozenset({"h1", "h2", "h4", "h5", "h6"})
    assert report.repairs == 2  # h6 on, h3 off


def test_an_extra_player_comes_off_by_who_checks_in_next():
    floors, report = _floors(
        *starters_act(),
        sub_in("home", "h6"),
        sub_in("home", "h1"),  # h1 checking in again means he was already off
        missed_2("home", "h6"),
        sub_out("home", "h2"),
        sub_in("home", "h1"),
    )
    on = floors[12][0]
    assert on == frozenset({"h2", "h3", "h4", "h5", "h6"})
    assert report.repairs == 1


def test_a_short_side_is_filled_by_the_bench_player_who_plays_next():
    floors, _ = _floors(
        *starters_act(),
        sub_out("home", "h1"),
        sub_in("home", "h6"),
        sub_out("home", "h6"),  # ...and h7's "subbing in" never came
        missed_2("away", "a1"),
        missed_2("home", "h7"),
    )
    assert floors[13][0] == frozenset({"h2", "h3", "h4", "h5", "h7"})


def test_the_assister_was_on_the_floor():
    floors, _ = _floors(
        *starters_act(),
        sub_out("home", "h1"),
        made_2("home", "h2", assist="h6"),
    )
    assert "h6" in floors[-1][0]


def test_a_technical_on_the_bench_puts_nobody_on_the_floor():
    floors, report = _floors(*starters_act(), technical("home", "h9"))
    assert floors[-1][0] == frozenset(HOME)
    assert report.repairs == 0


def test_a_side_with_four_starters_is_not_found():
    plays = [p for p in starters_act() if p.athlete_id != "h5"]
    _, report = _floors(*plays, play("timeout"))
    assert not report.starters_found
