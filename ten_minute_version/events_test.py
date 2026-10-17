import pytest

from .events import Kind, classify, offense
from .testing import play


@pytest.mark.parametrize(
    ("category", "play_type", "text", "kind"),
    [
        ("1001", "MadeFreeThrow", "made Free Throw.", Kind.MADE_FT),
        ("1004", "MadeFreeThrow", "missed Free Throw.", Kind.MISSED_FT),
        ("1002", "DunkShot", "makes 4-foot dunk", Kind.MADE_2),
        ("1003", "JumpShot", "makes 25-foot three point jumper", Kind.MADE_3),
        ("1005", "LayUpShot", "misses driving layup", Kind.MISSED_2),
        ("1006", "JumpShot", "misses 26-foot three point jumper", Kind.MISSED_3),
        ("1009", "Offensive Rebound", "", Kind.OFFENSIVE_REBOUND),
        ("1009", "Defensive Rebound", "", Kind.DEFENSIVE_REBOUND),
        ("1009", "Dead Ball Rebound", "", Kind.DEAD_BALL_REBOUND),
        ("1010", "Lost Ball Turnover", "bad pass turnover", Kind.TURNOVER),
        ("1007", "PersonalFoul", "Foul on X.", Kind.FOUL),
        ("1007", "Technical Foul", "Technical Foul on X.", Kind.TECHNICAL),
        ("1015", "Substitution", "X subbing in for Purdue", Kind.SUB_IN),
        ("1015", "Substitution", "X subbing out for Purdue", Kind.SUB_OUT),
        ("1018", "End Period", "End of 1st half", Kind.PERIOD_END),
        ("1020", "End Period", "End of 1st Overtime", Kind.PERIOD_END),
        ("1008", "OfficialTVTimeOut", "Official TV Timeout", Kind.TV_TIMEOUT),
        ("1008", "ShortTimeOut", "Purdue Timeout", Kind.TIMEOUT),
        ("1011", "Coach's Challenge (Stands)", "", Kind.OTHER),
    ],
)
def test_a_coded_play_is_read_from_its_category(category, play_type, text, kind):
    assert classify(play(text, "home", category, play_type)) == kind


@pytest.mark.parametrize(
    ("play_type", "text", "kind"),
    [
        ("JumpShot", "Naomi Panganiban missed Three Point Jumper.", Kind.MISSED_3),
        ("JumpShot", "X made Three Point Jumper. Assisted by Y.", Kind.MADE_3),
        ("LayUpShot", "Kiki Rice missed Layup.", Kind.MISSED_2),
        ("DunkShot", "Matheo Coffi makes 4-foot dunk", Kind.MADE_2),
        ("MadeFreeThrow", "Amanda Muse missed Free Throw.", Kind.MISSED_FT),
        ("MadeFreeThrow", "X made Free Throw.", Kind.MADE_FT),
    ],
)
def test_a_shot_without_a_category_is_read_from_its_words(play_type, text, kind):
    assert classify(play(text, "home", None, play_type)) == kind


def test_the_ball_is_the_side_making_its_own_play():
    assert offense(Kind.MISSED_3, "home") == "home"
    assert offense(Kind.DEFENSIVE_REBOUND, "away") == "away"
    assert offense(Kind.TURNOVER, "home") == "home"


def test_a_steal_or_block_means_the_other_side_had_it():
    assert offense(Kind.STEAL, "home") == "away"
    assert offense(Kind.BLOCK, "away") == "home"


def test_fouls_and_unsided_plays_say_nothing_about_the_ball():
    assert offense(Kind.FOUL, "home") is None
    assert offense(Kind.DEAD_BALL_REBOUND, "home") is None
    assert offense(Kind.MADE_2, None) is None
