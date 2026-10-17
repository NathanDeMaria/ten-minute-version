from .names import resolve_names
from .testing import foul, game, play


def _resolved(*plays):
    return resolve_names(game(*plays)).plays


def test_a_steal_gets_the_id_its_player_carries_elsewhere():
    *_, stolen = _resolved(
        foul("away", "a1"),
        play("Player a1 Steal.", "away", "1022", "Steal"),
    )
    assert stolen.athlete_id == "a1"
    assert stolen.participant_ids == ("a1",)


def test_an_old_feed_shot_gets_its_shooter_and_assister():
    *_, shot = _resolved(
        foul("home", "h1"),
        foul("home", "h2"),
        play("Player h1 made Jumper. Assisted by Player h2.", "home", None, "JumpShot"),
    )
    assert shot.athlete_id == "h1"
    assert shot.participant_ids == ("h1", "h2")


def test_todays_assist_wording_is_read_too():
    *_, shot = _resolved(
        foul("home", "h1"),
        foul("home", "h2"),
        play("Player h1 makes 6-foot jumper (Player h2 assists)", "home", "1002"),
    )
    assert shot.participant_ids == ("h1", "h2")


def test_only_names_on_the_plays_own_side_count():
    *_, stolen = _resolved(
        foul("home", "h1"),
        play("Player h1 Steal.", "away", "1022", "Steal"),
    )
    assert stolen.athlete_id is None


def test_the_longest_name_wins_and_a_name_must_end_where_it_says():
    plays = _resolved(
        foul("home", "h1"),
        foul("home", "h11"),
        play("Player h11 Steal.", "home", "1022", "Steal"),
        play("Player h1x Steal.", "home", "1022", "Steal"),
    )
    assert plays[2].athlete_id == "h11"
    assert plays[3].athlete_id is None


def test_a_play_that_already_has_its_ids_is_left_alone():
    original = foul("home", "h1")
    assert _resolved(original) == (original,)
