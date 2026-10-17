from .evaluate import _as_old_feed, accuracy, known_games
from .testing import (
    defensive_rebound,
    game,
    made_2,
    missed_2,
    play,
    starters_act,
    sub_in,
    sub_out,
)


def _known_game():
    return game(
        *starters_act(),
        made_2("home", "h1", assist="h2"),
        missed_2("away", "a1"),
        defensive_rebound("home", "h3"),
        sub_out("home", "h1"),
        sub_in("home", "h6"),
        missed_2("home", "h6"),
    )


def test_a_known_game_loses_its_substitutions_and_keeps_its_truth():
    (known,) = known_games([_known_game()])
    assert not any("subbing" in p.text for p in known.game.plays)
    assert len(known.truth) == 3
    assert known.truth[-1].floor[0] == frozenset({"h2", "h3", "h4", "h5", "h6"})
    assert known.box.starters["home"] == frozenset({"h1", "h2", "h3", "h4", "h5"})


def test_the_truth_scores_perfectly_against_itself():
    known = known_games([_known_game()])
    result = accuracy(known, lambda k: k.floors)
    assert (result.players_right, result.all_five_right) == (1.0, 1.0)


def test_a_game_without_substitutions_is_not_known():
    assert known_games([game(*starters_act(), missed_2("home", "h1"))]) == []


def test_the_old_feed_loses_the_ids_on_shots_and_steals_only():
    old = _as_old_feed(
        game(
            made_2("home", "h1", assist="h2"),
            play("Steal", "away", "1022", "Steal", "a1"),
        )
    )
    shot, steal = old.plays
    assert (shot.athlete_id, shot.participant_ids) == (None, ())
    assert steal.athlete_id is None
    kept = _as_old_feed(game(defensive_rebound("home", "h3"))).plays[0]
    assert kept.athlete_id == "h3"
