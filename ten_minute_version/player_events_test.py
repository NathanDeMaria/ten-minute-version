from .context import GameContext, Team
from .player_events import (
    COUNTS,
    box_agreement,
    event_rows,
    floor_possessions,
    game_events,
    mark_box,
)
from .testing import (
    defensive_rebound,
    foul,
    free_throw,
    game,
    made_2,
    made_3,
    missed_2,
    offensive_rebound,
    play,
    sub_in,
    sub_out,
    turnover,
)


def test_a_make_is_the_shooters_by_zone_and_an_assist_is_the_passers():
    counts = game_events(
        game(
            made_2("home", "h1", assist="h2"),
            made_3("home", "h2"),
            play("made", "home", "1002", "JumpShot", "h3"),
            missed_2("away", "a1"),
            play("missed three", "away", "1006", "JumpShot", "a2"),
        )
    )
    assert counts["home", "h1"]["made_rim_assisted"] == 1
    assert counts["home", "h2"]["assists_rim"] == 1
    assert counts["home", "h2"]["made_three_unassisted"] == 1
    assert counts["home", "h3"]["made_mid_unassisted"] == 1
    assert counts["away", "a1"]["missed_mid"] == 1
    assert counts["away", "a2"]["missed_three"] == 1


def test_a_free_throw_trip_runs_through_substitutions_but_not_other_plays():
    counts = game_events(
        game(
            foul("away", "a1"),
            free_throw("home", "h1"),
            sub_out("home", "h2"),
            sub_in("home", "h6"),
            free_throw("home", "h1", made=False),
            defensive_rebound("away", "a1"),
            foul("away", "a2"),
            free_throw("home", "h1"),
            free_throw("home", "h3"),
        )
    )
    h1 = counts["home", "h1"]
    assert (h1["ft_trips"], h1["ft_made"], h1["ft_missed"]) == (2, 2, 1)
    assert counts["home", "h3"]["ft_trips"] == 1
    assert counts["away", "a1"]["fouls"] == 1


def test_the_rest_of_the_box_is_whoever_espn_names():
    counts = game_events(
        game(
            foul("home", "h1"),
            turnover("home", "h1"),
            play("Player a1 Steal.", "away", "1022", "Steal"),
            foul("away", "a1"),
            offensive_rebound("home", "h2"),
            play("Block", "away", "1021", "Block Shot", "a2"),
            defensive_rebound("away", "a2"),
            play("Team Turnover.", "home", "1010", "Lost Ball Turnover"),
        )
    )
    assert counts["home", "h1"]["turnovers"] == 1
    # named only in its text, like every steal in the feed
    assert counts["away", "a1"]["steals"] == 1
    assert counts["home", "h2"]["offensive_rebounds"] == 1
    assert counts["away", "a2"]["blocks"] == 1
    assert counts["away", "a2"]["defensive_rebounds"] == 1
    assert sum(c["turnovers"] for c in counts.values()) == 1


def test_floor_possessions_weight_estimated_rows_and_skip_a_generic_side():
    rows = [
        {
            "game_id": "g1",
            "offense_side": "home",
            "offense_players": ["h1", "h2"],
            "defense_players": ["generic"],
            "possessions": 10,
        },
        {
            "game_id": "g1",
            "offense_side": "away",
            "offense_players": ["generic"],
            "offense_weights": [1.0],
            "defense_players": ["h1", "h3"],
            "defense_weights": [1.0, 0.25],
            "possessions": 8,
        },
    ]
    found = floor_possessions(rows)
    assert found["g1", "h1"] == ("home", 10.0, 8.0)
    assert found["g1", "h3"] == ("home", 0.0, 2.0)
    assert ("g1", "generic") not in found


def test_event_rows_list_everyone_on_the_floor_with_their_team():
    team = lambda t: Team(t, t, None, None, True)  # noqa: E731
    context = GameContext("g1", False, False, team("H"), team("A"))
    possessions = {
        ("g1", "h1"): ("home", 12.0, 11.0),
        ("g1", "h4"): ("home", 3.0, 2.5),
        ("g2", "x9"): ("home", 5.0, 5.0),
    }
    rows = event_rows(game(made_2("home", "h1")), possessions, context)
    by = {r["athlete_id"]: r for r in rows}
    assert set(by) == {"h1", "h4"}
    assert by["h1"]["made_rim_unassisted"] == 1
    assert by["h1"]["offense_possessions"] == 12.0
    assert by["h4"]["team_id"] == "H"
    assert all(by["h4"][count] == 0 for count in COUNTS)


def test_box_agreement_compares_each_stat_with_the_box():
    rows = [
        dict.fromkeys(COUNTS, 0)
        | {
            "game_id": "g1",
            "athlete_id": "h1",
            "made_three_assisted": 1,
            "missed_mid": 1,
        },
        dict.fromkeys(COUNTS, 0) | {"game_id": "g1", "athlete_id": "h2", "steals": 1},
        dict.fromkeys(COUNTS, 0) | {"game_id": "g9", "athlete_id": "x", "steals": 5},
    ]
    stats = ("fgm", "fga", "tpm", "tpa", "ftm", "fta", "oreb", "dreb", "ast", "stl")
    blank = dict.fromkeys((*stats, "blk", "tov", "pf"), 0.0)
    box = {
        ("g1", "h1"): blank | {"fgm": 1.0, "fga": 2.0, "tpm": 1.0, "tpa": 1.0},
        ("g1", "h2"): blank | {"stl": 2.0},
    }
    found = box_agreement(rows, box)
    assert found["fga"] == (1.0, 1.0)
    assert found["stl"] == (0.5, 0.5)


def test_mark_box_flags_a_game_whose_plays_dont_add_up():
    blank = dict.fromkeys(COUNTS, 0)
    rows = [
        blank | {"game_id": "g1", "athlete_id": "h1", "turnovers": 2, "fouls": 3},
        blank | {"game_id": "g2", "athlete_id": "h1", "turnovers": 4, "fouls": 6},
        blank | {"game_id": "g3", "athlete_id": "h1", "fouls": 1},
    ]
    line = {"tov": 2.0, "pf": 3.0, "dreb": 0.0}
    mark_box(rows, {("g1", "h1"): line, ("g2", "h1"): line})
    assert [r["box_ok"] for r in rows] == [True, False, None]
