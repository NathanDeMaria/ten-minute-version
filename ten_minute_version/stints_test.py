from .stints import GAME_SCHEMA, STINT_SCHEMA, game_stints, season_tables
from .testing import (
    defensive_rebound,
    foul,
    free_throw,
    game,
    made_2,
    made_3,
    missed_2,
    starters_act,
    sub_in,
    sub_out,
)


def _two_stints():
    """
    Home scores, away misses, home misses; then h6 comes in for h1 and away
    scores twice around a home miss.
    """
    return game(
        *starters_act(),
        made_2("home", "h1", clock=1180),
        missed_2("away", "a1", clock=1160),
        defensive_rebound("home", "h2", clock=1158),
        missed_2("home", "h2", clock=1140),
        defensive_rebound("away", "a2", clock=1138),
        foul("home", "h3", clock=1130),
        sub_out("home", "h1", clock=1130),
        sub_in("home", "h6", clock=1130),
        made_3("away", "a3", clock=1120),
        missed_2("home", "h6", clock=1100),
        defensive_rebound("away", "a1", clock=1098),
        made_2("away", "a1", clock=1080, score=(2, 5)),
    )


def test_a_stint_gives_a_row_for_each_side_that_had_the_ball():
    rows = game_stints(_two_stints()).rows
    assert [(r["stint"], r["offense_side"], r["possessions"]) for r in rows] == [
        (1, "home", 2),
        (1, "away", 1),
        (2, "away", 2),
        (2, "home", 1),
    ]
    first_home = rows[0]
    assert first_home["offense_players"] == ["h1", "h2", "h3", "h4", "h5"]
    assert first_home["defense_players"] == ["a1", "a2", "a3", "a4", "a5"]
    assert (first_home["points"], first_home["empty"]) == (2, 1)
    assert (first_home["ended_fg2"], first_home["ended_miss"]) == (1, 1)
    assert rows[2]["defense_players"] == ["h2", "h3", "h4", "h5", "h6"]


def test_the_endings_add_up_to_the_possessions():
    for row in game_stints(_two_stints()).rows:
        endings = sum(row[c] for c in row if c.startswith("ended_"))
        assert endings == row["possessions"]


def test_both_rows_of_a_stint_share_its_start_end_and_margin():
    rows = game_stints(_two_stints()).rows
    home, away = rows[2:4][1], rows[2:4][0]
    # The stint starts with the away possession the substitution happened in
    assert home["start_seconds"] == away["start_seconds"] == 62
    assert home["end_seconds"] == away["end_seconds"] == 120
    # Home led 2-0 when the second stint started
    assert (home["start_margin"], away["start_margin"]) == (2, -2)


def test_the_report_counts_points_and_compares_them_with_espns_final():
    report = game_stints(_two_stints()).report
    assert report["has_substitutions"] and report["starters_found"]
    assert (report["home_points"], report["away_points"]) == (2, 5)
    assert (report["home_final"], report["away_final"]) == (2, 5)
    assert (report["possessions"], report["possessions_dropped"]) == (6, 0)


def test_a_possession_that_isnt_five_on_five_is_dropped_and_counted():
    built = game_stints(
        game(
            *[p for p in starters_act() if p.athlete_id != "a5"],
            sub_in("home", "h6"),
            sub_out("home", "h1"),
            made_2("home", "h6"),
            missed_2("away", "a1"),
        )
    )
    assert built.rows == []
    assert built.report["possessions_dropped"] == 2


def test_a_game_without_substitutions_has_no_rows():
    built = game_stints(
        game(*starters_act(), made_2("home", "h1"), free_throw("home", "h1"))
    )
    assert built.rows == []
    assert not built.report["has_substitutions"]


def test_the_season_tables_have_their_schemas():
    stints, games, players = season_tables([_two_stints()])
    assert stints.schema.equals(STINT_SCHEMA)
    assert games.schema.equals(GAME_SCHEMA)
    assert stints.num_rows == 4 and games.num_rows == 1
    assert "h6" in players.column("athlete_id").to_pylist()
