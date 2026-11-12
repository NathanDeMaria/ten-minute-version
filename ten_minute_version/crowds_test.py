import math
from datetime import date

import pytest

from .crowds import (
    FLOOR,
    HALF_LIFE,
    MIN_GAMES,
    TYPICAL,
    HomeGame,
    Reading,
    carried,
    home_game,
    read_crowds,
    season_open,
    season_rows,
    with_season,
    write_crowds,
)


def _summary(attendance=None, neutral: bool = False, day: str = "2025-11-04T00:30Z"):
    def team(i: str, side: str) -> dict:
        return {"homeAway": side, "team": {"id": i, "displayName": f"{i} U"}}

    return {
        "header": {
            "competitions": [
                {
                    "date": day,
                    "neutralSite": neutral,
                    "competitors": [team("H", "home"), team("A", "away")],
                }
            ]
        },
        "gameInfo": {} if attendance is None else {"attendance": attendance},
    }


def test_a_home_game_is_its_home_sides_with_the_summarys_attendance():
    game = home_game("g1", _summary(4200))
    assert game == HomeGame("g1", date(2025, 11, 4), "H", "H U", 4200)
    assert home_game("g1", _summary()) == HomeGame(
        "g1", date(2025, 11, 4), "H", "H U", None
    )
    assert home_game("g1", _summary(4200, neutral=True)) is None
    assert home_game("g1", {}) is None


def _games(team: str, crowds: list[int | None], start: int = 1) -> list[HomeGame]:
    return [
        HomeGame(f"{team}{i}", date(2025, 11, start + i), team, f"{team} U", crowd)
        for i, crowd in enumerate(crowds)
    ]


def test_a_reading_waits_for_enough_games_and_skips_the_ones_without_a_number():
    rows = season_rows(
        "mens", 2025, _games("H", [2500, 0, None, 2500, 2500, 2500, 2500]), {}
    )
    # five games with a number: the 0 and the missing one made no row
    assert [r["games"] for r in rows] == [1, 2, 3, 4, 5]
    assert [r["crowd"] is None for r in rows] == [True] * (MIN_GAMES - 1) + [False]
    # the typical crowd, every night: a reading of 0
    assert rows[-1]["crowd"] == pytest.approx(0.0)
    # each row is dated the day after its game
    assert rows[0]["as_of"] == date(2025, 11, 2)
    assert rows[-1]["as_of"] == date(2025, 11, 8)


def test_a_later_game_weighs_more_and_the_reading_is_a_log():
    opening = {"H": Reading(5 * math.log(1000), 5.0, 5, "H U")}
    rows = season_rows("mens", 2025, _games("H", [4000]), opening)
    keep = 0.5 ** (1 / HALF_LIFE)
    expected = (5 * keep * math.log(1000) + math.log(4000)) / (5 * keep + 1)
    opened, played = rows
    assert opened["as_of"] == season_open(2025)
    assert opened["crowd"] == pytest.approx(math.log(1000 / TYPICAL["mens"]))
    assert played["crowd"] == pytest.approx(expected - math.log(TYPICAL["mens"]))
    assert played["games"] == 6


def test_a_team_opens_a_season_named_before_it_has_played_at_home():
    opening = {"H": Reading(5 * math.log(1000), 5.0, 5, "H U")}
    (row,) = season_rows("mens", 2025, [], opening)
    assert (row["team"], row["as_of"]) == ("H U", date(2025, 8, 1))
    # and a cut leaves out the games on or after it
    cut = season_rows("mens", 2025, _games("H", [900, 900]), opening, date(2025, 11, 2))
    assert [r["games"] for r in cut] == [5, 6]


def test_a_season_of_limited_crowds_is_read_from_its_own_alone():
    opening = {"H": Reading(5 * math.log(9000), 5.0, 5, "H U")}
    games = [
        HomeGame("a", date(2020, 12, 1), "H", "H U", 0),
        HomeGame("b", date(2020, 12, 5), "H", "H U", 3200),
        HomeGame("c", date(2020, 12, 9), "N", "N U", None),
    ]
    rows = {
        (r["team_id"], r["as_of"]): r for r in season_rows("mens", 2020, games, opening)
    }
    floor = math.log(FLOOR / TYPICAL["mens"])
    # everyone opens at the floor, whatever they drew before, and has a reading at once
    assert rows["H", date(2020, 8, 1)]["crowd"] == pytest.approx(floor)
    # an attendance of 0 is an empty building that season, not a missing number
    assert rows["H", date(2020, 12, 2)]["crowd"] == pytest.approx(floor)
    assert (
        floor < rows["H", date(2020, 12, 6)]["crowd"] < math.log(3200 / TYPICAL["mens"])
    )
    assert rows["N", date(2020, 12, 10)]["crowd"] == pytest.approx(floor)


def test_the_season_after_carries_on_from_the_one_before_the_limits(tmp_path):
    full = _games("H", [9000] * 5)
    rows = with_season([], "mens", 2019, [g for g in full], None)
    rows = with_season(
        rows, "mens", 2020, [HomeGame("x", date(2020, 12, 1), "H", "H U", 0)]
    )
    rows = with_season(rows, "mens", 2021, [])
    (opened,) = [r for r in rows if r["season"] == 2021]
    assert opened["crowd"] == pytest.approx(math.log(9000 / TYPICAL["mens"]))
    assert carried(rows, "mens", 2021)["H"].games == 5
    # another league's rows are not this one's to carry
    assert carried(rows, "womens", 2021) == {}
    # rebuilding a season replaces its rows, and the file reads back as written
    again = with_season(rows, "mens", 2020, [])
    assert len([r for r in again if r["season"] == 2020]) == 1
    path = tmp_path / "mens_crowds.csv"
    write_crowds(path, rows)
    back = read_crowds(path)
    assert [(r["team_id"], r["as_of"], r["games"]) for r in back] == [
        (r["team_id"], r["as_of"], r["games"]) for r in rows
    ]
    assert read_crowds(tmp_path / "nothing.csv") == []
