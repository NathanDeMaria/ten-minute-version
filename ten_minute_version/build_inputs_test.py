import pytest

from . import build


def _row(athlete: str, offense: float, defense: float = 0.0) -> dict:
    return {
        "athlete_id": athlete,
        "offense": offense,
        "defense": defense,
        "offense_possessions": 1000,
    }


def test_a_missing_season_is_an_error_naming_the_step():
    with pytest.raises(FileNotFoundError, match=r"\[2014\].*build.py season_ratings"):
        build._needed({2013: [], 2015: []}, range(2013, 2016), "x", "season_ratings")
    build._needed({2013: [], 2014: []}, range(2013, 2015), "x", "season_ratings")


def test_development_sees_only_the_seasons_before():
    # p improves 2 per 100 from 2013 to 2014, and 10 from 2014 to 2015
    ratings = {
        2013: [_row("p", 0.0)],
        2014: [_row("p", 2.0)],
        2015: [_row("p", 12.0)],
    }
    for_2015 = build.development_before(ratings, 2015)
    assert for_2015[1][0] == pytest.approx(0.02)
    # 2016 has seen 2015's jump too
    assert build.development_before(ratings, 2016)[2][0] == pytest.approx(0.10)
    # before anything's been rated twice, nothing is known of development
    assert build.development_before(ratings, 2014)[1] == (0.0, 0.0)


def test_a_backfill_without_season_ratings_refuses_to_run(tmp_path, monkeypatch):
    (tmp_path / "league=mens" / "season=2024").mkdir(parents=True)
    monkeypatch.setattr(build, "_spells", lambda league, tables: {2024: [], 2025: []})
    with pytest.raises(FileNotFoundError, match="season_ratings"):
        build.team_ratings("mens", first=2024, last=2025, tables=str(tmp_path))
