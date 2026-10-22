import numpy as np

from .recruiting import (
    Recruit,
    RecruitModel,
    features,
    newcomers,
    read_recruits,
)


def _row(athlete_id: str, total: float, possessions: int = 1000) -> dict:
    return {
        "athlete_id": athlete_id,
        "offense": total / 2,
        "defense": total / 2,
        "offense_possessions": possessions,
        "defense_possessions": possessions,
    }


def test_a_recruit_counts_in_his_class_season_and_the_next():
    recruit = Recruit("a", 2024, 1, 97)
    assert features(recruit, 2024) == (1.0, 1.0)
    assert features(recruit, 2025) == (1.0, 1.0)
    assert features(recruit, 2026) == (0.0, 0.0)
    assert features(None, 2024) == (0.0, 0.0)
    assert features(Recruit("b", 2024, 100, 80), 2024) == (1.0, 0.01)


def test_newcomers_are_who_last_season_didnt_rate():
    found = newcomers({2023: [_row("old", 1)], 2024: [_row("old", 2), _row("new", 3)]})
    assert [(n.season, n.athlete_id) for n in found] == [(2024, "new")]
    assert np.isclose(found[0].offense, 0.015)


def test_ranked_newcomers_start_higher(tmp_path):
    recruits = {f"r{i}": Recruit(f"r{i}", 2024, 1 + i, 95) for i in range(5)}
    ratings = {
        2023: [],
        2024: [_row(f"r{i}", 6 - i) for i in range(5)]
        + [_row(f"u{i}", 0) for i in range(20)],
    }
    model = RecruitModel.learn("mens", newcomers(ratings), recruits)
    top = sum(model.rating(recruits["r0"], 2024))
    low = sum(model.rating(recruits["r4"], 2024))
    unranked = sum(model.rating(None, 2024))
    assert top > low > unranked
    assert abs(unranked) < 1e-3
    path = model.save(tmp_path / "recruit.json")
    assert RecruitModel.load("mens", path) == model


def test_recruits_are_read_by_their_columns(tmp_path):
    (tmp_path / "mens").mkdir()
    (tmp_path / "mens" / "recruits.csv").write_text(
        "league,class_year,rank,recruit_id,name,grade,athlete_id,match\n"
        "mens,2024,1,9,A,97,5041939,team\n"
        "mens,2024,2,8,B,97,,\n"
    )
    assert read_recruits(tmp_path, "mens") == {
        "5041939": Recruit("5041939", 2024, 1, 97)
    }
