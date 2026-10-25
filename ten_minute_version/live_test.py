from datetime import date

import numpy as np

from .chain import ChainSettings, PlayerState, SeasonEnd, chained_priors
from .live import as_of, merged, prior_design
from .team_ratings import team_sides


def _row(day: date, team: str, net: float) -> dict:
    return {"league": "mens", "season": 2026, "as_of": day, "team_id": team, "net": net}


def test_a_rerun_morning_replaces_its_own_rows_and_keeps_the_rest():
    monday, tuesday = date(2026, 11, 9), date(2026, 11, 10)
    existing = [
        _row(monday, "A", 1.0),
        _row(tuesday, "A", 2.0),
        _row(tuesday, "B", 3.0),
    ]
    out = merged(existing, [_row(tuesday, "A", 5.0)])
    assert out == [_row(monday, "A", 1.0), _row(tuesday, "A", 5.0)]


def test_a_team_that_hasnt_played_is_its_roster_at_its_priors():
    design = prior_design(["p1", "p2", "p1"], ["T"])
    assert design.players == ["p1", "p2"] and design.rows == 0
    states = {"p1": PlayerState(2025, 0.03, 0.01, 0.0001, 0.0001, 2)}
    last = SeasonEnd(0.0, 0.01, {"T": (0.02, 0.0)})

    def newcomer(p, season):
        return (0.0, 0.0004), (0.0, 0.0004)

    mean, _ = chained_priors(
        design,
        2026,
        states,
        newcomer,
        np.zeros(design.columns),
        np.zeros(2, bool),
        ChainSettings(development=False),
        None,
        last,
    )
    offense, defense = team_sides(design, mean, "T", {"p1": 2.0, "p2": 3.0})
    assert np.isclose(offense, 2 * 0.03 + 0.02) and np.isclose(defense, 2 * 0.01)


def test_a_morning_is_today_unless_told():
    assert as_of("2026-11-03") == date(2026, 11, 3)
    assert as_of(None) == date.today()
