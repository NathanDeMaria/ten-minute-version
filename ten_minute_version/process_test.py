from datetime import date

import pytest

from .box_scores import BoxLine
from .context import GameContext, Team
from .luck import with_season
from .process import (
    AHEAD,
    MIN_ROWS,
    STATS,
    ProcessFit,
    Rating,
    Side,
    box_process,
    complete_games,
    expected_margin,
    fit_for,
    fit_process,
    fit_seasons,
    latest,
    load_fits,
    side_stats,
    team_totals,
    with_ahead,
)

# an ordinary side: 60 shots, 20 threes, 18 free throws, 64 points
BASE = {"points": 64.0, "fgm": 26.0, "fga": 60.0, "tpm": 7.0, "tpa": 20.0, "fta": 18.0,
        "oreb": 10.0, "dreb": 24.0, "ast": 13.0, "blk": 3.0, "tov": 12.0}  # fmt: skip
HOT = {**BASE, "fgm": 31.0, "points": 74.0}  # five more makes on the same shots
# a fit with the shape the real ones have: shooting discounted, rim protection
# and shot selection credited
FIT = ProcessFit(
    2025,
    sd=dict.fromkeys(STATS, 0.1),
    coef={
        **dict.fromkeys(STATS, 0.0),
        "efg": -0.1,
        "opp_efg": 0.15,
        "blk": 0.12,
        "tpar": 0.06,
    },
    rows=50_000,
)


def _line(game: str, team: str, player: str, **stats: float) -> BoxLine:
    return BoxLine(game, team, player, True, None, stats)


def _team(t: str) -> Team:
    return Team(t, f"{t} U", None, None, True)


def _contexts(neutral: bool = False) -> dict[str, GameContext]:
    return {"g1": GameContext("g1", neutral, False, _team("H"), _team("A"))}


def _game(home: dict, away: dict) -> list[BoxLine]:
    return [_line("g1", "H", "h1", **home), _line("g1", "A", "a1", **away)]


def _finals(home: dict, away: dict) -> dict[str, tuple[int, int]]:
    return {"g1": (int(home["points"]), int(away["points"]))}


def _read(home: dict, away: dict, fit: ProcessFit | None = FIT) -> dict[str, float]:
    return box_process(_game(home, away), _contexts(), _finals(home, away), fit)


def test_a_team_adds_up_over_its_players():
    lines = [
        _line("g1", "H", "h1", fgm=3.0, fga=5.0),
        _line("g1", "H", "h2", fgm=1.0, fga=4.0, blk=2.0),
    ]
    totals = team_totals(lines)[("g1", "H")]
    assert (totals["fgm"], totals["fga"], totals["blk"], totals["oreb"]) == (
        4.0,
        9.0,
        2.0,
        0.0,
    )


def test_the_rates_are_each_sides_view_of_the_game():
    rates = side_stats(BASE, {**BASE, "dreb": 30.0, "tpa": 30.0})
    assert rates is not None and set(rates) == set(STATS)
    assert rates["efg"] == pytest.approx((26 + 3.5) / 60)
    assert rates["orb"] == pytest.approx(10 / (10 + 30))
    assert rates["blk"] == pytest.approx(3 / (60 - 30))
    assert rates["tov"] == pytest.approx(12 / (60 - 10 + 12 + 0.475 * 18))


def test_a_side_with_no_shots_has_no_rates():
    assert side_stats({**BASE, "fga": 0.0}, BASE) is None


def test_a_box_score_missing_players_is_not_read():
    # the lines add up to 64-64, the game ended 70-64
    lines = _game(BASE, BASE)
    assert complete_games(lines, _contexts(), {"g1": (70, 64)}) == {}
    assert set(complete_games(lines, _contexts(), {"g1": (64, 64)})) == {"g1"}


def test_two_identical_sides_read_nothing_and_swapping_them_flips_the_sign():
    assert _read(BASE, BASE)["g1"] == pytest.approx(0.0)
    assert _read(HOT, BASE)["g1"] == pytest.approx(-_read(BASE, HOT)["g1"])


def test_a_home_side_that_shot_hot_has_part_of_its_margin_read_as_wont_repeat():
    assert _read(HOT, BASE)["g1"] > 0


def test_taking_threes_and_blocking_shots_read_as_strength_the_margin_missed():
    for change in ({"tpa": 30.0}, {"blk": 9.0}):
        assert _read({**BASE, **change}, BASE)["g1"] < 0


def test_nothing_is_read_without_a_fit():
    assert _read(HOT, BASE, fit=None) == {}


def test_a_season_not_fitted_yet_is_read_with_the_latest_fit_before_it():
    fits = {2020: FIT, 2023: FIT}
    assert fit_for(fits, 2023) is FIT and fit_for(fits, 2026) is fits[2023]
    assert fit_for(fits, 2019) is None


def test_the_shipped_fits_cover_both_leagues_through_the_live_season():
    for league in ("mens", "womens"):
        fits = load_fits(league)
        assert max(fits) >= 2026
        assert all(f.rows >= MIN_ROWS for f in fits.values())


def test_the_expected_margin_is_the_net_gap_at_the_pairs_pace_plus_the_home_edge():
    home, away = (
        Rating(date(2025, 11, 1), 10.0, 70.0, 3.0),
        Rating(date(2025, 11, 1), 0.0, 66.0, 3.0),
    )
    assert expected_margin(home, away, True, False) == pytest.approx(10 * 68 / 100 + 3)
    assert expected_margin(away, home, False, False) == pytest.approx(
        -10 * 68 / 100 - 3
    )
    assert expected_margin(home, away, True, True) == pytest.approx(10 * 68 / 100)


def test_a_game_reads_the_ratings_from_before_its_day():
    rows = [Rating(date(2025, 11, d), float(d), 70.0, 3.0) for d in (1, 5, 9)]
    # the 5th's row is for games after the 5th
    found = [latest(rows, date(2025, 11, d)) for d in (1, 5, 6)]
    assert [r.net if r else None for r in found] == [None, 1.0, 5.0]


def _sides(n_games: int, season: int = 2020) -> list[Side]:
    return [
        Side(
            season,
            date(2020, 11, 1 + i % 28),
            f"g{i}",
            "T",
            float(i % 3),
            {s: 0.5 + 0.01 * (i % 7) for s in STATS},
        )
        for i in range(n_games)
    ]


def test_a_side_is_scored_against_its_teams_next_games():
    rows = with_ahead(_sides(AHEAD + 2))
    assert len(rows) == 2  # the last five have no five after them
    side, ahead = rows[0]
    later = sorted(_sides(AHEAD + 2), key=lambda s: (s.day, s.game_id))
    assert ahead == pytest.approx(sum(s.over for s in later[1 : 1 + AHEAD]) / AHEAD)


def test_a_season_is_fitted_only_on_the_seasons_before_it():
    rows = [(s, s.over) for s in _sides(MIN_ROWS, 2020)] + [
        (s, -s.over) for s in _sides(10, 2021)
    ]
    fits = fit_seasons(rows, [2020, 2021, 2022])
    assert set(fits) == {2021, 2022}  # 2020 has nothing before it
    assert fits[2021].rows == MIN_ROWS and fits[2022].rows == MIN_ROWS + 10


def test_the_fit_finds_a_rate_that_carries_over():
    # every rate varies on its own cycle, so none stands in for another
    cycles = dict(zip(STATS, (3, 5, 7, 11, 13, 17, 19, 23, 29, 31)))
    sides = [
        Side(
            2020,
            date(2020, 11, 1),
            f"g{i}",
            "T",
            float(i % 37),
            {s: 0.01 * (i % c) for s, c in cycles.items()},
        )
        for i in range(5000)
    ]
    rows = [(s, 2.0 * s.rates["blk"]) for s in sides]
    fit = fit_process(2021, rows)
    assert fit.coef["blk"] == pytest.approx(2.0 * fit.sd["blk"], rel=1e-3)
    assert fit.coef["efg"] == pytest.approx(0.0, abs=1e-3)


def test_a_season_files_the_reading_beside_the_luck():
    contexts = {"g1": GameContext("g1", True, False, _team("H"), _team("A"))}
    rows = with_season(
        [], "mens", 2025, {}, _game(HOT, BASE), contexts, {"g1": date(2025, 11, 4)},
        finals=_finals(HOT, BASE), fits={2025: FIT},
    )  # fmt: skip
    # one game at a neutral site: filed as it is, not centered
    assert rows[0]["process"] == pytest.approx(_read(HOT, BASE)["g1"])
    without = with_season(
        [],
        "mens",
        2025,
        {},
        _game(HOT, BASE),
        contexts,
        {"g1": date(2025, 11, 4)},
        fits={2025: FIT},
    )
    assert without[0].get("process") is None  # no finals, no reading
