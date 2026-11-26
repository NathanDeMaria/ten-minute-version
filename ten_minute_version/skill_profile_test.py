import numpy as np
import pytest

from .player_events import DEFENSE, OFFENSE
from .shot_model import Shots
from .skill_profile import (
    DIMS,
    NOTHING,
    RHO,
    Population,
    ProfileState,
    adjust_makes,
    fit_population,
    noise,
    observe,
    posterior,
    priors,
    shot_shift,
)


def _counts(**given: float) -> tuple[np.ndarray, np.ndarray]:
    offense = np.array([[given.get(c, 0.0) for c in OFFENSE]])
    defense = np.array([[given.get(c, 0.0) for c in DEFENSE]])
    return offense, defense


def test_observe_turns_a_players_counts_into_rates_and_exposure():
    offense, defense = _counts(
        made_three_unassisted=2,
        made_three_assisted=4,
        missed_three=14,
        ft_made=6,
        ft_missed=2,
        assists_mid=1,
        assists_three=2,
        turnovers=5,
        steals=3,
    )
    x, den = observe(offense, defense, np.array([100.0]), np.array([50.0]))
    at = {d: (x[0, i], den[0, i]) for i, d in enumerate(DIMS)}
    assert at["three_att"] == pytest.approx((0.2, 100.0))
    assert at["three_pct"] == pytest.approx((0.3, 20.0))
    assert at["three_unast"] == pytest.approx((2 / 6, 6.0))
    assert at["ft_pct"] == pytest.approx((0.75, 8.0))
    assert at["ast_jumper"] == pytest.approx((0.03, 100.0))
    assert at["stl"] == pytest.approx((0.06, 50.0))
    assert at["rim_pct"] == (0.0, 0.0)


def test_noise_is_poisson_for_counts_binomial_for_shares_and_nothing_over_nothing():
    mean = np.zeros(len(DIMS))
    mean[DIMS.index("tov")], mean[DIMS.index("ft_pct")] = 0.2, 0.7
    den = np.zeros((1, len(DIMS)))
    den[0, DIMS.index("tov")], den[0, DIMS.index("ft_pct")] = 100.0, 50.0
    r = noise(mean, den)[0]
    assert r[DIMS.index("tov")] == pytest.approx(0.2 / 100)
    assert r[DIMS.index("ft_pct")] == pytest.approx(0.7 * 0.3 / 50)
    assert r[DIMS.index("blk")] == NOTHING


def test_fit_population_recovers_the_true_spread_and_link_under_the_noise():
    rng = np.random.default_rng(7)
    n, k = 4000, len(DIMS)
    mean = np.full(k, 0.3)
    true_cov = np.eye(k) * 0.002
    a, b = DIMS.index("three_pct"), DIMS.index("ft_pct")
    true_cov[a, b] = true_cov[b, a] = 0.0012
    skills = rng.multivariate_normal(mean, true_cov, size=n)
    den = np.full((n, k), 400.0)
    r = noise(mean, den)
    x = skills + rng.normal(size=(n, k)) * np.sqrt(r)
    found = fit_population(x, den)
    assert found.mean == pytest.approx(mean, abs=0.005)
    assert found.cov[a, a] == pytest.approx(0.002, rel=0.15)
    assert found.cov[a, b] == pytest.approx(0.0012, rel=0.2)


def test_a_linked_skill_moves_with_the_one_that_was_seen():
    mean = np.array([[0.3, 0.7]])
    cov = np.array([[[0.002, 0.001], [0.001, 0.002]]])
    x = np.array([[0.4, 0.0]])
    r = np.array([[0.002, NOTHING]])
    m, c = posterior(mean, cov, x, r)
    # half the way on the seen skill, and the link carries a quarter to the other
    assert m[0, 0] == pytest.approx(0.35)
    assert m[0, 1] == pytest.approx(0.7 + 0.001 / 0.004 * 0.1)
    assert c[0, 0, 0] == pytest.approx(0.001)


def test_a_carried_profile_is_pulled_toward_the_mean_and_widened():
    k = len(DIMS)
    population = Population(np.full(k, 0.3), np.eye(k) * 0.002)
    carried = ProfileState(population, {"p1": (np.full(k, 0.5), np.eye(k) * 0.0005)})
    mean, cov = priors(["p1", "new"], population, carried)
    assert mean[0, 0] == pytest.approx(0.3 + RHO * 0.2)
    assert cov[0, 0, 0] == pytest.approx(RHO**2 * 0.0005 + (1 - RHO**2) * 0.002)
    assert mean[1] == pytest.approx(population.mean)


def _threes(shooter: str, distances: list[float], made: list[bool]) -> Shots:
    n = len(distances)
    return Shots(
        shooter=np.array([shooter] * n, object),
        value=np.full(n, 3),
        made=np.array(made),
        shot_type=np.array(["JumpShot"] * n, object),
        x=np.full(n, 25.0),
        y=np.array(distances),
    )


def test_harder_shots_made_as_often_read_as_a_better_shooter():
    shots = _threes("s1", [22.0] * 4, [True, False, False, True])
    x, den = np.zeros((1, len(DIMS))), np.zeros((1, len(DIMS)))
    j = DIMS.index("three_pct")
    easy, _ = adjust_makes(x, den, ["s1"], shots, np.full(4, 0.5), np.ones(4, bool))
    hard, _ = adjust_makes(x, den, ["s1"], shots, np.full(4, 0.3), np.ones(4, bool))
    assert hard[0, j] > easy[0, j]


def test_a_shooter_at_the_populations_rate_gets_no_shift_and_a_stranger_none():
    k = len(DIMS)
    population = Population(np.full(k, 0.35), np.eye(k) * 0.002)
    shots = _threes("s1", [22.0, 24.0], [True, False])
    mean = np.tile(population.mean, (1, 1))
    shift, profiled = shot_shift(shots, ["s1"], mean, population)
    assert shift == pytest.approx([0.0, 0.0]) and profiled.all()
    better = mean.copy()
    better[0, DIMS.index("three_pct")] = 0.45
    shift, _ = shot_shift(shots, ["s1"], better, population)
    assert (shift > 0).all()
    _, profiled = shot_shift(shots, ["someone else"], mean, population)
    assert not profiled.any()
