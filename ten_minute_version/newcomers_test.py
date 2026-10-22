import math

import numpy as np

from .newcomers import NewcomerEstimate, NewcomerPrior, SidePrior, normal_eb
from .recruiting import Recruit

PRIOR = NewcomerPrior(
    "mens",
    SidePrior(5.0, -0.8, 0.5, -0.5, 2.5),
    SidePrior(4.0, -0.6, 0.0, -0.1, 1.5),
    0.6,
)


def test_a_ranked_recruits_mean_and_width_go_with_his_rank():
    top = Recruit("a", 2024, 1, 97)
    (mean, var) = PRIOR.side("offense", top, 2024)
    assert math.isclose(mean, 0.05)
    assert math.isclose(var, (math.exp(0.5 + 0.6) / 100) ** 2)
    low = Recruit("b", 2024, 100, 80)
    low_mean, low_var = PRIOR.side("offense", low, 2024)
    assert low_mean < mean and low_var < var


def test_unranked_and_out_of_class_newcomers_are_at_the_unranked_prior():
    unranked = (-0.005, 0.025**2)
    assert np.allclose(PRIOR.side("offense", None, 2024), unranked)
    old = Recruit("a", 2020, 1, 97)
    assert np.allclose(PRIOR.side("offense", old, 2024), unranked)


def test_the_prior_comes_back_from_its_estimates():
    rng = np.random.default_rng(11)
    found = []
    for i in range(6000):
        rank = int(rng.integers(1, 101)) if i % 2 else None
        noise = rng.uniform(4, 30)
        sides = []
        for p in (PRIOR.offense, PRIOR.defense):
            if rank is None:
                mean, sd = p.unranked_mean, p.unranked_sd
            else:
                mean = p.a + p.b * math.log(rank)
                sd = math.exp(p.log_sd + PRIOR.shape * (101 - rank) / 100)
            sides.append(rng.normal(mean, sd) + rng.normal(0, math.sqrt(noise)))
        found.append(
            NewcomerEstimate(2024, f"p{i}", rank, 1000.0, *sides, noise, noise, 0.0)
        )
    learned = NewcomerPrior.learn("mens", found)
    assert abs(learned.offense.a - 5.0) < 0.6
    assert abs(learned.offense.b + 0.8) < 0.2
    assert abs(learned.shape - 0.6) < 0.4
    assert abs(learned.offense.unranked_sd - 2.5) < 0.4
    assert abs(learned.defense.unranked_mean + 0.1) < 0.2


def test_normal_eb_takes_the_noise_out_of_the_spread():
    rng = np.random.default_rng(3)
    noise = rng.uniform(1, 16, 20000)
    x = rng.normal(2.0, 3.0, len(noise)) + rng.normal(0, np.sqrt(noise))
    mean, sd = normal_eb(x, noise)
    assert abs(mean - 2.0) < 0.1 and abs(sd - 3.0) < 0.15


def test_a_prior_saves_and_loads(tmp_path):
    path = PRIOR.save(tmp_path / "prior.json", seasons=[2024])
    assert NewcomerPrior.load("mens", path) == PRIOR
