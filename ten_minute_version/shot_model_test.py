import numpy as np

from .shot_model import (
    FREE_THROW,
    JUMPER,
    RIM,
    Shots,
    fit_shot_model,
    shooter_skills,
    with_skill,
)


def _shots(rows: list[tuple]) -> Shots:
    """(shooter, value, made, shot_type, x, y) per shot."""
    keys = ("shooter", "value", "made", "shot_type", "x", "y")
    return Shots.from_rows([dict(zip(keys, row)) for row in rows])


def _jumpers(rng: np.random.Generator, n: int = 4000) -> list[tuple]:
    out = []
    for _ in range(n):
        distance = rng.uniform(2, 30)
        made = rng.random() < 0.6 - 0.012 * distance
        out.append(
            ("p1", 3 if distance > 22 else 2, bool(made), "JumpShot", 25.0, distance)
        )
    return out


def test_a_jump_shots_chance_falls_with_distance():
    model = fit_shot_model(_shots(_jumpers(np.random.default_rng(0))))
    near, far = model.chance(
        _shots(
            [
                ("p1", 2, False, "JumpShot", 25.0, 5.0),
                ("p1", 3, False, "JumpShot", 25.0, 25.0),
            ]
        )
    )
    assert near > far
    assert abs(near - (0.6 - 0.012 * 5)) < 0.05


def test_a_deep_make_counts_as_a_miss_when_fitting():
    rows = (
        _jumpers(np.random.default_rng(1))
        + [("p1", 3, True, "JumpShot", 25.0, 45.0)] * 200
    )
    model = fit_shot_model(_shots(rows))
    (deep,) = model.chance(_shots([("p1", 3, False, "JumpShot", 25.0, 45.0)]))
    assert deep < 0.25


def test_rim_shots_and_free_throws_are_their_league_rates():
    rows: list[tuple] = [
        ("p1", 2, made, "DunkShot", 25.0, 1.0) for made in (True, True, True, False)
    ]
    rows += [("p1", 1, made, "MadeFreeThrow", None, None) for made in (True, False)]
    rows += _jumpers(np.random.default_rng(2), 200)
    model = fit_shot_model(_shots(rows))
    dunk, free_throw = model.chance(_shots([rows[0], rows[4]]))
    assert (dunk, free_throw) == (0.75, 0.5)


def test_the_kind_of_shot_is_what_a_skill_is_kept_for():
    shots = _shots(
        [
            ("p1", 2, True, "LayUpShot", 25.0, 1.0),
            ("p1", 3, True, "JumpShot", 25.0, 24.0),
            ("p1", 1, True, "MadeFreeThrow", None, None),
        ]
    )
    assert list(shots.group) == [RIM, JUMPER, FREE_THROW]


def test_a_shooter_who_beats_the_model_keeps_credit_shrunk_by_volume():
    shots = _shots(
        [("good", 3, True, "JumpShot", 25.0, 24.0)] * 20
        + [("bad", 3, False, "JumpShot", 25.0, 24.0)] * 20
    )
    chance = np.full(40, 0.35)
    light = shooter_skills(shots, chance, {JUMPER: 5.0})
    heavy = shooter_skills(shots, chance, {JUMPER: 500.0})
    assert (
        light[("good", JUMPER)] > heavy[("good", JUMPER)] > 0 > heavy[("bad", JUMPER)]
    )
    shifted = with_skill(shots, chance, light)
    assert shifted[0] > 0.35 > shifted[-1]
    assert np.allclose(with_skill(shots, chance, {}), 0.35)
