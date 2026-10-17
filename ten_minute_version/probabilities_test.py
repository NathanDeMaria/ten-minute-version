import numpy as np

from .events import classify
from .probabilities import FEATURES, OnFloorModel, _to_five, features, fit
from .testing import game, missed_2, starters_act


def test_a_side_is_scaled_to_five_with_actors_kept_at_one():
    p = np.array([1.0, 1.0, 0.6, 0.6, 0.3, 0.3, 0.2])
    pinned = np.array([True, True, False, False, False, False, False])
    out = _to_five(p, pinned)
    assert np.isclose(out.sum(), 5.0)
    assert (out[:2] == 1.0).all()
    assert (out <= 1.0).all()


def test_a_chance_scaled_past_one_is_capped_and_the_rest_rise():
    out = _to_five(np.array([0.9, 0.9, 0.9, 0.9, 0.05, 0.05]), np.zeros(6, dtype=bool))
    assert np.isclose(out.sum(), 5.0)
    assert np.allclose(out[:4], 1.0)
    assert np.allclose(out[4:], 0.5)


def test_the_fit_finds_a_feature_that_separates_on_from_off():
    rng = np.random.default_rng(0)
    x = np.zeros((2000, len(FEATURES)))
    x[:, 0] = 1.0
    x[:, 1] = rng.integers(0, 2, 2000)
    y = (rng.random(2000) < np.where(x[:, 1] == 1, 0.9, 0.1)).astype(float)
    model = fit("mens", [x], [y])
    in_estimate = model.coefficients[FEATURES.index("in_estimate")]
    assert 3.5 < in_estimate < 5.0  # log(0.9/0.1) * 2 = 4.39


def test_a_fit_round_trips_through_its_release(tmp_path):
    model = OnFloorModel("womens", tuple(float(i) for i in range(len(FEATURES))), True)
    path = model.save(tmp_path / "fit.json", seasons=[2025])
    assert OnFloorModel.load("womens", True, path) == model


def test_every_player_seen_gets_a_row_for_every_window():
    played = game(*starters_act(), missed_2("home", "h6"))
    kinds = [classify(p) for p in played.plays]
    floors = [(frozenset(), frozenset())] * len(kinds)
    frame = features(played, kinds, floors, floors, {"home": {}, "away": {}})
    assert frame.matrix.shape[1] == len(FEATURES)
    home_rows = frame.side == 0
    # six home players, ten away fouls and one shot (each foul a window)
    assert len(set(np.array(frame.players)[home_rows])) == 6
