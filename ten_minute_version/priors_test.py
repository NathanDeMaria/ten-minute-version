import numpy as np
import pyarrow as pa

from .box_scores import STATS, BoxLine
from .context import GameContext, Team
from .priors import BoxModel, box_prior, box_profile, exposures, last_season_prior
from .ratings import Design, Rows


def _line(
    game_id: str, player: str, minutes: float, points: float, position: str = "G"
) -> BoxLine:
    stats = {stat: 0.0 for stat in STATS} | {"minutes": minutes, "points": points}
    return BoxLine(game_id, "t", player, True, position, stats)


def test_rates_are_per_40_and_steadied_toward_the_league():
    lines = [
        _line("g1", "busy", 40, 40),
        _line("g1", "brief", 1, 10),
        _line("g1", "quiet", 40, 0),
    ]
    profile = box_profile(lines, {"g1"})
    points = {p: profile.features[i][0] for i, p in enumerate(profile.players)}
    # Ten points in a minute is 400 per 40 -- but a minute is next to nothing
    assert points["brief"] < 40
    assert points["busy"] > points["quiet"]


def test_lines_from_other_games_are_left_out():
    profile = box_profile([_line("g1", "a", 30, 10), _line("g2", "b", 30, 10)], {"g1"})
    assert profile.players == ["a"]


def _rows() -> Rows:
    five = ["p1", "p2", "p3", "p4", "p5"]
    other = ["q1", "q2", "q3", "q4", "q5"]
    keys = (
        "game_id",
        "stint",
        "offense_side",
        "offense_players",
        "defense_players",
        "possessions",
        "points",
    )
    table = pa.Table.from_pylist(
        [
            dict(zip(keys, ("g1", 1, "home", five, other, 10, 11))),
            dict(zip(keys, ("g1", 1, "away", other, five, 10, 9))),
        ]
    )
    team = Team("t", "t", None, None, True)
    return Rows.build(table, {"g1": GameContext("g1", False, False, team, team)})


def test_the_box_prior_follows_what_the_box_score_says():
    rows = _rows()
    design = Design.build(rows)
    players = design.players
    # Fitted ratings that rise with points per 40: the prior should too.
    scoring = {p: float(i) for i, p in enumerate(players)}
    fitted = np.zeros(design.columns)
    for i, p in enumerate(players):
        fitted[design.player_column(i, "offense")] = 0.01 * scoring[p]
    profile = box_profile(
        [_line("g1", p, 40, 10 * scoring[p]) for p in players], {"g1"}
    )
    prior = box_prior(design, fitted, profile, {p: 100.0 for p in players})
    offense = [prior[design.player_column(i, "offense")] for i in range(len(players))]
    assert np.corrcoef(offense, [scoring[p] for p in players])[0, 1] > 0.99


def test_last_season_is_scaled_and_a_newcomer_starts_at_zero():
    design = Design.build(_rows())
    prior = last_season_prior(design, {"p1": (0.04, -0.02)}, 0.5)
    i = design.players.index("p1")
    assert np.isclose(prior[design.player_column(i, "offense")], 0.02)
    assert np.isclose(prior[design.player_column(i, "defense")], -0.01)
    j = design.players.index("q1")
    assert prior[design.player_column(j, "offense")] == 0


def test_exposure_counts_both_ends():
    rows = _rows()
    seen = exposures(rows, np.array([True, True]))
    assert seen["p1"] == 20 and seen["q1"] == 20


def test_a_box_model_learned_on_one_season_rates_another(tmp_path):
    rows = _rows()
    design = Design.build(rows)
    players = design.players
    scoring = {p: float(i) for i, p in enumerate(players)}
    fitted = np.zeros(design.columns)
    for i, p in enumerate(players):
        fitted[design.player_column(i, "offense")] = 0.01 * scoring[p]
    learned_on = box_profile(
        [_line("g1", p, 40, 10 * scoring[p]) for p in players], {"g1"}
    )
    model = BoxModel.learn(
        "mens", design, fitted, learned_on, dict.fromkeys(players, 100.0)
    )
    path = model.save(tmp_path / "box.json", season=2025)
    assert BoxModel.load("mens", path) == model
    # Another season: p1 scores most now, and a player with no box score
    # starts at zero.
    lines = [_line("g2", p, 40, 0) for p in players if p != "q5"]
    lines[0] = _line("g2", players[0], 40, 200)
    prior = model.prior(design, box_profile(lines, {"g2"}))
    offense = [prior[design.player_column(i, "offense")] for i in range(len(players))]
    assert offense[0] == max(offense)
    assert prior[design.player_column(players.index("q5"), "offense")] == 0
