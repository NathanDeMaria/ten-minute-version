import numpy as np
import pyarrow as pa

from .context import GameContext, Team
from .ratings import Design, Rows, game_margins, penalties, solve, targets


def _team(team_id: str) -> Team:
    return Team(team_id, team_id, None, None, True)


def _stints(rows: list[tuple]) -> pa.Table:
    """(game_id, stint, offense_side, offense, defense, possessions, points) per row."""
    keys = (
        "game_id",
        "stint",
        "offense_side",
        "offense_players",
        "defense_players",
        "possessions",
        "points",
    )
    return pa.Table.from_pylist([dict(zip(keys, row)) for row in rows])


HOME = ["h1", "h2", "h3", "h4", "h5"]
AWAY = ["a1", "a2", "a3", "a4", "a5"]


def _contexts(neutral: bool = False) -> dict[str, GameContext]:
    return {
        g: GameContext(g, neutral, False, _team("H"), _team("A")) for g in ("g1", "g2")
    }


def test_the_home_edge_is_both_ends_and_nothing_at_a_neutral_site():
    table = _stints(
        [("g1", 1, "home", HOME, AWAY, 10, 12), ("g1", 1, "away", AWAY, HOME, 10, 9)]
    )
    assert list(Rows.build(table, _contexts()).home) == [1.0, -1.0]
    assert list(Rows.build(table, _contexts(neutral=True)).home) == [0.0, 0.0]


def test_a_player_whose_lineups_score_more_rates_higher():
    rng = np.random.default_rng(3)
    good, plain = ["h1", "h2", "h3", "h4", "h5"], ["h6", "h2", "h3", "h4", "h5"]
    rows = []
    for stint in range(400):
        five = good if stint % 2 else plain
        expected = 1.2 if five is good else 1.0
        rows.append(
            ("g1", stint, "home", five, AWAY, 20, float(rng.poisson(expected * 20)))
        )
        rows.append(("g1", stint, "away", AWAY, five, 20, float(rng.poisson(20))))
    built = Rows.build(_stints(rows), _contexts(neutral=True))
    design = Design.build(built)
    b = solve(
        design,
        built.points / built.possessions,
        built.possessions,
        penalties(design, player=10.0),
    )
    at = {p: i for i, p in enumerate(design.players)}
    rating = lambda p: b[design.player_column(at[p], "offense")]  # noqa: E731
    assert rating("h1") - rating("h6") > 0.1


def test_with_nothing_to_go_on_a_rating_is_its_prior():
    built = Rows.build(_stints([("g1", 1, "home", HOME, AWAY, 10, 10)]), _contexts())
    design = Design.build(built)
    prior = np.zeros(design.columns)
    prior[design.player_column(0, "offense")] = 0.05
    b = solve(
        design,
        built.points / built.possessions,
        np.zeros(len(built)),
        penalties(design, player=100.0),
        prior=prior,
    )
    assert np.isclose(b[design.player_column(0, "offense")], 0.05)


def test_team_columns_carry_the_offense_and_defense_teams():
    built = Rows.build(_stints([("g1", 1, "home", HOME, AWAY, 10, 10)]), _contexts())
    design = Design.build(built, players=False, teams=True)
    assert design.teams == ["A", "H"]
    entries = set(zip(design.col.tolist(), design.val.tolist()))
    assert (design.team_column(1, "offense"), 1.0) in entries
    assert (design.team_column(0, "defense"), -1.0) in entries


def test_a_games_margin_is_home_minus_away():
    built = Rows.build(
        _stints(
            [
                ("g1", 1, "home", HOME, AWAY, 10, 12),
                ("g1", 1, "away", AWAY, HOME, 10, 9),
            ]
        ),
        _contexts(),
    )
    predicted, actual = game_margins(
        built, np.array([1.1, 1.0]), np.array(["g1"], object)
    )
    assert np.isclose(predicted[0], 1.0) and actual[0] == 3.0


def test_with_team_columns_a_rating_carries_a_fifth_of_its_teams():
    from .ratings import ratings_table

    built = Rows.build(
        _stints(
            [
                ("g1", 1, "home", HOME, AWAY, 10, 12),
                ("g1", 1, "away", AWAY, HOME, 10, 9),
            ]
        ),
        _contexts(),
    )
    design = Design.build(built, teams=True)
    b = np.zeros(design.columns)
    h1 = design.players.index("h1")
    b[design.player_column(h1, "offense")] = 0.02
    b[design.team_column(design.teams.index("H"), "offense")] = 0.05
    table = ratings_table("mens", 2025, built, design, b, {}, {"H": "Home"})
    row = next(r for r in table.to_pylist() if r["athlete_id"] == "h1")
    assert np.isclose(row["offense_vs_team"], 2.0)
    assert np.isclose(row["offense"], 2.0 + 5.0 / 5)
    assert row["team"] == "Home"


def test_a_generic_sides_shooters_are_one_shooter():
    """Generic shooters -- a different one every few shots -- who make a
    fifth of their threes against Division I's two fifths are expected to
    make fewer than the league, not shrunk each to its average."""
    rng = np.random.default_rng(4)
    games = [str(g) for g in range(1, 41)]
    stints, shots = [], []
    for g in games:
        stints += [
            (g, 1, "home", HOME, ["generic"], 10, 0),
            (g, 1, "away", ["generic"], HOME, 10, 0),
        ]
        for i in range(30):
            for side, shooter, rate in (
                ("home", HOME[i % 5], 0.4),
                ("away", f"x{g}-{i % 3}", 0.2),
            ):
                shots.append(
                    {
                        "game_id": g,
                        "stint": 1,
                        "offense_side": side,
                        "shooter": shooter,
                        "value": 3,
                        "made": bool(rng.random() < rate),
                        "shot_type": "JumpShot",
                        "x": 25.0,
                        "y": float(rng.uniform(22.0, 26.0)),
                    }
                )
    contexts = {g: GameContext(g, False, False, _team("H"), _team("G")) for g in games}
    rows = Rows.build(_stints(stints), contexts)
    found = targets(rows, shots, set(games))
    away = ~np.asarray(rows.offense_home)
    expected, shooter = found.expected[3][away], found.shooter[3][away]
    assert shooter.mean() < expected.mean() - 0.5
