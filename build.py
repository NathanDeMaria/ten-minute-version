"""
Build the possession tables for a season, and say how far to trust them.

    uv run python build.py stints --league mens --season 2025 --root ./plays

writes `{out}/league={league}/season={season}/{stints,games,players}.parquet`
and prints the checks: how many games had substitutions, the share of
possessions that came out five on five (or five against a generic team from
outside Division I), how often the points counted match ESPN's final, and
possessions per side per game.

    uv run python build.py evaluate --league mens --season 2025 --train 2024
    uv run python build.py fit --league mens --seasons 2024,2025

score the estimated lineups on games whose real ones are known (see
`ten_minute_version.evaluate`), and fit the on-floor probabilities that ship
in `ten_minute_version/releases/`.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from fire import Fire

from ten_minute_version.estimate import Rules, nearest_action_floors
from ten_minute_version.evaluate import (
    Known,
    accuracy,
    calibration,
    chances_for,
    estimate_season,
    fit_on_floor,
    known_games,
)
from ten_minute_version.opponents import appearances
from ten_minute_version.plays import read_games
from ten_minute_version.probabilities import FEATURES, OnFloorModel
from ten_minute_version.stints import season_tables


def stints(league: str, season: int, root: str = "./plays", out: str = "./stints"):
    # Read twice: once to count each player's games, which says which sides
    # are teams from outside Division I, then to build.
    seen = appearances(read_games(root, league, season))
    games = read_games(root, league, season)
    stint_table, game_table, player_table = season_tables(games, seen)
    directory = Path(out) / f"league={league}" / f"season={season}"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(stint_table, directory / "stints.parquet", compression="zstd")
    pq.write_table(game_table, directory / "games.parquet", compression="zstd")
    pq.write_table(player_table, directory / "players.parquet", compression="zstd")
    print(json.dumps(checks(stint_table, game_table), indent=2))


def checks(stint_table: pa.Table, game_table: pa.Table) -> dict:
    games = game_table.to_pylist()
    lined = [game for game in games if game["has_substitutions"]]
    n_lined = max(len(lined), 1)
    possessions = sum(game["possessions"] for game in lined)
    dropped = sum(game["possessions_dropped"] for game in lined)
    points_match = sum(
        (game["home_points"], game["away_points"])
        == (game["home_final"], game["away_final"])
        for game in lined
    )
    generic = [game for game in lined if game["generic_side"]]
    return {
        "games": len(games),
        "games_with_substitutions": len(lined),
        "starters_found": round(sum(g["starters_found"] for g in lined) / n_lined, 4),
        "repairs_per_game": round(sum(g["repairs"] for g in lined) / n_lined, 2),
        "possessions_per_side_per_game": round(possessions / n_lined / 2, 1),
        "possessions_five_on_five": round(1 - dropped / max(possessions, 1), 4),
        "games_with_a_generic_side": len(generic),
        "possessions_in_them": sum(game["possessions"] for game in generic),
        "points_match_final": round(points_match / n_lined, 4),
        "stint_rows": stint_table.num_rows,
        "stint_rows_per_game": round(stint_table.num_rows / n_lined, 1),
    }


# Each rule switched on in turn, so the table says what each one adds.
_LADDER = [
    ("persistent, swap where forced", Rules(None, False, False, 0.0, 0.0, 0.0)),
    ("+ backdate swaps a third", Rules(0.36, False, False, 0.0, 0.0, 0.0)),
    ("+ snap to TV timeouts", Rules(0.36, True, False, 0.0, 0.0, 0.0)),
    ("+ starters back at half", Rules(0.36, True, True, 0.0, 0.0, 0.0)),
    ("+ swap like for like", Rules(0.36, True, True, 0.0, 400.0, 0.0)),
    ("+ foul trouble sits", Rules()),
]


def evaluate(
    league: str,
    season: int,
    train: int | None = None,
    root: str = "./plays",
    old_feed: bool = False,
):
    """
    Hard estimates rule by rule, then the on-floor probabilities -- fitted on
    `train` when given (an honest out-of-season score), otherwise the
    shipped release.
    """
    known = known_games(read_games(root, league, season), old_feed)
    print(
        f"{len(known)} games with known lineups, {league} {season}"
        + (", ids stripped as the old feed" if old_feed else "")
    )
    rows = [("nearest action", accuracy(known, _by_game(known, None)))]
    for name, rules in _LADDER:
        rows.append((name, accuracy(known, _by_game(known, rules))))
    rows.append(
        (
            "+ box-score starters and minutes",
            accuracy(known, _by_game(known, Rules(), box=True)),
        )
    )
    print(f"{'hard estimate':<34}{'players right':>14}{'all five right':>16}")
    for name, result in rows:
        print(f"{name:<34}{result.players_right:>14.1%}{result.all_five_right:>16.1%}")

    trained = known_games(read_games(root, league, train), old_feed) if train else None
    print(f"\n{'brier':<34}{'probabilities':>14}{'hard estimate':>16}")
    for box in (False, True):
        if trained is not None:
            model = fit_on_floor(league, trained, box=box)
        else:
            model = OnFloorModel.load(league, box)
        chances = chances_for(model, known, box=box)
        result = calibration(
            known,
            lambda k: chances[k.game.game_id],
            _by_game(known, Rules(), box=box),
        )
        name = "with box scores" if box else "without box scores"
        print(f"{name:<34}{result.brier:>14.4f}{result.hard_brier:>16.4f}")
        if not box:
            print("  calibration (mean chance -> share really on):")
            for mean, real, n in result.deciles:
                print(f"    {mean:5.2f} -> {real:5.2f}  ({n})")


def _by_game(known: list[Known], rules: Rules | None, box: bool = False):
    """Each game's floors under `rules` (None for nearest action), by game."""
    if rules is None:
        floors = [nearest_action_floors(k.game, k.kinds) for k in known]
    else:
        _, floors = estimate_season(known, rules, box)
    by_id = dict(zip((k.game.game_id for k in known), floors))
    return lambda k: by_id[k.game.game_id]


def fit(league: str, seasons, root: str = "./plays", old_feed: bool = False):
    """
    Fit the on-floor probabilities on known games and write the releases.
    `--seasons 2024,2025` (Fire reads that as a tuple) or one season.
    """
    years = [
        int(year)
        for year in (seasons if isinstance(seasons, (tuple, list)) else [seasons])
    ]
    known = [
        k
        for year in years
        for k in known_games(read_games(root, league, year), old_feed)
    ]
    for box in (False, True):
        model = fit_on_floor(league, known, box=box)
        path = model.save(seasons=years, games=len(known), old_feed=old_feed)
        print(f"wrote {path}")
        for name, value in zip(FEATURES, model.coefficients):
            print(f"    {name:<26}{value:+.3f}")


if __name__ == "__main__":
    Fire({"stints": stints, "evaluate": evaluate, "fit": fit})
