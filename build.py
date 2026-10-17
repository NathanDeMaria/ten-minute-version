"""
Build the possession tables for a season, and say how far to trust them.

    uv run python build.py stints --league mens --season 2025 --root ./plays

writes `{out}/league={league}/season={season}/{stints,games,players}.parquet`
and prints the checks: how many games had substitutions, the share of
possessions that came out five on five, how often the points counted match
ESPN's final, and possessions per side per game.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from fire import Fire

from ten_minute_version.plays import read_games
from ten_minute_version.stints import season_tables


def stints(league: str, season: int, root: str = "./plays", out: str = "./stints"):
    games = read_games(root, league, season)
    stint_table, game_table, player_table = season_tables(games)
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
    return {
        "games": len(games),
        "games_with_substitutions": len(lined),
        "starters_found": round(sum(g["starters_found"] for g in lined) / n_lined, 4),
        "repairs_per_game": round(sum(g["repairs"] for g in lined) / n_lined, 2),
        "possessions_per_side_per_game": round(possessions / n_lined / 2, 1),
        "possessions_five_on_five": round(1 - dropped / max(possessions, 1), 4),
        "points_match_final": round(points_match / n_lined, 4),
        "stint_rows": stint_table.num_rows,
        "stint_rows_per_game": round(stint_table.num_rows / n_lined, 1),
    }


if __name__ == "__main__":
    Fire({"stints": stints})
