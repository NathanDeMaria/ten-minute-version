from datetime import date

import pyarrow as pa
import pyarrow.parquet as pq

from .plays import read_games


def _write_day(root, league, season, day, rows):
    directory = root / f"league={league}" / f"season={season}" / f"date={day}"
    directory.mkdir(parents=True)
    # The columns endgame writes that this package reads, plus one it doesn't
    table = pa.Table.from_pylist(
        [
            {
                "game_id": game_id,
                "play_number": number,
                "period": 1,
                "clock_seconds": 1200 - number,
                "side": "home",
                "play_category_id": "1002",
                "play_type": "JumpShot",
                "text": f"play {number}",
                "athlete_id": "p1",
                "athlete_name": "Player One",
                "participant_ids": ["p1"],
                "home_score": 2 * number,
                "away_score": 0,
                "shot_x": 19.0,
                "shot_y": 12.0,
                "scoring_play": True,
            }
            for game_id, number in rows
        ]
    )
    pq.write_table(table, directory / "data.parquet")


def test_games_come_back_in_date_order_with_their_plays_in_order(tmp_path):
    _write_day(tmp_path, "mens", 2025, "2026-01-22", [("g3", 1)])
    _write_day(tmp_path, "mens", 2025, "2026-01-21", [("g2", 2), ("g1", 1), ("g2", 1)])
    games = list(read_games(tmp_path, "mens", 2025))
    assert [(g.date, g.game_id, len(g.plays)) for g in games] == [
        (date(2026, 1, 21), "g1", 1),
        (date(2026, 1, 21), "g2", 2),
        (date(2026, 1, 22), "g3", 1),
    ]
    g2 = games[1]
    assert [p.text for p in g2.plays] == ["play 1", "play 2"]
    assert g2.plays[0].participant_ids == ("p1",)
    assert g2.plays[0].clock_seconds == 1199
    assert (g2.plays[0].x, g2.plays[0].y) == (19.0, 12.0)


def test_a_season_with_nothing_processed_has_no_games(tmp_path):
    assert list(read_games(tmp_path, "womens", 2025)) == []
