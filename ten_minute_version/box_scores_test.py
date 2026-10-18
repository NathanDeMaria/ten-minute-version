from .box_scores import parse_box_lines

_KEYS = [
    "minutes",
    "points",
    "fieldGoalsMade-fieldGoalsAttempted",
    "threePointFieldGoalsMade-threePointFieldGoalsAttempted",
    "freeThrowsMade-freeThrowsAttempted",
    "rebounds",
    "assists",
    "turnovers",
    "steals",
    "blocks",
    "offensiveRebounds",
    "defensiveRebounds",
    "fouls",
]


def _athlete(athlete_id: str | None, **entry) -> dict:
    athlete = {"position": {"abbreviation": "G"}}
    if athlete_id:
        athlete["id"] = athlete_id
    return {
        "athlete": athlete,
        "starter": True,
        "didNotPlay": False,
        "stats": [
            "30",
            "12",
            "4-9",
            "2-5",
            "2-2",
            "6",
            "3",
            "1",
            "2",
            "0",
            "1",
            "5",
            "2",
        ],
        **entry,
    }


def _summary(*athletes: dict) -> dict:
    return {
        "boxscore": {
            "players": [
                {
                    "team": {"id": "2509"},
                    "statistics": [{"keys": _KEYS, "athletes": list(athletes)}],
                }
            ]
        }
    }


def test_a_line_splits_the_pairs_and_keeps_the_rest():
    (line,) = parse_box_lines("g1", _summary(_athlete("p1")))
    assert (line.team_id, line.athlete_id, line.starter, line.position) == (
        "2509",
        "p1",
        True,
        "G",
    )
    assert (
        line.stats["fgm"],
        line.stats["fga"],
        line.stats["tpa"],
        line.stats["dreb"],
    ) == (4, 9, 5, 5)


def test_players_without_stats_or_an_id_have_no_line():
    lines = parse_box_lines(
        "g1",
        _summary(
            _athlete("p1"), _athlete("p2", didNotPlay=True, stats=[]), _athlete(None)
        ),
    )
    assert [line.athlete_id for line in lines] == ["p1"]
