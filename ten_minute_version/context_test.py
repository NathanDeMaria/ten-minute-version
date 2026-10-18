import gzip
import json

from .context import parse_context, read_context


def _summary(neutral: bool = False, away_groups: dict | None = None) -> dict:
    d1 = {"id": "50", "name": "NCAA Division I"}
    return {
        "header": {
            "competitions": [
                {
                    "neutralSite": neutral,
                    "conferenceCompetition": True,
                    "competitors": [
                        {
                            "homeAway": "home",
                            "team": {
                                "id": "2509",
                                "displayName": "Purdue Boilermakers",
                                "groups": {
                                    "id": "7",
                                    "parent": d1,
                                    "slug": "big-ten-conference",
                                },
                            },
                        },
                        {
                            "homeAway": "away",
                            "team": {
                                "id": "275",
                                "displayName": "Wisconsin Badgers",
                                "groups": away_groups,
                            },
                        },
                    ],
                }
            ]
        }
    }


def test_the_teams_and_where_come_from_the_header():
    context = parse_context("g1", _summary(neutral=True))
    assert context is not None
    assert (context.neutral_site, context.conference_game) == (True, True)
    assert (context.home.team_id, context.home.conference) == (
        "2509",
        "big-ten-conference",
    )
    assert context.home.division_one
    assert context.team("away").name == "Wisconsin Badgers"


def test_a_team_outside_division_one_says_so():
    outside = {"id": "99", "parent": {"id": "51", "name": "NCAA Division II"}}
    context = parse_context("g1", _summary(away_groups=outside))
    assert context is not None
    assert not context.away.division_one
    # and a team with no groups at all isn't D1 either
    no_groups = parse_context("g1", _summary())
    assert no_groups is not None and not no_groups.away.division_one


def test_a_summary_without_both_teams_is_skipped():
    summary = _summary()
    summary["header"]["competitions"][0]["competitors"].pop()
    assert parse_context("g1", summary) is None


def test_a_season_is_read_from_its_directory(tmp_path):
    season = tmp_path / "mens" / "2025"
    season.mkdir(parents=True)
    with gzip.open(season / "401.json.gz", "wt") as file:
        json.dump(_summary(), file)
    assert list(read_context(tmp_path, "mens", 2025)) == ["401"]
    assert read_context(tmp_path, "womens", 2025) == {}
