"""
The coming season's rosters, from ESPN's team roster endpoint: who's on each
Division I team before it plays, for the preseason projection (`preseason`).

ESPN serves only the current roster, so this can only be fetched for the
season about to start (or under way). Each team's response is kept whole,
as `{root}/{league}/{season}/{team}.json`, so reading it differently later
costs ESPN nothing.
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Iterable

import pyarrow as pa

_URL = (
    "https://site.api.espn.com/apis/site/v2/sports/basketball"
    "/{league}-college-basketball/teams/{team}/roster"
)
_HEADERS = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64)"}

ROSTER_SCHEMA = pa.schema(
    [
        pa.field("team_id", pa.string(), nullable=False),
        pa.field("athlete_id", pa.string(), nullable=False),
        pa.field("name", pa.string()),
        # ESPN's years of eligibility used: 1 for a freshman
        pa.field("experience", pa.int32()),
        pa.field("position", pa.string()),
        pa.field("height_inches", pa.float64()),
        # ESPN's injury status, if it lists one ("Out", "Day-To-Day", ...)
        pa.field("injury", pa.string()),
    ]
)


def raw_path(root: str | Path, league: str, season: int, team: str) -> Path:
    return Path(root) / league / str(season) / f"{team}.json"


def fetch(
    root: str | Path, league: str, season: int, team: str, refetch: bool = False
) -> dict:
    """A team's roster response, fetched and kept if it isn't already. Some
    teams' rosters are still last season's for a while: `current` says
    whether one is for `season`."""
    path = raw_path(root, league, season, team)
    if refetch or not path.is_file():
        request = urllib.request.Request(
            _URL.format(league=league, team=team), headers=_HEADERS
        )
        body = _read(request)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        time.sleep(0.3)
    return json.loads(path.read_text())


def current(response: dict, season: int) -> bool:
    """Whether a roster response is for `season` (ESPN names a season for the
    year it ends in)."""
    return response.get("season", {}).get("year") == season + 1


def _read(request, tries: int = 6, wait: float = 2.0) -> bytes:
    problem = ""
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            if error.code < 500:
                raise
            problem = f"HTTP {error.code}"
        except urllib.error.URLError as error:
            problem = str(error.reason)
        time.sleep(wait * 2**attempt)
    raise RuntimeError(f"{request.full_url}: {problem} after {tries} tries")


def roster_rows(team: str, response: dict) -> Iterable[dict]:
    for athlete in response.get("athletes", []):
        if not athlete.get("id"):
            continue
        injuries = athlete.get("injuries") or []
        injury = (injuries[0].get("status") or "") if injuries else ""
        yield {
            "team_id": team,
            "athlete_id": str(athlete["id"]),
            "name": athlete.get("displayName"),
            "experience": (athlete.get("experience") or {}).get("years"),
            "position": (athlete.get("position") or {}).get("abbreviation"),
            "height_inches": athlete.get("height"),
            "injury": injury or None,
        }


def roster_table(rows: Iterable[dict]) -> pa.Table:
    return pa.Table.from_pylist(list(rows), schema=ROSTER_SCHEMA)
