"""
What a newcomer is worth before he's played: his high school recruiting rank.

A player last season didn't rate has nothing to carry over, and the carryover
test (`carryover`) counted every one of them alike -- a top-five recruit and
a walk-on both "a newcomer". But ranked freshmen rate well above unranked
ones in their first season (in the ratings `season_ratings` writes, top-25
recruits come in about 5 points per 100 above unranked freshmen for mens, 7
for womens), so a newcomer's rank says where to start him.

The recruits are ESPN's top 100 of each high school class, as so-high-school
builds them, matched to ESPN athlete ids: `{root}/{league}/recruits.csv`,
read by its columns (`make recruits` copies them from S3). A newcomer
counts as ranked if his class is this season's or last season's (a redshirt
or a late start).

`RecruitModel` is a regression of every newcomer's first-season rating on
whether he was ranked and how high, weighted by his possessions, for
offense and defense: learned on the `season_ratings` of every season it's
given, and shipped in `releases/`.
"""

import csv
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_RELEASES = Path(__file__).parent / "releases"
FEATURES = ("ranked", "rank_score")


@dataclass(frozen=True)
class Recruit:
    athlete_id: str
    class_year: int
    rank: int
    grade: int


def read_recruits(root: str | Path, league: str) -> dict[str, Recruit]:
    """The matched recruits of `league`, by ESPN athlete id."""
    found = {}
    with (Path(root) / league / "recruits.csv").open() as file:
        for row in csv.DictReader(file):
            if row["athlete_id"]:
                found[row["athlete_id"]] = Recruit(
                    row["athlete_id"],
                    int(row["class_year"]),
                    int(row["rank"]),
                    int(row["grade"] or 0),
                )
    return found


def features(recruit: Recruit | None, season: int) -> tuple[float, float]:
    """(ranked, rank score) for a newcomer in `season`: 1 and (101 - rank) /
    100 if he's a recruit of this class or last, else 0 and 0."""
    if recruit is None or recruit.class_year not in (season, season - 1):
        return 0.0, 0.0
    return 1.0, (101 - recruit.rank) / 100


@dataclass(frozen=True)
class Newcomer:
    """A player in his first rated season, as `RecruitModel` learns from him."""

    season: int
    athlete_id: str
    offense: float  # per possession, a fifth of his team's included
    defense: float
    possessions: float


def newcomers(tables: Mapping[int, Iterable[Mapping]]) -> list[Newcomer]:
    """
    Every player in a season's ratings (`season_ratings.parquet` rows) who
    wasn't in the season before's: seasons with no season before are left
    out, since there's no telling who's new.
    """
    seen = {season: {r["athlete_id"] for r in rows} for season, rows in tables.items()}
    out = []
    for season, rows in tables.items():
        if season - 1 not in seen:
            continue
        for r in rows:
            if r["athlete_id"] in seen[season - 1]:
                continue
            out.append(
                Newcomer(
                    season,
                    r["athlete_id"],
                    r["offense"] / 100,
                    r["defense"] / 100,
                    float(min(r["offense_possessions"], r["defense_possessions"])),
                )
            )
    return out


@dataclass(frozen=True)
class RecruitModel:
    """A newcomer's (offense, defense) per possession from his recruiting rank:
    intercept, then one coefficient per `FEATURES`."""

    league: str
    offense: tuple[float, ...]
    defense: tuple[float, ...]

    @staticmethod
    def release_path(league: str) -> Path:
        return _RELEASES / f"recruit-prior-{league}.json"

    @classmethod
    def learn(
        cls,
        league: str,
        found: Iterable[Newcomer],
        recruits: Mapping[str, Recruit],
    ) -> RecruitModel:
        found = list(found)
        x = np.array(
            [(1.0, *features(recruits.get(n.athlete_id), n.season)) for n in found]
        )
        w = np.array([n.possessions for n in found])
        beta = {}
        for side in ("offense", "defense"):
            y = np.array([getattr(n, side) for n in found])
            beta[side] = np.linalg.solve(
                (x * w[:, None]).T @ x + 1e-6 * np.eye(x.shape[1]),
                (x * w[:, None]).T @ y,
            )
        return cls(
            league,
            tuple(map(float, beta["offense"])),
            tuple(map(float, beta["defense"])),
        )

    def rating(self, recruit: Recruit | None, season: int) -> tuple[float, float]:
        """What a newcomer in `season` is expected to be, per possession."""
        x = np.array((1.0, *features(recruit, season)))
        return float(x @ np.array(self.offense)), float(x @ np.array(self.defense))

    def save(self, path: Path | None = None, **about) -> Path:
        path = path or self.release_path(self.league)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = ("intercept", *FEATURES)
        path.write_text(
            json.dumps(
                {
                    "league": self.league,
                    "features": FEATURES,
                    "offense": dict(zip(names, self.offense)),
                    "defense": dict(zip(names, self.defense)),
                    **about,
                },
                indent=2,
            )
            + "\n"
        )
        return path

    @classmethod
    def load(cls, league: str, path: Path | None = None) -> RecruitModel:
        path = path or cls.release_path(league)
        raw = json.loads(path.read_text())
        if tuple(raw["features"]) != FEATURES:
            raise ValueError(f"{path} was learned on other features: {raw['features']}")
        names = ("intercept", *FEATURES)
        return cls(
            raw["league"],
            tuple(raw["offense"][n] for n in names),
            tuple(raw["defense"][n] for n in names),
        )
