"""
Each game's shooting luck, for a model that learns from results to discount:
Cassandra's Glicko, which otherwise takes a final margin as all signal.

A game's luck is each side's points on threes and free throws less what the
shot model expected of its shooters (`ratings.Targets`: the-archer's chance
from where and how each shot was taken, plus the shooter's own skill), the
home side's less the away side's. Twos are left alone: a layup is one league
chance, so finishing and rim protection would go with it, and the defense
has least say over threes and free throws -- the same split the ratings'
target ("half on 3s and FTs") makes.

The shot model knows nothing of where a game is played, and home sides
shoot better than it expects of them: 1.3 points a game mens, 1.1 womens.
That's the home edge, not luck, and the model reading this has its own; so
a game at a home site is filed less the home sides' average luck over the
same fit's games (`centered`), and a neutral site's as it is.

A game's luck is as known soon after it: the shot model and the shooters'
skills are fitted on the games before the first cut after it -- the first
weekly fit whose training games include it, in the backfill (`build.luck`);
the morning after, live (`build.live`). It doesn't change after that, so a
replay sees what a morning would have.

Measured on 2014-2025, each team's luck so far this season predicts the
next game's residual of Cassandra's `glicko_players` against it (-0.0011
logit per point mens, -0.0016 womens, t -6): a team that's been lucky is
overrated. Taking part of each game's luck out of the margin Glicko learns
from, at its 2026-10-28 fit, Brier on 2014+ games against none:

    share       0.3        0.4        0.5        0.6        1 (uncentered)
    mens     -0.00035   -0.00038   -0.00037   -0.00031   +0.0008
    womens   -0.00038   -0.00044   -0.00045   -0.00041   +0.0004

Uncentered, 0.4 is -0.00027 mens and -0.00040 womens. All of it is too
much: the shot model's expectation misses some of what's real, as the
ratings' targets found.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.csv

from .box_scores import BoxLine
from .context import GameContext
from .flow import GameFlow, flow_values
from .ratings import Rows, Targets

LUCK_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("game_id", pa.string(), nullable=False),
        pa.field("date", pa.date32(), nullable=False),
        pa.field("home_team_id", pa.string(), nullable=False),
        pa.field("home", pa.string()),
        pa.field("away_team_id", pa.string(), nullable=False),
        pa.field("away", pa.string()),
        # points, the home side's luck less the away side's
        pa.field("luck", pa.float32(), nullable=False),
        # the cut it was fitted at: on the games before it
        pa.field("as_of", pa.date32(), nullable=False),
        # `flow`'s readings of the game, each in points toward the home
        # side; empty for a game whose plays don't give them
        pa.field("jumpers", pa.float32()),
        pa.field("over_lead", pa.float32()),
        pa.field("closing", pa.float32()),
        # where `luck` came from: the shot model (`SHOTS`, and what a row
        # from before this column is), or the box score (`BOX`)
        pa.field("source", pa.string()),
    ]
)

# The kinds of shot whose luck is counted: threes and free throws.
LUCKY = (3, 1)

SHOTS = "shots"
BOX = "box"

# A box score's threes and free throws: the columns of `box_scores.BoxLine.
# stats`, the points each is worth, about the league's make rate, and how
# many attempts of that are mixed into a season's own.
_BOX_KINDS = (("tpm", "tpa", 3, 0.34, 2000.0), ("ftm", "fta", 1, 0.70, 2000.0))


def game_luck(rows: Rows, fitted: Targets, games: Iterable[str]) -> dict[str, float]:
    """Each of `games`' luck in points, home side less away, from `fitted`'s
    shot model and skills."""
    per_row = sum(fitted.actual[v] - fitted.shooter[v] for v in LUCKY)
    points = per_row * rows.possessions * np.where(rows.offense_home, 1.0, -1.0)
    wanted = set(games)
    found: dict[str, float] = defaultdict(float)
    for game_id, value in zip(rows.game_id, points):
        if game_id in wanted:
            found[game_id] += float(value)
    return dict(found)


def centered(
    luck: Mapping[str, float], games: Iterable[str], contexts: Mapping[str, GameContext]
) -> dict[str, float]:
    """`games`' luck less the home sides' average over all of `luck`'s games
    at a home site -- what the same fit says of everyone -- a neutral site's
    as it is. A game without a context is left out, as `luck_rows` would."""
    home = [
        v for g, v in luck.items() if g in contexts and not contexts[g].neutral_site
    ]
    edge = float(np.mean(home)) if home else 0.0
    return {
        g: luck[g] - (0.0 if contexts[g].neutral_site else edge)
        for g in games
        if g in luck and g in contexts
    }


def luck_rows(
    league: str,
    season: int,
    as_of: date,
    luck: Mapping[str, float],
    contexts: Mapping[str, GameContext],
    day: Mapping[str, date],
    source: str = SHOTS,
) -> list[dict]:
    """The file's rows for `luck`'s games; a game without a context has no
    teams to file it under, and is left out."""
    return [
        {
            "league": league,
            "season": season,
            "game_id": game_id,
            "date": day[game_id],
            "home_team_id": contexts[game_id].home.team_id,
            "home": contexts[game_id].home.name,
            "away_team_id": contexts[game_id].away.team_id,
            "away": contexts[game_id].away.name,
            "luck": value,
            "as_of": as_of,
            "source": source,
        }
        for game_id, value in sorted(luck.items())
        if game_id in contexts
    ]


def box_luck(
    lines: Iterable[BoxLine], contexts: Mapping[str, GameContext]
) -> dict[str, float]:
    """Each game's luck as its box score has it, for a game the shot model
    never saw -- one with no plays, as every game before 2013-14 is: each
    side's points on threes and free throws less the make rates of `lines`'
    own games times its attempts, the home side's less the away's.

    The best estimate without the shots: it knows how many went in and not
    who took them or from where. A game is left out without a context, or
    unless both of its teams have a line."""
    totals: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for line in lines:
        team = totals[line.game_id, line.team_id]
        for made, attempted, *_ in _BOX_KINDS:
            team[made] += line.stats.get(made, 0.0)
            team[attempted] += line.stats.get(attempted, 0.0)
    rates = {
        made: (sum(t[made] for t in totals.values()) + prior * weight)
        / (sum(t[attempted] for t in totals.values()) + weight)
        for made, attempted, _, prior, weight in _BOX_KINDS
    }

    def beyond(team: Mapping[str, float]) -> float:
        return sum(
            points * (team[made] - rates[made] * team[attempted])
            for made, attempted, points, *_ in _BOX_KINDS
        )

    found = {}
    for game_id, context in contexts.items():
        home = totals.get((game_id, context.home.team_id))
        away = totals.get((game_id, context.away.team_id))
        if home is not None and away is not None:
            found[game_id] = beyond(home) - beyond(away)
    return found


def with_flow(
    rows: list[dict], values: Mapping[str, Mapping[str, float]], replace: bool = False
) -> list[dict]:
    """`rows` with `values`' columns (`flow.flow_values`: by column, then
    game) filled in where a game has one. A reading already filed stays as
    first known, like the luck beside it, unless `replace`."""
    filled = []
    for row in rows:
        row = dict(row)
        for column, by_game in values.items():
            if row["game_id"] in by_game and (replace or row.get(column) is None):
                row[column] = by_game[row["game_id"]]
        filled.append(row)
    return filled


def with_season(
    known: list[dict],
    league: str,
    season: int,
    flows: Mapping[str, GameFlow],
    lines: Iterable[BoxLine],
    contexts: Mapping[str, GameContext],
    day: Mapping[str, date],
    as_of: date | None = None,
    replace: bool = False,
) -> list[dict]:
    """The file's rows with what a season's plays and box scores add to the
    shot model's luck: `flow`'s readings on every game that has them, and a
    row of box-score luck (`box_luck`) for each game the file has none for.

    Every game given is as known at one cut, so each column is centered
    (`centered`) on its own home sides' average over them. A box-score row
    is filed as of `as_of`, or the day after its game without one. `replace`
    rebuilds the season: its box-score rows and flow readings are dropped
    first, the shot model's luck kept.
    """

    def mine(row: dict) -> bool:
        return row["league"] == league and row["season"] == season

    if replace:
        known = [r for r in known if not (mine(r) and r.get("source") == BOX)]
    have = {r["game_id"] for r in known if r["league"] == league}
    boxed = box_luck(lines, contexts)
    new = sorted(g for g in boxed if g not in have and g in day)
    added = [
        row
        for game_id, value in centered(boxed, new, contexts).items()
        for row in luck_rows(
            league,
            season,
            as_of or day[game_id] + timedelta(days=1),
            {game_id: value},
            contexts,
            day,
            source=BOX,
        )
    ]
    values = {
        column: centered(by_game, by_game, contexts)
        for column, by_game in flow_values(flows, league).items()
    }
    return [
        with_flow([row], values, replace)[0] if mine(row) else row
        for row in merged_luck(known, added)
    ]


def merged_luck(existing: list[dict], found: list[dict]) -> list[dict]:
    """The file's rows with `found`'s games added: a game already there
    keeps its luck, as first known."""
    have = {(r["league"], r["game_id"]) for r in existing}
    added = [r for r in found if (r["league"], r["game_id"]) not in have]
    return sorted(
        existing + added,
        key=lambda r: (r["league"], r["season"], r["date"], r["game_id"]),
    )


def read_luck(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return pyarrow.csv.read_csv(
        path, convert_options=pyarrow.csv.ConvertOptions(column_types=LUCK_SCHEMA)
    ).to_pylist()


def write_luck(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pyarrow.csv.write_csv(pa.Table.from_pylist(rows, LUCK_SCHEMA), path)
