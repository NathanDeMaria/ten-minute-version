"""
Build the possession tables for a season, and say how far to trust them.

    uv run python build.py stints --league mens --season 2025 --root ./plays

writes `{out}/league={league}/season={season}/{stints,games,players,shots}.parquet`
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
from collections import Counter
from collections.abc import Mapping
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute  # noqa: F401  (pa.compute)
import pyarrow.parquet as pq
from fire import Fire

from ten_minute_version.box_scores import read_box_lines, season_minutes
from ten_minute_version.carryover import (
    GameTerms,
    Result,
    compare,
    game_terms,
    variants,
)
from ten_minute_version.context import context_table, read_context
from ten_minute_version.estimate import Rules, nearest_action_floors, season_context
from ten_minute_version.estimated import estimated_tables
from ten_minute_version.evaluate import (
    Known,
    accuracy,
    calibration,
    chances_for,
    estimate_season,
    fit_on_floor,
    known_games,
)
from ten_minute_version.lineups import SIDES
from ten_minute_version.opponents import (
    Opponents,
    appearances,
    outside_side,
    schedules,
)
from ten_minute_version.plays import read_games
from ten_minute_version.preseason import FEATURES as MINUTES_FEATURES
from ten_minute_version.preseason import (
    History,
    MinutesModel,
    Spell,
    project_teams,
    team_term,
)
from ten_minute_version.preseason import features as minutes_features
from ten_minute_version.preseason import roster as season_roster
from ten_minute_version.priors import FEATURES as BOX_FEATURES
from ten_minute_version.priors import (
    BoxModel,
    box_profile,
    compare_priors,
    exposures,
    player_ratings,
)
from ten_minute_version.probabilities import FEATURES, OnFloorModel
from ten_minute_version.ratings import (
    TARGETS,
    Design,
    Rows,
    cross_validate,
    penalties,
    ratings_table,
    solve,
    targets,
)
from ten_minute_version.recruiting import (
    Recruit,
    RecruitModel,
    newcomers,
    read_recruits,
)
from ten_minute_version.recruiting import (
    features as recruit_features,
)
from ten_minute_version.rosters import current as roster_current
from ten_minute_version.rosters import fetch as fetch_roster
from ten_minute_version.rosters import roster_rows, roster_table
from ten_minute_version.stints import season_tables


def stints(
    league: str,
    season: int,
    root: str = "./plays",
    summaries: str = "./summaries",
    out: str = "./stints",
):
    # Read twice: once to count each player's games, which says which sides
    # are teams from outside Division I where a game has no summary, then to
    # build.
    seen = appearances(read_games(root, league, season))
    opponents = Opponents(seen, read_context(summaries, league, season))
    tables = season_tables(read_games(root, league, season), opponents)
    directory = Path(out) / f"league={league}" / f"season={season}"
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables._asdict().items():
        pq.write_table(table, directory / f"{name}.parquet", compression="zstd")
    print(json.dumps(checks(tables.stints, tables.games), indent=2))


def context(
    league: str, season: int, summaries: str = "./summaries", out: str = "./stints"
):
    """
    Write who played each game and where (`context.parquet`) where the
    season's stint tables go, and check the side `opponents` plays as generic
    against ESPN's own Division I flag, with the teams they disagree on --
    mostly ones ESPN files under the D1 conference they're in now.
    """
    contexts = read_context(summaries, league, season)
    directory = Path(out) / f"league={league}" / f"season={season}"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        context_table(contexts), directory / "context.parquet", compression="zstd"
    )
    found = schedules(contexts.values())
    agreement: Counter[str] = Counter()
    disputed: Counter[str] = Counter()
    for game in contexts.values():
        outside = [side for side in SIDES if not game.team(side).division_one]
        called = outside_side(game, found)
        if called is None and not outside:
            verdict = "both D1, none generic"
        elif called is not None and outside == [called]:
            verdict = "outside team generic"
        elif called is None:
            verdict = "ESPN's outside team kept as players"
        else:
            verdict = "ESPN's D1 team generic"
        agreement[verdict] += 1
        for side in outside if called is None else [called]:
            if verdict.startswith("ESPN"):
                team = game.team(side)
                schedule = found[team.team_id]
                disputed[
                    f"{team.name} ({schedule.games} games,"
                    f" {schedule.conference_games} in conference)"
                ] += 1
    stinted = directory / "games.parquet"
    games = pq.read_table(stinted).to_pylist() if stinted.exists() else []
    print(
        json.dumps(
            {
                "summaries": len(contexts),
                "games_with_plays": len(games),
                "with_a_summary": sum(game["game_id"] in contexts for game in games),
                "neutral_site": sum(c.neutral_site for c in contexts.values()),
                "conference_game": sum(c.conference_game for c in contexts.values()),
                "generic_side_vs_espn": dict(agreement),
                "disputed": dict(disputed.most_common()),
            },
            indent=2,
        )
    )


def ratings(
    league: str,
    season: int,
    summaries: str = "./summaries",
    tables: str = "./stints",
    folds: int = 5,
    team_penalty: float = 30,
    anchored_penalty: float = 1000,
):
    """
    Score the rating models and targets on held-out games, then fit the best
    on every game and write `ratings.parquet` next to the season's tables.
    """
    directory = Path(tables) / f"league={league}" / f"season={season}"
    contexts = read_context(summaries, league, season)
    stints = pq.read_table(directory / "stints.parquet")
    # A game ESPN has no summary for has no teams and no venue: left out.
    games = stints.column("game_id").to_pylist()
    stints = stints.filter(pa.array([game_id in contexts for game_id in games]))
    rows = Rows.build(stints, contexts)
    shot_rows = pq.read_table(directory / "shots.parquet").to_pylist()
    scores = cross_validate(rows, shot_rows, folds=folds)
    print(f"{len(rows)} stint rows, {len(set(rows.game_id))} games, {folds} folds")
    best = next(score for score in scores if score.model == "players")
    # The best penalty for each model and target, with how sure its gap to
    # the best players model is: a t statistic on the same held-out games.
    shown: set[tuple[str, str]] = set()
    print(
        f"{'model':9s} {'target':22s} {'penalty':>7s} {'margin RMSE':>11s}"
        f" {'MAE':>6s} {'t vs best':>9s}"
    )
    for score in scores:
        if (score.model, score.target) in shown:
            continue
        shown.add((score.model, score.target))
        t = "" if score is best else f"{score.paired_t(best):.1f}"
        print(
            f"{score.model:9s} {score.target:22s} {score.penalty:7g}"
            f" {score.rmse:11.3f} {score.mae:6.3f} {t:>9s}"
        )
    # The ratings written come from the best target, fitted with team
    # columns: players pulled toward their team rather than toward zero beat
    # players alone on held-out games (`priors`), at these penalties. Pass
    # team_penalty=0 for the players model the table above picked.
    fitted = targets(rows, shot_rows, set(rows.game_id))
    design = Design.build(rows, teams=team_penalty > 0)
    player_penalty = anchored_penalty if team_penalty > 0 else best.penalty
    b = solve(
        design,
        TARGETS[best.target](fitted),
        rows.possessions,
        penalties(design, player=player_penalty, team=team_penalty),
    )
    names = {
        r["athlete_id"]: r["athlete_name"]
        for r in pq.read_table(directory / "players.parquet").to_pylist()
    }
    team_names = {
        c.team(side).team_id: c.team(side).name
        for c in contexts.values()
        for side in ("home", "away")
    }
    table = ratings_table(league, season, rows, design, b, names, team_names)
    pq.write_table(table, directory / "ratings.parquet", compression="zstd")
    home = b[design.home] * 100
    print(
        f"\nwritten: {best.target}, player penalty {player_penalty:g}, team penalty"
        f" {team_penalty:g}. Home edge {home:+.1f} per 100 possessions at each end,"
        f" {2 * home * 0.69:+.1f} points a game"
    )
    played = [r for r in table.to_pylist() if r["offense_possessions"] >= 1000]
    for r in sorted(played, key=lambda r: -r["total"])[:15]:
        print(
            f"  {r['athlete_name'] or r['athlete_id']:24s} {r['team'] or '':28s}"
            f" off {r['offense']:+5.1f} def {r['defense']:+5.1f} total {r['total']:+5.1f}"
        )


def _anchored_fit(
    league: str,
    season: int,
    target: str,
    player_penalty: float,
    team_penalty: float,
    summaries: str,
    tables: str,
    box: BoxModel | None = None,
):
    """
    A season's team-anchored fit on all its games, real and estimated --
    pulled toward `box`'s box-score prior if given: (rows, shots, design, b),
    or None if the season isn't built.
    """
    built = _season_rows(league, season, summaries, tables, with_estimated=True)
    if built is None:
        return None
    rows, shot_rows = built
    y = TARGETS[target](targets(rows, shot_rows, set(rows.game_id)))
    design = Design.build(rows, teams=True)
    pen = penalties(design, player=player_penalty, team=team_penalty)
    prior = None
    if box is not None:
        profile = box_profile(
            read_box_lines(summaries, league, season), set(rows.game_id)
        )
        prior = box.prior(design, profile)
    return rows, shot_rows, design, solve(design, y, rows.possessions, pen, prior=prior)


def box_model(
    league: str,
    season: int = 2025,
    target: str = "half on 3s and FTs",
    player_penalty: float = 1000,
    team_penalty: float = 30,
    summaries: str = "./summaries",
    tables: str = "./stints",
):
    """
    Learn the box-score prior `season_ratings` uses (`priors.BoxModel`) from
    a season with real lineups, and write it to `releases/`: commit what
    this changes.
    """
    built = _anchored_fit(
        league, season, target, player_penalty, team_penalty, summaries, tables
    )
    assert built is not None, f"no tables for {league} {season}"
    rows, _, design, b = built
    games = set(rows.game_id)
    profile = box_profile(read_box_lines(summaries, league, season), games)
    exposure = exposures(rows, np.ones(len(rows), bool))
    model = BoxModel.learn(league, design, b, profile, exposure)
    path = model.save(
        season=season,
        games=len(games),
        target=target,
        player_penalty=player_penalty,
        team_penalty=team_penalty,
    )
    print(f"wrote {path}")
    names = ("intercept", *BOX_FEATURES)
    for name, o, d in zip(names, model.offense, model.defense):
        print(f"    {name:<22}off {100 * o:+6.2f}  def {100 * d:+6.2f}")


def season_ratings(
    league: str,
    first: int = 2013,
    last: int = 2025,
    target: str = "half on 3s and FTs",
    player_penalty: float = 1000,
    team_penalty: float = 30,
    box_prior: bool = True,
    top: int = 15,
    min_possessions: int = 1000,
    summaries: str = "./summaries",
    tables: str = "./stints",
):
    """
    Every season's players rated from all of its games -- real lineups where
    the feed has them, estimated possessions where it doesn't -- with the
    team columns `ratings` writes and, unless `box_prior` is off, pulled
    toward what the box score says (`box_model`'s release) rather than
    toward their team alone. To `season_ratings.parquet` next to each
    season's tables; and each season's best by total, among players on the
    floor for at least `min_possessions` possessions at each end.
    """
    box = BoxModel.load(league) if box_prior else None
    for season in range(first, last + 1):
        built = _anchored_fit(
            league,
            season,
            target,
            player_penalty,
            team_penalty,
            summaries,
            tables,
            box,
        )
        if built is None:
            print(f"{league} {season}: no tables")
            continue
        rows, _, design, b = built
        table = _ratings_table(league, season, rows, design, b, summaries, tables)
        directory = Path(tables) / f"league={league}" / f"season={season}"
        pq.write_table(table, directory / "season_ratings.parquet", compression="zstd")
        played = [
            r
            for r in table.to_pylist()
            if min(r["offense_possessions"], r["defense_possessions"])
            >= min_possessions
        ]
        print(
            f"\n{league} {season}-{(season + 1) % 100:02d}: {len(set(rows.game_id))}"
            f" games, {len(played)} players with {min_possessions}+ possessions",
            flush=True,
        )
        for r in sorted(played, key=lambda r: -r["total"])[:top]:
            print(
                f"  {r['athlete_name'] or r['athlete_id']:24s} {r['team'] or '':28s}"
                f" off {r['offense']:+5.1f} def {r['defense']:+5.1f}"
                f" total {r['total']:+5.1f}",
                flush=True,
            )


def _ratings_table(league, season, rows, design, b, summaries, tables) -> pa.Table:
    directory = Path(tables) / f"league={league}" / f"season={season}"
    names = {
        r["athlete_id"]: r["athlete_name"]
        for r in pq.read_table(directory / "players.parquet").to_pylist()
    }
    contexts = read_context(summaries, league, season)
    team_names = {
        c.team(side).team_id: c.team(side).name
        for c in contexts.values()
        for side in ("home", "away")
    }
    return ratings_table(league, season, rows, design, b, names, team_names)


def estimated(
    league: str,
    season: int,
    root: str = "./plays",
    summaries: str = "./summaries",
    out: str = "./stints",
):
    """
    Possession rows from estimated lineups for the season's games without
    substitutions (`ten_minute_version.estimated`): `estimated.parquet` and
    `estimated_shots.parquet` next to the stint tables.
    """
    contexts = read_context(summaries, league, season)
    box_lines = list(read_box_lines(summaries, league, season))
    seen = appearances(read_games(root, league, season))
    roles = season_context(read_games(root, league, season))
    model = OnFloorModel.load(league, with_box_scores=True)
    rows, shots = estimated_tables(
        read_games(root, league, season),
        box_lines,
        contexts,
        model,
        roles,
        Opponents(seen, contexts),
    )
    directory = Path(out) / f"league={league}" / f"season={season}"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(rows, directory / "estimated.parquet", compression="zstd")
    pq.write_table(shots, directory / "estimated_shots.parquet", compression="zstd")
    games = len(set(rows.column("game_id").to_pylist()))
    print(
        f"{rows.num_rows} estimated possessions in {games} games, {shots.num_rows} shots"
    )


def _season_rows(
    league: str, season: int, summaries: str, tables: str, with_estimated: bool = False
) -> tuple[Rows, list[dict]] | None:
    """
    A season's stint rows (games with a summary only) and shots, if built --
    and with the estimated possessions of its games without substitutions,
    if asked for and built.
    """
    directory = Path(tables) / f"league={league}" / f"season={season}"
    if not (directory / "shots.parquet").exists():
        return None
    contexts = read_context(summaries, league, season)
    stints = pq.read_table(directory / "stints.parquet")
    shots = pq.read_table(directory / "shots.parquet")
    columns = [
        "game_id",
        "stint",
        "offense_side",
        "offense_players",
        "offense_weights",
        "defense_players",
        "defense_weights",
        "possessions",
        "points",
    ]
    weights = pa.list_(pa.float32())
    stints = stints.append_column(
        "offense_weights",
        pa.array(
            [[1.0] * len(f) for f in stints.column("offense_players").to_pylist()],
            weights,
        ),
    ).append_column(
        "defense_weights",
        pa.array(
            [[1.0] * len(f) for f in stints.column("defense_players").to_pylist()],
            weights,
        ),
    )
    tables_ = [stints.select(columns).cast(_ROW_SCHEMA)]
    if with_estimated and (directory / "estimated.parquet").exists():
        tables_.append(
            pq.read_table(directory / "estimated.parquet")
            .select(columns)
            .cast(_ROW_SCHEMA)
        )
        shots = pa.concat_tables(
            [shots, pq.read_table(directory / "estimated_shots.parquet")]
        )
    combined = pa.concat_tables(tables_)
    games = combined.column("game_id").to_pylist()
    combined = combined.filter(pa.array([game_id in contexts for game_id in games]))
    return Rows.build(combined, contexts), shots.to_pylist()


_ROW_SCHEMA = pa.schema(
    [
        ("game_id", pa.string()),
        ("stint", pa.int16()),
        ("offense_side", pa.string()),
        ("offense_players", pa.list_(pa.string())),
        ("offense_weights", pa.list_(pa.float32())),
        ("defense_players", pa.list_(pa.string())),
        ("defense_weights", pa.list_(pa.float32())),
        ("possessions", pa.int16()),
        ("points", pa.int16()),
    ]
)


def priors(
    league: str,
    season: int,
    target: str = "half on 3s and FTs",
    penalty: float = 300,
    summaries: str = "./summaries",
    tables: str = "./stints",
    folds: int = 5,
):
    """
    Score each prior on the ratings against none, on held-out games.
    `target` and `penalty` are what last season's ratings are fitted with,
    for the last-season priors: by default the best of `ratings` on 2025-26.
    Last season is fitted twice where its estimated possessions are built
    (`estimated`): from its real lineups alone, and with them added.
    """
    built = _season_rows(league, season, summaries, tables)
    assert built is not None, f"no tables for {league} {season}: make stints first"
    rows, shot_rows = built
    box_lines = list(read_box_lines(summaries, league, season))
    lasts: dict[str, dict[str, tuple[float, float]]] = {}
    last_directory = Path(tables) / f"league={league}" / f"season={season - 1}"
    for which, with_estimated in (("real lineups", False), ("with estimated", True)):
        if with_estimated and not (last_directory / "estimated.parquet").exists():
            continue
        previous = _season_rows(league, season - 1, summaries, tables, with_estimated)
        if previous is None:
            continue
        last_rows, last_shots = previous
        design = Design.build(last_rows)
        y = TARGETS[target](targets(last_rows, last_shots, set(last_rows.game_id)))
        b = solve(design, y, last_rows.possessions, penalties(design, player=penalty))
        lasts[which] = player_ratings(design, b)
        print(
            f"last season, {which}: {len(lasts[which])} players from"
            f" {len(set(last_rows.game_id))} games, {len(last_rows)} rows",
            flush=True,
        )
    scores = compare_priors(
        rows, shot_rows, box_lines, lasts, target=target, folds=folds
    )
    none = next(score for score in scores if score.target == "none")
    print(
        f"{len(rows)} stint rows, {len(set(rows.game_id))} games, {folds} folds, target {target}"
    )
    print(
        f"{'prior':58s} {'penalty':>7s} {'margin RMSE':>11s} {'MAE':>6s} {'t vs none':>9s}"
    )
    shown: set[str] = set()
    for score in scores:
        if score.target in shown:
            continue
        shown.add(score.target)
        t = "" if score is none else f"{score.paired_t(none):.1f}"
        print(
            f"{score.target:58s} {score.penalty:7g} {score.rmse:11.3f} {score.mae:6.3f} {t:>9s}"
        )


def carryover(
    league: str,
    season: int,
    target: str = "half on 3s and FTs",
    penalty: float = 300,
    team_penalty: float = 30,
    tune_until: str = "",
    test_until: str = "",
    box_prior: bool = True,
    summaries: str = "./summaries",
    tables: str = "./stints",
):
    """
    Whether last season's ratings predict this season's early games
    (`ten_minute_version.carryover`): team carryover, last season's players
    from its real lineups alone (where it has any) and with its estimated
    possessions, as `season_ratings` writes them (team-anchored and pulled
    toward the box score, unless `box_prior` is off), and each with the team
    carryover. Weights are fitted on the
    games through `tune_until` (default: the season's first two weeks, to
    Nov 16 in 2025-26 and Dec 8 in 2020-21, which started late) and scored on
    the rest through `test_until` (default: Dec 31).
    """
    results, tuned, scored = _carryover(
        league,
        season,
        target,
        penalty,
        team_penalty,
        tune_until,
        test_until,
        summaries,
        tables,
        box=BoxModel.load(league) if box_prior else None,
    )
    print(f"weights fitted on {tuned} games, scored on {scored}")
    print(f"{'from last season':34s} {'margin RMSE':>11s} {'t vs team':>9s}")
    for result in results:
        t = "" if result.name == "team carryover" else f"{result.t_vs_team:.1f}"
        print(f"{result.name:34s} {result.rmse:11.3f} {t:>9s}")


def _rating_tables(league: str, tables: str) -> dict[int, list[dict]]:
    """Every season's `season_ratings.parquet` rows, by season."""
    found = {}
    for path in sorted(
        Path(tables).glob(f"league={league}/season=*/season_ratings.parquet")
    ):
        found[int(path.parent.name.split("=")[1])] = pq.read_table(path).to_pylist()
    return found


def _recruit_model(
    league: str,
    ratings: Mapping[int, list[dict]],
    recruits: Mapping[str, Recruit],
    leave_out: int | None = None,
) -> RecruitModel:
    """`RecruitModel` learned on every season's newcomers but `leave_out`'s."""
    found = [n for n in newcomers(ratings) if n.season != leave_out]
    return RecruitModel.learn(league, found, recruits)


def recruit_model(league: str, recruits: str = "./recruits", tables: str = "./stints"):
    """
    Learn what a newcomer's recruiting rank says of his first season
    (`recruiting.RecruitModel`) from every season's `season_ratings` (run
    that first), and write it to `releases/`: commit what this changes.
    """
    ratings = _rating_tables(league, tables)
    found = read_recruits(recruits, league)
    model = _recruit_model(league, ratings, found)
    path = model.save(
        seasons=sorted(ratings),
        newcomers=len(newcomers(ratings)),
        ranked=sum(
            recruit_features(found.get(n.athlete_id), n.season)[0] == 1
            for n in newcomers(ratings)
        ),
    )
    print(f"wrote {path}")
    for label, recruit in (
        ("unranked", None),
        ("#100", Recruit("", 0, 100, 0)),
        ("#50", Recruit("", 0, 50, 0)),
        ("#1", Recruit("", 0, 1, 0)),
    ):
        o, d = model.rating(recruit, 0)
        print(
            f"    {label:9s} off {100 * o:+5.1f} def {100 * d:+5.1f} total {100 * (o + d):+5.1f}"
        )


def minutes(
    league: str, season: int, summaries: str = "./summaries", tables: str = "./stints"
):
    """
    Each player's minutes for each team in a season, from the box scores
    (`box_scores.season_minutes`): `minutes.parquet` next to the season's
    tables, for the preseason projection.
    """
    table = season_minutes(summaries, league, season)
    directory = Path(tables) / f"league={league}" / f"season={season}"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / "minutes.parquet", compression="zstd")
    print(f"{table.num_rows} player-teams")


def _spells(league: str, tables: str) -> dict[int, list[Spell]]:
    """Every season's `minutes.parquet`, by season."""
    found = {}
    for path in sorted(Path(tables).glob(f"league={league}/season=*/minutes.parquet")):
        found[int(path.parent.name.split("=")[1])] = [
            Spell(r["athlete_id"], r["team_id"], r["share"], r["first_date"])
            for r in pq.read_table(path).to_pylist()
        ]
    return found


def _team_ratings(
    league: str,
    season: int,
    target: str,
    team_penalty: float,
    summaries: str,
    tables: str,
) -> dict[str, tuple[float, float]]:
    """A season's team-only ratings, per possession, as carryover uses them."""
    built = _season_rows(league, season, summaries, tables, with_estimated=True)
    assert built is not None, f"no tables for {league} {season}"
    rows, shots = built
    y = TARGETS[target](targets(rows, shots, set(rows.game_id)))
    design = Design.build(rows, players=False, teams=True)
    b = solve(
        design, y, rows.possessions, penalties(design, player=0, team=team_penalty)
    )
    return {
        team: (
            float(b[design.team_column(i, "offense")]),
            float(b[design.team_column(i, "defense")]),
        )
        for i, team in enumerate(design.teams)
    }


def _histories(league: str, tables: str, spells, found):
    """`preseason.History` for any season, from what the seasons before it
    left: their minutes, their `season_ratings`, the recruits."""
    ratings = {
        season: {
            r["athlete_id"]: (r["offense"] / 100, r["defense"] / 100) for r in rows
        }
        for season, rows in _rating_tables(league, tables).items()
    }
    played: dict[str, set[int]] = {}
    for season, season_spells in spells.items():
        for spell in season_spells:
            played.setdefault(spell.athlete_id, set()).add(season)

    def history(season: int) -> History:
        last_spells: dict[str, list[Spell]] = {}
        for spell in spells.get(season - 1, []):
            last_spells.setdefault(spell.athlete_id, []).append(spell)
        return History(last_spells, ratings.get(season - 1, {}), found, played)

    return history


def _minutes_examples(spells, histories):
    """Each season's (features, shares) for `MinutesModel`: everyone who
    played, by what was known before it."""
    examples = {}
    for season, history in histories.items():
        players = season_roster(spells[season])
        share = {(s.athlete_id, s.team_id): s.share for s in spells[season]}
        x = np.array([minutes_features(p, t, season, history) for p, t in players])
        examples[season] = (x, np.array([share[k] for k in players]))
    return examples


def rosters(
    league: str,
    season: int = 2026,
    root: str = "./rosters",
    summaries: str = "./summaries",
    tables: str = "./stints",
    refetch: bool = False,
):
    """
    The coming season's rosters, for every team ESPN filed in Division I
    the season before (`rosters`): `roster.parquet` in the season's tables.
    ESPN only serves current rosters, so `season` is the one about to start;
    a team whose roster ESPN hasn't moved on to it yet is left out, and
    named.
    """
    contexts = read_context(summaries, league, season - 1)
    teams = sorted(
        {
            c.team(side).team_id
            for c in contexts.values()
            for side in ("home", "away")
            if c.team(side).division_one
        }
    )
    rows = []
    stale = []
    for k, team in enumerate(teams):
        response = fetch_roster(root, league, season, team, refetch)
        if roster_current(response, season):
            rows += roster_rows(team, response)
        else:
            stale.append(team)
        if (k + 1) % 50 == 0:
            print(f"  {k + 1} of {len(teams)} teams", flush=True)
    table = roster_table(rows)
    directory = Path(tables) / f"league={league}" / f"season={season}"
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, directory / "roster.parquet", compression="zstd")
    print(f"{len(teams) - len(stale)} teams, {table.num_rows} players")
    if stale:
        print(
            f"  {len(stale)} teams' rosters are still last season's, left out"
            f" (rerun with --refetch later): {', '.join(stale)}"
        )


def project(
    league: str,
    season: int = 2026,
    first: int = 2014,
    top: int = 25,
    recruits: str = "./recruits",
    summaries: str = "./summaries",
    tables: str = "./stints",
):
    """
    Every team's projected offense and defense before `season` starts
    (`preseason`), from its roster (`rosters`), last season's ratings
    (`season_ratings`), recruits and minutes: `projection.parquet` in the
    season's tables, and the best `top` printed with who carries them.
    The minutes model is learned on every season from `first`.
    """
    spells = _spells(league, tables)
    found = read_recruits(recruits, league)
    history = _histories(league, tables, spells, found)
    past = {s: history(s) for s in range(first, season) if s in spells}
    examples = _minutes_examples(spells, past)
    model = MinutesModel.learn(
        np.vstack([x for x, _ in examples.values()]),
        np.concatenate([y for _, y in examples.values()]),
    )
    recruit = RecruitModel.load(league)
    directory = Path(tables) / f"league={league}" / f"season={season}"
    listed = pq.read_table(directory / "roster.parquet").to_pylist()
    now = history(season)
    players = sorted((r["athlete_id"], r["team_id"]) for r in listed)
    x = np.array([minutes_features(p, t, season, now) for p, t in players])
    shares = model.project(players, x)
    teams = project_teams(shares, season, now, recruit)
    names = {r["athlete_id"]: r["name"] for r in listed}
    contexts = read_context(summaries, league, season - 1)
    team_names = {
        c.team(side).team_id: c.team(side).name
        for c in contexts.values()
        for side in ("home", "away")
    }
    carried: dict[str, list[tuple[float, str, float, float, str]]] = {}
    for (player, team), share in shares.items():
        rating = now.last_ratings.get(player)
        last = {s.team_id for s in now.last_spells.get(player, ())}
        how = "newcomer" if not last else "returning" if team in last else "transfer"
        if rating is None:
            rating = recruit.rating(now.recruits.get(player), season)
            if recruit_features(now.recruits.get(player), season)[0]:
                how = f"recruit #{now.recruits[player].rank}"
        total = 100 * sum(rating)
        carried.setdefault(team, []).append(
            (share * total, names.get(player) or player, total, share, how)
        )
    ranked = sorted(
        ((team, 100 * o, 100 * d) for team, (o, d) in teams.items()),
        key=lambda r: -(r[1] + r[2]),
    )
    rows = [
        {
            "team_id": team,
            "team": team_names.get(team),
            "offense": o,
            "defense": d,
            "net": o + d,
        }
        for team, o, d in ranked
    ]
    pq.write_table(
        pa.Table.from_pylist(rows), directory / "projection.parquet", compression="zstd"
    )
    print(f"{league} {season}-{(season + 1) % 100:02d}, {len(rows)} teams; per 100")
    print(f"{'':4s} {'team':28s} {'net':>6s} {'off':>6s} {'def':>6s}  carried by")
    for rank, (team, o, d) in enumerate(ranked[:top], start=1):
        best = sorted(carried[team], reverse=True)[:3]
        who = "; ".join(
            f"{name} {total:+.1f} x{share:.2f} ({how})"
            for _, name, total, share, how in best
        )
        print(
            f"{rank:3d}. {team_names.get(team) or team:28s} {o + d:+6.1f}"
            f" {o:+6.1f} {d:+6.1f}  {who}"
        )


def preseasons(
    league: str,
    first: int = 2014,
    last: int = 2025,
    target: str = "half on 3s and FTs",
    team_penalty: float = 30,
    recruits: str = "./recruits",
    summaries: str = "./summaries",
    tables: str = "./stints",
):
    """
    How well each season's teams are known before it starts
    (`ten_minute_version.preseason`): every season from `first` to `last`,
    its early games (to Dec 31) predicted from last season's team ratings,
    from its rosters projected (players' ratings times projected minutes),
    and from both -- with the whole season's roster, and with only who's
    played by the end of the first two weeks. The minutes and recruit
    models are learned on every season but the one predicted. Run
    `season_ratings` and `minutes` for every season first.
    """
    spells = _spells(league, tables)
    found = read_recruits(recruits, league)
    history = _histories(league, tables, spells, found)
    seasons = range(first, last + 1)
    histories = {season: history(season) for season in seasons}
    examples = _minutes_examples(spells, histories)
    rating_tables = _rating_tables(league, tables)
    names = [
        "team carryover",
        "projected roster",
        "team + projected roster",
        "team + projected, early roster",
    ]
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in names}
    print(
        f"{'season':>6s} {'games':>6s} {'teams':>7s} {'roster':>7s} {'both':>7s}"
        f" {'t':>6s} {'early':>7s} {'t':>6s}"
    )

    def gap_t(e: np.ndarray, reference: np.ndarray) -> float:
        gap = e**2 - reference**2
        return float(gap.mean() / (gap.std(ddof=1) / np.sqrt(len(gap))))

    def line(head: str, n: int, errors: dict[str, np.ndarray]) -> str:
        rmse = {k: np.sqrt(np.mean(v**2)) for k, v in errors.items()}
        team = errors["team carryover"]
        return (
            f"{head:>6s} {n:6d} {rmse[names[0]]:7.2f} {rmse[names[1]]:7.2f}"
            f" {rmse[names[2]]:7.2f} {gap_t(errors[names[2]], team):6.1f}"
            f" {rmse[names[3]]:7.2f} {gap_t(errors[names[3]], team):6.1f}"
        )

    for season in seasons:
        others = [s for s in seasons if s != season]
        model = MinutesModel.learn(
            np.vstack([examples[s][0] for s in others]),
            np.concatenate([examples[s][1] for s in others]),
        )
        recruit = _recruit_model(league, rating_tables, found, leave_out=season)
        built = _season_rows(league, season, summaries, tables, with_estimated=True)
        assert built is not None, f"no tables for {league} {season}"
        rows, _ = built
        directory = Path(tables) / f"league={league}" / f"season={season}"
        day = {
            r["game_id"]: r["date"]
            for r in pq.read_table(directory / "games.parquet").to_pylist()
        }
        games = sorted({g for g in rows.game_id if day[g] <= date(season, 12, 31)})
        tune_end = min(day[g] for g in games) + timedelta(days=13)
        teams = _team_ratings(
            league, season - 1, target, team_penalty, summaries, tables
        )
        terms = game_terms(rows, games, teams, {})
        generic = histories[season].last_ratings.get("generic", (0.0, 0.0))
        extra = {}
        for name, by in (("projected", None), ("projected, early", tune_end)):
            players = season_roster(spells[season], by)
            x = np.array(
                [minutes_features(p, t, season, histories[season]) for p, t in players]
            )
            projected = project_teams(
                model.project(players, x), season, histories[season], recruit
            )
            extra[name] = team_term(rows, games, projected, generic)
        both = GameTerms(terms.games, terms.actual, {**terms.terms, **extra})
        results = compare(
            both,
            day,
            tune_end,
            {
                names[0]: ["home", "balance", "team"],
                names[1]: ["home", "balance", "projected"],
                names[2]: ["home", "balance", "team", "projected"],
                names[3]: ["home", "balance", "team", "projected, early"],
            },
        )
        errors = {r.name: r.errors for r in results}
        for name in names:
            pooled[name].append(errors[name])
        print(line(str(season), len(errors[names[0]]), errors), flush=True)
    together = {name: np.concatenate(parts) for name, parts in pooled.items()}
    print(line("all", len(together[names[0]]), together))
    print("\nminutes model, every season:")
    model = MinutesModel.learn(
        np.vstack([examples[s][0] for s in seasons]),
        np.concatenate([examples[s][1] for s in seasons]),
    )
    for name, value in zip(("intercept", *MINUTES_FEATURES), model.coefficients):
        print(f"    {name:<16}{value:+.3f}")


def carryovers(
    league: str,
    first: int = 2014,
    last: int = 2025,
    target: str = "half on 3s and FTs",
    penalty: float = 300,
    team_penalty: float = 30,
    box_prior: bool = True,
    recruit_prior: bool = True,
    recruits: str = "./recruits",
    summaries: str = "./summaries",
    tables: str = "./stints",
):
    """
    `carryover` for every season from `first` to `last`, each from the one
    before, a line a season -- and over all of them, every season's scored
    games together. Four ways to predict: last season's teams, teams and
    players fitted without a prior, teams and players as `season_ratings`
    writes them (with the box-score prior, unless `box_prior` is off), and
    those with each ranked newcomer at what his recruiting rank says
    (unless `recruit_prior` is off; learned from every season but the one
    predicted, so run `season_ratings` first). Each `t` is against the
    column before it.
    """
    box = BoxModel.load(league) if box_prior else None
    names = ["team carryover", "team + players, with estimated"]
    if box is not None:
        names.append("team + players, box prior")
    found: dict[str, Recruit] = {}
    ratings: dict[int, list[dict]] = {}
    if box is not None and recruit_prior:
        names.append("team + players, box prior + recruits")
        found = read_recruits(recruits, league)
        ratings = _rating_tables(league, tables)
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in names}
    labels = ("teams", "+ players", "+ box prior", "+ recruits")[: len(names)]
    print(
        f"{'season':>6s} {'games':>6s} "
        + " ".join(
            f"{label:>11s}" + ("" if k == 0 else f" {'t':>6s}")
            for k, label in enumerate(labels)
        )
    )

    def line(head: str, n: int, errors: list[np.ndarray]) -> str:
        cells = []
        for k, e in enumerate(errors):
            cells.append(f"{np.sqrt(np.mean(e**2)):11.2f}")
            if k:
                gap = e**2 - errors[k - 1] ** 2
                cells.append(
                    f"{gap.mean() / (gap.std(ddof=1) / np.sqrt(len(gap))):6.1f}"
                )
        return f"{head:>6s} {n:6d} " + " ".join(cells)

    for season in range(first, last + 1):
        results, _, scored = _carryover(
            league,
            season,
            target,
            penalty,
            team_penalty,
            "",
            "",
            summaries,
            tables,
            quiet=True,
            box=box,
            recruits=found,
            recruit_model=(
                _recruit_model(league, ratings, found, leave_out=season)
                if found
                else None
            ),
        )
        by = {r.name: r for r in results}
        for name in names:
            pooled[name].append(by[name].errors)
        print(
            line(str(season), scored, [by[name].errors for name in names]), flush=True
        )
    together = [np.concatenate(pooled[name]) for name in names]
    print(line("all", len(together[0]), together))


def _carryover(
    league: str,
    season: int,
    target: str,
    penalty: float,
    team_penalty: float,
    tune_until: str,
    test_until: str,
    summaries: str,
    tables: str,
    quiet: bool = False,
    box: BoxModel | None = None,
    anchored_penalty: float = 1000,
    recruits: Mapping[str, Recruit] | None = None,
    recruit_model: RecruitModel | None = None,
) -> tuple[list[Result], int, int]:
    """`carryover`'s results, and how many games it tuned and scored on."""
    test_end = date.fromisoformat(test_until) if test_until else date(season, 12, 31)
    last_directory = Path(tables) / f"league={league}" / f"season={season - 1}"
    lined = pq.read_metadata(last_directory / "stints.parquet").num_rows > 0
    players: dict[str, dict[str, tuple[float, float]]] = {}
    teams: dict[str, tuple[float, float]] = {}
    for name, with_estimated in (("real lineups", False), ("with estimated", True)):
        if not with_estimated and not lined:
            continue
        built = _season_rows(league, season - 1, summaries, tables, with_estimated)
        assert built is not None, f"no tables for {league} {season - 1}"
        last_rows, last_shots = built
        y = TARGETS[target](targets(last_rows, last_shots, set(last_rows.game_id)))
        design = Design.build(last_rows)
        b = solve(design, y, last_rows.possessions, penalties(design, player=penalty))
        players[name] = player_ratings(design, b)
        if with_estimated:
            team_design = Design.build(last_rows, players=False, teams=True)
            pen = penalties(team_design, player=0, team=team_penalty)
            tb = solve(team_design, y, last_rows.possessions, pen)
            teams = {
                t: (
                    tb[team_design.team_column(i, "offense")],
                    tb[team_design.team_column(i, "defense")],
                )
                for i, t in enumerate(team_design.teams)
            }
        if not quiet:
            print(
                f"last season, {name}: {len(players[name])} players,"
                f" {len(set(last_rows.game_id))} games",
                flush=True,
            )
    if box is not None:
        built = _anchored_fit(
            league,
            season - 1,
            target,
            anchored_penalty,
            team_penalty,
            summaries,
            tables,
            box,
        )
        assert built is not None, f"no tables for {league} {season - 1}"
        last_rows, _, design, b = built
        table = ratings_table(league, season - 1, last_rows, design, b, {}, {})
        # per possession, a fifth of his team's included, as the players'
        # sums in `game_terms` need
        players["box prior"] = {
            r["athlete_id"]: (r["offense"] / 100, r["defense"] / 100)
            for r in table.to_pylist()
        }
    # This season's lineups as well: estimated, before there were real ones.
    built = _season_rows(league, season, summaries, tables, with_estimated=True)
    assert built is not None, f"no tables for {league} {season}"
    rows, _ = built
    directory = Path(tables) / f"league={league}" / f"season={season}"
    day = {
        r["game_id"]: r["date"]
        for r in pq.read_table(directory / "games.parquet").to_pylist()
    }
    games = sorted({g for g in rows.game_id if day[g] <= test_end})
    if recruit_model is not None and "box prior" in players:
        # This season's ranked newcomers, at what their rank says; the rest
        # stay newcomers, counted alike.
        rated = players["box prior"]
        ranked = {
            p: recruit_model.rating(recruits.get(p) if recruits else None, season)
            for five in (*rows.offense, *rows.defense)
            for p in five
            if p not in rated
            and recruit_features((recruits or {}).get(p), season)[0] == 1
        }
        players["box prior + recruits"] = {**rated, **ranked}
    first = min(day[g] for g in games)
    tune_end = (
        date.fromisoformat(tune_until) if tune_until else first + timedelta(days=13)
    )
    terms = game_terms(rows, games, teams, players)
    results = compare(terms, day, tune_end, variants(list(players)))
    tuned = sum(day[g] <= tune_end for g in games)
    return results, tuned, len(games) - tuned


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
    Fire(
        {
            "stints": stints,
            "context": context,
            "ratings": ratings,
            "season_ratings": season_ratings,
            "box_model": box_model,
            "recruit_model": recruit_model,
            "minutes": minutes,
            "preseasons": preseasons,
            "rosters": rosters,
            "project": project,
            "priors": priors,
            "estimated": estimated,
            "carryover": carryover,
            "carryovers": carryovers,
            "evaluate": evaluate,
            "fit": fit,
        }
    )
