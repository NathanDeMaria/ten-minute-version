"""
What to shrink a player's rating toward, instead of zero.

A ridge fit pulls every player toward average in proportion to how little
he played, and with ~1,600 possessions for the median player that pull is
strong. A prior says where to pull him instead. Three are tried, each
scored against none on the same held-out games (`compare_priors`):

- **team**: the team's own offense and defense, fitted alongside the
  players (team columns with their own, lighter penalty), so a player is
  pulled toward his teammates rather than toward an average D1 player;
- **box score**: what his box score says -- a regression of the ratings
  fitted without a prior on per-40-minute box-score rates, his minutes, how
  often he started and his position (a statistical plus-minus), all from
  the training games;
- **last season**: his rating a season before, scaled down by a factor
  chosen on held-out games -- and that alongside the box score, as one more
  feature of the same regression.

A prior only moves where the fit starts from; a player who played enough
still ends up where his lineups put him.
"""

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from .box_scores import STATS, BoxLine
from .ratings import (
    TARGETS,
    Design,
    Rows,
    Score,
    game_margins,
    penalties,
    solve,
    targets,
)

# Per-40-minute box-score rates, each pulled toward the league's by this
# many minutes of it, so a player's few minutes can't give him wild ones.
_STEADYING_MINUTES = 40.0
_RATES = tuple(stat for stat in STATS if stat != "minutes")
FEATURES = (
    *(f"{stat}_per_40" for stat in _RATES),
    "minutes_per_game",
    "started",
    "forward",
    "center",
)


@dataclass(frozen=True)
class BoxProfile:
    """Each player's box-score features, over a set of games."""

    players: list[str]
    features: np.ndarray  # players x FEATURES
    minutes: np.ndarray


def box_profile(lines: Iterable[BoxLine], games: set[str]) -> BoxProfile:
    totals: dict[str, dict[str, float]] = {}
    for line in lines:
        if line.game_id not in games:
            continue
        seen = totals.setdefault(line.athlete_id, {stat: 0.0 for stat in STATS})
        for stat in STATS:
            seen[stat] += line.stats[stat]
        seen["games"] = seen.get("games", 0.0) + 1
        seen["started"] = seen.get("started", 0.0) + line.starter
        seen[f"position_{line.position}"] = (
            seen.get(f"position_{line.position}", 0.0) + 1
        )
    players = sorted(totals)
    minutes = np.array([max(totals[p]["minutes"], 0.0) for p in players])
    league = {
        stat: sum(t[stat] for t in totals.values()) / max(minutes.sum(), 1.0)
        for stat in _RATES
    }
    rows = []
    for p, played in zip(players, minutes):
        t = totals[p]
        rates = [
            40
            * (t[stat] + _STEADYING_MINUTES * league[stat])
            / (played + _STEADYING_MINUTES)
            for stat in _RATES
        ]
        appeared = t["games"]
        rows.append(
            [
                *rates,
                played / appeared,
                t["started"] / appeared,
                t.get("position_F", 0.0) / appeared,
                t.get("position_C", 0.0) / appeared,
            ]
        )
    return BoxProfile(players, np.array(rows), minutes)


def box_prior(
    design: Design,
    fitted: np.ndarray,
    profile: BoxProfile,
    exposure: Mapping[str, float],
    extra: Mapping[str, Sequence[float]] | None = None,
) -> np.ndarray:
    """
    A prior vector for `design`: each player's offense and defense as the
    box score predicts them, by weighted least squares of the ratings
    `fitted` without a prior on his box-score features (and `extra`
    features per player, if given), weighted by his possessions.
    """
    at = {p: i for i, p in enumerate(profile.players)}
    known = [(i, p) for i, p in enumerate(design.players) if p in at]
    x = profile.features[[at[p] for _, p in known]]
    if extra:
        x = np.hstack([x, np.array([extra[p] for _, p in known])])
    mean, sd = x.mean(axis=0), x.std(axis=0)
    sd[sd == 0] = 1.0
    x = np.hstack([np.ones((len(known), 1)), (x - mean) / sd])
    w = np.array([exposure.get(p, 0.0) for _, p in known])
    prior = np.zeros(design.columns)
    for side in ("offense", "defense"):
        columns = [design.player_column(i, side) for i, _ in known]
        y = fitted[columns]
        beta = np.linalg.solve(
            (x * w[:, None]).T @ x + 1e-3 * np.eye(x.shape[1]), (x * w[:, None]).T @ y
        )
        prior[columns] = x @ beta
    return prior


def last_season_prior(
    design: Design, last: Mapping[str, tuple[float, float]], scale: float
) -> np.ndarray:
    """A prior vector: each player's (offense, defense) a season before, times `scale`."""
    prior = np.zeros(design.columns)
    for i, p in enumerate(design.players):
        if p in last:
            offense, defense = last[p]
            prior[design.player_column(i, "offense")] = scale * offense
            prior[design.player_column(i, "defense")] = scale * defense
    return prior


def exposures(rows: Rows, train: np.ndarray) -> dict[str, float]:
    """Possessions on the floor, offense and defense, in the training rows."""
    seen: dict[str, float] = {}
    for i in np.flatnonzero(train):
        for players, weights in (
            (rows.offense[i], rows.offense_weights[i]),
            (rows.defense[i], rows.defense_weights[i]),
        ):
            for p, w in zip(players, weights):
                seen[p] = seen.get(p, 0.0) + w * rows.possessions[i]
    return seen


def player_ratings(design: Design, b: np.ndarray) -> dict[str, tuple[float, float]]:
    """(offense, defense) per possession for every player in `design`."""
    return {
        p: (
            float(b[design.player_column(i, "offense")]),
            float(b[design.player_column(i, "defense")]),
        )
        for i, p in enumerate(design.players)
    }


def compare_priors(
    rows: Rows,
    shot_rows: Sequence[Mapping],
    box_lines: Sequence[BoxLine],
    lasts: Mapping[str, Mapping[str, tuple[float, float]]],
    *,
    target: str,
    folds: int = 5,
    player_penalties: Sequence[float] = (300, 1000, 3000),
    team_penalties: Sequence[float] = (10, 30, 100),
    scales: Sequence[float] = (0.25, 0.5),
) -> list[Score]:
    """
    The players model with each prior, scored as `ratings.cross_validate`
    scores models: on the same folds of held-out games, by the error in
    their home margins. "none" is the no-prior fit.

    The box-score and last-season priors are tried on their own and on top
    of the team columns, where they say where a player sits relative to his
    team. A score's `penalty` is the player penalty; the team penalty is in
    its name.
    """
    games = np.array(sorted(set(rows.game_id)), object)
    fold_of = dict(
        zip(games, np.random.default_rng(1989).permutation(len(games)) % folds)
    )
    row_fold = np.array([fold_of[g] for g in rows.game_id])
    plain = Design.build(rows)
    with_teams = Design.build(rows, teams=True)
    errors: dict[tuple[str, float], list[np.ndarray]] = {}
    # The box-score regression learns from the fit at the middle penalty:
    # shrunk enough to be steady, not so much it's all zero.
    middle = sorted(player_penalties)[len(player_penalties) // 2]

    for fold in range(folds):
        train = row_fold != fold
        train_games = set(rows.game_id[train])
        test_games = games[np.array([fold_of[g] == fold for g in games])]
        y = TARGETS[target](targets(rows, shot_rows, train_games))
        weight = rows.possessions * train
        profile = box_profile(box_lines, train_games)
        exposure = exposures(rows, train)

        def run(name: str, design: Design, team_lam: float, prior_of=None) -> None:
            """Fit `name` at every player penalty; `prior_of(design, anchor)`
            builds its prior from the no-prior fit at the middle penalty."""
            anchor = None
            if prior_of is not None:
                pen = penalties(design, player=middle, team=team_lam)
                anchor = solve(design, y, weight, pen)
            prior = prior_of(design, anchor) if prior_of is not None else None
            start = None
            for lam in player_penalties:
                pen = penalties(design, player=lam, team=team_lam)
                b = solve(design, y, weight, pen, prior=prior, start=start)
                start = b
                predicted, actual = game_margins(rows, design.times(b), test_games)
                errors.setdefault((name, lam), []).append(predicted - actual)

        def box(design: Design, anchor: np.ndarray) -> np.ndarray:
            return box_prior(design, anchor, profile, exposure)

        run("none", plain, 0.0)
        run("box score", plain, 0.0, box)
        for team_lam in team_penalties:
            label = f"team penalty {team_lam:g}"
            run(f"team ({label})", with_teams, team_lam)
            run(f"team + box score ({label})", with_teams, team_lam, box)
        # Each set of last season's ratings, on its own (scaled), on top of
        # the team columns, and as features of the box-score regression.
        best_team = sorted(team_penalties)[len(team_penalties) // 2]
        label = f"team penalty {best_team:g}"
        for which, last in lasts.items():
            tag = f" [{which}]" if len(lasts) > 1 else ""
            extra = {
                p: [*last.get(p, (0.0, 0.0)), float(p in last)] for p in profile.players
            }

            def box_and_last(
                design: Design, anchor: np.ndarray, extra=extra
            ) -> np.ndarray:
                return box_prior(design, anchor, profile, exposure, extra)

            def scaled(scale: float, last=last):
                return lambda design, anchor: last_season_prior(design, last, scale)

            for scale in scales:
                run(f"last season x{scale:g}{tag}", plain, 0.0, scaled(scale))
                run(
                    f"team + last season x{scale:g}{tag} ({label})",
                    with_teams,
                    best_team,
                    scaled(scale),
                )
            run(
                f"team + box score + last season{tag} ({label})",
                with_teams,
                best_team,
                box_and_last,
            )
    scores = []
    for (name, lam), parts in errors.items():
        e = np.concatenate(parts)
        rmse, mae = float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e)))
        scores.append(Score("players", name, lam, rmse, mae, len(e), e))
    return sorted(scores, key=lambda s: s.rmse)
