"""
Player ratings from the stint table: regularised adjusted plus-minus.

Each stint row is a side's possessions against a defense. Its points per
possession are modelled as

    baseline + home edge + sum(offense ratings of the five) - sum(defense ratings of the five)

and fitted by ridge regression weighted by possessions, so a player's
ratings are what's left of his lineups' results once his teammates and
opponents are accounted for, shrunk toward zero by how little he played. A
positive defense rating is points kept off the board. Ratings are reported
per 100 possessions.

- **Home edge** is +h for the home side's offense and -h for the away
  side's, nothing at a neutral site (from `context`).
- **A generic team** (`opponents`) is one player, `GENERIC`, rated like any
  other.
- **The target** is the row's points, its expected points, or a blend
  (`Targets`): expected points take the luck out of shooting, but only a
  shooter's luck -- a shooter better than the model's average keeps credit
  for it (`shot_model`).

Solved by conjugate gradient on the normal equations, with nothing but
numpy: ~370,000 rows and ~10,000 ratings fit in seconds.
"""

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pyarrow as pa

from .context import GameContext
from .shot_model import (
    Shots,
    choose_shrinkage,
    fit_shot_model,
    shooter_skills,
    with_skill,
)


@dataclass(frozen=True)
class Rows:
    """The stint table as arrays, one entry per row."""

    game_id: np.ndarray
    offense_home: np.ndarray  # bool
    offense: list[list[str]]
    defense: list[list[str]]
    # How much of each listed player was on the floor: 1 for everyone in a
    # stint, and for an estimated possession (`estimated`) his chance of
    # being there, adding up to five a side.
    offense_weights: list[list[float]]
    defense_weights: list[list[float]]
    possessions: np.ndarray
    points: np.ndarray
    # +1 home offense, -1 away offense, 0 at a neutral site
    home: np.ndarray
    # the teams on offense and defense, "" where the game has no context
    offense_team: np.ndarray
    defense_team: np.ndarray
    # (game_id, stint, offense_side) -> row, to put shots on their rows
    index: Mapping[tuple[str, int, str], int]

    @classmethod
    def build(cls, stints: pa.Table, contexts: Mapping[str, GameContext]) -> "Rows":
        t = stints.to_pydict()
        n = len(t["game_id"])
        home_side = np.array([side == "home" for side in t["offense_side"]])
        home = np.zeros(n)
        offense_team = np.full(n, "", object)
        defense_team = np.full(n, "", object)
        for i, (game_id, side) in enumerate(zip(t["game_id"], t["offense_side"])):
            context = contexts.get(game_id)
            if context is None:
                continue
            if not context.neutral_site:
                home[i] = 1.0 if side == "home" else -1.0
            other = "away" if side == "home" else "home"
            offense_team[i] = context.team(side).team_id
            defense_team[i] = context.team(other).team_id

        def weights(column: str, players: str) -> list[list[float]]:
            if column in t:
                return [list(w) for w in t[column]]
            return [[1.0] * len(five) for five in t[players]]

        return cls(
            game_id=np.array(t["game_id"], object),
            offense_home=home_side,
            offense=t["offense_players"],
            defense=t["defense_players"],
            offense_weights=weights("offense_weights", "offense_players"),
            defense_weights=weights("defense_weights", "defense_players"),
            possessions=np.array(t["possessions"], float),
            points=np.array(t["points"], float),
            home=home,
            offense_team=offense_team,
            defense_team=defense_team,
            index={
                (g, s, side): i
                for i, (g, s, side) in enumerate(
                    zip(t["game_id"], t["stint"], t["offense_side"])
                )
            },
        )

    def __len__(self) -> int:
        return len(self.possessions)


@dataclass(frozen=True)
class Design:
    """
    The regression's matrix, as its nonzero entries. Columns: each player's
    offense then defense rating, each team's (if any), the baseline, the
    home edge.
    """

    row: np.ndarray
    col: np.ndarray
    val: np.ndarray
    rows: int
    players: list[str]
    teams: list[str]

    @property
    def columns(self) -> int:
        return 2 * len(self.players) + 2 * len(self.teams) + 2

    @property
    def baseline(self) -> int:
        return self.columns - 2

    @property
    def home(self) -> int:
        return self.columns - 1

    def player_column(self, i: int, side: str) -> int:
        return 2 * i + (side == "defense")

    def team_column(self, i: int, side: str) -> int:
        return 2 * len(self.players) + 2 * i + (side == "defense")

    @classmethod
    def build(
        cls, rows: Rows, *, players: bool = True, teams: bool = False
    ) -> "Design":
        player_ids = (
            sorted({p for five in (*rows.offense, *rows.defense) for p in five})
            if players
            else []
        )
        team_ids = (
            sorted({t for t in (*rows.offense_team, *rows.defense_team) if t})
            if teams
            else []
        )
        player_at = {p: i for i, p in enumerate(player_ids)}
        team_at = {t: i for i, t in enumerate(team_ids)}
        design = cls(
            np.empty(0, int),
            np.empty(0, int),
            np.empty(0),
            len(rows),
            player_ids,
            team_ids,
        )
        r: list[int] = []
        c: list[int] = []
        v: list[float] = []

        def add(row: int, column: int, value: float) -> None:
            r.append(row)
            c.append(column)
            v.append(value)

        for i in range(len(rows)):
            if players:
                for p, w in zip(rows.offense[i], rows.offense_weights[i]):
                    add(i, design.player_column(player_at[p], "offense"), w)
                for p, w in zip(rows.defense[i], rows.defense_weights[i]):
                    add(i, design.player_column(player_at[p], "defense"), -w)
            if teams:
                if rows.offense_team[i]:
                    add(
                        i,
                        design.team_column(team_at[rows.offense_team[i]], "offense"),
                        1.0,
                    )
                if rows.defense_team[i]:
                    add(
                        i,
                        design.team_column(team_at[rows.defense_team[i]], "defense"),
                        -1.0,
                    )
            add(i, design.baseline, 1.0)
            if rows.home[i]:
                add(i, design.home, rows.home[i])
        return cls(
            np.array(r), np.array(c), np.array(v), len(rows), player_ids, team_ids
        )

    def times(self, b: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.row, weights=self.val * b[self.col], minlength=self.rows
        )

    def transpose_times(self, v: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.col, weights=self.val * v[self.row], minlength=self.columns
        )

    def column_squares(self, w: np.ndarray) -> np.ndarray:
        return np.bincount(
            self.col, weights=self.val**2 * w[self.row], minlength=self.columns
        )


def penalties(design: Design, player: float, team: float = 0.0) -> np.ndarray:
    """The ridge penalty on each column: none on the baseline or home edge."""
    pen = np.zeros(design.columns)
    pen[: 2 * len(design.players)] = player
    pen[2 * len(design.players) : design.baseline] = team
    return pen


def solve(
    design: Design,
    target: np.ndarray,
    weight: np.ndarray,
    penalty: np.ndarray,
    prior: np.ndarray | None = None,
    start: np.ndarray | None = None,
    tolerance: float = 1e-7,
) -> np.ndarray:
    """
    Minimise sum(weight * (target - Xb)^2) + sum(penalty * (b - prior)^2)
    by conjugate gradient, preconditioned by the diagonal.
    """
    prior = np.zeros(design.columns) if prior is None else prior
    # A column nothing in the weighted rows touches, and nothing penalises,
    # would make the system singular: pin it to its prior.
    diagonal = design.column_squares(weight) + penalty
    free = diagonal > 0
    diagonal = np.where(free, diagonal, 1.0)

    def apply(b: np.ndarray) -> np.ndarray:
        out = design.transpose_times(weight * design.times(b)) + penalty * b
        return np.where(free, out, b)

    rhs = design.transpose_times(weight * target) + penalty * prior
    rhs = np.where(free, rhs, prior)
    b = prior.copy() if start is None else start.copy()
    residual = rhs - apply(b)
    z = residual / diagonal
    direction = z.copy()
    rz = residual @ z
    scale = np.sqrt(rhs @ rhs) or 1.0
    for _ in range(5000):
        if np.sqrt(residual @ residual) < tolerance * scale:
            break
        step = apply(direction)
        alpha = rz / (direction @ step)
        b += alpha * direction
        residual -= alpha * step
        z = residual / diagonal
        rz_next = residual @ z
        direction = z + (rz_next / rz) * direction
        rz = rz_next
    return b


@dataclass(frozen=True)
class Targets:
    """
    Each row's points per possession, split by the kind of shot they came
    from (2s, 3s, free throws), as they happened and as expected -- at the
    league chance from where and how each shot was taken, and at its
    shooter's own chance (`shot_model.with_skill`). A target is a mix of
    these (`TARGETS`).
    """

    actual: Mapping[int, np.ndarray]
    expected: Mapping[int, np.ndarray]
    shooter: Mapping[int, np.ndarray]

    @property
    def points(self) -> np.ndarray:
        return sum(self.actual.values(), np.zeros_like(self.actual[2]))


def targets(rows: Rows, shot_rows: Sequence[Mapping], train_games: set[str]) -> Targets:
    """
    The rows' targets, with the shot model and shooters' skills fitted on
    `train_games` only, so a held-out game's makes can't leak into the
    targets of the games it's scored against.
    """
    shots = Shots.from_rows(shot_rows)
    row_of = np.array(
        [
            rows.index.get((s["game_id"], s["stint"], s["offense_side"]), -1)
            for s in shot_rows
        ]
    )
    games = np.array([s["game_id"] for s in shot_rows], object)
    train = np.array([g in train_games for g in games]) & (row_of >= 0)
    model = fit_shot_model(shots.take(train))
    chance = model.chance(shots)
    half = np.array([int(g) % 2 == 0 for g in games])
    shrinkage = choose_shrinkage(shots.take(train), chance[train], half[train])
    skills = shooter_skills(shots.take(train), chance[train], shrinkage)
    skilled = with_skill(shots, chance, skills)

    def per_row(points: np.ndarray, value: int) -> np.ndarray:
        keep = (row_of >= 0) & (shots.value == value)
        total = np.bincount(row_of[keep], weights=points[keep], minlength=len(rows))
        return total / rows.possessions

    values = (2, 3, 1)
    return Targets(
        actual={v: per_row(shots.value * shots.made, v) for v in values},
        expected={v: per_row(shots.value * chance, v) for v in values},
        shooter={v: per_row(shots.value * skilled, v) for v in values},
    )


def game_margins(
    rows: Rows, prediction: np.ndarray, games: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Predicted and actual home margins over each of `games`' stint-table
    possessions: points per possession times possessions, home minus away.
    """
    sign = np.where(rows.offense_home, 1.0, -1.0)
    at = {g: i for i, g in enumerate(games)}
    index = np.array([at.get(g, -1) for g in rows.game_id])
    keep = index >= 0
    predicted = np.bincount(
        index[keep],
        weights=(sign * rows.possessions * prediction)[keep],
        minlength=len(games),
    )
    actual = np.bincount(
        index[keep], weights=(sign * rows.points)[keep], minlength=len(games)
    )
    return predicted, actual


# What a rating can be fitted to, from a row's `Targets`. Expected points
# for every shot take out more than luck: a layup is one league chance, so
# finishing and rim protection go with it. Threes and free throws are where
# the defense has least say and the shooter's luck most.
TARGETS = {
    "points": lambda t: t.points,
    "expected": lambda t: t.expected[2] + t.expected[3] + t.expected[1],
    "expected, shooter": lambda t: t.shooter[2] + t.shooter[3] + t.shooter[1],
    "half and half": lambda t: (
        0.5 * t.points + 0.5 * (t.shooter[2] + t.shooter[3] + t.shooter[1])
    ),
    "expected 3s and FTs": lambda t: t.actual[2] + t.shooter[3] + t.shooter[1],
    "expected 3s": lambda t: t.actual[2] + t.shooter[3] + t.actual[1],
    "half on 3s and FTs": lambda t: (
        t.actual[2]
        + 0.5 * (t.actual[3] + t.shooter[3])
        + 0.5 * (t.actual[1] + t.shooter[1])
    ),
}


@dataclass(frozen=True)
class Score:
    model: str
    target: str
    penalty: float
    # Over every held-out game: the error in its predicted home margin.
    rmse: float
    mae: float
    games: int
    # Each held-out game's error, in the same game order for every score, so
    # two can be compared game by game (`paired_t`).
    errors: np.ndarray

    def paired_t(self, other: "Score") -> float:
        """
        How sure the squared-error gap to `other` is, as a t statistic on
        the same games: positive when this one's errors are bigger.
        """
        gap = self.errors**2 - other.errors**2
        return float(gap.mean() / (gap.std(ddof=1) / np.sqrt(len(gap))))


def cross_validate(
    rows: Rows,
    shot_rows: Sequence[Mapping],
    *,
    folds: int = 5,
    player_penalties: Sequence[float] = (30, 100, 300, 1000),
    team_penalties: Sequence[float] = (0.3, 1, 3, 10, 30),
    targets_to_try: Sequence[str] = tuple(TARGETS),
) -> list[Score]:
    """
    Every model and target scored on held-out games: each fold's games are
    predicted from ratings fitted on the rest, with their actual lineups and
    possessions, and the home margin compared with what happened. A target
    is only what's fitted -- every model is scored on real points.

    The models: the baseline and home edge alone; teams (each team's
    offense and defense, nothing about who's on the floor); and players.
    """
    games = np.array(sorted(set(rows.game_id)), object)
    fold_of = dict(
        zip(games, np.random.default_rng(1989).permutation(len(games)) % folds)
    )
    row_fold = np.array([fold_of[g] for g in rows.game_id])
    designs = {
        "baseline": Design.build(rows, players=False),
        "teams": Design.build(rows, players=False, teams=True),
        "players": Design.build(rows),
    }
    grids = {
        "baseline": (0.0,),
        "teams": tuple(team_penalties),
        "players": tuple(player_penalties),
    }
    errors: dict[tuple[str, str, float], list[np.ndarray]] = {}
    for fold in range(folds):
        train = row_fold != fold
        train_games = set(rows.game_id[train])
        test_games = games[np.array([fold_of[g] == fold for g in games])]
        fitted = targets(rows, shot_rows, train_games)
        weight = rows.possessions * train
        for model, design in designs.items():
            for target_name in targets_to_try:
                y = TARGETS[target_name](fitted)
                start = None
                for lam in grids[model]:
                    pen = penalties(design, player=lam, team=lam)
                    if model == "players":
                        pen = penalties(design, player=lam)
                    b = solve(design, y, weight, pen, start=start)
                    start = b
                    predicted, actual = game_margins(rows, design.times(b), test_games)
                    errors.setdefault((model, target_name, lam), []).append(
                        predicted - actual
                    )
    scores = []
    for (model, target_name, lam), parts in errors.items():
        e = np.concatenate(parts)
        rmse, mae = float(np.sqrt(np.mean(e**2))), float(np.mean(np.abs(e)))
        scores.append(Score(model, target_name, lam, rmse, mae, len(e), e))
    return sorted(scores, key=lambda s: s.rmse)


RATINGS_SCHEMA = pa.schema(
    [
        pa.field("league", pa.string(), nullable=False),
        pa.field("season", pa.int16(), nullable=False),
        pa.field("athlete_id", pa.string(), nullable=False),
        pa.field("athlete_name", pa.string()),
        # The team he was on the floor for most, and how much.
        pa.field("team_id", pa.string()),
        pa.field("team", pa.string()),
        pa.field("offense_possessions", pa.int32(), nullable=False),
        pa.field("defense_possessions", pa.int32(), nullable=False),
        # Per 100 possessions; defense is points kept off the board. With
        # team columns in the fit, these are his rating against his team's
        # plus a fifth of his team's, so a lineup's five still add up to
        # what the fit predicts for it...
        pa.field("offense", pa.float32(), nullable=False),
        pa.field("defense", pa.float32(), nullable=False),
        pa.field("total", pa.float32(), nullable=False),
        # ...and these his rating against his team's alone (the same as the
        # three above without team columns).
        pa.field("offense_vs_team", pa.float32(), nullable=False),
        pa.field("defense_vs_team", pa.float32(), nullable=False),
    ]
)


def ratings_table(
    league: str,
    season: int,
    rows: Rows,
    design: Design,
    b: np.ndarray,
    names: Mapping[str, str | None],
    team_names: Mapping[str, str | None],
) -> pa.Table:
    offense_possessions: dict[str, float] = {}
    defense_possessions: dict[str, float] = {}
    team_time: dict[str, dict[str, float]] = {}
    for i in range(len(rows)):
        n = rows.possessions[i]
        for players, weights, team, seen in (
            (
                rows.offense[i],
                rows.offense_weights[i],
                rows.offense_team[i],
                offense_possessions,
            ),
            (
                rows.defense[i],
                rows.defense_weights[i],
                rows.defense_team[i],
                defense_possessions,
            ),
        ):
            for p, w in zip(players, weights):
                seen[p] = seen.get(p, 0.0) + w * n
                if team:
                    by_team = team_time.setdefault(p, {})
                    by_team[team] = by_team.get(team, 0.0) + w * n
    team_at = {t: i for i, t in enumerate(design.teams)}
    out = []
    for i, p in enumerate(design.players):
        offense_vs_team = 100 * b[design.player_column(i, "offense")]
        defense_vs_team = 100 * b[design.player_column(i, "defense")]
        teams = team_time.get(p, {})
        team = max(teams, key=lambda t: (teams[t], t)) if teams else None
        offense, defense = offense_vs_team, defense_vs_team
        if team in team_at:
            offense += 100 * b[design.team_column(team_at[team], "offense")] / 5
            defense += 100 * b[design.team_column(team_at[team], "defense")] / 5
        out.append(
            {
                "league": league,
                "season": season,
                "athlete_id": p,
                "athlete_name": names.get(p),
                "team_id": team,
                "team": team_names.get(team) if team else None,
                "offense_possessions": int(offense_possessions.get(p, 0)),
                "defense_possessions": int(defense_possessions.get(p, 0)),
                "offense": offense,
                "defense": defense,
                "total": offense + defense,
                "offense_vs_team": offense_vs_team,
                "defense_vs_team": defense_vs_team,
            }
        )
    return pa.Table.from_pylist(out, schema=RATINGS_SCHEMA)
