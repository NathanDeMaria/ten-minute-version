"""
A latent skill profile for every player: what he does and how well, as one
vector of true skills, each seen through its own noisy rate and all of them
linked.

**The skills** (`DIMS`), from each player's events (`player_events`): how
often he shoots at the rim, from mid-range and from three; how often each
goes in; what share of his makes there he made unassisted; his free-throw
trips and percentage; his assists, to the rim and to jumpers; turnovers;
offensive and defensive rebounds, blocks, steals and fouls. Counts are per
possession on the floor, the rest shares.

**Seen with noise.** A player's rate is his true skill plus sampling noise
whose variance is known from his exposure -- the population's rate over his
possessions for a count (Poisson), its p(1-p) over his attempts for a share
(binomial) -- so a few attempts say little and many say a lot. A make rate is
difficulty-adjusted: the league's rate in that zone plus his makes less what
the shot model expected of his shots (`shot_model`), per attempt. Taken raw,
a big who dunks would get the dunk's higher chance twice.

**Linked.** The skills' true covariance across players (`fit_population`:
the rates' covariance less their noise, kept positive definite) lets one
skill inform another: free-throw percentage and shot volume say a lot about a
shooter's three-point percentage, which his own threes pin slowly. A
player's profile is the Gaussian posterior (`posterior`) of his skills given
his rates.

**Carried.** At a season's end each player's posterior becomes the next
season's prior, pulled `RHO` of the way back from the population's mean and
widened to match (an AR(1) whose spread stays the population's).

**Measured, 2022-2025, last season's population.** Predicting a player's
rates in the other half of a season's games from one half, the linked
profile beats shrinking each skill on its own on every one of the 19, both
leagues: by 0.15 of the true spread on average, most on make rates (three-
point 0.8, free throws 0.27) and self-creation; across a summer, keeping 0.9
of a profile's distance from the mean predicts best (0.8 and 1.0 close, 0.5
much worse). As a shooter's skill on the shot model's chance -- skills from
half a season's games, log loss of the other half's shots -- it beats
`shot_model.shooter_skills` (each kind shrunk alone, nothing carried) at every
zone in every season: per 1,000 shots, mens -0.51 threes, -0.28 mid-range,
-1.08 at the rim, -3.77 free throws; womens -1.18, -1.15, -2.10, -3.60.
`ratings.targets` takes it for its shooters.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from .player_events import DEFENSE as DEF_COUNTS
from .player_events import OFFENSE as OFF_COUNTS
from .shot_model import FREE_THROW, RIM, Shots

DIMS = (
    "rim_att", "mid_att", "three_att",
    "rim_pct", "mid_pct", "three_pct",
    "rim_unast", "mid_unast", "three_unast",
    "ft_rate", "ft_pct",
    "ast_rim", "ast_jumper", "tov", "oreb",
    "dreb", "blk", "stl", "pf",
)  # fmt: skip
SHARES = frozenset(
    {
        "rim_pct",
        "mid_pct",
        "three_pct",
        "rim_unast",
        "mid_unast",
        "three_unast",
        "ft_pct",
    }
)
_IS_SHARE = np.array([d in SHARES for d in DIMS])
# Each zone's make-rate dim, for a shooter's skill on a shot.
MAKES = {"rim": "rim_pct", "mid": "mid_pct", "three": "three_pct", "ft": "ft_pct"}
RHO = 0.9
# Noise variance for a rate there's nothing to see of.
NOTHING = 1e6


@dataclass(frozen=True)
class Population:
    """The skills' mean and true covariance across players."""

    mean: np.ndarray
    cov: np.ndarray


@dataclass(frozen=True)
class ProfileState:
    """What a season leaves the next: its population, and every player's
    posterior (mean, covariance) at its end."""

    population: Population
    players: Mapping[str, tuple[np.ndarray, np.ndarray]]


@dataclass(frozen=True)
class SeasonEvents:
    """A season's player-game events (`player_events`), the games whose
    plays agree with their box score and with possessions on the floor."""

    game: np.ndarray  # object
    athlete: np.ndarray  # object
    offense: np.ndarray  # rows x OFF_COUNTS
    defense: np.ndarray  # rows x DEF_COUNTS
    offense_possessions: np.ndarray
    defense_possessions: np.ndarray

    def totals(
        self, games: set[str]
    ) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """(athletes, offense counts, defense counts, offense possessions,
        defense possessions) summed over `games`."""
        keep = np.array([g in games for g in self.game], bool)
        athletes, inv = np.unique(self.athlete[keep], return_inverse=True)
        n = len(athletes)

        def total(x: np.ndarray) -> np.ndarray:
            return np.bincount(inv, weights=x[keep], minlength=n).astype(float)

        offense = np.column_stack(
            [total(self.offense[:, k]) for k in range(len(OFF_COUNTS))]
        )
        defense = np.column_stack(
            [total(self.defense[:, k]) for k in range(len(DEF_COUNTS))]
        )
        return (
            [str(a) for a in athletes],
            offense,
            defense,
            total(self.offense_possessions),
            total(self.defense_possessions),
        )


def read_season_events(tables: str | Path, league: str, season: int) -> SeasonEvents:
    path = Path(tables) / f"league={league}" / f"season={season}" / "events.parquet"
    t = pq.read_table(path).to_pydict()
    keep = np.array(
        [
            ok is not False and o + d > 0
            for ok, o, d in zip(
                t["box_ok"], t["offense_possessions"], t["defense_possessions"]
            )
        ],
        bool,
    )

    def column(name: str) -> np.ndarray:
        return np.array(t[name], float)[keep]

    return SeasonEvents(
        game=np.array(t["game_id"], object)[keep],
        athlete=np.array(t["athlete_id"], object)[keep],
        offense=np.column_stack([column(c) for c in OFF_COUNTS]),
        defense=np.column_stack([column(c) for c in DEF_COUNTS]),
        offense_possessions=column("offense_possessions"),
        defense_possessions=column("defense_possessions"),
    )


def observe(
    offense: np.ndarray,
    defense: np.ndarray,
    offense_possessions: np.ndarray,
    defense_possessions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Each player's raw rate on every dim and what it's over (possessions
    or attempts): two arrays, players x DIMS. A rate over nothing is 0."""
    at = {c: offense[:, i] for i, c in enumerate(OFF_COUNTS)}
    num = np.zeros((len(offense_possessions), len(DIMS)))
    den = np.zeros_like(num)

    def put(dim: str, count: np.ndarray, over: np.ndarray) -> None:
        j = DIMS.index(dim)
        num[:, j], den[:, j] = count, over

    for zone in ("rim", "mid", "three"):
        unassisted = at[f"made_{zone}_unassisted"]
        made = unassisted + at[f"made_{zone}_assisted"]
        attempts = made + at[f"missed_{zone}"]
        put(f"{zone}_att", attempts, offense_possessions)
        put(f"{zone}_pct", made, attempts)
        put(f"{zone}_unast", unassisted, made)
    free_throws = at["ft_made"] + at["ft_missed"]
    put("ft_rate", free_throws, offense_possessions)
    put("ft_pct", at["ft_made"], free_throws)
    put("ast_rim", at["assists_rim"], offense_possessions)
    put("ast_jumper", at["assists_mid"] + at["assists_three"], offense_possessions)
    put("tov", at["turnovers"], offense_possessions)
    put("oreb", at["offensive_rebounds"], offense_possessions)
    for dim, count in zip(("dreb", "blk", "stl", "pf"), DEF_COUNTS):
        put(dim, defense[:, DEF_COUNTS.index(count)], defense_possessions)
    x = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    return x, den


def noise(mean: np.ndarray, den: np.ndarray) -> np.ndarray:
    """Each rate's sampling variance at the population's mean: mean/n for a
    count per possession, mean(1-mean)/n for a share; NOTHING over nothing."""
    unit = np.where(_IS_SHARE, mean * (1 - mean), mean)
    return np.where(den > 0, unit / np.maximum(den, 1e-12), NOTHING)


def zones(shots: Shots) -> np.ndarray:
    """Each shot's zone: "rim", "mid", "three" or "ft"."""
    group = shots.group
    return np.where(
        group == FREE_THROW,
        "ft",
        np.where(group == RIM, "rim", np.where(shots.value == 3, "three", "mid")),
    )


def adjust_makes(
    x: np.ndarray,
    den: np.ndarray,
    athletes: Sequence[str],
    shots: Shots,
    chance: np.ndarray,
    counted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """`x` and `den` with each make-rate dim difficulty-adjusted, over the
    shots `counted`: the zone's league rate plus (makes - the shot model's
    expected makes) per attempt, over those attempts."""
    x, den = x.copy(), den.copy()
    at = {a: i for i, a in enumerate(athletes)}
    shooter = np.array([at.get(s, -1) if s is not None else -1 for s in shots.shooter])
    zone = zones(shots)
    for z, dim in MAKES.items():
        j = DIMS.index(dim)
        sel = counted & (zone == z) & (shooter >= 0)
        n = len(athletes)
        attempts = np.bincount(shooter[sel], minlength=n).astype(float)
        made = np.bincount(
            shooter[sel], weights=shots.made[sel].astype(float), minlength=n
        )
        expected = np.bincount(shooter[sel], weights=chance[sel], minlength=n)
        league = made.sum() / max(attempts.sum(), 1.0)
        seen = attempts > 0
        x[seen, j] = np.clip(
            league + (made[seen] - expected[seen]) / attempts[seen], 1e-3, 1 - 1e-3
        )
        den[:, j] = attempts
    return x, den


def fit_population(x: np.ndarray, den: np.ndarray, min_info: float = 4.0) -> Population:
    """The skills' mean (pooled) and true covariance: each pair's moments over
    players seen well on both (noise under 1/`min_info` of the spread), less
    the mean noise on the diagonal, made positive definite."""
    k = x.shape[1]
    mean = (x * den).sum(0) / np.maximum(den.sum(0), 1e-12)
    r = noise(mean, den)
    seen = den > 0
    spread = np.array(
        [x[seen[:, j], j].var() if seen[:, j].any() else 0.0 for j in range(k)]
    )
    good = seen & (r < spread[None, :] / min_info)
    cov = np.zeros((k, k))
    for a in range(k):
        for b in range(a, k):
            both = good[:, a] & good[:, b]
            if both.sum() < 30:
                continue
            c = float(((x[both, a] - mean[a]) * (x[both, b] - mean[b])).mean())
            if a == b:
                c -= float(r[both, a].mean())
            cov[a, b] = cov[b, a] = c
    w, v = np.linalg.eigh(cov)
    w = np.maximum(w, 1e-4 * max(float(w.max()), 1e-12))
    return Population(mean, (v * w) @ v.T)


def posterior(
    mean: np.ndarray, cov: np.ndarray, x: np.ndarray, r: np.ndarray, chunk: int = 20000
) -> tuple[np.ndarray, np.ndarray]:
    """Each player's posterior (mean, covariance) from his prior (mean
    players x k, covariance players x k x k) and rates `x` with noise
    variances `r`: one Gaussian update, in chunks."""
    n, k = x.shape
    out_mean, out_cov = np.zeros((n, k)), np.zeros((n, k, k))
    eye = np.eye(k)
    for a in range(0, n, chunk):
        sl = slice(a, a + chunk)
        prior = cov[sl]
        joint = prior + r[sl][:, :, None] * eye
        gain = np.linalg.solve(joint, prior).transpose(0, 2, 1)
        out_mean[sl] = mean[sl] + np.einsum("ijk,ik->ij", gain, x[sl] - mean[sl])
        out_cov[sl] = prior - np.einsum("ijk,ikl->ijl", gain, prior)
    return out_mean, out_cov


def priors(
    athletes: Sequence[str], population: Population, carried: ProfileState | None
) -> tuple[np.ndarray, np.ndarray]:
    """Each athlete's prior: his last posterior pulled `RHO` toward the
    population's mean and widened, or the population's."""
    k = len(population.mean)
    mean = np.tile(population.mean, (len(athletes), 1))
    cov = np.tile(population.cov, (len(athletes), 1, 1))
    players = carried.players if carried is not None else {}
    for i, a in enumerate(athletes):
        last = players.get(a)
        if last is not None:
            mean[i] = population.mean + RHO * (last[0] - population.mean)
            cov[i] = RHO**2 * last[1] + (1 - RHO**2) * population.cov
    assert mean.shape[1] == k
    return mean, cov


def profiles(
    events: SeasonEvents,
    games: set[str],
    shots: Shots,
    chance: np.ndarray,
    counted: np.ndarray,
    population: Population,
    carried: ProfileState | None,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Every athlete's posterior from his events in `games` and his shots
    `counted` (make rates difficulty-adjusted on `chance`): (athletes,
    mean, covariance) and the rates and their exposure behind them."""
    athletes, offense, defense, op, dp = events.totals(games)
    x, den = observe(offense, defense, op, dp)
    x, den = adjust_makes(x, den, athletes, shots, chance, counted)
    prior_mean, prior_cov = priors(athletes, population, carried)
    mean, cov = posterior(prior_mean, prior_cov, x, noise(population.mean, den))
    return athletes, mean, cov, x, den


def shot_shift(
    shots: Shots, athletes: Sequence[str], mean: np.ndarray, population: Population
) -> tuple[np.ndarray, np.ndarray]:
    """Each shot's log-odds shift by its shooter's make rate in its zone, over
    the population's, and whether he has a profile."""
    at = {a: i for i, a in enumerate(athletes)}
    shooter = np.array([at.get(s, -1) if s is not None else -1 for s in shots.shooter])
    zone = zones(shots)
    shift = np.zeros(len(shooter))
    for z, dim in MAKES.items():
        j = DIMS.index(dim)
        sel = (zone == z) & (shooter >= 0)
        shift[sel] = _logit(mean[shooter[sel], j]) - _logit(population.mean[j])
    return shift, shooter >= 0


def season_end(
    events: SeasonEvents,
    shots: Shots,
    chance: np.ndarray,
    counted: np.ndarray,
    carried: ProfileState | None,
) -> ProfileState:
    """What a season leaves the next: its own population (every game), and
    each player's posterior on the season's prior (last season's population
    and carried profiles; this one's, for a first season). A player not seen
    this season is pulled toward the mean once more."""
    games = set(events.game)
    athletes, offense, defense, op, dp = events.totals(games)
    x, den = observe(offense, defense, op, dp)
    x, den = adjust_makes(x, den, athletes, shots, chance, counted)
    population = fit_population(x, den)
    prior_population = carried.population if carried is not None else population
    prior_mean, prior_cov = priors(athletes, prior_population, carried)
    mean, cov = posterior(prior_mean, prior_cov, x, noise(prior_population.mean, den))
    players: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    if carried is not None:
        m0, c0 = prior_population.mean, prior_population.cov
        for a, (m, c) in carried.players.items():
            players[a] = (m0 + RHO * (m - m0), RHO**2 * c + (1 - RHO**2) * c0)
    for i, a in enumerate(athletes):
        players[a] = (mean[i], cov[i])
    return ProfileState(population, players)


def _logit(p: np.ndarray | float) -> np.ndarray:
    q = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))
