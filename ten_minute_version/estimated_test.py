from .box_scores import STATS, BoxLine
from .context import GameContext, Team
from .estimated import _five, box_score


def _line(team: str, player: str, minutes: float, starter: bool) -> BoxLine:
    stats = {stat: 0.0 for stat in STATS} | {"minutes": minutes}
    return BoxLine("g1", team, player, starter, "G", stats)


_CONTEXT = GameContext(
    "g1",
    False,
    False,
    Team("H", "H", None, None, True),
    Team("A", "A", None, None, True),
)


def _side(team: str, prefix: str, bench_minutes: float = 0.0) -> list[BoxLine]:
    lines = [
        _line(team, f"{prefix}{i}", 36.0 if i < 5 else bench_minutes, i < 5)
        for i in range(7)
    ]
    return lines


def test_a_box_score_gives_starters_and_shares_of_the_game():
    box = box_score(_side("H", "h", 10.0) + _side("A", "a", 10.0), _CONTEXT)
    assert box is not None
    assert box.starters["home"] == frozenset({"h0", "h1", "h2", "h3", "h4"})
    # Five 36s and two 10s: a 40-minute game
    assert abs(box.minutes["home"]["h0"] - 0.9) < 1e-9
    assert abs(sum(box.minutes["away"].values()) - 5) < 1e-9


def test_a_box_score_without_five_starters_a_side_is_no_use():
    home = _side("H", "h")
    home[0] = _line("H", "h0", 36.0, False)
    assert box_score(home + _side("A", "a"), _CONTEXT) is None


def test_a_side_with_more_than_five_on_is_scaled_to_five():
    six = {f"p{i}": 1.0 for i in range(6)}
    scaled = _five(six)
    assert abs(sum(scaled.values()) - 5) < 1e-9
    assert _five({"p1": 0.5}) == {"p1": 0.5}
