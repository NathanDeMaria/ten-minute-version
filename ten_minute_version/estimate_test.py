from .estimate import (
    BoxScore,
    Rules,
    Windows,
    estimate_floors,
    nearest_action_floors,
)
from .events import classify
from .testing import AWAY, foul, game, made_2, missed_2, starters_act

# Every rule off: a swap happens at the play that forces it, and the leaver
# is whoever's next action is furthest away.
BARE = Rules(None, False, False, 0.0, 0.0, 0.0)


def _estimate(*plays, rules=BARE, box=None):
    played = game(*plays)
    kinds = [classify(p) for p in played.plays]
    return estimate_floors(played, kinds, box=box, rules=rules)


def test_a_dead_ball_ends_a_window():
    played = game(missed_2("home", "h1"), foul("away", "a1"), missed_2("home", "h2"))
    windows = Windows(played, [classify(p) for p in played.plays])
    assert windows.of_play == [0, 0, 1]


def test_the_starters_are_the_first_five_to_act():
    floors = _estimate(*starters_act(), missed_2("home", "h6", clock=600))
    assert floors[0] == (frozenset({"h1", "h2", "h3", "h4", "h5"}), frozenset(AWAY))


def _later(players, clock):
    """Each player acts once more, a dead ball between each."""
    for n, player in enumerate(players):
        yield foul("away", "a5", clock=clock - 2 * n)
        yield missed_2("home", player, clock=clock - 2 * n - 1)


def test_the_lineup_only_changes_when_someone_off_it_acts():
    floors = _estimate(
        *starters_act(),
        missed_2("home", "h2", clock=1100),
        foul("away", "a1", clock=1050),
        missed_2("home", "h6", clock=1000),
        *_later(("h2", "h3", "h4", "h5"), clock=900),
    )
    assert "h6" not in floors[10][0]
    # h1 never acts again, so h1 is the one who made room
    assert floors[12][0] == frozenset({"h2", "h3", "h4", "h5", "h6"})


def test_a_swap_is_backdated_a_third_of_the_way_to_the_leavers_last_action():
    floors = _estimate(
        *starters_act(),
        missed_2("home", "h1", clock=1190),  # h1's last action, 10s in
        foul("away", "a1", clock=1180),
        foul("away", "a2", clock=1160),
        foul("away", "a3", clock=1150),
        missed_2("home", "h6", clock=1100),  # 100s in: the swap lands at ~42s
        *(missed_2("home", p, clock=1000) for p in ("h2", "h3", "h4", "h5")),
        rules=Rules(0.36, False, False, 0.0, 0.0, 0.0),
    )
    assert "h6" not in floors[12][0]  # the foul 40s in
    assert "h6" in floors[13][0]  # the foul 50s in


def test_the_starters_come_back_out_for_the_second_half():
    floors = _estimate(
        *starters_act(),
        missed_2("home", "h6", clock=600),
        *(missed_2("home", p, clock=500) for p in ("h2", "h3", "h4", "h5")),
        foul("away", "a1", clock=100),
        missed_2("home", "h2", period=2, clock=1150),
        missed_2("home", "h1", period=2, clock=1000),
        rules=Rules(None, False, True, 0.0, 0.0, 0.0),
    )
    assert "h6" in floors[15][0]
    assert floors[16][0] == frozenset({"h1", "h2", "h3", "h4", "h5"})


def test_a_player_in_foul_trouble_is_the_one_to_leave():
    plays = [
        *starters_act(),
        foul("home", "h1", clock=1100),  # h1's second foul of the half
        missed_2("home", "h6", clock=1000),
        *_later(("h2", "h3", "h4"), clock=800),
        *_later(("h1",), clock=750),
        *_later(("h5",), clock=600),
    ]
    at_h6 = 11
    plain = _estimate(*plays)
    assert "h5" not in plain[at_h6][0]  # next action furthest away
    careful = _estimate(*plays, rules=Rules(None, False, False, 0.0, 0.0, 300.0))
    assert "h1" not in careful[at_h6][0]
    assert "h5" in careful[at_h6][0]


def test_box_score_starters_are_taken_over_the_first_to_act():
    box = BoxScore(
        starters={
            "home": frozenset({"h2", "h3", "h4", "h5", "h6"}),
            "away": frozenset(AWAY),
        },
        minutes={"home": {}, "away": {}},
    )
    floors = _estimate(
        foul("away", "a1"),
        made_2("home", "h6"),
        *(missed_2("home", p) for p in ("h1", "h2", "h3", "h4", "h5")),
        box=box,
    )
    assert floors[0][0] == frozenset({"h2", "h3", "h4", "h5", "h6"})


def test_nearest_action_puts_whoever_acted_in_the_window_on():
    played = game(*starters_act(), missed_2("home", "h6"))
    floors = nearest_action_floors(played, [classify(p) for p in played.plays])
    assert "h6" in floors[-1][0]
    assert len(floors[-1][0]) == 5
