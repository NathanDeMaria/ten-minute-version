from .opponents import appearances, generic_side
from .testing import AWAY, HOME, foul, game, made_2, play, starters_act


def _unlinked(side: str):
    """A made shot by a player ESPN has no record of: a name, no id."""
    return play("Pat Nobody made Layup.", side, "1002", "LayUpShot")


def _season(home_games: int, away_games: int) -> dict[str, int]:
    """Season appearances: every home player in `home_games`, away in `away_games`."""
    return {**{p: home_games for p in HOME}, **{p: away_games for p in AWAY}}


def test_a_side_whose_players_play_a_few_games_is_generic():
    # A team from outside D1: its players turn up only against D1 teams.
    assert generic_side(game(*starters_act()), _season(30, 3)) == "away"
    assert generic_side(game(*starters_act()), _season(3, 30)) == "home"


def test_a_side_with_one_unlinked_player_is_not():
    # A D1 team with a player ESPN hasn't linked: the rest still play all season.
    plays = [p for p in starters_act() if p.athlete_id != "a5"]
    plays.append(_unlinked("away"))
    assert generic_side(game(*plays), _season(30, 28)) is None


def test_a_side_with_no_identified_player_is_generic():
    plays = [foul("home", p) for p in HOME] + [_unlinked("away")]
    assert generic_side(game(*plays), _season(30, 0)) == "away"


def test_nobody_is_generic_when_everyone_has_played_as_often():
    # Opening day: one game each, so nothing tells the teams apart yet.
    assert generic_side(game(*starters_act()), _season(1, 1)) is None


def test_appearances_count_games_not_plays():
    one = game(*starters_act(), made_2("home", "h1"), made_2("home", "h1"))
    seen = appearances([one, game(foul("home", "h1"))])
    assert (seen["h1"], seen["h2"], seen["a1"]) == (2, 1, 1)
