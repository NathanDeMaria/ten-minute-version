from .context import GameContext, Team
from .opponents import Opponents, appearances, generic_side, outside_side, schedules
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


def _team(team_id: str) -> Team:
    return Team(team_id, team_id, None, None, division_one=True)


def _game(game_id: str, home: str, away: str, conference: bool = False):
    return GameContext(game_id, False, conference, _team(home), _team(away))


def _schedule(*games: GameContext) -> dict[str, GameContext]:
    return {g.game_id: g for g in games}


def test_a_team_that_plays_a_few_of_the_seasons_games_is_generic():
    # D1 teams d1 and d2 play ten games each; o plays d1 once.
    played = [_game(f"g{i}", "d1", "d2") for i in range(9)]
    visit = _game("visit", "d1", "o")
    found = schedules([*played, visit])
    assert outside_side(visit, found) == "away"
    assert outside_side(played[0], found) is None


def test_a_team_that_played_in_its_conference_is_not():
    # 2020-21: a D1 team whose season stopped after three games, one in its
    # conference, against teams that played ten.
    played = [_game(f"g{i}", "d1", "d2") for i in range(8)]
    short = [_game("c", "d1", "s", conference=True), _game("n", "s", "d2")]
    found = schedules([*played, *short, _game("x", "s", "d2")])
    assert all(outside_side(g, found) is None for g in short)


def test_the_schedule_decides_where_a_game_has_a_summary():
    # The players say away is from outside; the schedule, where the game is
    # one of d2's ten, says it isn't.
    contexts = _schedule(*[_game(f"g{i}", "d1", "d2") for i in range(10)])
    seen = _season(30, 3)
    assert Opponents(seen, contexts).generic_side(game(*starters_act())) is None
    # Without the summary, the players decide.
    assert Opponents(seen).generic_side(game(*starters_act())) == "away"
