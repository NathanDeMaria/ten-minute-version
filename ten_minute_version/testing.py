"""
Plays for tests, written the way the feed writes them, so a test can spell
out a game in a few lines without the parquet in between.
"""

from datetime import date

from .plays import Game, Play


def play(
    text: str = "",
    side: str | None = None,
    category: str | None = None,
    play_type: str | None = None,
    athlete: str | None = None,
    participants: tuple[str, ...] = (),
    period: int = 1,
    clock: int = 1200,
    score: tuple[int, int] | None = None,
) -> Play:
    return Play(
        period=period,
        clock_seconds=clock,
        side=side,
        category=category,
        play_type=play_type,
        text=text,
        athlete_id=athlete,
        athlete_name=f"Player {athlete}" if athlete else None,
        participant_ids=participants or ((athlete,) if athlete else ()),
        home_score=score[0] if score else None,
        away_score=score[1] if score else None,
    )


def sub_in(side: str, player: str, **kw) -> Play:
    return play(f"{player} subbing in", side, "1015", "Substitution", player, **kw)


def sub_out(side: str, player: str, **kw) -> Play:
    return play(f"{player} subbing out", side, "1015", "Substitution", player, **kw)


def made_2(side: str, player: str, assist: str | None = None, **kw) -> Play:
    participants = (player, assist) if assist else (player,)
    return play("made", side, "1002", "LayUpShot", player, participants, **kw)


def made_3(side: str, player: str, **kw) -> Play:
    return play("made three", side, "1003", "JumpShot", player, **kw)


def missed_2(side: str, player: str, **kw) -> Play:
    return play("missed", side, "1005", "JumpShot", player, **kw)


def free_throw(side: str, player: str, made: bool = True, **kw) -> Play:
    category = "1001" if made else "1004"
    return play("free throw", side, category, "MadeFreeThrow", player, **kw)


def defensive_rebound(side: str, player: str | None = None, **kw) -> Play:
    return play("rebound", side, "1009", "Defensive Rebound", player, **kw)


def offensive_rebound(side: str, player: str | None = None, **kw) -> Play:
    return play("rebound", side, "1009", "Offensive Rebound", player, **kw)


def turnover(side: str, player: str, **kw) -> Play:
    return play("turnover", side, "1010", "Lost Ball Turnover", player, **kw)


def steal(side: str, **kw) -> Play:
    # A steal names its player only in the text, like the feed's.
    return play("Somebody Steal.", side, "1022", "Steal", **kw)


def foul(side: str, player: str, **kw) -> Play:
    return play(f"Foul on {player}.", side, "1007", "PersonalFoul", player, **kw)


def technical(side: str, player: str | None = None, **kw) -> Play:
    return play("Technical Foul", side, "1007", "Technical Foul", player, **kw)


def period_end(**kw) -> Play:
    return play("End of 1st half", None, "1018", "End Period", **kw)


# Five a side, ids that say which side they're on: h1-h5 start for home,
# a1-a5 for away, and h6/a6 come off the bench.
HOME = ("h1", "h2", "h3", "h4", "h5")
AWAY = ("a1", "a2", "a3", "a4", "a5")


def starters_act() -> list[Play]:
    """A play by every starter, so each is seen on the floor first."""
    return [foul("home", p) for p in HOME] + [foul("away", p) for p in AWAY]


def game(*plays: Play, league: str = "mens") -> Game:
    return Game(league, 2025, date(2026, 1, 21), "g1", tuple(plays))
