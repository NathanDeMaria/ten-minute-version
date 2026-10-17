"""
Player ids for the plays that only name their player.

Some plays carry a player's name in their text but no id: every steal
("Fletcher Loyer Steal."), and -- in the feed before 2022-23 -- every shot
and free throw, with the assist only in the words ("Gary Harris made Jumper.
Assisted by Keith Appling."). The same players carry ids on their rebounds,
fouls, turnovers and blocks, with their names alongside, so a game's own
plays say which id each name is.

A play is given the longest name, of a player on its own side, that its text
starts with; an assister, the name in "(X assists)" or "Assisted by X". A
player who never does anything that carries an id in the game stays unnamed.

On the 2013-14 to 2019-20 feed this takes the plays credited to a player
from ~65 per team per game to ~155, the density the feed has had since
2022-23. On today's feed it adds the steals, and the shooter and assister of
the few hundred shots a season that arrive without ids.
"""

import re
from dataclasses import replace

from .plays import Game, Play

_ASSISTS = (
    re.compile(r"\(([^()]+?) assists\)"),
    re.compile(r"Assisted by (.+?)\.?$"),
)


def resolve_names(game: Game) -> Game:
    """The game with ids filled in wherever a play names a known player."""
    names: dict[str, dict[str, str]] = {"home": {}, "away": {}}
    for play in game.plays:
        if play.side in names and play.athlete_id and play.athlete_name:
            names[play.side][play.athlete_name] = play.athlete_id
    plays = tuple(
        _resolve(play, names[play.side]) if play.side in names else play
        for play in game.plays
    )
    return replace(game, plays=plays)


def _resolve(play: Play, names: dict[str, str]) -> Play:
    athlete = play.athlete_id or _named_at_start(play.text, names)
    participants = list(play.participant_ids)
    if athlete and not participants:
        participants = [athlete]
    if len(participants) == 1:
        assister = _assister(play.text, names)
        if assister and assister != athlete:
            participants.append(assister)
    if athlete == play.athlete_id and len(participants) == len(play.participant_ids):
        return play
    return replace(play, athlete_id=athlete, participant_ids=tuple(participants))


def _named_at_start(text: str, names: dict[str, str]) -> str | None:
    best = None
    for name in names:
        # A whole name, not the front of a longer one: "Smith" isn't the
        # start of "Smithson Steal."
        if text.startswith(name) and not text[len(name) : len(name) + 1].isalnum():
            if best is None or len(name) > len(best):
                best = name
    return names[best] if best else None


def _assister(text: str, names: dict[str, str]) -> str | None:
    for pattern in _ASSISTS:
        match = pattern.search(text)
        if match:
            return names.get(match.group(1))
    return None
