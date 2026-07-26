"""Qualified OpenSpiel semantic-family task catalog."""

from arena.adapters.task_openspiel.adapter import (
    FROZEN_GAMES,
    OpenSpielPackager,
    OpenSpielSequentialAEC,
    OpenSpielSimultaneousParallel,
    OpenSpielTicTacToeAEC,
    frozen_game_spec,
    interaction_for_game,
)

__all__ = [
    "FROZEN_GAMES",
    "OpenSpielPackager",
    "OpenSpielSequentialAEC",
    "OpenSpielSimultaneousParallel",
    "OpenSpielTicTacToeAEC",
    "frozen_game_spec",
    "interaction_for_game",
]
