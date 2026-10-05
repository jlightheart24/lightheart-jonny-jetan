"""Stable action-ordering wrappers and student ranking-key stubs."""

from collections.abc import Callable
from typing import Any

import hw06_config
from jetan import JetanBoard, Move, PieceType, Player
from student_strategies import _PIECE_WEIGHTS

# Larger than any ordinary capture score, so game-winning captures rank first.
_WINNING_CAPTURE = 100.0

OrderingFunction = Callable[
    [JetanBoard, tuple[Move, ...], Player], tuple[Move, ...]
]


def ordering_key_1(
    board: JetanBoard, action: Move, perspective: Player
) -> Any:
    """Return the first ranking key; larger keys are searched first at MAX.

    Capture value (most valuable victim, least valuable attacker): captures of
    heavier pieces first, preferring lighter attackers; quiet moves score 0.
    Signed by the mover so the opponent's best captures sort first at MIN.
    """
    mover = board.at(action.source)
    victim = board.at(action.destination)
    assert mover is not None
    if victim is None:
        return 0.0
    if victim.kind is PieceType.PRINCESS or (
        victim.kind is PieceType.CHIEF and mover.kind is PieceType.CHIEF
    ):
        score = _WINNING_CAPTURE
    else:
        score = _PIECE_WEIGHTS[victim.kind] - _PIECE_WEIGHTS[mover.kind] / 10
    return score if mover.player is perspective else -score


def ordering_key_2(
    board: JetanBoard, action: Move, perspective: Player
) -> Any:
    """Return a distinct ranking key; larger keys are searched first at MAX.

    One-ply lookahead: score the successor with the selected evaluator from the
    fixed root perspective. Terminal successors use their exact utility, which
    lies outside the evaluator's range, so wins and losses rank at the ends.
    """
    successor = board.result(action)
    if successor.is_terminal():
        return float(successor.utility(perspective))
    return float(hw06_config.SELECTED_EVALUATOR(successor, perspective))


def _stable_order(
    board: JetanBoard,
    actions: tuple[Move, ...],
    perspective: Player,
    key: Callable[[JetanBoard, Move, Player], Any],
) -> tuple[Move, ...]:
    """Rank the supplied canonical tuple while retaining input-order ties."""
    return tuple(
        sorted(
            actions,
            key=lambda action: key(board, action, perspective),
            reverse=board.player() is perspective,
        )
    )


def order_actions_1(
    board: JetanBoard, actions: tuple[Move, ...], perspective: Player
) -> tuple[Move, ...]:
    """Apply the first student key to the supplied canonical action tuple."""
    return _stable_order(board, actions, perspective, ordering_key_1)


def order_actions_2(
    board: JetanBoard, actions: tuple[Move, ...], perspective: Player
) -> tuple[Move, ...]:
    """Apply the second student key to the supplied canonical action tuple."""
    return _stable_order(board, actions, perspective, ordering_key_2)
