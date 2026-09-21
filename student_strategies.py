"""Implement student evaluation, prompting, parsing, and fallback functions."""

import json
import math
from collections.abc import Sequence

from jetan import BOARD_SIZE, JetanBoard, Location, Move, PieceType, Player

# Relative piece weights (Evaluation 1): hand-estimated from each piece's
# movement range/power, since Jetan has no official point chart.
_PIECE_WEIGHTS: dict[PieceType, int] = {
    PieceType.PANTHAN: 1,
    PieceType.WARRIOR: 2,
    PieceType.PADWAR: 2,
    PieceType.THOAT: 3,
    PieceType.DWAR: 4,
    PieceType.FLIER: 4,
    PieceType.CHIEF: 5,
    PieceType.PRINCESS: 6,
}
# 2 Warrior + 2 Padwar + 2 Dwar + 2 Flier + 1 Chief + 1 Princess + 2 Thoat
# + 8 Panthan: the weighted value of one full starting army.
_FULL_ARMY_VALUE = 49
# 19 non-Princess pieces per army, each able to advance at most BOARD_SIZE - 1
# rows from its own back rank.
_MAX_ADVANCEMENT_PER_ARMY = 19 * (BOARD_SIZE - 1)


def _material_total(board: JetanBoard, player: Player) -> int:
    return sum(_PIECE_WEIGHTS[piece.kind] for _, piece in board.pieces(player))


def evaluate_position_1(board: JetanBoard, perspective: Player) -> float:
    my_total = _material_total(board, perspective)
    opponent_total = _material_total(board, perspective.opponent)
    return (my_total - opponent_total) / _FULL_ARMY_VALUE


def _advancement(location: Location, player: Player) -> int:
    _, y = location
    return y if player is Player.ORANGE else (BOARD_SIZE - 1 - y)


def _advancement_total(board: JetanBoard, player: Player) -> int:
    return sum(
        _advancement(location, player)
        for location, piece in board.pieces(player)
        if piece.kind is not PieceType.PRINCESS
    )


def _advancement_score(board: JetanBoard, perspective: Player) -> float:
    my_advancement = _advancement_total(board, perspective)
    opponent_advancement = _advancement_total(board, perspective.opponent)
    return (my_advancement - opponent_advancement) / _MAX_ADVANCEMENT_PER_ARMY


def evaluate_position_2(board: JetanBoard, perspective: Player) -> float:
    material_score = evaluate_position_1(board, perspective)
    advancement_score = _advancement_score(board, perspective)
    return 0.7 * material_score + 0.3 * advancement_score


def _attacked_value(board: JetanBoard, defender: Player) -> int:
    """Weighted value of defender's pieces sitting on a square the opponent attacks."""
    attacked_squares = board.attacked_locations(defender.opponent)
    return sum(
        _PIECE_WEIGHTS[piece.kind]
        for location, piece in board.pieces(defender)
        if location in attacked_squares
    )


def _threat_score(board: JetanBoard, perspective: Player) -> float:
    my_pieces_attacked = _attacked_value(board, perspective)
    opponent_pieces_attacked = _attacked_value(board, perspective.opponent)
    return (opponent_pieces_attacked - my_pieces_attacked) / _FULL_ARMY_VALUE


def evaluate_position_3(board: JetanBoard, perspective: Player) -> float:
    material_score = evaluate_position_1(board, perspective)
    advancement_score = _advancement_score(board, perspective)
    threat_score = _threat_score(board, perspective)
    return 0.5 * material_score + 0.2 * advancement_score + 0.3 * threat_score


_EVALUATION_SYSTEM_PROMPT = """You output only raw JSON, no reasoning, no explanation, no thinking, no markdown.
The response must match this exact schema: {"score": NUMBER}
NUMBER must be a plain finite number between -1 and 1, from the requested
player's perspective: 1 means that player is winning, -1 means that player is
losing, and 0 means the position is even.
/no_think"""

_PIECE_NAMES: dict[PieceType, str] = {
    PieceType.WARRIOR: "Warrior",
    PieceType.PADWAR: "Padwar",
    PieceType.DWAR: "Dwar",
    PieceType.FLIER: "Flier",
    PieceType.CHIEF: "Chief",
    PieceType.PRINCESS: "Princess",
    PieceType.THOAT: "Thoat",
    PieceType.PANTHAN: "Panthan",
}


def _piece_list(board: JetanBoard, player: Player) -> str:
    """Compact 'Name@col-row' listing; avoids the raw ASCII grid, which this
    model reasons over at extreme (unbounded) length instead of just scoring."""
    parts = []
    for (x, y), piece in board.pieces(player):
        column = chr(ord("a") + x)
        parts.append(f"{_PIECE_NAMES[piece.kind]}@{column}{y}")
    return ", ".join(parts)


_RESPONSE_PREFIX = '{"score": '


def build_evaluation_prompt(
    board: JetanBoard, perspective: Player
) -> Sequence[dict[str, str]]:
    user_content = (
        f"Jetan position. Orange pieces: {_piece_list(board, Player.ORANGE)}.\n"
        f"Black pieces: {_piece_list(board, Player.BLACK)}.\n"
        f"Score for {perspective.name.title()}, 1=winning -1=losing 0=even. "
        'Respond with only {"score": NUMBER}.'
    )
    # Seed the response with an assistant-message prefix. The server treats
    # this as a continuation rather than a fresh turn, which roughly halves
    # the reasoning tokens needed before it reaches a final answer (measured
    # ~750 -> ~319 completion tokens on the live course endpoint).
    return (
        {"role": "system", "content": _EVALUATION_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": _RESPONSE_PREFIX},
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    seen: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key in response: {key!r}")
        seen[key] = value
    return seen


def parse_evaluation_response(response: str) -> float:
    # The client only returns the continuation after our assistant-message
    # prefix (see build_evaluation_prompt), so glue it back on before parsing.
    parsed = json.loads(
        _RESPONSE_PREFIX + response.strip(), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(parsed, dict) or set(parsed) != {"score"}:
        raise ValueError("response must be a JSON object with exactly one 'score' key")
    score = parsed["score"]
    if isinstance(score, bool) or not isinstance(score, (int, float)):
        raise ValueError("score must be a number")
    value = float(score)
    if not math.isfinite(value):
        raise ValueError("score must be finite")
    if not -1.0 <= value <= 1.0:
        raise ValueError("score must be within [-1, 1]")
    return value


_MOVE_SYSTEM_PROMPT = """You choose one move in the game of Jetan.
You output only raw JSON, no reasoning, no explanation, no thinking, no markdown.
The response must match this exact schema: {"move": "SRC-DST"}
SRC-DST must be exactly one of the strings given in the legal move list.
/no_think"""

_MOVE_RESPONSE_PREFIX = '{"move": "'


def build_move_prompt(
    board: JetanBoard,
    actions: tuple[Move, ...],
    recent_moves: tuple[Move, ...],
) -> Sequence[dict[str, str]]:
    move_list = ", ".join(str(action) for action in actions)
    recent = ", ".join(str(move) for move in recent_moves) if recent_moves else "none"
    escape_used = {Player.ORANGE: board.used_escape[0], Player.BLACK: board.used_escape[1]}
    user_content = (
        f"Jetan position. Orange pieces: {_piece_list(board, Player.ORANGE)}.\n"
        f"Black pieces: {_piece_list(board, Player.BLACK)}.\n"
        f"It is {board.player().name.title()}'s turn.\n"
        f"Princess escape already used: Orange={escape_used[Player.ORANGE]}, "
        f"Black={escape_used[Player.BLACK]}.\n"
        f"Recent moves (oldest first): {recent}.\n"
        f"Legal moves: {move_list}.\n"
        'Choose one move from the legal move list. Respond with only {"move": "SRC-DST"}.'
    )
    return (
        {"role": "system", "content": _MOVE_SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": _MOVE_RESPONSE_PREFIX},
    )


def parse_move_response(response: str, actions: tuple[Move, ...]) -> Move:
    # The client only returns the continuation after our assistant-message
    # prefix (see build_move_prompt), so glue it back on before parsing.
    parsed = json.loads(
        _MOVE_RESPONSE_PREFIX + response.strip(), object_pairs_hook=_reject_duplicate_keys
    )
    if not isinstance(parsed, dict) or set(parsed) != {"move"}:
        raise ValueError("response must be a JSON object with exactly one 'move' key")
    move_text = parsed["move"]
    if not isinstance(move_text, str):
        raise ValueError("move must be a string")
    by_notation = {str(action): action for action in actions}
    if move_text not in by_notation:
        raise ValueError(f"move {move_text!r} is not in the legal action list")
    return by_notation[move_text]


def choose_fallback(board: JetanBoard, actions: tuple[Move, ...]) -> Move:
    """Deterministic one-ply greedy fallback using our strongest evaluator."""
    perspective = board.player()
    best_action = actions[0]
    best_value = -math.inf
    for action in actions:
        result = board.result(action)
        if result.is_terminal():
            value = float(result.utility(perspective))
        else:
            value = evaluate_position_3(result, perspective)
        if value > best_value:
            best_value = value
            best_action = action
    return best_action
