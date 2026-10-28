"""Hand-crafted features describing a candidate move in a position.

The style model is a softmax over legal moves: each move gets a score
w . f(board, move) and the player's actual move should score highest. The
features capture both "objective" things (does it hang a piece, win
material) and habits (which pieces you like moving, favourite squares,
how often you castle, check or trade).
"""
import chess

VALUES = {chess.PAWN: 1, chess.KNIGHT: 3, chess.BISHOP: 3,
          chess.ROOK: 5, chess.QUEEN: 9, chess.KING: 0}
PHASES = 3  # opening, middlegame, endgame

# Dense feature names; each is also crossed with the game phase.
DENSE = [
    "bias_piece_P", "bias_piece_N", "bias_piece_B", "bias_piece_R",
    "bias_piece_Q", "bias_piece_K",
    "capture", "capture_value", "net_material", "recapture",
    "gives_check", "castle_k", "castle_q", "promotion",
    "pawn_advance", "develops_minor", "backward",
    "centralise", "approach_king", "king_walk",
    "hangs_moved", "hangs_other", "threatens", "trade_offer",
    "early_queen", "same_piece_again", "pawn_storm", "edge_knight",
]
N_DENSE = len(DENSE) * PHASES
# Sparse "favourite squares": piece type x destination square x colour-relative.
SQ_OFFSET = N_DENSE
N_FEATURES = SQ_OFFSET + 6 * 64
IDX = {name: i for i, name in enumerate(DENSE)}
CENTER_DIST = [max(abs(chess.square_file(s) * 2 - 7), abs(chess.square_rank(s) * 2 - 7)) // 2
               for s in chess.SQUARES]


def game_phase(board):
    if board.fullmove_number <= 10:
        return 0
    material = sum(VALUES[p.piece_type] for p in board.piece_map().values()
                   if p.piece_type not in (chess.PAWN, chess.KING))
    return 2 if material <= 26 else 1


def _rel_rank(sq, color):
    r = chess.square_rank(sq)
    return r if color == chess.WHITE else 7 - r


def _en_prise_value(board, sq, color):
    """Rough value lost if the piece of `color` on `sq` can be taken."""
    piece = board.piece_at(sq)
    if piece is None or piece.piece_type == chess.KING:
        return 0
    attackers = board.attackers(not color, sq)
    if not attackers:
        return 0
    value = VALUES[piece.piece_type]
    if not board.attackers(color, sq):
        return value
    cheapest = min(VALUES[board.piece_type_at(a)] or 100 for a in attackers)
    return max(0, value - cheapest)


def move_features(board, move, phase=None):
    """Return a sparse feature dict {index: value} for `move` on `board`."""
    color = board.turn
    if phase is None:
        phase = game_phase(board)
    piece = board.piece_at(move.from_square)
    pt = piece.piece_type
    last_move = board.move_stack[-1] if board.move_stack else None
    f = {}

    def add(name, value=1.0):
        if value:
            f[IDX[name] * PHASES + phase] = f.get(IDX[name] * PHASES + phase, 0.0) + value

    add("bias_piece_" + piece.symbol().upper())
    is_capture = board.is_capture(move)
    captured_value = 0
    if is_capture:
        captured = board.piece_at(move.to_square)
        captured_value = VALUES[captured.piece_type] if captured else 1  # en passant
        add("capture")
        add("capture_value", captured_value / 9)
        if last_move is not None and last_move.to_square == move.to_square:
            add("recapture")
    if board.gives_check(move):
        add("gives_check")
    if board.is_kingside_castling(move):
        add("castle_k")
    elif board.is_queenside_castling(move):
        add("castle_q")
    elif pt == chess.KING and phase < 2:
        add("king_walk")
    if move.promotion:
        add("promotion")
    fr, tr = _rel_rank(move.from_square, color), _rel_rank(move.to_square, color)
    if pt == chess.PAWN:
        add("pawn_advance", tr / 7)
        their_king = board.king(not color)
        if their_king is not None and abs(chess.square_file(their_king) - chess.square_file(move.to_square)) <= 1:
            add("pawn_storm")
    if pt in (chess.KNIGHT, chess.BISHOP) and fr == 0 and phase == 0:
        add("develops_minor")
    if tr < fr and pt != chess.PAWN:
        add("backward")
    if pt == chess.KNIGHT and chess.square_file(move.to_square) in (0, 7):
        add("edge_knight")
    add("centralise", (CENTER_DIST[move.from_square] - CENTER_DIST[move.to_square]) / 3)
    their_king = board.king(not color)
    if their_king is not None:
        add("approach_king", (chess.square_distance(move.from_square, their_king)
                              - chess.square_distance(move.to_square, their_king)) / 7)
    if pt == chess.QUEEN and phase == 0:
        add("early_queen")
    if len(board.move_stack) >= 2:
        if board.move_stack[-2].to_square == move.from_square and phase == 0:
            add("same_piece_again")

    # Look at the position after the move for tactics-ish features.
    board.push(move)
    try:
        hang = _en_prise_value(board, move.to_square, color)
        add("hangs_moved", hang / 9)
        other = 0
        for sq in board.pieces(chess.KNIGHT, color) | board.pieces(chess.BISHOP, color) \
                | board.pieces(chess.ROOK, color) | board.pieces(chess.QUEEN, color):
            if sq != move.to_square:
                other = max(other, _en_prise_value(board, sq, color))
        add("hangs_other", other / 9)
        add("net_material", (captured_value - hang) / 9)
        if is_capture and hang and captured_value and VALUES[pt] == captured_value:
            add("trade_offer")
        threat = 0
        for sq in board.attacks(move.to_square):
            target = board.piece_at(sq)
            if target and target.color != color and target.piece_type != chess.KING:
                threat = max(threat, _en_prise_value(board, sq, not color))
        add("threatens", threat / 9)
    finally:
        board.pop()

    to_sq = move.to_square if color == chess.WHITE else chess.square_mirror(move.to_square)
    f[SQ_OFFSET + (pt - 1) * 64 + to_sq] = 1.0
    return f
