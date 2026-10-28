"""The bot: plays your moves from memory, otherwise moves the way you would."""
import json
import math
import random

import chess

from .features import VALUES, move_features
from .train import position_key

MATE = 100000


class StyleBot:
    def __init__(self, profile_path, temperature=1.0, strength=0.5, use_book=True, seed=None):
        """
        temperature: 0 = always your most likely move; 1 = sample like you do;
                     higher = more random.
        strength:    0 = pure imitation; higher values blend in a small tactical
                     search so the bot blunders less than the raw model.
        """
        with open(profile_path, encoding="utf-8") as fh:
            profile = json.load(fh)
        self.user = profile["user"]
        self.book = profile["book"]
        self.weights = profile["weights"]
        self.stats = profile["stats"]
        self.temperature = temperature
        self.strength = strength
        self.use_book = use_book
        self.rng = random.Random(seed)
        self.last_source = None

    # ---- move selection -------------------------------------------------
    def choose_move(self, board):
        legal = list(board.legal_moves)
        if not legal:
            return None
        if self.use_book:
            move = self._book_move(board, legal)
            if move is not None:
                self.last_source = "book"
                return move
        self.last_source = "model"
        scores = self.move_scores(board, legal)
        return self._sample(legal, scores)

    def _book_move(self, board, legal):
        entry = self.book.get(position_key(board))
        if not entry:
            return None
        legal_uci = {m.uci(): m for m in legal}
        options = [(legal_uci[u], c) for u, c in entry.items() if u in legal_uci]
        if not options:
            return None
        moves, counts = zip(*options)
        return self._sample(list(moves), [math.log(c) for c in counts])

    def move_scores(self, board, legal):
        w = self.weights
        scores = [sum(w[k] * v for k, v in move_features(board, m).items()) for m in legal]
        if self.strength > 0:
            for i, m in enumerate(legal):
                board.push(m)
                if board.is_checkmate():
                    value = MATE
                else:
                    value = -_search(board, 1, -MATE, MATE)
                board.pop()
                # 1 pawn of evaluation ~ `strength` logits.
                scores[i] += self.strength * max(-20.0, min(20.0, value / 100))
                if value >= MATE:
                    scores[i] += 1000  # never miss a mate in one
        return scores

    def _sample(self, moves, scores):
        if self.temperature <= 0:
            return moves[scores.index(max(scores))]
        m = max(scores)
        weights = [math.exp((s - m) / self.temperature) for s in scores]
        return self.rng.choices(moves, weights)[0]


# ---- a tiny material search used only as a blunder check ----------------
def _evaluate(board):
    """Material from the side to move's point of view, in centipawns."""
    score = 0
    for piece in board.piece_map().values():
        v = VALUES[piece.piece_type] * 100
        score += v if piece.color == board.turn else -v
    return score


def _search(board, depth, alpha, beta):
    if board.is_checkmate():
        return -MATE
    if board.is_stalemate() or board.is_insufficient_material():
        return 0
    if depth == 0:
        return _quiesce(board, alpha, beta, 4)
    for move in _ordered(board, board.legal_moves):
        board.push(move)
        score = -_search(board, depth - 1, -beta, -alpha)
        board.pop()
        if score >= beta:
            return score
        alpha = max(alpha, score)
    return alpha


def _quiesce(board, alpha, beta, depth):
    stand = _evaluate(board)
    if stand >= beta or depth == 0:
        return stand
    alpha = max(alpha, stand)
    for move in _ordered(board, (m for m in board.legal_moves if board.is_capture(m))):
        board.push(move)
        score = -_quiesce(board, -beta, -alpha, depth - 1)
        board.pop()
        if score >= beta:
            return score
        alpha = max(alpha, score)
    return alpha


def _ordered(board, moves):
    def key(m):
        victim = board.piece_type_at(m.to_square)
        return -(VALUES[victim] * 10 - VALUES[board.piece_type_at(m.from_square)]) if victim else 0
    return sorted(moves, key=key)
