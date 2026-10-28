"""Check the browser bot (web/bot.js) computes exactly what the Python bot does.

Usage: python -m tests.test_web_parity data/games.pgn [n_positions]
Needs node and the chess.js@0.10.3 npm package (set CHESSJS to its path).
"""
import json
import random
import subprocess
import sys

import chess.pgn

from chessbot.bot import MATE, _search
from chessbot.features import move_features
from chessbot.train import position_key


def main():
    pgn, n = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 300
    rng = random.Random(0)
    cases = []
    with open(pgn, encoding="utf-8") as fh:
        while len(cases) < n:
            game = chess.pgn.read_game(fh)
            if game is None:
                break
            if game.headers.get("Variant") or rng.random() < 0.9:
                continue
            moves = list(game.mainline_moves())
            if not moves:
                continue
            board = game.board()
            for m in moves[:rng.randrange(len(moves))]:
                board.push(m)
            if board.is_game_over():
                continue
            feats, search = {}, {}
            for m in board.legal_moves:
                feats[m.uci()] = {str(k): v for k, v in move_features(board, m).items()}
                if len(cases) % 3 == 0:
                    board.push(m)
                    search[m.uci()] = MATE if board.is_checkmate() else -_search(board, 1, -MATE, MATE)
                    board.pop()
            cases.append({"moves": [m.uci() for m in board.move_stack], "key": position_key(board),
                          "features": feats, "search": search or None})
    result = subprocess.run(["node", "tests/parity.js"], input=json.dumps(cases), text=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
