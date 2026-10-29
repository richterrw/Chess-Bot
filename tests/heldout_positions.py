"""Write a sample of Robbie's moves from the held-out games as JSON.

Uses the same game split as chessbot/nn_data.py, so none of these games
were seen by the neural network during training.
Usage: python -m tests.heldout_positions data/games.pgn out.json [n]
"""
import io
import json
import random
import sys

import chess
import chess.pgn

from chessbot.nn_data import TCS, _parse, split_games
from chessbot.train import time_class


def main():
    pgn, out, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3000
    user = "robbierichter"
    games = [g for g in split_games(pgn) if _parse((g, user)) is not None]
    rng = random.Random(0)
    held = [g for g in games if rng.random() < 0.1]
    pick = random.Random(1)
    cases = []
    for text in held:
        game = chess.pgn.read_game(io.StringIO(text))
        h = game.headers
        color = chess.WHITE if h["White"].lower() == user else chess.BLACK
        elo = h.get("WhiteElo" if color == chess.WHITE else "BlackElo", "")
        tc = TCS.get(time_class(h.get("TimeControl", "")), 1)
        moves = list(game.mainline_moves())
        for i, m in enumerate(moves):
            if (i % 2 == 0) == (color == chess.WHITE) and pick.random() < 0.15:
                cases.append({"moves": [x.uci() for x in moves[:i]], "played": m.uci(), "tc": tc,
                              "rating": int(elo) if elo.isdigit() else 1100,
                              "tcName": time_class(h.get("TimeControl", ""))})
    pick.shuffle(cases)
    json.dump(cases[:n], open(out, "w"))
    print(f"{len(held)} held-out games, wrote {min(n, len(cases))} positions")


if __name__ == "__main__":
    main()
