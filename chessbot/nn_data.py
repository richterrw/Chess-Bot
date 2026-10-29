"""Turn PGN games into training tensors for the neural move predictor.

Every position is shown from the side to move (the board is flipped for
Black), so "our" pieces are always moving up the board. Each sample records
the mover's time control, rating and whether the mover is the player we are
imitating, so one network learns how people at this level play in general
and how Robbie in particular differs.
"""
import argparse
import multiprocessing as mp
import random

import chess
import chess.pgn
import numpy as np

from .train import time_class

N_PLANES = 21          # board planes; scalar inputs are added inside the model
TCS = {"bullet": 0, "blitz": 1, "rapid": 2, "daily": 2}


def flip_sq(sq, white):
    return sq if white else chess.square_mirror(sq)


def encode(board, history):
    """21x64 uint8 planes for `board`, side to move at the bottom.

    history: list of moves played so far (python-chess Moves)."""
    planes = np.zeros((N_PLANES, 64), dtype=np.uint8)
    us = board.turn
    white = us == chess.WHITE
    for sq, piece in board.piece_map().items():
        idx = (piece.piece_type - 1) + (0 if piece.color == us else 6)
        planes[idx, flip_sq(sq, white)] = 1
    them = not us
    planes[12, :] = board.has_kingside_castling_rights(us)
    planes[13, :] = board.has_queenside_castling_rights(us)
    planes[14, :] = board.has_kingside_castling_rights(them)
    planes[15, :] = board.has_queenside_castling_rights(them)
    if board.ep_square is not None and board.has_legal_en_passant():
        planes[16, flip_sq(board.ep_square, white)] = 1
    if history:
        planes[17, flip_sq(history[-1].from_square, white)] = 1
        planes[18, flip_sq(history[-1].to_square, white)] = 1
    if len(history) >= 2:
        planes[19, flip_sq(history[-2].from_square, white)] = 1
        planes[20, flip_sq(history[-2].to_square, white)] = 1
    return planes


def move_index(move, white):
    return flip_sq(move.from_square, white) * 64 + flip_sq(move.to_square, white)


def _parse(args):
    pgn_text, user = args
    import io
    game = chess.pgn.read_game(io.StringIO(pgn_text))
    if game is None:
        return None
    h = game.headers
    if h.get("Variant") or h.get("SetUp") == "1":
        return None
    is_white_user = h.get("White", "").lower() == user
    is_black_user = h.get("Black", "").lower() == user
    if not (is_white_user or is_black_user):
        return None
    tc = TCS.get(time_class(h.get("TimeControl", "")), 1)
    elos = {}
    for color, key in ((chess.WHITE, "WhiteElo"), (chess.BLACK, "BlackElo")):
        v = h.get(key, "")
        elos[color] = int(v) if v.isdigit() else 1100
    board = game.board()
    hist = []
    xs, ys, meta = [], [], []
    for move in game.mainline_moves():
        mover = board.turn
        robbie = is_white_user if mover == chess.WHITE else is_black_user
        xs.append(np.packbits(encode(board, hist)))
        ys.append(move_index(move, mover == chess.WHITE))
        meta.append((tc, elos[mover], int(robbie), len(hist)))
        board.push(move)
        hist.append(move)
    if not xs:
        return None
    return np.stack(xs), np.array(ys, dtype=np.int16), np.array(meta, dtype=np.int16)


def split_games(pgn_path):
    text = open(pgn_path, encoding="utf-8", errors="replace").read()
    parts = text.split("\n[Event ")
    return [parts[0]] + ["[Event " + p for p in parts[1:]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pgn", default="data/games.pgn")
    ap.add_argument("--user", default="robbierichter")
    ap.add_argument("--out", default="data/nn_data.npz")
    ap.add_argument("--test-fraction", type=float, default=0.1)
    args = ap.parse_args()
    games = split_games(args.pgn)
    with mp.Pool(4) as pool:
        results = pool.map(_parse, [(g, args.user.lower()) for g in games], chunksize=64)
    results = [r for r in results if r is not None]
    rng = random.Random(0)
    test_flags = [rng.random() < args.test_fraction for _ in results]
    X = np.concatenate([r[0] for r in results])
    Y = np.concatenate([r[1] for r in results])
    M = np.concatenate([r[2] for r in results])
    G = np.concatenate([np.full(len(r[1]), i, dtype=np.int32) for i, r in enumerate(results)])
    T = np.array(test_flags, dtype=bool)[G]
    np.savez(args.out, X=X, Y=Y, M=M, G=G, T=T)
    print(f"{len(results)} games, {len(Y)} positions ({int(M[:, 2].sum())} of Robbie's moves), "
          f"{int(T.sum())} positions held out for testing")


if __name__ == "__main__":
    main()
