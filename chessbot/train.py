"""Learn a player's style from their PGN games.

Produces a profile JSON with:
  * book   - every position you reached (as the side to move) and the moves
             you chose there, with counts. The bot replays these directly.
  * model  - weights of a softmax move-choice model trained to predict your
             move in every position, used when the bot is "out of book".
  * stats  - a human-readable summary of how you play.
"""
import argparse
import collections
import io
import json
import math
import random
import sys
import time

import chess
import chess.pgn

from .features import N_FEATURES, DENSE, PHASES, SQ_OFFSET, move_features

BOOK_MAX_PLY = 40          # always store positions from the first 20 moves
BOOK_MIN_LATE_COUNT = 2    # later positions only if reached at least twice


def position_key(board):
    """FEN without the move counters, so transpositions share an entry."""
    return " ".join(board.fen().split(" ")[:4])


def iter_games(pgn_path, user, time_classes=None, rated_only=False):
    user = user.lower()
    with open(pgn_path, encoding="utf-8", errors="replace") as fh:
        while True:
            game = chess.pgn.read_game(fh)
            if game is None:
                return
            h = game.headers
            if h.get("Variant") or h.get("SetUp") == "1":
                continue  # skip chess960 / odd starting positions
            if h.get("White", "").lower() == user:
                color = chess.WHITE
            elif h.get("Black", "").lower() == user:
                color = chess.BLACK
            else:
                continue
            tc = time_class(h.get("TimeControl", ""))
            if time_classes and tc not in time_classes:
                continue
            yield game, color, tc


def time_class(tc):
    """Mirror chess.com's bullet/blitz/rapid/daily buckets."""
    if not tc or tc == "-" or "/" in tc:
        return "daily"
    base, _, inc = tc.partition("+")
    try:
        total = int(base) + 40 * int(inc or 0)
    except ValueError:
        return "unknown"
    if total < 180:
        return "bullet"
    if total < 600:
        return "blitz"
    return "rapid"


def train(pgn_path, user, out_path, max_positions=60000, epochs=3, time_classes=None, seed=0):
    rng = random.Random(seed)
    book = collections.defaultdict(collections.Counter)
    samples = []  # (features of each legal move, index of the move played)
    stats = collections.Counter()
    ratings = collections.defaultdict(list)
    openings = {chess.WHITE: collections.Counter(), chess.BLACK: collections.Counter()}
    first_moves = collections.Counter()
    replies = collections.defaultdict(collections.Counter)
    castle_moves = []
    seen_positions = 0
    t0 = time.time()

    for n, (game, color, tc) in enumerate(iter_games(pgn_path, user, time_classes), 1):
        h = game.headers
        side = "white" if color == chess.WHITE else "black"
        stats["games"] += 1
        stats[f"games_{side}"] += 1
        stats[f"tc_{tc}"] += 1
        result = h.get("Result", "*")
        won = (result == "1-0") == (color == chess.WHITE) and result != "1/2-1/2" and result != "*"
        stats["wins" if won else "draws" if result == "1/2-1/2" else "losses"] += 1
        elo = h.get("WhiteElo" if color == chess.WHITE else "BlackElo")
        if elo and elo.isdigit():
            ratings[tc].append(int(elo))
        name = h.get("ECOUrl", "").rsplit("/", 1)[-1].replace("-", " ") or h.get("Opening", "")
        if name:
            openings[color][name[:60]] += 1

        board = game.board()
        castled = False
        for ply, move in enumerate(game.mainline_moves()):
            if board.turn == color:
                key = position_key(board)
                if ply < BOOK_MAX_PLY:
                    book[key][move.uci()] += 1
                else:
                    book["~" + key][move.uci()] += 1  # tentative, pruned later
                if ply == 0:
                    first_moves[board.san(move)] += 1
                elif ply == 1:
                    replies[board.peek().uci()][board.san(move)] += 1
                if board.is_castling(move) and not castled:
                    castled = True
                    castle_moves.append(board.fullmove_number)
                    stats["castle_kingside" if board.is_kingside_castling(move) else "castle_queenside"] += 1
                if board.is_capture(move):
                    stats["captures"] += 1
                if board.gives_check(move):
                    stats["checks"] += 1
                if board.piece_type_at(move.from_square) == chess.QUEEN and board.fullmove_number <= 6:
                    stats["early_queen_moves"] += 1
                stats["moves"] += 1
                # Reservoir-sample positions for the move-choice model.
                legal = list(board.legal_moves)
                if len(legal) > 1:
                    seen_positions += 1
                    if len(samples) < max_positions:
                        samples.append((board.copy(), move))
                    else:
                        j = rng.randrange(seen_positions)
                        if j < max_positions:
                            samples[j] = (board.copy(), move)
            board.push(move)
        if not castled:
            stats["never_castled"] += 1
        stats["plies"] += board.ply()
        if n % 500 == 0:
            print(f"  parsed {n} games ({time.time() - t0:.0f}s)", file=sys.stderr)

    if not stats["games"]:
        sys.exit(f"No games for '{user}' found in {pgn_path}")
    print(f"Parsed {stats['games']} games, {stats['moves']} of your moves", file=sys.stderr)

    # Keep late-game positions only when they recur (true habits, not noise).
    final_book = {}
    for key, counter in book.items():
        if key.startswith("~"):
            if sum(counter.values()) < BOOK_MIN_LATE_COUNT:
                continue
            real = key[1:]
            merged = final_book.setdefault(real, collections.Counter())
            merged.update(counter)
        else:
            final_book.setdefault(key, collections.Counter()).update(counter)

    print(f"Training move-choice model on {len(samples)} positions...", file=sys.stderr)
    weights, accuracy = fit_model(samples, epochs, rng)

    games = stats["games"]
    summary = {
        "user": user,
        "games": games,
        "as_white": stats["games_white"], "as_black": stats["games_black"],
        "score": {"wins": stats["wins"], "draws": stats["draws"], "losses": stats["losses"]},
        "time_classes": {k[3:]: v for k, v in stats.items() if k.startswith("tc_")},
        "ratings": {tc: {"peak": max(r), "latest": r[-1], "average": round(sum(r) / len(r))}
                    for tc, r in ratings.items() if r},
        "favourite_first_moves": first_moves.most_common(5),
        "replies_as_black": {m: c.most_common(3) for m, c in replies.items()
                             if sum(c.values()) >= 5},
        "top_openings_white": openings[chess.WHITE].most_common(10),
        "top_openings_black": openings[chess.BLACK].most_common(10),
        "castling_rate": round(1 - stats["never_castled"] / games, 3),
        "kingside_castle_share": round(stats["castle_kingside"] /
                                       max(1, stats["castle_kingside"] + stats["castle_queenside"]), 3),
        "average_castle_move": round(sum(castle_moves) / len(castle_moves), 1) if castle_moves else None,
        "capture_rate": round(stats["captures"] / stats["moves"], 3),
        "check_rate": round(stats["checks"] / stats["moves"], 3),
        "early_queen_moves_per_game": round(stats["early_queen_moves"] / games, 2),
        "average_game_length_moves": round(stats["plies"] / games / 2, 1),
        "model_top1_accuracy": round(accuracy, 3),
        "book_positions": len(final_book),
        "style_weights": {name: [round(weights[i * PHASES + p], 3) for p in range(PHASES)]
                          for i, name in enumerate(DENSE)},
    }
    profile = {
        "version": 1,
        "user": user,
        "book": {k: dict(v) for k, v in final_book.items()},
        "weights": [round(w, 5) for w in weights],
        "stats": summary,
    }
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(profile, fh)
    print(f"Wrote profile to {out_path}", file=sys.stderr)
    return profile


def fit_model(samples, epochs, rng, lr=0.1, l2=1e-4, holdout=0.1):
    """Conditional-logit (softmax over legal moves) trained with SGD."""
    t0 = time.time()
    data = []
    for i, (board, played) in enumerate(samples):
        moves = list(board.legal_moves)
        feats = [move_features(board, m) for m in moves]
        data.append((feats, moves.index(played)))
        if (i + 1) % 5000 == 0:
            print(f"  featurised {i + 1}/{len(samples)} ({time.time() - t0:.0f}s)", file=sys.stderr)
    rng.shuffle(data)
    n_test = int(len(data) * holdout) if len(data) >= 50 else 0
    test, train_set = data[:n_test], data[n_test:]

    w = [0.0] * N_FEATURES
    for epoch in range(epochs):
        rng.shuffle(train_set)
        step = lr / (1 + epoch)
        loss = 0.0
        for feats, target in train_set:
            scores = [sum(w[k] * v for k, v in f.items()) for f in feats]
            m = max(scores)
            exps = [math.exp(s - m) for s in scores]
            z = sum(exps)
            loss -= math.log(exps[target] / z)
            grad = collections.defaultdict(float)
            for f, e in zip(feats, exps):
                p = e / z
                for k, v in f.items():
                    grad[k] -= p * v
            for k, v in feats[target].items():
                grad[k] += v
            for k, g in grad.items():
                # Square-preference features are noisier: regularise them harder.
                reg = l2 * (10 if k >= SQ_OFFSET else 1)
                w[k] += step * (g - reg * w[k])
        print(f"  epoch {epoch + 1}: train loss {loss / max(1, len(train_set)):.3f}", file=sys.stderr)

    eval_set = test or train_set
    correct = 0
    for feats, target in eval_set:
        scores = [sum(w[k] * v for k, v in f.items()) for f in feats]
        correct += scores.index(max(scores)) == target
    accuracy = correct / max(1, len(eval_set))
    print(f"  model predicts your exact move {accuracy:.1%} of the time "
          f"({'held-out' if test else 'train'} positions)", file=sys.stderr)
    return w, accuracy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pgn", default="data/games.pgn")
    ap.add_argument("--user", default="robbierichter")
    ap.add_argument("--out", default="data/profile.json")
    ap.add_argument("--max-positions", type=int, default=60000,
                    help="positions sampled to train the move model (more = slower, better)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--time-classes", nargs="*", choices=["bullet", "blitz", "rapid", "daily"],
                    help="only learn from these time controls (default: all)")
    args = ap.parse_args()
    profile = train(args.pgn, args.user, args.out, args.max_positions, args.epochs,
                    set(args.time_classes) if args.time_classes else None)
    s = dict(profile["stats"])
    s.pop("style_weights")
    print(json.dumps(s, indent=2))


if __name__ == "__main__":
    main()
