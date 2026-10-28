"""Play against your style bot in the terminal."""
import argparse

import chess

from .bot import StyleBot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="data/profile.json")
    ap.add_argument("--color", choices=["white", "black"], default="white",
                    help="the colour YOU play")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--strength", type=float, default=0.5)
    args = ap.parse_args()

    bot = StyleBot(args.profile, args.temperature, args.strength)
    human = chess.WHITE if args.color == "white" else chess.BLACK
    board = chess.Board()
    print(f"You are playing {bot.user}-bot. Enter moves in SAN (e4, Nf3, O-O) or UCI (e2e4). "
          "'quit' to stop, 'undo' to take back.")
    while not board.is_game_over():
        print()
        print(board.unicode(invert_color=True, borders=True, orientation=human))
        if board.turn == human:
            text = input("Your move: ").strip()
            if text == "quit":
                return
            if text == "undo" and len(board.move_stack) >= 2:
                board.pop(); board.pop()
                continue
            try:
                board.push(board.parse_san(text))
            except ValueError:
                try:
                    board.push_uci(text)
                except ValueError:
                    print("Illegal or unrecognised move.")
        else:
            move = bot.choose_move(board)
            print(f"{bot.user}-bot plays {board.san(move)}  ({bot.last_source})")
            board.push(move)
    print(board)
    print("Game over:", board.result(), board.outcome().termination.name)


if __name__ == "__main__":
    main()
