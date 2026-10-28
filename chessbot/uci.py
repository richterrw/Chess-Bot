"""UCI protocol wrapper so the bot can run in any chess GUI or lichess-bot."""
import argparse
import sys

import chess

from .bot import StyleBot


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--profile", default="data/profile.json")
    args = ap.parse_args()
    opts = {"Temperature": 1.0, "Strength": 0.5, "OwnBook": True}
    bot = None
    board = chess.Board()

    def out(line):
        sys.stdout.write(line + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        parts = line.strip().split()
        if not parts:
            continue
        cmd = parts[0]
        if cmd == "uci":
            out("id name StyleBot")
            out("id author chess-bot")
            out("option name Temperature type string default 1.0")
            out("option name Strength type string default 0.5")
            out("option name OwnBook type check default true")
            out("uciok")
        elif cmd == "setoption" and "name" in parts and "value" in parts:
            name = " ".join(parts[parts.index("name") + 1:parts.index("value")])
            value = " ".join(parts[parts.index("value") + 1:])
            if name == "OwnBook":
                opts[name] = value.lower() == "true"
            elif name in opts:
                opts[name] = float(value)
            bot = None
        elif cmd == "isready":
            if bot is None:
                bot = StyleBot(args.profile, opts["Temperature"], opts["Strength"], opts["OwnBook"])
            out("readyok")
        elif cmd == "ucinewgame":
            board = chess.Board()
        elif cmd == "position":
            if parts[1] == "startpos":
                board = chess.Board()
                rest = parts[2:]
            else:
                idx = parts.index("moves") if "moves" in parts else len(parts)
                board = chess.Board(" ".join(parts[2:idx]))
                rest = parts[idx:]
            if rest and rest[0] == "moves":
                for uci in rest[1:]:
                    board.push_uci(uci)
        elif cmd == "go":
            if bot is None:
                bot = StyleBot(args.profile, opts["Temperature"], opts["Strength"], opts["OwnBook"])
            move = bot.choose_move(board)
            out(f"bestmove {move.uci() if move else '0000'}")
        elif cmd == "quit":
            break


if __name__ == "__main__":
    main()
