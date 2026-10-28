"""Convert trained profiles into compact JSON files for the browser version."""
import argparse
import json
import os

STAT_KEYS = ["games", "as_white", "as_black", "score", "ratings", "favourite_first_moves",
             "top_openings_white", "top_openings_black", "castling_rate", "capture_rate",
             "check_rate", "average_game_length_moves", "model_top1_accuracy"]


def export(profile_path, out_path, label, min_count=1):
    with open(profile_path, encoding="utf-8") as fh:
        profile = json.load(fh)
    book = {}
    for key, moves in profile["book"].items():
        # Positions reached only once are rarely reached again; drop them to keep the page fast.
        if sum(moves.values()) < min_count:
            continue
        ranked = sorted(moves.items(), key=lambda kv: -kv[1])
        book[key] = " ".join(f"{u}:{c}" for u, c in ranked)
    stats = {k: profile["stats"].get(k) for k in STAT_KEYS}
    out = {"user": profile["user"], "label": label, "weights": profile["weights"],
           "book": book, "stats": stats}
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, separators=(",", ":"))
    print(f"{out_path}: {len(book)} positions, {os.path.getsize(out_path) / 1e6:.1f} MB")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="web/data")
    ap.add_argument("--min-count", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for tc in ("bullet", "blitz", "rapid"):
        path = os.path.join(args.data, f"profile_{tc}.json")
        if os.path.exists(path):
            export(path, os.path.join(args.out, f"{tc}.json"), tc.capitalize(), args.min_count)


if __name__ == "__main__":
    main()
