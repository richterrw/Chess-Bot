"""Download every game a player has played on chess.com via the public API."""
import argparse
import json
import os
import sys
import time
import urllib.request

API = "https://api.chess.com/pub/player/{user}/games/archives"
HEADERS = {"User-Agent": "chess-bot style trainer (github.com/richterrw/chess-bot)"}


def _get(url, as_json=True, retries=4):
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
            return json.loads(body) if as_json else body
        except Exception as exc:  # network hiccups, 429 rate limits
            if attempt == retries - 1:
                raise
            wait = 2 ** (attempt + 1)
            print(f"  retrying {url} in {wait}s ({exc})", file=sys.stderr)
            time.sleep(wait)


def fetch_all(user, out_path):
    archives = _get(API.format(user=user.lower()))["archives"]
    print(f"Found {len(archives)} monthly archives for {user}")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    total = 0
    with open(out_path, "w", encoding="utf-8") as out:
        for i, url in enumerate(archives, 1):
            pgn = _get(url + "/pgn", as_json=False)
            n = pgn.count("[Event ")
            total += n
            print(f"  [{i}/{len(archives)}] {url.rsplit('/', 2)[-2:]} -> {n} games")
            out.write(pgn.strip() + "\n\n")
            time.sleep(0.3)  # be polite to the API
    print(f"Saved {total} games to {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--user", default="robbierichter")
    ap.add_argument("--out", default="data/games.pgn")
    args = ap.parse_args()
    fetch_all(args.user, args.out)


if __name__ == "__main__":
    main()
