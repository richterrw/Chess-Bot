# Chess Bot: plays like robbierichter

A chess bot trained on every game from a chess.com account. It plays the
moves you actually played, and when it reaches a position you've never seen,
it picks the move you'd most likely choose.

## How it imitates you

1. **Your personal book.** Every position you reached in your first 20 moves,
   plus every later position you reached at least twice, is stored with
   the moves you chose there and how often. In those positions the bot picks
   one of your moves, weighted by how often you played it. So it opens like
   you, answers 1.e4 like you, and repeats your habits.
2. **A model of how you choose moves.** Out of book, a softmax model trained on
   your games scores each legal move. Its features include which piece moves,
   captures, checks, castling timing, pawn storms, early queen moves, hanging
   pieces, threats, favourite squares, and so on, each split by game phase.
   The model learns how much *you* care about each one.
3. **Optional blunder check.** `strength` mixes in a small tactical search.
   At `0` the bot is pure imitation. Higher values make it blunder less than
   the raw model. It never misses a mate-in-one when `strength > 0`.

## Play in your browser

`web/` is a version that runs entirely in the browser, with three bots trained
separately on Robbie's bullet, blitz and rapid games. To run it locally:

```bash
cd web && python -m http.server 8000
```

Then open http://localhost:8000. `web/bot.js` is a line-for-line port of the
Python bot; `python -m tests.test_web_parity data/games.pgn` checks that both
compute identical features and search scores.

## Public website and friends' results

`.github/workflows/pages.yml` publishes `web/` to GitHub Pages on every push
(turn it on under Settings → Pages → Source: GitHub Actions). Players enter
their name, and each finished game (win, loss, draw, resignation or timeout,
with the full PGN) is saved to a Google Sheet. Setup steps are in
[results/README.md](results/README.md).

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Download all your games from chess.com (public API, no login needed)
python -m chessbot.fetch --user robbierichter          # -> data/games.pgn

# 2. Learn your style (a couple of minutes for thousands of games)
python -m chessbot.train --user robbierichter          # -> data/profile.json
#    only learn from one time control, e.g. your blitz games:
python -m chessbot.train --time-classes blitz

#    or train one bot per time control and export them for the browser:
for tc in bullet blitz rapid; do
  python -m chessbot.train --time-classes $tc --out data/profile_$tc.json
done
#    and a "peak" bot that only learns from games played at your highest ratings:
python -m chessbot.train --top-percent 20 --time-classes bullet blitz rapid --out data/profile_peak.json
python -m chessbot.export_web

# 3. Play against yourself in the terminal (add --profile data/profile_blitz.json etc.)
python -m chessbot.play --color white
```

Training prints a summary of your play: openings, replies to 1.e4/1.d4,
castling habits, capture/check rate, ratings, and how often the model
predicts your exact move.

### Knobs

| Option          | Meaning                                                            |
|-----------------|--------------------------------------------------------------------|
| `--temperature` | `0` = always your most likely move, `1` = sample like you (default) |
| `--strength`    | `0` = pure imitation, `0.5` default, `2+` = noticeably stronger     |

### The neural network (browser bots)

The browser bots predict moves with a small residual network (6 blocks, 64
channels, 463k weights) trained on every move of every game: first on both
players' moves, so it learns how people at this level play, then fine-tuned
with an "is Robbie" input switched on. On Robbie's moves in held-out games it
picks his exact move 45.5% of the time (top-3: 69%), versus 37.9% for the
hand-made feature model. It runs in plain JavaScript (`web/movenet.js`).

```bash
python -m chessbot.nn_data      # games -> data/nn_data.npz
python -m chessbot.nn_train     # ~50 min on 4 CPU cores -> data/movenet.pt
python -m chessbot.nn_export    # -> web/data/movenet.bin + .json
```

A short tactics search then removes moves that lose material. Measured on
231 middlegame positions from Robbie's held-out blitz games (blunder = loses
2+ pawns, judged by a 4-ply search):

| Level            | Search          | Same move as Robbie | Blunders |
|------------------|-----------------|---------------------|----------|
| Robbie himself   |                 | -                   | 10.8%    |
| Rushed           | 1 ply, 200cp    | 35.5%               | 8.2%     |
| Like Robbie      | 2 plies, 150cp  | 39.0%               | 4.3%     |
| Sharp            | 3 plies, 100cp  | 34.6%               | 1.3%     |
| Max              | 4 plies, 50cp   | 35.9%               | 0%       |
| old feature bot  | 1 ply           | 25.5%               | 10.4%    |

### Use it in a chess GUI or on lichess

`python -m chessbot.uci --profile data/profile.json` speaks the UCI protocol,
so you can load it as an engine in Arena, Cute Chess, BanksiaGUI, and so on,
or run it as a lichess bot account with
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot). UCI options:
`Temperature`, `Strength`, `OwnBook`.

(chess.com doesn't let third-party bots play on the site, so lichess or a
local GUI is the way to play against it online.)
