// Reads cases from stdin (JSON) and checks web/bot.js against Python's output.
const { Chess } = require(process.env.CHESSJS || 'chess.js');
const bot = require('../web/bot.js');
const cases = JSON.parse(require('fs').readFileSync(0, 'utf8'));
let bad = 0, checked = 0;
for (const c of cases) {
  const chess = new Chess();
  const history = [];
  for (const u of c.moves) {
    const m = chess.move({ from: u.slice(0, 2), to: u.slice(2, 4), promotion: u[4] });
    history.push(m);
  }
  const moves = chess.moves({ verbose: true });
  const key = bot.positionKey(chess, moves);
  if (key !== c.key) { bad++; console.log('key', key, '!=', c.key); }
  const b = bot.board64(chess), phase = bot.gamePhase(chess, b);
  for (const m of moves) {
    const u = bot.uci(m), want = c.features[u];
    checked++;
    if (!want) { bad++; console.log('missing move', u); continue; }
    const got = bot.moveFeatures(chess, b, m, phase, history);
    const keys = new Set([...Object.keys(got), ...Object.keys(want)]);
    for (const k of keys) {
      if (Math.abs((got[k] || 0) - (want[k] || 0)) > 1e-9) {
        bad++; console.log(c.key, u, 'feature', k, got[k], want[k]); break;
      }
    }
    if (c.search) {
      chess.move(m);
      const v = chess.in_checkmate() ? bot.MATE : -bot.search(chess, 1, -bot.MATE, bot.MATE);
      chess.undo();
      if (v !== c.search[u]) { bad++; console.log(c.key, u, 'search', v, c.search[u]); }
    }
  }
  if (moves.length !== Object.keys(c.features).length) { bad++; console.log('move count', c.key); }
}
console.log(`checked ${checked} moves in ${cases.length} positions, ${bad} mismatches`);
process.exit(bad ? 1 : 0);
