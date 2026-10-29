/* Browser forward pass of the neural move predictor (chessbot/nn_model.py).
 * Plain JavaScript, no libraries: about 30 million multiply-adds per move. */
(function (root) {
  'use strict';
  var PIECES = { p: 0, n: 1, b: 2, r: 3, q: 4, k: 5 };

  /* meta: the JSON written by chessbot/nn_export.py (weights in base64). */
  function MoveNet(meta) {
    var bytes = typeof atob === 'function'
      ? Uint8Array.from(atob(meta.weights), function (c) { return c.charCodeAt(0); })
      : new Uint8Array(Buffer.from(meta.weights, 'base64'));
    var f = new Float32Array(bytes.buffer), off = 0, ch = meta.channels;
    function take(n) { var a = f.subarray(off, off + n); off += n; return a; }
    function conv(cin, cout, k) { return { cin: cin, cout: cout, k: k, w: take(cout * cin * k * k), b: take(cout) }; }
    this.stem = conv(meta.inputs, ch, 3);
    this.blocks = [];
    for (var i = 0; i < meta.blocks; i++) this.blocks.push([conv(ch, ch, 3), conv(ch, ch, 3)]);
    this.head = conv(ch, 64, 1);
    if (off !== f.length) throw new Error('movenet: weight file does not match its description');
  }

  /* x: Float32Array [cin*64]; returns Float32Array [cout*64]. */
  function conv3(L, x, residual, relu) {
    var out = new Float32Array(L.cout * 64), w = L.w, cin = L.cin;
    for (var o = 0; o < L.cout; o++) {
      var acc = out.subarray(o * 64, o * 64 + 64);
      acc.fill(L.b[o]);
      for (var c = 0; c < cin; c++) {
        var xi = c * 64, wi = (o * cin + c) * 9;
        for (var ky = 0; ky < 3; ky++) {
          for (var kx = 0; kx < 3; kx++) {
            var wv = w[wi + ky * 3 + kx];
            if (wv === 0) continue;
            var dy = ky - 1, dx = kx - 1;
            var y0 = Math.max(0, -dy), y1 = Math.min(8, 8 - dy), x0 = Math.max(0, -dx), x1 = Math.min(8, 8 - dx);
            for (var yy = y0; yy < y1; yy++) {
              var row = yy * 8, src = xi + (yy + dy) * 8 + dx;
              for (var xx = x0; xx < x1; xx++) acc[row + xx] += wv * x[src + xx];
            }
          }
        }
      }
      if (residual) for (var j = 0; j < 64; j++) acc[j] += residual[o * 64 + j];
      if (relu) for (var q = 0; q < 64; q++) if (acc[q] < 0) acc[q] = 0;
    }
    return out;
  }

  function conv1(L, x) {
    var out = new Float32Array(L.cout * 64);
    for (var o = 0; o < L.cout; o++) {
      for (var s = 0; s < 64; s++) {
        var v = L.b[o];
        for (var c = 0; c < L.cin; c++) v += L.w[o * L.cin + c] * x[c * 64 + s];
        out[o * 64 + s] = v;
      }
    }
    return out;
  }

  /* Returns Float32Array(4096) of logits indexed from*64 + to (mover's frame). */
  MoveNet.prototype.forward = function (input) {
    var x = conv3(this.stem, input, null, true);
    for (var i = 0; i < this.blocks.length; i++) {
      var y = conv3(this.blocks[i][0], x, null, true);
      x = conv3(this.blocks[i][1], y, x, true);
    }
    var h = conv1(this.head, x), logits = new Float32Array(4096);
    for (var to = 0; to < 64; to++) for (var from = 0; from < 64; from++) logits[from * 64 + to] = h[to * 64 + from];
    return logits;
  };

  function sqIndex(name) { return (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97); }

  /* Same planes as chessbot/nn_data.encode, plus the 5 constant inputs. */
  function encode(chess, history, opts) {
    var fen = chess.fen().split(' '), us = fen[1], white = us === 'w';
    var x = new Float32Array(26 * 64);
    function flip(s) { return white ? s : s ^ 56; }
    var board = chess.board();
    for (var r = 0; r < 8; r++) for (var f = 0; f < 8; f++) {
      var p = board[r][f];
      if (!p) continue;
      var idx = PIECES[p.type] + (p.color === us ? 0 : 6);
      x[idx * 64 + flip((7 - r) * 8 + f)] = 1;
    }
    var cast = fen[2], mine = white ? ['K', 'Q'] : ['k', 'q'], theirs = white ? ['k', 'q'] : ['K', 'Q'];
    var flags = [cast.indexOf(mine[0]) >= 0, cast.indexOf(mine[1]) >= 0, cast.indexOf(theirs[0]) >= 0, cast.indexOf(theirs[1]) >= 0];
    for (var k = 0; k < 4; k++) if (flags[k]) for (var s = 0; s < 64; s++) x[(12 + k) * 64 + s] = 1;
    if (fen[3] !== '-' && opts.legalEp) x[16 * 64 + flip(sqIndex(fen[3]))] = 1;
    var n = history.length;
    if (n >= 1) { x[17 * 64 + flip(sqIndex(history[n - 1].from))] = 1; x[18 * 64 + flip(sqIndex(history[n - 1].to))] = 1; }
    if (n >= 2) { x[19 * 64 + flip(sqIndex(history[n - 2].from))] = 1; x[20 * 64 + flip(sqIndex(history[n - 2].to))] = 1; }
    var extra = [opts.tc === 0 ? 1 : 0, opts.tc === 1 ? 1 : 0, opts.tc === 2 ? 1 : 0, (opts.rating - 1100) / 300, opts.robbie ? 1 : 0];
    for (var e = 0; e < 5; e++) for (var t = 0; t < 64; t++) x[(21 + e) * 64 + t] = extra[e];
    return x;
  }

  /* Probability of each legal move (chess.js verbose moves) under the net.
   * Underpromotions share their square pair with the queen promotion and
   * are almost never right, so they get a tiny share. */
  MoveNet.prototype.moveProbs = function (chess, moves, history, opts) {
    var white = chess.turn() === 'w';
    var legalEp = moves.some(function (m) { return m.flags.indexOf('e') >= 0; });
    var logits = this.forward(encode(chess, history, { tc: opts.tc, rating: opts.rating, robbie: opts.robbie, legalEp: legalEp }));
    var raw = moves.map(function (m) {
      var from = sqIndex(m.from), to = sqIndex(m.to);
      if (!white) { from ^= 56; to ^= 56; }
      return logits[from * 64 + to] - (m.promotion && m.promotion !== 'q' ? 8 : 0);
    });
    var mx = Math.max.apply(null, raw), sum = 0;
    var ex = raw.map(function (v) { var e = Math.exp(v - mx); sum += e; return e; });
    return ex.map(function (e) { return e / sum; });
  };

  var api = { MoveNet: MoveNet, encode: encode };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ChessMoveNet = api;
})(this);
