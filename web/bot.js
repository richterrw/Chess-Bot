/* Browser port of chessbot/features.py and chessbot/bot.py.
 * Must stay feature-for-feature identical to the Python version, because the
 * weights were trained there. tests/test_web_parity.py checks this. */
(function (root) {
  'use strict';
  var VALUES = { p: 1, n: 3, b: 3, r: 5, q: 9, k: 0 };
  var PT_INDEX = { p: 1, n: 2, b: 3, r: 4, q: 5, k: 6 };
  var DENSE = [
    'bias_piece_P', 'bias_piece_N', 'bias_piece_B', 'bias_piece_R',
    'bias_piece_Q', 'bias_piece_K',
    'capture', 'capture_value', 'net_material', 'recapture',
    'gives_check', 'castle_k', 'castle_q', 'promotion',
    'pawn_advance', 'develops_minor', 'backward',
    'centralise', 'approach_king', 'king_walk',
    'hangs_moved', 'hangs_other', 'threatens', 'trade_offer',
    'early_queen', 'same_piece_again', 'pawn_storm', 'edge_knight'
  ];
  var PHASES = 3;
  var SQ_OFFSET = DENSE.length * PHASES;
  var IDX = {};
  DENSE.forEach(function (n, i) { IDX[n] = i; });
  var MATE = 100000;

  function sq(name) { return (name.charCodeAt(1) - 49) * 8 + (name.charCodeAt(0) - 97); }
  function file(s) { return s & 7; }
  function rank(s) { return s >> 3; }
  function dist(a, b) { return Math.max(Math.abs(file(a) - file(b)), Math.abs(rank(a) - rank(b))); }
  var CENTER_DIST = [];
  for (var s0 = 0; s0 < 64; s0++) {
    CENTER_DIST.push(Math.floor(Math.max(Math.abs(file(s0) * 2 - 7), Math.abs(rank(s0) * 2 - 7)) / 2));
  }

  /* 64-entry array (a1 = 0) of {type, color} or null. */
  function board64(chess) {
    var b = chess.board(), out = new Array(64);
    for (var r = 0; r < 8; r++) for (var f = 0; f < 8; f++) out[(7 - r) * 8 + f] = b[r][f];
    return out;
  }

  var KNIGHT = [[1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1], [-1, 2]];
  var KING = [[1, 0], [1, 1], [0, 1], [-1, 1], [-1, 0], [-1, -1], [0, -1], [1, -1]];
  var ROOK_D = [[1, 0], [-1, 0], [0, 1], [0, -1]];
  var BISHOP_D = [[1, 1], [1, -1], [-1, 1], [-1, -1]];

  function onBoard(f, r) { return f >= 0 && f < 8 && r >= 0 && r < 8; }

  /* Squares attacked by the piece standing on s (like python-chess board.attacks). */
  function attacksFrom(b, s) {
    var p = b[s], out = [];
    if (!p) return out;
    var f = file(s), r = rank(s);
    function step(list) {
      list.forEach(function (d) { if (onBoard(f + d[0], r + d[1])) out.push((r + d[1]) * 8 + f + d[0]); });
    }
    function slide(list) {
      list.forEach(function (d) {
        var ff = f + d[0], rr = r + d[1];
        while (onBoard(ff, rr)) {
          out.push(rr * 8 + ff);
          if (b[rr * 8 + ff]) break;
          ff += d[0]; rr += d[1];
        }
      });
    }
    if (p.type === 'p') {
      var dr = p.color === 'w' ? 1 : -1;
      step([[-1, dr], [1, dr]]);
    } else if (p.type === 'n') step(KNIGHT);
    else if (p.type === 'k') step(KING);
    else {
      if (p.type !== 'b') slide(ROOK_D);
      if (p.type !== 'r') slide(BISHOP_D);
    }
    return out;
  }

  /* Squares of pieces of `color` attacking s (like python-chess board.attackers). */
  function attackers(b, color, s) {
    var out = [];
    for (var i = 0; i < 64; i++) {
      var p = b[i];
      if (!p || p.color !== color) continue;
      if (p.type === 'p' || p.type === 'n' || p.type === 'k') {
        var df = Math.abs(file(i) - file(s)), drr = rank(s) - rank(i);
        if (p.type === 'p') { if (df === 1 && drr === (color === 'w' ? 1 : -1)) out.push(i); }
        else if (p.type === 'n') { if ((df === 1 && Math.abs(drr) === 2) || (df === 2 && Math.abs(drr) === 1)) out.push(i); }
        else if (Math.max(df, Math.abs(drr)) === 1) out.push(i);
      } else if (attacksFrom(b, i).indexOf(s) >= 0) out.push(i);
    }
    return out;
  }

  function enPriseValue(b, s, color) {
    var p = b[s];
    if (!p || p.type === 'k') return 0;
    var them = color === 'w' ? 'b' : 'w';
    var att = attackers(b, them, s);
    if (!att.length) return 0;
    var value = VALUES[p.type];
    if (!attackers(b, color, s).length) return value;
    var cheapest = Math.min.apply(null, att.map(function (a) { return VALUES[b[a].type] || 100; }));
    return Math.max(0, value - cheapest);
  }

  function fullmove(chess) { return parseInt(chess.fen().split(' ')[5], 10); }

  function gamePhase(chess, b) {
    if (fullmove(chess) <= 10) return 0;
    var m = 0;
    for (var i = 0; i < 64; i++) if (b[i] && b[i].type !== 'p' && b[i].type !== 'k') m += VALUES[b[i].type];
    return m <= 26 ? 2 : 1;
  }

  function relRank(s, color) { return color === 'w' ? rank(s) : 7 - rank(s); }

  function kingSquare(b, color) {
    for (var i = 0; i < 64; i++) if (b[i] && b[i].type === 'k' && b[i].color === color) return i;
    return null;
  }

  /* history: array of {from, to} for the moves played so far (oldest first). */
  function moveFeatures(chess, b, move, phase, history) {
    var color = move.color, them = color === 'w' ? 'b' : 'w';
    var from = sq(move.from), to = sq(move.to), pt = move.piece;
    var last = history.length ? history[history.length - 1] : null;
    var f = {};
    function add(name, v) {
      if (v === undefined) v = 1;
      if (!v) return;
      var k = IDX[name] * PHASES + phase;
      f[k] = (f[k] || 0) + v;
    }
    add('bias_piece_' + pt.toUpperCase());
    var isCapture = move.flags.indexOf('c') >= 0 || move.flags.indexOf('e') >= 0;
    var capturedValue = 0;
    if (isCapture) {
      capturedValue = move.flags.indexOf('e') >= 0 ? 1 : VALUES[b[to].type];
      add('capture');
      add('capture_value', capturedValue / 9);
      if (last && sq(last.to) === to) add('recapture');
    }
    var kside = move.flags.indexOf('k') >= 0, qside = move.flags.indexOf('q') >= 0;
    if (kside) add('castle_k');
    else if (qside) add('castle_q');
    else if (pt === 'k' && phase < 2) add('king_walk');
    if (move.promotion) add('promotion');
    var fr = relRank(from, color), tr = relRank(to, color);
    var theirKing = kingSquare(b, them);
    if (pt === 'p') {
      add('pawn_advance', tr / 7);
      if (theirKing !== null && Math.abs(file(theirKing) - file(to)) <= 1) add('pawn_storm');
    }
    if ((pt === 'n' || pt === 'b') && fr === 0 && phase === 0) add('develops_minor');
    if (tr < fr && pt !== 'p') add('backward');
    if (pt === 'n' && (file(to) === 0 || file(to) === 7)) add('edge_knight');
    add('centralise', (CENTER_DIST[from] - CENTER_DIST[to]) / 3);
    if (theirKing !== null) add('approach_king', (dist(from, theirKing) - dist(to, theirKing)) / 7);
    if (pt === 'q' && phase === 0) add('early_queen');
    if (history.length >= 2 && sq(history[history.length - 2].to) === from && phase === 0) add('same_piece_again');

    chess.move(move);
    var check = chess.in_check();
    var a = board64(chess);
    chess.undo();
    if (check) add('gives_check');
    var hang = enPriseValue(a, to, color);
    add('hangs_moved', hang / 9);
    var other = 0;
    for (var i = 0; i < 64; i++) {
      var p = a[i];
      if (p && p.color === color && i !== to && 'nbrq'.indexOf(p.type) >= 0) other = Math.max(other, enPriseValue(a, i, color));
    }
    add('hangs_other', other / 9);
    add('net_material', (capturedValue - hang) / 9);
    if (isCapture && hang && capturedValue && VALUES[pt] === capturedValue) add('trade_offer');
    var threat = 0;
    attacksFrom(a, to).forEach(function (t) {
      var tp = a[t];
      if (tp && tp.color !== color && tp.type !== 'k') threat = Math.max(threat, enPriseValue(a, t, them));
    });
    add('threatens', threat / 9);

    var rel = color === 'w' ? to : to ^ 56;
    f[SQ_OFFSET + (PT_INDEX[pt] - 1) * 64 + rel] = 1;
    return f;
  }

  /* ---- tiny material search, used only as a blunder check ----
   * Uses its own fast move generator on a 64-square array: chess.js's
   * move()/moves() are far too slow to call thousands of times per move. */
  function posFromChess(chess) {
    var f = chess.fen().split(' ');
    return { b: board64(chess), turn: f[1], castling: f[2] === '-' ? '' : f[2],
             ep: f[3] === '-' ? -1 : sq(f[3]) };
  }

  function isAttacked(b, s, by) {
    var f = file(s), r = rank(s), i, d, ff, rr, p;
    var pr = by === 'w' ? r - 1 : r + 1;
    for (i = -1; i <= 1; i += 2) {
      if (onBoard(f + i, pr)) { p = b[pr * 8 + f + i]; if (p && p.color === by && p.type === 'p') return true; }
    }
    for (i = 0; i < 8; i++) {
      d = KNIGHT[i];
      if (onBoard(f + d[0], r + d[1])) { p = b[(r + d[1]) * 8 + f + d[0]]; if (p && p.color === by && p.type === 'n') return true; }
      d = KING[i];
      if (onBoard(f + d[0], r + d[1])) { p = b[(r + d[1]) * 8 + f + d[0]]; if (p && p.color === by && p.type === 'k') return true; }
    }
    for (i = 0; i < 8; i++) {
      d = i < 4 ? ROOK_D[i] : BISHOP_D[i - 4];
      ff = f + d[0]; rr = r + d[1];
      while (onBoard(ff, rr)) {
        p = b[rr * 8 + ff];
        if (p) {
          if (p.color === by && (p.type === 'q' || p.type === (i < 4 ? 'r' : 'b'))) return true;
          break;
        }
        ff += d[0]; rr += d[1];
      }
    }
    return false;
  }

  function pseudoMoves(pos, capturesOnly) {
    var b = pos.b, us = pos.turn, out = [];
    function add(from, to, extra) {
      var m = { from: from, to: to, piece: b[from].type, captured: b[to] ? b[to].type : null };
      if (extra) for (var k in extra) m[k] = extra[k];
      out.push(m);
    }
    function addPawn(from, to, cap) {
      if (rank(to) === 7 || rank(to) === 0) {
        ['q', 'r', 'b', 'n'].forEach(function (pr) { add(from, to, { promotion: pr }); });
      } else add(from, to);
    }
    for (var s = 0; s < 64; s++) {
      var p = b[s];
      if (!p || p.color !== us) continue;
      var f = file(s), r = rank(s), i, d, ff, rr, t;
      if (p.type === 'p') {
        var dir = us === 'w' ? 1 : -1;
        t = s + 8 * dir;
        if (!capturesOnly && !b[t]) {
          addPawn(s, t);
          if (r === (us === 'w' ? 1 : 6) && !b[t + 8 * dir]) add(s, t + 8 * dir);
        }
        for (i = -1; i <= 1; i += 2) {
          if (!onBoard(f + i, r + dir)) continue;
          t = (r + dir) * 8 + f + i;
          if (b[t] && b[t].color !== us) addPawn(s, t);
          else if (t === pos.ep && !b[t]) add(s, t, { ep: true, captured: 'p' });
        }
      } else if (p.type === 'n' || p.type === 'k') {
        var list = p.type === 'n' ? KNIGHT : KING;
        for (i = 0; i < 8; i++) {
          d = list[i];
          if (!onBoard(f + d[0], r + d[1])) continue;
          t = (r + d[1]) * 8 + f + d[0];
          if (b[t] ? b[t].color !== us : !capturesOnly) add(s, t);
        }
      } else {
        var dirs = p.type === 'r' ? ROOK_D : p.type === 'b' ? BISHOP_D : ROOK_D.concat(BISHOP_D);
        for (i = 0; i < dirs.length; i++) {
          d = dirs[i]; ff = f + d[0]; rr = r + d[1];
          while (onBoard(ff, rr)) {
            t = rr * 8 + ff;
            if (b[t]) { if (b[t].color !== us) add(s, t); break; }
            if (!capturesOnly) add(s, t);
            ff += d[0]; rr += d[1];
          }
        }
      }
    }
    if (!capturesOnly) {
      var them = us === 'w' ? 'b' : 'w', base = us === 'w' ? 0 : 56;
      var kingSide = us === 'w' ? 'K' : 'k', queenSide = us === 'w' ? 'Q' : 'q';
      var king = b[base + 4], rook;
      if (king && king.type === 'k' && king.color === us && !isAttacked(b, base + 4, them)) {
        rook = b[base + 7];
        if (pos.castling.indexOf(kingSide) >= 0 && rook && rook.type === 'r' && rook.color === us &&
            !b[base + 5] && !b[base + 6] && !isAttacked(b, base + 5, them) && !isAttacked(b, base + 6, them)) {
          add(base + 4, base + 6, { castle: base + 7 });
        }
        rook = b[base];
        if (pos.castling.indexOf(queenSide) >= 0 && rook && rook.type === 'r' && rook.color === us &&
            !b[base + 1] && !b[base + 2] && !b[base + 3] && !isAttacked(b, base + 3, them) && !isAttacked(b, base + 2, them)) {
          add(base + 4, base + 2, { castle: base });
        }
      }
    }
    return out;
  }

  function makeMove(pos, m) {
    var b = pos.b, us = pos.turn;
    var undo = { m: m, castling: pos.castling, ep: pos.ep, toPiece: b[m.to], fromPiece: b[m.from], epPiece: null };
    b[m.to] = m.promotion ? { type: m.promotion, color: us } : b[m.from];
    b[m.from] = null;
    if (m.ep) { var cs = m.to + (us === 'w' ? -8 : 8); undo.epPiece = b[cs]; undo.epSq = cs; b[cs] = null; }
    if (m.castle !== undefined) {
      var rookTo = m.to > m.from ? m.from + 1 : m.from - 1;
      b[rookTo] = b[m.castle]; b[m.castle] = null;
    }
    pos.ep = m.piece === 'p' && Math.abs(m.to - m.from) === 16 ? (m.to + m.from) / 2 : -1;
    var c = pos.castling;
    [[4, 'KQ'], [0, 'Q'], [7, 'K'], [60, 'kq'], [56, 'q'], [63, 'k']].forEach(function (e) {
      if (m.from === e[0] || m.to === e[0]) for (var j = 0; j < e[1].length; j++) c = c.replace(e[1][j], '');
    });
    pos.castling = c;
    pos.turn = us === 'w' ? 'b' : 'w';
    return undo;
  }

  function unmakeMove(pos, u) {
    var b = pos.b, m = u.m;
    b[m.from] = u.fromPiece; b[m.to] = u.toPiece;
    if (m.ep) b[u.epSq] = u.epPiece;
    if (m.castle !== undefined) {
      var rookTo = m.to > m.from ? m.from + 1 : m.from - 1;
      b[m.castle] = b[rookTo]; b[rookTo] = null;
    }
    pos.castling = u.castling; pos.ep = u.ep;
    pos.turn = pos.turn === 'w' ? 'b' : 'w';
  }

  function legalMoves(pos, capturesOnly) {
    var us = pos.turn, them = us === 'w' ? 'b' : 'w';
    return pseudoMoves(pos, capturesOnly).filter(function (m) {
      var u = makeMove(pos, m), k = kingSquare(pos.b, us);
      var ok = k === null || !isAttacked(pos.b, k, them);
      unmakeMove(pos, u);
      return ok;
    });
  }

  function inCheck(pos) {
    var k = kingSquare(pos.b, pos.turn);
    return k !== null && isAttacked(pos.b, k, pos.turn === 'w' ? 'b' : 'w');
  }

  /* Port of python-chess is_insufficient_material. */
  function insufficient(pos) {
    var b = pos.b, cnt = { w: { all: 0 }, b: { all: 0 } }, bishopsLight = false, bishopsDark = false;
    var pawns = 0, knights = 0;
    for (var i = 0; i < 64; i++) {
      var p = b[i];
      if (!p) continue;
      cnt[p.color].all++;
      cnt[p.color][p.type] = (cnt[p.color][p.type] || 0) + 1;
      if (p.type === 'p') pawns++;
      if (p.type === 'n') knights++;
      if (p.type === 'b') { if ((file(i) + rank(i)) % 2) bishopsLight = true; else bishopsDark = true; }
    }
    function side(c) {
      var o = c === 'w' ? 'b' : 'w', mine = cnt[c], theirs = cnt[o];
      if (mine.p || mine.r || mine.q) return false;
      if (mine.n) return mine.all <= 2 && theirs.all - (theirs.k || 0) - (theirs.q || 0) === 0;
      if (mine.b) return (!bishopsLight || !bishopsDark) && !pawns && !knights;
      return true;
    }
    return side('w') && side('b');
  }

  function evalPos(pos) {
    var score = 0;
    for (var i = 0; i < 64; i++) {
      var p = pos.b[i];
      if (p) score += (p.color === pos.turn ? 1 : -1) * VALUES[p.type] * 100;
    }
    return score;
  }
  function ordered(moves) {
    return moves.map(function (m, i) {
      var key = m.captured && !m.ep ? -(VALUES[m.captured] * 10 - VALUES[m.piece]) : 0;
      return [key, i, m];
    }).sort(function (x, y) { return x[0] - y[0] || x[1] - y[1]; }).map(function (x) { return x[2]; });
  }
  function searchPos(pos, depth, alpha, beta) {
    var moves = legalMoves(pos, false);
    if (!moves.length) return inCheck(pos) ? -MATE : 0;
    if (insufficient(pos)) return 0;
    if (depth === 0) return quiesce(pos, alpha, beta, 4);
    var list = ordered(moves);
    for (var i = 0; i < list.length; i++) {
      var u = makeMove(pos, list[i]);
      var score = -searchPos(pos, depth - 1, -beta, -alpha);
      unmakeMove(pos, u);
      if (score >= beta) return score;
      alpha = Math.max(alpha, score);
    }
    return alpha;
  }
  function quiesce(pos, alpha, beta, depth) {
    var stand = evalPos(pos);
    if (stand >= beta || depth === 0) return stand;
    alpha = Math.max(alpha, stand);
    var list = ordered(legalMoves(pos, true));
    for (var i = 0; i < list.length; i++) {
      var u = makeMove(pos, list[i]);
      var score = -quiesce(pos, -beta, -alpha, depth - 1);
      unmakeMove(pos, u);
      if (score >= beta) return score;
      alpha = Math.max(alpha, score);
    }
    return alpha;
  }
  function search(chess, depth, alpha, beta) { return searchPos(posFromChess(chess), depth, alpha, beta); }

  /* ---- deeper search used by the "Like Robbie" and "Tougher" levels ----
   * Material plus piece-square tables, alpha-beta with quiescence, and a
   * time limit. The bot then picks, among the moves that search says are
   * not much worse than the best, the one Robbie's style model likes most. */
  var VAL2 = { p: 100, n: 320, b: 330, r: 500, q: 900, k: 0 };
  // Tables are written from White's side, rank 8 first (Simplified Evaluation Function).
  var PST = {
    p: [0,0,0,0,0,0,0,0, 50,50,50,50,50,50,50,50, 10,10,20,30,30,20,10,10, 5,5,10,25,25,10,5,5,
        0,0,0,20,20,0,0,0, 5,-5,-10,0,0,-10,-5,5, 5,10,10,-20,-20,10,10,5, 0,0,0,0,0,0,0,0],
    n: [-50,-40,-30,-30,-30,-30,-40,-50, -40,-20,0,0,0,0,-20,-40, -30,0,10,15,15,10,0,-30, -30,5,15,20,20,15,5,-30,
        -30,0,15,20,20,15,0,-30, -30,5,10,15,15,10,5,-30, -40,-20,0,5,5,0,-20,-40, -50,-40,-30,-30,-30,-30,-40,-50],
    b: [-20,-10,-10,-10,-10,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,10,10,5,0,-10, -10,5,5,10,10,5,5,-10,
        -10,0,10,10,10,10,0,-10, -10,10,10,10,10,10,10,-10, -10,5,0,0,0,0,5,-10, -20,-10,-10,-10,-10,-10,-10,-20],
    r: [0,0,0,0,0,0,0,0, 5,10,10,10,10,10,10,5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5,
        -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, -5,0,0,0,0,0,0,-5, 0,0,0,5,5,0,0,0],
    q: [-20,-10,-10,-5,-5,-10,-10,-20, -10,0,0,0,0,0,0,-10, -10,0,5,5,5,5,0,-10, -5,0,5,5,5,5,0,-5,
        0,0,5,5,5,5,0,-5, -10,5,5,5,5,5,0,-10, -10,0,5,0,0,0,0,-10, -20,-10,-10,-5,-5,-10,-10,-20],
    k: [-30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30, -30,-40,-40,-50,-50,-40,-40,-30,
        -20,-30,-30,-40,-40,-30,-30,-20, -10,-20,-20,-20,-20,-20,-20,-10, 20,20,0,0,0,0,20,20, 20,30,10,0,0,10,30,20],
    ke: [-50,-40,-30,-20,-20,-30,-40,-50, -30,-20,-10,0,0,-10,-20,-30, -30,-10,20,30,30,20,-10,-30, -30,-10,30,40,40,30,-10,-30,
         -30,-10,30,40,40,30,-10,-30, -30,-10,20,30,30,20,-10,-30, -30,-30,0,0,0,0,-30,-30, -50,-30,-30,-30,-30,-30,-30,-50]
  };
  function pstIndex(s, color) { return color === 'w' ? (7 - rank(s)) * 8 + file(s) : rank(s) * 8 + file(s); }

  function evalPos2(pos) {
    var b = pos.b, score = 0, nonPawn = 0, i, p;
    for (i = 0; i < 64; i++) { p = b[i]; if (p && p.type !== 'p' && p.type !== 'k') nonPawn += VAL2[p.type]; }
    var kingTable = nonPawn <= 1300 ? 'ke' : 'k';
    for (i = 0; i < 64; i++) {
      p = b[i];
      if (!p) continue;
      var v = VAL2[p.type] + PST[p.type === 'k' ? kingTable : p.type][pstIndex(i, p.color)];
      score += p.color === pos.turn ? v : -v;
    }
    return score;
  }

  function Abort() {}

  function orderMoves(moves) {
    return moves.map(function (m) {
      var key = 0;
      if (m.captured) key -= 10 * VAL2[m.captured] - VAL2[m.piece] / 10 + 1000;
      if (m.promotion) key -= VAL2[m.promotion];
      return [key, m];
    }).sort(function (x, y) { return x[0] - y[0]; }).map(function (x) { return x[1]; });
  }

  function Searcher(deadline) { this.deadline = deadline; this.nodes = 0; }
  Searcher.prototype.tick = function () {
    if ((++this.nodes & 1023) === 0 && Date.now() > this.deadline) throw new Abort();
  };
  Searcher.prototype.negamax = function (pos, depth, alpha, beta, ply) {
    this.tick();
    var moves = legalMoves(pos, false);
    if (!moves.length) return inCheck(pos) ? -(MATE - ply) : 0;
    if (insufficient(pos)) return 0;
    if (depth <= 0) return this.quiesce(pos, alpha, beta, ply, 6);
    var list = orderMoves(moves), best = -Infinity;
    for (var i = 0; i < list.length; i++) {
      var u = makeMove(pos, list[i]);
      var score = -this.negamax(pos, depth - 1, -beta, -alpha, ply + 1);
      unmakeMove(pos, u);
      if (score > best) best = score;
      if (score > alpha) alpha = score;
      if (alpha >= beta) break;
    }
    return best;
  };
  Searcher.prototype.quiesce = function (pos, alpha, beta, ply, qdepth) {
    this.tick();
    var stand = evalPos2(pos);
    if (stand >= beta || qdepth === 0) return stand;
    if (stand > alpha) alpha = stand;
    var list = orderMoves(legalMoves(pos, true));
    for (var i = 0; i < list.length; i++) {
      var u = makeMove(pos, list[i]);
      var score = -this.quiesce(pos, -beta, -alpha, ply + 1, qdepth - 1);
      unmakeMove(pos, u);
      if (score >= beta) return score;
      if (score > alpha) alpha = score;
    }
    return alpha;
  };

  /* Values (centipawns, from the mover's side) of every root move, exact for
   * moves within `margin` of the best and an upper bound for the rest.
   * Searches deeper while time allows; returns the last finished depth. */
  function rootValues(chess, order, maxDepth, margin, timeMs) {
    var pos = posFromChess(chess);
    var internal = legalMoves(pos, false);
    function find(m) {
      for (var i = 0; i < internal.length; i++) {
        var x = internal[i];
        if (squareName(x.from) === m.from && squareName(x.to) === m.to && (x.promotion || '') === (m.promotion || '')) return x;
      }
      return null;
    }
    var pairs = order.map(function (m) { return [m, find(m)]; });
    var searcher = new Searcher(Date.now() + timeMs);
    var done = null, depthDone = 0;
    for (var d = 1; d <= maxDepth; d++) {
      var vals = new Array(pairs.length), best = -Infinity;
      try {
        for (var i = 0; i < pairs.length; i++) {
          var u = makeMove(pos, pairs[i][1]);
          var floor = best === -Infinity ? -Infinity : best - margin - 1;
          var v;
          try { v = -searcher.negamax(pos, d - 1, -Infinity, -floor, 1); }
          finally { unmakeMove(pos, u); }
          vals[i] = v;
          if (v > best) best = v;
        }
      } catch (e) {
        if (e instanceof Abort) break;
        throw e;
      }
      done = vals; depthDone = d;
      // Search the most promising moves first next time (keeps style order among ties).
      var idx = pairs.map(function (_, k) { return k; }).sort(function (a, b) { return vals[b] - vals[a] || a - b; });
      pairs = idx.map(function (k) { return pairs[k]; });
      done = idx.map(function (k) { return vals[k]; });
    }
    return { moves: pairs.map(function (p) { return p[0]; }), values: done, depth: depthDone, nodes: searcher.nodes };
  }

  function squareName(s) { return 'abcdefgh'[file(s)] + (rank(s) + 1); }

  /* FEN key matching python-chess: en passant square only if a capture is legal. */
  function positionKey(chess, moves) {
    var parts = chess.fen().split(' ');
    if (parts[3] !== '-') {
      var ep = parts[3];
      var legalEp = moves.some(function (m) { return m.flags.indexOf('e') >= 0 && m.to === ep; });
      if (!legalEp) parts[3] = '-';
    }
    return parts.slice(0, 4).join(' ');
  }

  function uci(m) { return m.from + m.to + (m.promotion || ''); }

  function StyleBot(profile, opts) {
    opts = opts || {};
    this.profile = profile;
    this.weights = profile.weights;
    this.book = profile.book;
    this.temperature = opts.temperature === undefined ? 1 : opts.temperature;
    this.strength = opts.strength === undefined ? 0.5 : opts.strength;
    this.random = opts.random || Math.random;
    // A search level ({depth, margin, penalty, timeMs}) replaces the old shallow check.
    this.level = opts.level || null;
    // Optional neural move predictor (web/movenet.js) with {tc, rating} for its inputs.
    // When present it replaces the hand-made style features.
    this.net = opts.net || null;
    this.netInputs = opts.netInputs || null;
    this.netTemperature = opts.netTemperature === undefined ? 1 : opts.netTemperature;
    // Remembered moves are picked with weight count^bookSharpness, so the main
    // line is near-automatic (100 vs 8 games -> 99.9%) and only real splits vary.
    this.bookSharpness = opts.bookSharpness === undefined ? 3 : opts.bookSharpness;
  }

  /* Style scores only (no search): how much Robbie's model likes each move. */
  StyleBot.prototype.styleScores = function (chess, moves, history) {
    if (this.net) {
      var probs = this.net.moveProbs(chess, moves, history,
        { tc: this.netInputs.tc, rating: this.netInputs.rating, robbie: true });
      return probs.map(function (p) { return Math.log(Math.max(p, 1e-9)); });
    }
    var b = board64(chess), phase = gamePhase(chess, b), w = this.weights;
    return moves.map(function (m) {
      var f = moveFeatures(chess, b, m, phase, history), s = 0;
      for (var k in f) s += w[k] * f[k];
      return s;
    });
  };

  /* Search every move (best first, by Robbie's taste), then return the
   * moves within `margin` centipawns of the best with their values. */
  StyleBot.prototype.searchMoves = function (chess, moves, style) {
    var lvl = this.level;
    var order = moves.map(function (m, i) { return [style[i], i]; })
      .sort(function (a, b) { return b[0] - a[0]; }).map(function (x) { return moves[x[1]]; });
    var res = rootValues(chess, order, lvl.depth, lvl.margin, lvl.timeMs);
    if (!res.values) return null;
    var best = Math.max.apply(null, res.values), ok = [];
    res.moves.forEach(function (m, i) { if (res.values[i] >= best - lvl.margin) ok.push({ move: m, value: res.values[i] }); });
    this.lastSearch = { depth: res.depth, nodes: res.nodes, candidates: ok.length, best: best };
    return { best: best, ok: ok };
  };

  StyleBot.prototype.scores = function (chess, moves, history) {
    var b = board64(chess), phase = gamePhase(chess, b), w = this.weights, self = this;
    return moves.map(function (m) {
      var f = moveFeatures(chess, b, m, phase, history), s = 0;
      for (var k in f) s += w[k] * f[k];
      if (self.strength > 0) {
        chess.move(m);
        var value = chess.in_checkmate() ? MATE : -search(chess, 1, -MATE, MATE);
        chess.undo();
        s += self.strength * Math.max(-20, Math.min(20, value / 100));
        if (value >= MATE) s += 1000;
      }
      return s;
    });
  };

  StyleBot.prototype.sample = function (items, scores) {
    if (this.temperature <= 0) return items[scores.indexOf(Math.max.apply(null, scores))];
    var m = Math.max.apply(null, scores), t = this.temperature;
    var ws = scores.map(function (s) { return Math.exp((s - m) / t); });
    var total = ws.reduce(function (x, y) { return x + y; }, 0), r = this.random() * total;
    for (var i = 0; i < items.length; i++) { r -= ws[i]; if (r <= 0) return items[i]; }
    return items[items.length - 1];
  };

  /* Returns {move, source: 'book'|'model', count, total}. */
  StyleBot.prototype.choose = function (chess, history) {
    var moves = chess.moves({ verbose: true });
    if (!moves.length) return null;
    var self = this, lvl = this.level;
    var style = lvl || this.net ? this.styleScores(chess, moves, history) : null;
    var found = lvl ? this.searchMoves(chess, moves, style) : null;
    var safe = null;
    if (found) {
      safe = {};
      found.ok.forEach(function (c) { safe[uci(c.move)] = c.value; });
    }
    var entry = this.book[positionKey(chess, moves)];
    if (entry) {
      var counts = {}, total = 0;
      entry.split(' ').forEach(function (pair) {
        var p = pair.split(':'); counts[p[0]] = parseInt(p[1], 10);
      });
      // With search on, skip remembered moves that search says lose material.
      var opts = moves.filter(function (m) { return counts[uci(m)] && (!safe || uci(m) in safe); });
      if (opts.length) {
        opts.forEach(function (m) { total += counts[uci(m)]; });
        var sharp = this.bookSharpness;
        var pick = this.sample(opts, opts.map(function (m) { return sharp * Math.log(counts[uci(m)]); }));
        return { move: pick, source: 'book', count: counts[uci(pick)], total: total };
      }
    }
    if (found) {
      // Style decides among safe moves; each pawn given up costs `penalty` logits.
      var cands = found.ok.map(function (c) { return c.move; });
      var scores = found.ok.map(function (c) {
        return style[moves.indexOf(c.move)] + lvl.penalty * (c.value - found.best) / 100;
      });
      var chosen = self.sample(cands, scores);
      return { move: chosen, source: 'model', prob: this.net ? Math.exp(style[moves.indexOf(chosen)]) : undefined };
    }
    if (this.net) {
      var t = this.netTemperature;
      var pickNet = t <= 0 ? moves[style.indexOf(Math.max.apply(null, style))]
        : this.sample(moves, style.map(function (v) { return v / t; }));
      return { move: pickNet, source: 'model', prob: Math.exp(style[moves.indexOf(pickNet)]) };
    }
    return { move: this.sample(moves, this.scores(chess, moves, history)), source: 'model' };
  };

  var api = { rootValues: rootValues, evalPos2: evalPos2, StyleBot: StyleBot, moveFeatures: moveFeatures, board64: board64, gamePhase: gamePhase,
              positionKey: positionKey, search: search, uci: uci, MATE: MATE };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.ChessStyleBot = api;
})(this);
