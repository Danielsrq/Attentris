// Attempts to RECOVER the missing Zenith garbage by search.
//
//   node tools/recover.mjs data/replays/<run>.ttr [maxPlacements]
//
// The tower's garbage is not in the replay, so it cannot be read -- but it may be
// inferable. Two facts make a search viable:
//
//   1. A wrong or missing row breaks the board AT that placement (tools/sensitivity),
//      so the fitness signal is immediate and local rather than delayed.
//   2. The player's inputs were aimed at the true board. Inject the right garbage and
//      their pieces keep landing flush; inject anything else and holes appear at once.
//
// So: run forward until a sealed hole appears and persists. That is the engine telling
// us a row is missing just behind us. Back up, try every (placement, amount, column)
// in a small window, keep whichever survives longest, and continue. tetrp's
// checkpoint/restore makes the backtracking cheap.
//
// If the search can push the clean window materially past the no-injection baseline,
// the garbage is recoverable and the dataset grows. If it cannot, the failure is
// informative: it means the board does not constrain the garbage enough to identify
// it, and no amount of cleverness with these replays will.
import { readFileSync } from 'node:fs';
import { parseReplay, prepareReplay, Reconstruction } from '../vendor/tetrp/src/replay/index.js';

const MAX_AMT = Number(process.env.MAX_AMT ?? 4);   // tallest garbage event considered
const AMOUNTS = Array.from({length: MAX_AMT}, (_, i) => i + 1);
const COLUMNS = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9];

// Candidate shapes for one garbage event, as a column per row in insertion order.
//
// CLEAN: every row shares a hole column. This is the overwhelmingly likely shape,
// because messiness_change = 0.1 means the hole stays put between rows 90% of the
// time. Tried first, and it is what has been accepted in every recovery so far.
//
// CHEESE: the hole moves once and then stays. messiness_nosame = true means it moves
// to a DIFFERENT column, so a single transition at row k to any other column covers
// nearly all of the remaining probability mass. Enumerating a column per row instead
// would be 10^N -- 10,000 shapes for four rows -- where this is 9(N-1) per start
// column, 280 for four rows, and captures the same cases that actually occur.
function cleanShapes(amt) {
  return COLUMNS.map(c => new Array(amt).fill(c));
}
function cheeseShapes(amt) {
  if (amt < 2) return [];
  const out = [];
  for (const c of COLUMNS)
    for (let k = 1; k < amt; k++)
      for (const c2 of COLUMNS) {
        if (c2 === c) continue;
        const p = new Array(amt).fill(c);
        for (let j = k; j < amt; j++) p[j] = c2;
        out.push(p);
      }
  return out;
}
const BACKTRACK = Number(process.env.BACKTRACK ?? 8);  // look-back window on a break
const CONFIRM = Number(process.env.CONFIRM ?? 6);      // holes surviving this = damage
const MIN_GAIN = Number(process.env.MIN_GAIN ?? 2);    // reject gains at or below this

function sealed(board) {
  const H = board.rows.length, W = board.width;
  const open = new Uint8Array(H * W), st = [];
  for (let x = 0; x < W; x++) if (board.rows[0][x] == null) { open[x] = 1; st.push([x, 0]); }
  while (st.length) {
    const [x, y] = st.pop();
    for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
      if (open[ny*W+nx] || board.rows[ny][nx] != null) continue;
      open[ny*W+nx] = 1; st.push([nx, ny]);
    }
  }
  let n = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    if (board.rows[y][x] == null && !open[y*W+x]) n++;
  return n;
}

const manual = e => { e.takeDamage = () => {}; e.continuousGarbage = () => {}; return e; };

/** Steps one placement. Returns null at end of run. */
function step(r) {
  const before = r.engine.state.stats.pieces;
  while (r.advance()) {
    const s = r.engine.state;
    if (s.stats.pieces > before) return {pieces: s.stats.pieces, sealed: sealed(s.board), alive: s.playing};
    if (!s.playing) return {pieces: s.stats.pieces, sealed: sealed(s.board), alive: false};
  }
  return null;
}

/** Runs to the end from a checkpoint and returns the recovery cut, measured exactly as
 *  the rest of the pipeline does: walk back from the final board to the last clean one.
 *  A donation is covered and reopened, so its episode ends; a missing garbage row
 *  starts an episode that never closes. Scoring on "first hole that sticks" instead
 *  makes the search chase donations, which is what an earlier version of this did. */
function reachFrom(cp, shape, limit) {
  const r = Reconstruction.restore(cp);
  manual(r.engine);
  if (shape) for (const col of shape) r.engine.insertGarbage(col);
  const seq = [];
  for (;;) {
    const t = step(r);
    if (!t || !t.alive) break;
    seq.push(t.sealed);
    if (seq.length >= limit) break;
  }
  let i = seq.length - 1;
  while (i >= 0 && seq[i] > 0) i--;
  return i + 1;                       // placements of clean run from this checkpoint
}

const path = process.argv[2];
const CAP = Number(process.argv[3] ?? 400);
const tl = prepareReplay(parseReplay(readFileSync(path, 'utf8')).rounds[0].players[0]);

// Baseline: never inject, which is what the shipped dataset does.
let r = new Reconstruction(tl); manual(r.engine);
const cps = [r.checkpoint()];
let base = 0;
const seq0 = [];
for (;;) {
  const t = step(r);
  if (!t || !t.alive) break;
  base++; cps.push(r.checkpoint()); seq0.push(t.sealed);
  if (base >= CAP) break;
}
let bi = seq0.length - 1;
while (bi >= 0 && seq0[bi] > 0) bi--;
const baseCut = bi + 1;

console.log(path.split(/[\\/]/).pop());
console.log('  baseline: clean for %d placements (run reaches %d)', baseCut, base);
const nClean = AMOUNTS.reduce((n, a) => n + cleanShapes(a).length, 0);
const nCheese = AMOUNTS.reduce((n, a) => n + cheeseShapes(a).length, 0);
console.log('  per break: %d clean shapes then %d cheese shapes, x%d positions',
  nClean, nCheese, BACKTRACK);

let at = baseCut, reach = base, accepted = [], guard = 0, tried = 0;
while (at < reach && at < CAP && guard++ < 40) {
  let best = {gain: 0, shape: null, from: null};
  // Phase 1 is clean blocks, phase 2 cheese. Cheese is 14x the work, so it is only
  // reached when no clean block helps -- which is the case a search that could not
  // express cheese would simply report as "nothing helped".
  for (const phase of ['clean', 'cheese']) {
    for (let back = 1; back <= BACKTRACK; back++) {
      const from = at - back;
      if (from < 0 || !cps[from]) continue;
      const without = reachFrom(cps[from], null, CAP - from);
      for (const amt of AMOUNTS) {
        for (const shape of (phase === 'clean' ? cleanShapes(amt) : cheeseShapes(amt))) {
          tried++;
          const with_ = reachFrom(cps[from], shape, CAP - from);
          if (with_ - without > best.gain)
            best = {gain: with_ - without, shape, from, reach: from + with_, phase};
        }
      }
    }
    if (best.shape) break;
  }
  if (!best.shape || best.gain <= MIN_GAIN) {
    console.log('  stopped at placement ' + at + ': best candidate gained only ' + best.gain +
      (best.shape ? ' (' + best.shape.join(',') + ')' : ' -- nothing helped at all'));
    break;
  }
  const uniform = best.shape.every(c => c === best.shape[0]);
  accepted.push({at: best.from, amt: best.shape.length, col: best.shape[0],
                 shape: best.shape, cheese: !uniform, gain: best.gain});
  console.log('  + %s: %d row(s) [%s] after placement %d  -> clean window %d -> %d',
    uniform ? 'clean ' : 'CHEESE', best.shape.length, best.shape.join(','),
    best.from, at, best.reach);
  // rebuild checkpoints past the accepted injection
  const rr = Reconstruction.restore(cps[best.from]); manual(rr.engine);
  for (const col of best.shape) rr.engine.insertGarbage(col);
  let p = best.from;
  cps.length = best.from + 1;
  for (;;) {
    const t = step(rr);
    if (!t || !t.alive) break;
    p++; cps[p] = rr.checkpoint();
    if (p >= CAP) break;
  }
  reach = p;                    // the injected run survives further than the baseline
  at = best.reach;
}
// Machine-readable, so tools/harvest.mjs can replay the inferred garbage.
if (process.env.OUT) {
  const fs = await import('node:fs');
  fs.appendFileSync(process.env.OUT, JSON.stringify({
    replay: path.split(/[\\/]/).pop().replace(/\.ttr$/, ''),
    baseline: baseCut, recovered: at, survives: reach,
    injections: accepted.map(a => ({after: a.at, amt: a.amt, col: a.col,
                                    shape: a.shape, cheese: a.cheese})),
  }) + '\n');
}
console.log('\n  candidates evaluated: %d', tried);
console.log('  accepted injections : %d', accepted.length);
console.log('  clean window: ' + baseCut + ' -> ' + at + ' placements  (+' + (at - baseCut) + ')');
console.log('  run now survives ' + reach + ' placements (was ' + base + ' with no garbage)');
