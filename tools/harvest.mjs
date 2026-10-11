// Two-tier harvest of Zenith (state -> action) pairs for training data.
//
// Zenith replays omit most of the garbage a player receives: the tower's environmental
// garbage is generated client-side and is absent from the file (measurably so --
// stats.garbage.received runs 3-8x the total recorded in interaction events, where
// League always runs BELOW it because of cancellation). So a full reconstruction
// desyncs the moment garbage first lands.
//
// While NOTHING has entered the board, though, the sim is driven purely by the piece
// sequence and the inputs, both of which are exact. The only question is where that
// window closes, and it can be bracketed from both sides:
//
//   strict   -- recorded garbage inserted. Our cancellation is far weaker than a pro's,
//               so the first insertion fires early. Pessimistic: certainly clean.
//   extended -- garbage suppressed entirely (everything cancelled). Optimistic, so the
//               point where the stack degenerates estimates where reality's garbage
//               actually began landing. Cut one placement before that.
//
// Truth lies between; every sample is tagged so either tier can be used alone.
import { readFileSync, writeFileSync } from 'node:fs';
// tetrp, vendored under vendor/tetrp -- a third-party reimplementation of TETR.IO's
// engine. It is what turns a replay's keypresses into board states; the .ttr files
// themselves contain no board data at all. See docs/tetrp.md.
import { parseReplay, prepareReplay, Reconstruction } from '../vendor/tetrp/src/replay/index.js';
import * as B from '../vendor/tetrp/src/board.js';

// Cut thresholds. Height alone is a poor signal: a pro upstacking deliberately is
// tall and CLEAN, while a reconstruction that has lost sync is hole-rich because the
// inputs are aimed at rows that do not exist on our board. Sealed holes separate the
// two -- and they must be SEALED, not merely covered, because an overhang is what an
// all-spin player builds on purpose.
//
// Neither threshold can be calibrated: the recorded garbage never exceeds these
// players' attack output at any moment, so nothing in the file says when the
// unrecorded environmental garbage actually landed. Both are judgement, so each
// sample also carries its own `sealed` and `height` for re-cutting later.
const DANGER = 17;
const SEALED_MAX = 8;
const rowStr = row => row.map(c => c == null ? '.' : (c === 'gb' ? 'G' : c)).join('');
const height = b => 40 - b.rows.findIndex(row => row.some(c => c != null));

/** Empty cells with no 4-connected path to open air. */
function sealed(board) {
  const H = board.rows.length, W = board.width;
  const open = new Uint8Array(H * W), stack = [];
  for (let x = 0; x < W; x++) if (board.rows[0][x] == null) { open[x] = 1; stack.push([x, 0]); }
  while (stack.length) {
    const [x, y] = stack.pop();
    for (const [dx, dy] of [[1,0],[-1,0],[0,1],[0,-1]]) {
      const nx = x + dx, ny = y + dy;
      if (nx < 0 || nx >= W || ny < 0 || ny >= H) continue;
      if (open[ny * W + nx] || board.rows[ny][nx] != null) continue;
      open[ny * W + nx] = 1; stack.push([nx, ny]);
    }
  }
  let n = 0;
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++)
    if (board.rows[y][x] == null && !open[y * W + x]) n++;
  return n;
}

function run(tl, suppress, plan = []) {
  const r = new Reconstruction(tl), e = r.engine;
  const pending = plan.slice().sort((x, y) => x.after - y.after);
  let injected = 0;
  // The incoming-garbage queue, as a player sees it beside the board: it rises with
  // each recorded packet and falls by the attack that placement produced. Surplus
  // attack is NOT banked -- it goes up the tower and is gone.
  //
  // There is deliberately no deadline here. An earlier version assumed a packet
  // entered the board 40 frames after arriving, and cut the run at the first one to
  // expire; measured against real play that fired about five times too early, because
  // packets survive a median of 72 frames and are cancelled anyway. Worse, in 30 of
  // 50 runs the recorded queue was completely EMPTY at the moment the board actually
  // broke, so recorded garbage cannot be what breaks it.
  //
  // What the queue is good for is the question "does it still reach zero". A strong
  // player drains it hundreds of times a run; when it stops draining they have been
  // overwhelmed, and that moment tracks the board's own verdict at r = 0.989. Both
  // are symptoms of the same event rather than one causing the other.
  const arrivals = tl.events.filter(ev => ev.type === 'receive')
    .map(ev => ({frame: ev.frame, amt: ev.data.amt}));
  let ai = 0, attackSeen = 0, lastZero = -1;
  const queue = [];
  // Keep the real insertGarbage for OUR injections, and silence only tetrp's own
  // automatic tanking. Replacing insertGarbage outright would swallow them too.
  const realInsert = e.insertGarbage.bind(e);
  if (suppress) { e.takeDamage = () => {}; e.continuousGarbage = () => {}; }
  const samples = [];
  let open = null, placed = null, atLock = null, firstTank = null, dangerAt = null;
  let holeRun = 0;

  const origLock = e.lock.bind(e);
  e.lock = function () {
    const p = e.state.piece;
    // board.commit resolves the fractional y with ceil, not round -- a soft-dropped
    // piece can sit well below .5 and would otherwise land a row too high.
    if (p && !p.sleeping)
      // Absolute cells as the engine itself resolves them. (x, y, r) here is
      // pivot-relative and rotation-table specific; a consumer with its own SRS
      // convention would have to re-derive it, so ship the resolved cells too.
      placed = {type: p.type, x: p.x, y: Math.ceil(p.y), r: p.r,
                spin: p.spin, kick: p.kick, soft: p.softDropped,
                cells: B.cells(p).map(([cx, cy]) => [cx, Math.ceil(cy)])};
    // Queue and hold AS THEY STAND WHEN THE PIECE LANDS. The sample's own queue/hold
    // are the state at spawn, which is what a learner needs (state -> action), but a
    // hold swap happens between the two: the held piece has moved into the hold slot
    // and, if that slot was empty, the swap consumed a queue entry. Anything drawing
    // the placed piece next to the spawn-time queue shows two different instants.
    if (p && !p.sleeping)
      atLock = {queue: e.state.bag.queue.slice(0, 6), hold: e.state.hold.piece,
                hold_locked: !!e.state.hold.locked};
    return origLock();
  };

  // Snapshot at the spawn boundary, for the same reason as the lock wrapper: the
  // trace is replayed AFTER the frame has finished, so reading live state then can
  // pick up a hold that happened later in the same advance and record the swapped-in
  // piece as the one that spawned.
  const snapshot = () => {
    const s = e.state;
    return {frame: s.frame, board: s.board.rows.map(rowStr),
            piece: s.piece ? s.piece.type : null,
            queue: s.bag.queue.slice(0, 6), hold: s.hold.piece, holds: 0};
  };
  let pendingSpawn = null;
  const origSpawn = e.spawn.bind(e);
  e.spawn = function (...args) {
    const out = origSpawn(...args);
    pendingSpawn = snapshot();
    return out;
  };
  // The engine spawns its first piece in the constructor, before the wrapper existed.
  open = snapshot();

  while (r.advance()) {
    const s = e.state;
    while (ai < arrivals.length && arrivals[ai].frame <= s.frame)
      queue.push({amt: arrivals[ai++].amt});
    const produced = s.attack ? s.attack.totals.generated : 0;
    let budget = produced - attackSeen;
    attackSeen = produced;
    while (budget > 0 && queue.length) {          // oldest packet cancels first
      const take = Math.min(budget, queue[0].amt);
      queue[0].amt -= take; budget -= take;
      if (queue[0].amt === 0) queue.shift();
    }
    // advance() drains engine.trace into r.transitions, so read that.
    for (const t of r.transitions) {
      if (t.type === 'hold' && open) open.holds++;
      else if (t.type === 'spawn') {
        if (!open) { open = pendingSpawn || snapshot(); pendingSpawn = null; }
        else open.piece ??= t.piece;
      }
      else if (t.type === 'lock' && open && placed) {
        open.sealed = sealed(e.state.board);
        open.height = Math.max(0, height(e.state.board));
        // Consecutive placements ending here on which the board held a sealed hole.
        // A hole covered on purpose and reopened resets this; one that stays does not.
        holeRun = open.sealed > 0 ? holeRun + 1 : 0;
        open.hole_run = holeRun;
        open.q_depth = queue.reduce((n, p) => n + p.amt, 0);
        if (open.q_depth === 0) lastZero = samples.length;
        open.recovered = injected > 0;
        open.garbage_rows = injected;
        samples.push({n: t.placementIndex - 1, frame: open.frame, board: open.board,
          piece: open.piece ?? placed.type, queue: open.queue, hold: open.hold,
          used_hold: open.holds % 2 === 1, holds: open.holds,
          place: placed, cleared: 0, cleared_rows: [], spin: t.spin,
          lock_queue: atLock ? atLock.queue : open.queue,
          lock_hold: atLock ? atLock.hold : open.hold,
          lock_hold_locked: atLock ? atLock.hold_locked : false,
          sealed: open.sealed, height: open.height, hole_run: open.hole_run,
          q_depth: open.q_depth, recovered: open.recovered,
          garbage_rows: open.garbage_rows});
        // Inferred garbage lands immediately after the placement it was attributed to.

        open = null; placed = null; atLock = null;
      } else if (t.type === 'remove-lines' && samples.length) {
        const last = samples[samples.length - 1];
        last.cleared = t.rows.length;
        // Row indices into THIS sample's board (piece added, before removal).
        last.cleared_rows = t.rows.slice();
      }
    }
    // Inferred garbage is applied at a FRAME boundary, once advance() has completed,
    // which is where tools/recover.mjs injected it when it measured the improvement.
    // Injecting mid-advance -- between the lock and the next spawn -- lands the rows
    // against a different engine state and does not reproduce that result.
    // `after: k` means "after k placements", and samples.length is exactly that count.
    let addedNow = 0;
    while (pending.length && pending[0].after <= samples.length) {
      const p = pending.shift();
      // `shape` gives a hole column per row, so cheese replays correctly; older
      // plans without it are a clean block of `amt` rows at `col`.
      const cols = p.shape ?? new Array(p.amt).fill(p.col);
      for (const c of cols) { if (!realInsert(c)) break; injected++; addedNow++; }
    }
    // Board snapshots are taken inside e.spawn, which runs DURING advance(), while the
    // injection runs after it returns -- so without this the snapshot for the next
    // placement predates its own garbage, and the piece after that clears it before
    // any snapshot sees it. The rows were always on the board (the run survives
    // hundreds of placements longer); they were simply invisible in the dataset.
    if (addedNow) {
      const now = e.state.board.rows.map(rowStr);
      if (open) open.board = now;
      if (pendingSpawn) pendingSpawn.board = now;
    }
    if (!open && s.piece && !s.piece.sleeping) { open = pendingSpawn || snapshot(); pendingSpawn = null; }
    // No cut is applied here any more. Two independent candidate signals -- the first
    // garbage the sim inserts, and the onset of persistent holes -- correlate at
    // r = 0.07 across these runs, so neither localises where the reconstruction
    // diverges, and a threshold picked from either would be arbitrary. Every sample
    // carries its own evidence instead, and the run ends where it ends.
    if (false) dangerAt = samples.length;
    if (!suppress && firstTank === null && s.attack && s.attack.totals.tanked > 0) firstTank = samples.length;
    if (!s.playing) break;
  }
  // Stamped after the fact: a sample cannot know whether the queue drains again.
  for (const s of samples) s.q_last_zero = lastZero;
  return {samples, firstTank, dangerAt, total: samples.length};
}

// Optional: inferred garbage from tools/recover.mjs. Without it the harvest suppresses
// garbage entirely (the historical behaviour); with it, the rows the search recovered
// are replayed, so the board carries real garbage and the clean window extends.
const PLAN = new Map();
if (process.env.INJECTIONS) {
  for (const line of readFileSync(process.env.INJECTIONS, 'utf8').trim().split(/\r?\n/)) {
    if (!line) continue;
    const r = JSON.parse(line);
    PLAN.set(r.replay, r.injections ?? []);
  }
}

const out = [], rows = [];
let broken = 0;
for (const path of process.argv.slice(3)) {
  const file = path.split(/[\/]/).pop().replace(/\.ttr$/, '');
  // {user}_{mods}__{id}. The mods token is generic rather than a closed list: Zenith mods
  // combine, and a run can be allspin+volatile+expert. Underscores are the field
  // separator, so tools/fetch_replays.py joins multiple mods with '-' and strips the
  // underscore out of names like allspin_reversed -> allspinrev. Verified against all
  // existing files: every one parses to the same (user, mods, id) as the old closed list.
  const m = /^(.+?)_+([a-z0-9]+(?:-[a-z0-9]+)*)_+([0-9a-f]+)$/.exec(file) || [];
  let tl;
  try { tl = prepareReplay(parseReplay(readFileSync(path, 'utf8')).rounds[0].players[0]); }
  catch (err) { rows.push({file, error: (err.code || err.name) + ': ' + err.message}); continue; }
  const plan = PLAN.get(file) ?? [];
  const strict = run(tl, false), ext = run(tl, true, plan);
  const extCut0 = Math.max(0, (ext.dangerAt === null ? ext.total : ext.dangerAt) - 1);
  // Continuity cut. The piece that spawns next must be the head of the queue as it
  // stood when the previous piece locked. An initial hold applied inside spawn()
  // breaks that: the engine's own spawn event already reports the swapped-in piece,
  // so the piece that was really drawn is not observable and the sample cannot be
  // vouched for. Rather than ship it, the run ends there.
  let extCut = extCut0;
  for (let i = 0; i + 1 < ext.samples.length && ext.samples[i + 1].n < extCut0; i++) {
    if (ext.samples[i + 1].piece !== ext.samples[i].lock_queue[0]) {
      extCut = ext.samples[i + 1].n;
      broken++;
      break;
    }
  }
  const strictCut = Math.min(strict.firstTank === null ? strict.total : strict.firstTank, extCut);
  for (const s of ext.samples) {
    if (s.n >= extCut) break;
    out.push({replay: file, user: m[1] || file, mods: m[2] || '?',
              tier: s.n < strictCut ? 'strict' : 'extended', ...s});
  }
  rows.push({file, strict: strictCut, extended: extCut});
}
writeFileSync(process.argv[2], out.map(o => JSON.stringify(o)).join('\n') + '\n');

const skipped = rows.filter(r => r.error);
const good = rows.filter(r => !r.error);
let a = 0, b = 0;
for (const r of good) { a += r.strict; b += r.extended; }
const byUser = {};
for (const o of out) (byUser[o.user] ??= {strict: 0, extended: 0})[o.tier]++;
console.log(['user'.padEnd(12), 'strict'.padStart(8), 'extended'.padStart(9), 'total'.padStart(8)].join(''));
console.log('-'.repeat(38));
for (const [u, v] of Object.entries(byUser).sort((x, y) => (y[1].strict + y[1].extended) - (x[1].strict + x[1].extended)))
  console.log(u.padEnd(12) + String(v.strict).padStart(8) + String(v.extended).padStart(9) + String(v.strict + v.extended).padStart(8));
console.log('-'.repeat(38));
console.log('TOTAL'.padEnd(12) + String(a).padStart(8) + String(b - a).padStart(9) + String(b).padStart(8));
console.log('\n' + out.length + ' samples from ' + good.length + ' replays -> ' + process.argv[2]);
for (const k of skipped) console.log('  SKIPPED ' + k.file + ' -- ' + k.error);
if (broken) console.log('  ' + broken + ' run(s) truncated at a spawn-continuity break');
