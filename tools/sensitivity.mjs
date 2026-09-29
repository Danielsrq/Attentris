// How wrong does ONE misplaced garbage row make the reconstruction?
//
//   node tools/sensitivity.mjs data/replays/<run>.ttr
//
// Garbage is not a local edit: pushing a row shifts the entire stack up by one, so
// every later piece lands a row higher than it did for the player. If a single
// spurious row is enough to break the reconstruction, then partial garbage is strictly
// worse than none, and reconstructing it is all-or-nothing -- which is the difference
// between "we have 57% of the rows" being useful and being useless.
//
// The test injects exactly one row at a chosen placement into an otherwise clean run,
// and reports where the board stops recovering compared with not injecting at all.
import { readFileSync } from 'node:fs';
import { parseReplay, prepareReplay, Reconstruction } from '../vendor/tetrp/src/replay/index.js';

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

/** Injects one row with its hole at `hole` after placement `at`; -1 injects nothing. */
function trial(tl, at, hole) {
  const r = new Reconstruction(tl), e = r.engine;
  e.takeDamage = () => {};
  e.continuousGarbage = () => {};
  let pieces = 0, done = false;
  const seq = [];
  while (r.advance()) {
    const s = e.state;
    let locked = false;
    for (const t of r.transitions) if (t.type === 'lock') locked = true;
    if (locked) {
      pieces = s.stats.pieces;
      if (!done && at >= 0 && pieces >= at) { e.insertGarbage(hole); done = true; }
      seq.push(sealed(s.board));
    }
    if (!s.playing) break;
  }
  let i = seq.length - 1;
  while (i >= 0 && seq[i] > 0) i--;
  return {cut: i + 1, pieces};
}

const path = process.argv[2];
const tl = prepareReplay(parseReplay(readFileSync(path, 'utf8')).rounds[0].players[0]);
const base = trial(tl, -1, 0);
console.log(path.split(/[\\/]/).pop());
console.log('  baseline (no garbage at all): cut %d, survived %d placements',
  base.cut, base.pieces);
console.log();
console.log(['inject 1 row at'.padEnd(18), 'cut'.padStart(6), 'survived'.padStart(10),
  'cut lost'.padStart(10)].join(''));
console.log('-'.repeat(46));
for (const at of [25, 50, 100, 150, 200, 300, 400]) {
  if (at >= base.pieces) break;
  const t = trial(tl, at, 4);
  console.log(String(at).padEnd(18) + String(t.cut).padStart(6) +
    String(t.pieces).padStart(10) + String(base.cut - t.cut).padStart(10));
}
console.log();
console.log('  "cut lost" is how many placements of usable data one single wrong row costs.');
