// Attempts to reconstruct Zenith garbage, and measures whether it helps.
//
//   node tools/inject.mjs data/replays/<run>.ttr [W ...]
//
// THE MODEL. A recorded packet arrives at frame F carrying `amt` lines with their hole
// in column `x`. A timer W runs. Once it has expired the packet enters the board at the
// next PLACEMENT, not the next frame -- so the player always gets at least one more
// piece after the timer runs out, and if the attack from that piece covers the packet
// they are safe. That grace move is why packets are observed surviving far longer than
// any fixed frame deadline would allow.
//
// THE TEST. Injecting garbage changes where every later piece lands, so this cannot be
// evaluated by post-processing a dataset -- the inputs must be re-simulated, which is
// what tetrp is for. The measure is where the board stops recovering from its holes:
// a correct injection makes our board match the one the player could see, so their
// inputs stop producing holes and the cut moves LATER. A wrong one makes it earlier.
//
// The baseline is W = Infinity, i.e. never inject, which is what the shipped dataset
// does. Any W that beats that baseline is reconstructing something real.
import { readFileSync } from 'node:fs';
import { parseReplay, prepareReplay, Reconstruction } from '../vendor/tetrp/src/replay/index.js';

/** Empty cells with no 4-connected path to open air. */
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

function arrivalsOf(path) {
  const doc = JSON.parse(readFileSync(path, 'utf8'));
  return doc.replay.events
    .filter(e => e.type === 'ige' && e.data?.type === 'interaction'
                 && e.data.data?.type === 'garbage')
    .map(e => ({frame: e.frame, amt: e.data.data.amt, x: e.data.data.x}))
    .sort((a, b) => a.frame - b.frame);
}

function trial(tl, arrivals, W) {
  const r = new Reconstruction(tl), e = r.engine;
  // Drive insertion entirely from here: tetrp's own tanking would apply the 1v1
  // schedule on top of ours.
  e.takeDamage = () => {};
  e.continuousGarbage = () => {};

  const queue = [];
  let ai = 0, attackSeen = 0, pieces = 0, inserted = 0, seq = [];
  while (r.advance()) {
    const s = e.state;
    while (ai < arrivals.length && arrivals[ai].frame <= s.frame) queue.push({...arrivals[ai++]});

    let locked = false;
    for (const t of r.transitions) if (t.type === 'lock') locked = true;
    if (locked) {
      pieces = s.stats.pieces;
      // 1. attack from THIS placement cancels what is waiting, oldest first
      const produced = s.attack ? s.attack.totals.generated : 0;
      let budget = produced - attackSeen; attackSeen = produced;
      while (budget > 0 && queue.length) {
        const take = Math.min(budget, queue[0].amt);
        queue[0].amt -= take; budget -= take;
        if (queue[0].amt === 0) queue.shift();
      }
      // 2. whatever has outlived the timer enters now -- the grace move has passed
      while (queue.length && Number.isFinite(W) && queue[0].frame + W <= s.frame) {
        const p = queue.shift();
        const hole = Number.isInteger(p.x) && p.x >= 0 ? p.x : 0;
        for (let i = 0; i < p.amt; i++) { if (!e.insertGarbage(hole)) break; inserted++; }
      }
      seq.push(sealed(s.board));
    }
    if (!s.playing) break;
  }
  let i = seq.length - 1;
  while (i >= 0 && seq[i] > 0) i--;
  return {cut: i + 1, pieces, inserted, reason: e.state.reason};
}

const [path, ...rest] = process.argv.slice(2);
const Ws = rest.length ? rest.map(Number) : [0, 30, 60, 120, 240, 480, 960, Infinity];
const doc = parseReplay(readFileSync(path, 'utf8'));
const tl = prepareReplay(doc.rounds[0].players[0]);
const arrivals = arrivalsOf(path);
const stats = doc.rounds[0].players[0].stream?.results?.stats
  ?? JSON.parse(readFileSync(path, 'utf8')).replay.results.stats;

console.log(path.split(/[\\/]/).pop());
console.log('  real run: %d placements, %d garbage lines received; recorded packets: %d lines',
  stats.piecesplaced, stats.garbage?.received ?? 0,
  arrivals.reduce((n, a) => n + a.amt, 0));
console.log();
console.log(['W (frames)'.padEnd(12), 'cut'.padStart(6), 'survived'.padStart(10),
  'injected'.padStart(10), '  ended'].join(''));
console.log('-'.repeat(52));
let base = null;
for (const W of Ws) {
  const t = trial(tl, arrivals, W);
  if (!Number.isFinite(W)) base = t.cut;
  const tag = Number.isFinite(W) ? String(W) : 'never (baseline)';
  console.log(tag.padEnd(12) + String(t.cut).padStart(6) + String(t.pieces).padStart(10) +
    String(t.inserted).padStart(10) + '  ' + (t.reason ?? '-'));
}
if (base !== null) console.log('\n  baseline cut (no garbage) = %d; a better W pushes it higher.', base);
