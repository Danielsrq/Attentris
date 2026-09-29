// Independent geometric check of the harvested dataset, using the engine's own cell
// geometry rather than a re-derived SRS table: drop the recorded piece onto the
// recorded board and confirm it fits and explains exactly the recorded line clears.
import { readFileSync } from 'node:fs';
import * as B from '../vendor/tetrp/src/board.js';
const lines = readFileSync(process.argv[2], 'utf8').trim().split('\n');
let ok = 0, oob = 0, wrong = 0; const ex = [];
for (const line of lines) {
  const r = JSON.parse(line), p = r.place;
  const grid = r.board.map(s => s.split(''));
  const cells = B.cells({type: p.type, x: p.x, y: p.y, r: p.r});
  let fits = true;
  for (const c of cells) {
    const x = c.x ?? c[0], y = Math.ceil(c.y ?? c[1]);
    if (!(x >= 0 && x < 10 && y >= 0 && y < 40) || grid[y][x] !== '.') { fits = false; break; }
  }
  if (!fits) { oob++; if (ex.length < 3) ex.push(['nofit', r.replay, r.n, p, JSON.stringify(cells)]); continue; }
  for (const c of cells) grid[Math.ceil(c.y ?? c[1])][(c.x ?? c[0])] = '#';
  const full = grid.filter(row => !row.includes('.')).length;
  if (full === r.cleared) ok++;
  else { wrong++; if (ex.length < 3) ex.push([`cleared ${full}!=${r.cleared}`, r.replay, r.n, p]); }
}
console.log(`samples ${lines.length} | ok ${ok} | does-not-fit ${oob} | wrong-clears ${wrong}`);
for (const e of ex) console.log('  ', JSON.stringify(e).slice(0, 220));
