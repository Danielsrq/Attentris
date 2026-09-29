// Re-cuts the harvested file by a rule, without re-harvesting.
//
// Usage: node scripts/zcut.mjs <in.jsonl> <out.jsonl> <rule>
//   recover       truncate at the hole episode the board never recovers from. Strong
//                 players bury cells on purpose and reopen them, so a hole alone means
//                 nothing; an episode that never clears is a lost garbage row. No
//                 threshold: the cut is found by walking back to the last clean board.
//   queue         every sample before the first RECORDED garbage packet whose
//                 cancellation deadline passed with lines still in the queue. This is
//                 the only rule here that is a statement about garbage rather than an
//                 opinion about board shape, and it agrees with the engine's own
//                 tanking at r = 0.94 (mean difference 2.4 placements).
//   strict        every sample before the first garbage the sim inserts
//   holes:K       ...and stop once a sealed hole has survived K placements
//   height:N      ...and stop once the stack passes N
//   none          everything, to the topout
//
// A rule truncates each run at the first sample that trips it: samples after a
// divergence are not salvageable by filtering individually, because the board they
// were recorded against is already wrong.
import { readFileSync, writeFileSync } from 'node:fs';
const [inPath, outPath, rule = 'none'] = process.argv.slice(2);
const recs = readFileSync(inPath, 'utf8').trim().split('\n').map(l => JSON.parse(l));
const [kind, argRaw] = rule.split(':');
const arg = Number(argRaw);
const trips = r => {
  if (kind === 'recover') return r.n >= breakAt.get(r.replay);
  if (kind === 'queue') return r.q_overdue > 0;
  if (kind === 'strict') return r.tier !== 'strict';
  if (kind === 'holes') return r.hole_run >= arg;
  if (kind === 'height') return r.height > arg;
  return false;
};
// 'recover' needs the whole run before it can place the cut, so it is resolved up
// front rather than streamed like the threshold rules.
const breakAt = new Map();
if (kind === 'recover') {
  const byRun = new Map();
  for (const r of recs) (byRun.get(r.replay) ?? byRun.set(r.replay, []).get(r.replay)).push(r);
  for (const [name, rs] of byRun) {
    let i = rs.length - 1;
    while (i >= 0 && rs[i].sealed > 0) i--;
    breakAt.set(name, i + 1 < rs.length ? rs[i + 1].n : Infinity);
  }
}

const out = [];
const runs = new Map();
for (const r of recs) {
  if (!runs.has(r.replay)) runs.set(r.replay, {done: false, n: 0});
  const st = runs.get(r.replay);
  if (st.done) continue;
  if (trips(r)) { st.done = true; continue; }
  st.n++; out.push(r);
}
writeFileSync(outPath, out.map(o => JSON.stringify(o)).join('\n') + '\n');
const holeFree = out.filter(o => o.sealed === 0).length;
const med = a => a.slice().sort((x, y) => x - y)[Math.floor(a.length / 2)] ?? 0;
console.log('%s -> %d samples from %d runs | median run %d | %s%% hole-free',
  rule.padEnd(12), out.length, [...runs.values()].filter(v => v.n > 0).length,
  med([...runs.values()].map(v => v.n).filter(Boolean)),
  (100 * holeFree / Math.max(1, out.length)).toFixed(1));
