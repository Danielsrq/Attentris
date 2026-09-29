// How long does the reconstruction survive after the cut?
//
// If the cut were arbitrary, the gap to the topout would scale with run length --
// a cut at a fixed FRACTION of the run. If it is a real event (garbage lands, the
// board becomes unrecoverable, the stack drowns), the gap should be roughly constant
// and independent of how long the run was. That is a check on the cut that uses
// nothing the cut itself was derived from.
import { readFileSync } from 'node:fs';
import { parseReplay, prepareReplay, Reconstruction } from '../vendor/tetrp/src/replay/index.js';
function sealed(b){const H=b.rows.length,W=b.width,o=new Uint8Array(H*W),st=[];
  for(let x=0;x<W;x++)if(b.rows[0][x]==null){o[x]=1;st.push([x,0]);}
  while(st.length){const[x,y]=st.pop();
    for(const[dx,dy]of[[1,0],[-1,0],[0,1],[0,-1]]){const nx=x+dx,ny=y+dy;
      if(nx<0||nx>=W||ny<0||ny>=H||o[ny*W+nx]||b.rows[ny][nx]!=null)continue;
      o[ny*W+nx]=1;st.push([nx,ny]);}}
  let n=0;for(let y=0;y<H;y++)for(let x=0;x<W;x++)if(b.rows[y][x]==null&&!o[y*W+x])n++;return n;}
const rows=[];
for (const path of process.argv.slice(2)) {
  const file = path.split(/[\/]/).pop().replace(/\.ttr$/,'');
  let tl;
  try { tl = prepareReplay(parseReplay(readFileSync(path,'utf8')).rounds[0].players[0]); }
  catch { continue; }
  const r = new Reconstruction(tl); r.engine.insertGarbage=()=>true;
  let pieces=0; const seq=[];
  while(r.advance()){const s=r.engine.state;
    if(s.stats.pieces>pieces){pieces=s.stats.pieces;seq.push(sealed(s.board));}
    if(!s.playing)break;}
  let i=seq.length-1; while(i>=0&&seq[i]>0) i--;
  const cut=i+1;
  rows.push({file, cut, end:seq.length, gap:seq.length-cut});
}
const gaps=rows.map(r=>r.gap), lens=rows.map(r=>r.end);
const srt=[...gaps].sort((a,b)=>a-b);
const pct=p=>srt[Math.floor(srt.length*p)];
const mean=a=>a.reduce((p,c)=>p+c,0)/a.length;
const mg=mean(gaps), ml=mean(lens);
let n=0,dg=0,dl=0;
for(let i=0;i<gaps.length;i++){n+=(gaps[i]-mg)*(lens[i]-ml);dg+=(gaps[i]-mg)**2;dl+=(lens[i]-ml)**2;}
console.log('placements from the cut to the topout, across ' + rows.length + ' runs:');
console.log('  min ' + srt[0] + '  p25 ' + pct(.25) + '  median ' + pct(.5) +
  '  p75 ' + pct(.75) + '  p90 ' + pct(.90) + '  max ' + srt[srt.length-1] +
  '  mean ' + mg.toFixed(1) + '  sd ' + Math.sqrt(dg/gaps.length).toFixed(1));
console.log('  within 40: ' + gaps.filter(g=>g<=40).length + ' of ' + gaps.length +
  ' (' + (100*gaps.filter(g=>g<=40).length/gaps.length).toFixed(0) + '%)');
console.log('');
console.log('correlation between gap and run length: r = ' + (n/Math.sqrt(dg*dl)).toFixed(3));
console.log('  (near zero = a fixed event; near one = a fixed fraction of the run)');
console.log('  run length spans ' + Math.min(...lens) + '..' + Math.max(...lens) +
  '; a fixed FRACTION would give gaps spanning ~' +
  Math.round(Math.min(...lens)*mg/ml) + '..' + Math.round(Math.max(...lens)*mg/ml));
const wide = rows.filter(r=>r.gap>40).sort((a,b)=>b.gap-a.gap);
console.log('');
console.log(wide.length + ' runs exceed 40:');
for (const r of wide)
  console.log('   ' + r.file.padEnd(30) + ' cut ' + String(r.cut).padStart(4) +
    ' of ' + String(r.end).padStart(4) + '   gap ' + r.gap);
