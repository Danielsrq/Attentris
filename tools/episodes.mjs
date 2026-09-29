// Episodes of holes: how long does the board stay holed before it is clean again?
// A donation is covered and reopened, so its episode is short. A lost garbage row is
// never reopened, so its episode runs to the end of the reconstruction.
import { readFileSync } from 'node:fs';
import { parseReplay, prepareReplay, Reconstruction } from '../vendor/tetrp/src/replay/index.js';
function sealed(b){const H=b.rows.length,W=b.width,o=new Uint8Array(H*W),st=[];
  for(let x=0;x<W;x++)if(b.rows[0][x]==null){o[x]=1;st.push([x,0]);}
  while(st.length){const[x,y]=st.pop();
    for(const[dx,dy]of[[1,0],[-1,0],[0,1],[0,-1]]){const nx=x+dx,ny=y+dy;
      if(nx<0||nx>=W||ny<0||ny>=H||o[ny*W+nx]||b.rows[ny][nx]!=null)continue;
      o[ny*W+nx]=1;st.push([nx,ny]);}}
  let n=0;for(let y=0;y<H;y++)for(let x=0;x<W;x++)if(b.rows[y][x]==null&&!o[y*W+x])n++;return n;}
const tl = prepareReplay(parseReplay(readFileSync(process.argv[2],'utf8')).rounds[0].players[0]);
const r = new Reconstruction(tl); r.engine.insertGarbage=()=>true;
let pieces=0; const seq=[];
while(r.advance()){const s=r.engine.state;
  if(s.stats.pieces>pieces){pieces=s.stats.pieces;seq.push(sealed(s.board));}
  if(!s.playing)break;}
const spells=[]; let start=-1;
for(let i=0;i<seq.length;i++){
  if(seq[i]>0 && start<0) start=i;
  if(seq[i]===0 && start>=0){spells.push({from:start+1,to:i,len:i-start,peak:Math.max(...seq.slice(start,i))});start=-1;}
}
if(start>=0) spells.push({from:start+1,to:seq.length,len:seq.length-start,peak:Math.max(...seq.slice(start)),open:true});
console.log('run length ' + seq.length + ' placements, ' + spells.length + ' hole episodes');
console.log(['from'.padStart(6),'to'.padStart(6),'len'.padStart(6),'peak'.padStart(6)].join('') + '');
for (const x of spells)
  console.log(String(x.from).padStart(6) + String(x.to).padStart(6) +
    String(x.len).padStart(6) + String(x.peak).padStart(6) +
    (x.open ? '   <- never recovered: the cut' : ''));
const rec = spells.filter(x => !x.open);
if (rec.length)
  console.log('');
if (rec.length)
  console.log('largest episode that DID recover: ' + Math.max(...rec.map(x => x.peak)) +
    ' cells; the unrecovered one peaks at ' + (spells.find(x => x.open)?.peak ?? 0));
