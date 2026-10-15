#!/bin/sh
# After an AtTenTris-ToMe training run: play tests (64 games, normal, no garbage: depth 1,
# depth 2 beam 4, depth 3 beam 8), single-game latency at depth 1, structure recovery
# (tome_iou.py) and the blob viewer (tome_viz.py, the viewer's 48 positions + 24 random). All CPU.
#   sh tome_post.sh [model name, default model_tome]  -> results_<name>.txt, <name>_viz.html
NAME=${1:-model_tome}
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
PYC="$CPUPY"
M="$DIR/data/$NAME.pt"
SHORT=$(echo "$NAME" | sed 's/^model_//')
OUT="$DIR/results_$SHORT.txt"
until [ -f "$M.log.json" ] && "$PYG" -c "import json,sys; sys.exit(0 if len(json.load(open(sys.argv[1]))) >= 8 else 1)" "$M.log.json"; do sleep 30; done
sleep 30          # the log gets its 8th entry right after the final checkpoint is saved; let the process exit
: > "$OUT"
"$PYG" -c "import json,sys; [print('epoch %d: val top1 %.3f / %.3f' % (r['epoch'], r['val']['normal']['top1'], r['val']['gate4']['top1'])) for r in json.load(open(sys.argv[1]))]" "$M.log.json" | tee -a "$OUT"
for db in "1 8" "2 4" "3 8"; do
    set -- $db
    echo "$SHORT: $("$PYC" "$DIR/play.py" --model "$M" --games 64 --depth $1 --beam $2 | grep "^AtTenTris")" | tee -a "$OUT"
done
echo "$SHORT latency: $("$PYC" "$DIR/play.py" --model "$M" --games 8 --jobs 1 --depth 1 | grep "^AtTenTris")" | tee -a "$OUT"
"$PYG" "$DIR/tome_iou.py" --model "$M" --n 2000 | grep -v "positions$" | tee -a "$OUT"
"$PYG" "$DIR/tome_viz.py" --model "$M" --random 24 --out "$DIR/${SHORT}_viz.html" | tee -a "$OUT"
echo "$SHORT post done" | tee -a "$OUT"
