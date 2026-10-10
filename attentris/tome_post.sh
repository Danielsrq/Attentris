#!/bin/sh
# After AtTenTris-ToMe finishes training (tome.sh): play tests (64 games, normal, no garbage:
# depth 1, depth 2 beam 4, depth 3 beam 8), single-game latency at depth 1, structure recovery
# (tome_iou.py), and the blob viewer (tome_viz.py). All CPU; waits for the 8th epoch first.
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
M="$DIR/data/model_tome.pt"
OUT="$DIR/results_tome.txt"
until [ -f "$M.log.json" ] && "$PYG" -c "import json,sys; sys.exit(0 if len(json.load(open(sys.argv[1]))) >= 8 else 1)" "$M.log.json"; do sleep 30; done
sleep 30          # the log gets its 8th entry right after the final checkpoint is saved; let the process exit
: > "$OUT"
grep "^epoch" "$DIR/train_tome.out" | tee -a "$OUT"
for db in "1 8" "2 4" "3 8"; do
    set -- $db
    echo "tome: $("$PYC" "$DIR/play.py" --model "$M" --games 64 --depth $1 --beam $2 | grep "^AtTenTris")" | tee -a "$OUT"
done
echo "tome latency: $("$PYC" "$DIR/play.py" --model "$M" --games 8 --jobs 1 --depth 1 | grep "^AtTenTris")" | tee -a "$OUT"
"$PYG" "$DIR/tome_iou.py" --model "$M" --n 2000 | grep -v "positions$" | tee -a "$OUT"
"$PYG" "$DIR/tome_viz.py" --model "$M" --out "$DIR/tome_viz.html" | tee -a "$OUT"
echo "tome post done" | tee -a "$OUT"
