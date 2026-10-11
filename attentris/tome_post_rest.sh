#!/bin/sh
# Finish a tome_post.sh run from the depth-3 games on (the first run's pool workers died when RAM ran
# out): depth 3 beam 8, single-game latency, structure recovery, blob viewer. Appends to the same files.
#   sh tome_post_rest.sh model_tome_v4
NAME=${1:-model_tome}
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
M="$DIR/data/$NAME.pt"
SHORT=$(echo "$NAME" | sed 's/^model_//')
OUT="$DIR/results_$SHORT.txt"
echo "$SHORT: $("$PYC" "$DIR/play.py" --model "$M" --games 64 --depth 3 --beam 8 | grep "^AtTenTris")" | tee -a "$OUT"
echo "$SHORT latency: $("$PYC" "$DIR/play.py" --model "$M" --games 8 --jobs 1 --depth 1 | grep "^AtTenTris")" | tee -a "$OUT"
"$PYG" "$DIR/tome_iou.py" --model "$M" --n 2000 | grep -v "positions$" | tee -a "$OUT"
"$PYG" "$DIR/tome_viz.py" --model "$M" --random 24 --out "$DIR/${SHORT}_viz.html" | tee -a "$OUT"
echo "$SHORT post done" | tee -a "$OUT"
