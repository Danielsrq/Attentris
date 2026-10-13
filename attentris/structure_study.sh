#!/bin/sh
# Does incidental structure recovery make a ToMe model choose better? (structure_study.py)
# Each ToMe model plays 64 games (normal style, depth 1, no garbage, fresh seeds 38001+) with every state
# dumped; the states are labelled by the teacher itself (CEM-depth6, normal.txt, beam 64, every move
# scored), packed and pre-tokenized per model; then structure_study.py scores each model on its own
# positions against the teacher, with v5.1 dagger-ft on the same positions as a difficulty control.
# Resumable per model and step; CPU (play, label, pack) then the analysis.   sh structure_study.sh > structure_study.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
BOT="$DIR/../bot/target/release/bot.exe"
D="$DIR/data/structure_study"
WD=$(cygpath -m "$D")
W=$(cygpath -m "$DIR/../bot/weights/normal.txt")
OUT="$DIR/results_structure_study.txt"
mkdir "$DIR/.structure_study.lock" 2>/dev/null || { echo "structure_study.sh is already running"; exit 1; }
trap 'rmdir "$DIR/.structure_study.lock"' EXIT
mkdir -p "$D"
MODELS=${MODELS:-"tome_v42_l4g_dagger_ft tome_v42_l4h_dagger_ft tome_v42_l6g_dagger_ft tome_v4"}
spec=""
for m in $MODELS; do
    f="$D/$m"
    if ! grep -q "^AtTenTris" "$f.games" 2>/dev/null; then
        rm -f "$f.games" "$f.jsonl"
        echo "$m: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --games 64 --depth 1 --style normal --seed 38001 \
            --out "$WD/$m.games" --dump "$WD/$m.jsonl" | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-120)" | tee -a "$OUT"
    fi
    if [ ! -f "$f.labelled.jsonl" ]; then
        "$BOT" label --mode normal --weights "$W" --label normal --threads 16 --all-roots 1 --beam 64 \
            --out "$WD/$m.labelled.jsonl.part" < "$f.jsonl" > "$f.label.log" 2>&1 && mv "$f.labelled.jsonl.part" "$f.labelled.jsonl"
    fi
    if [ ! -f "$D/pack_$m.npz" ]; then
        rm -rf "$D/raw_$m"; mkdir -p "$D/raw_$m"
        "$PYG" -c "
import json, sys
src, out = sys.argv[1], sys.argv[2]
fs = [open('%s/normal_s%d.jsonl' % (out, i), 'w') for i in range(16)]
for l in open(src):
    fs[json.loads(l)['seed'] % 16].write(l)
" "$WD/$m.labelled.jsonl" "$WD/raw_$m"
        "$PYG" "$DIR/pack.py" --only "$WD/raw_$m" --name "pack_$m" --out "$WD" --jobs 16 | tail -1
        "$PYG" "$DIR/pretok.py" --data "$WD" --files "pack_$m.npz" | tail -1
    fi
    spec="$spec --tome $m=$(cygpath -m "$DIR/data/model_$m.pt")=$WD/pack_$m.npz"
done
echo "== structure study ($(date '+%H:%M'))" | tee -a "$OUT"
"$PYG" "$DIR/structure_study.py" $spec --control "$(cygpath -m "$DIR/data/model_v51_dagger_ft.pt")" --device cuda | tee -a "$OUT"
echo "structure study done"
