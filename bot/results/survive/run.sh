#!/bin/sh
# CEM-survival: tune for survival against cheese. Normal rules, garbage every 10 pieces of 3-4 rows,
# half the batches cheese (a new hole every row), clean batches mostly in the previous column.
# Fitness VS = (attack + garbage cleared) over the full piece budget (TETR.IO's versus score
# without PPS; dying early scores 0 for the rest). From normal.txt (the CEM-depth6 teacher).
# Then normal.txt vs survive.txt at the teacher's search (beam 64, depth 6), 64 games each with no
# garbage, 80/20 and 50/50 clean/cheese, on seeds 33001+ (the tune plays 1000-3950). Resumable:
#   THREADS=16 sh run.sh          run or resume
#   powershell -File stop.ps1     stop now (the generation in progress reruns)
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
PYC="$DIR/../../../external/tetrisbattle/.venv/Scripts/python.exe"
VS="$DIR/../../../attentris/vs_score.py"
T=${THREADS:-16}
G50="10,3-4,cheese0.5,0.2"
G20="10,3-4,cheese0.2,0.2"
WD=$(cygpath -m "$DIR")

echo "== CEM-survival: CEM under normal rules with garbage $G50, fitness VS, from normal.txt"
"$BOT" tune --algo cem --mode normal --garbage $G50 --fitness vs --weights "$DIR/../../weights/normal.txt" --freeze default \
    --gens 30 --pop 32 --elite 8 --games 12 --pieces 500 --beam 24 --depth 4 --threads $T --save "$DIR/survive.txt"

# eval NAME WEIGHTS TAG GARBAGE(or -)
eval_run() {
    name=$1; wf=$2; tag=$3; garb=$4
    out="$WD/eval_${name}_${tag}.jsonl"
    gflag=""; [ "$garb" != "-" ] && gflag="--garbage $garb"
    [ -f "$out" ] || "$BOT" play --mode normal $gflag --weights "$wf" --games 64 --pieces 500 --seed 33001 --threads $T \
        --label "$name" --game-log "$out.part" > /dev/null 2>&1 && mv "$out.part" "$out"
}
for tag in none cheese20 cheese50; do
    case $tag in none) g=- ;; cheese20) g=$G20 ;; cheese50) g=$G50 ;; esac
    eval_run normal "$DIR/../../weights/normal.txt" $tag $g
    eval_run survive "$DIR/survive.txt" $tag $g
done
"$PYC" "$VS" \
    "CEM-depth6 (normal.txt), no garbage=$WD/eval_normal_none.jsonl" "CEM-survival-depth6, no garbage=$WD/eval_survive_none.jsonl" \
    "CEM-depth6, 80/20 clean/cheese=$WD/eval_normal_cheese20.jsonl" "CEM-survival-depth6, 80/20=$WD/eval_survive_cheese20.jsonl" \
    "CEM-depth6, 50/50 clean/cheese=$WD/eval_normal_cheese50.jsonl" "CEM-survival-depth6, 50/50=$WD/eval_survive_cheese50.jsonl" \
    | tee "$DIR/eval.txt"
echo "all done"
