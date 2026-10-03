#!/bin/sh
# Gate rules (training): only chains opened by a full spin double/triple score; quads
# join for 1; a chain of two or more pays the sum (TSD 4, TST 10). The tuned weights are
# then evaluated under burst2 and normal rules. Every step resumes:
#   THREADS=14 sh run.sh          run or resume
#   powershell -File stop.ps1     stop now (the generation in progress reruns)
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
W="$DIR/../../weights"
DATA="$DIR/../../../data/bot"
T=${THREADS:-14}

echo "== CEM under gate rules, from the burst2 weights (masked)"
"$BOT" tune --algo cem --mode gate --weights "$DIR/../burst2/burst2.txt" --freeze default --gens 30 --pop 32 --elite 8 \
    --games 6 --pieces 300 --beam 24 --depth 4 --threads $T --save "$DIR/gate.txt"

heldout() {
    name=$1; wf=$2; mode=$3
    out="$DIR/heldout_${name}_${mode}"
    if [ ! -f "$out.txt" ]; then
        "$BOT" play --mode "$mode" --weights "$wf" --games 20 --pieces 500 --threads $T --label "$name" --seq-out "$out.seq.csv" 2>/dev/null \
            | grep -E "^$name |clears per 100|^burst|template" > "$out.part" && mv "$out.part" "$out.txt"
    fi
    cat "$out.txt"
}
for mode in gate burst2 normal; do
    echo "== held-out under $mode rules"
    heldout gate "$DIR/gate.txt" $mode
    heldout burst2 "$DIR/../burst2/burst2.txt" $mode
done

echo "== replays: gate bot under gate rules and burst2 rules"
if [ ! -f "$DATA/gate.jsonl" ]; then
    "$BOT" play --mode gate --weights "$DIR/gate.txt" --games 5 --pieces 500 --threads $T --label gate --out "$DATA/gate.jsonl" | tail -2
fi
if [ ! -f "$DATA/gate_burst2.jsonl" ]; then
    "$BOT" play --mode burst2 --weights "$DIR/gate.txt" --games 5 --pieces 500 --threads $T --label gate_burst2 --out "$DATA/gate_burst2.jsonl" | tail -2
fi
echo "all done"
