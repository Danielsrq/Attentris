#!/bin/sh
# Gate rules, round 2: a chain pays only once it holds two spin clears (quads hold the
# gate open and add 1, but cannot complete a chain); V measures board health after the
# T-spin chain (static prophecy masks + banded rescan). 12 games per candidate: the
# reward is sparse. B2B weights frozen too (B2B pays nothing under gate). Resumable:
#   THREADS=14 sh run.sh          run or resume
#   powershell -File stop.ps1     stop now (the generation in progress reruns)
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
DATA="$DIR/../../../data/bot"
T=${THREADS:-14}

echo "== CEM under gate rules (round 2), from the burst2 weights"
"$BOT" tune --algo cem --mode gate --weights "$DIR/../burst2/burst2.txt" --freeze default,b2b,b2b_level,b2b_next \
    --gens 30 --pop 32 --elite 8 --games 12 --pieces 300 --beam 24 --depth 4 --threads $T --save "$DIR/gate2.txt"

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
    heldout gate2 "$DIR/gate2.txt" $mode
done
heldout burst2 "$DIR/../burst2/burst2.txt" gate

echo "== replays: gate2 bot under gate rules"
if [ ! -f "$DATA/gate2.jsonl" ]; then
    "$BOT" play --mode gate --weights "$DIR/gate2.txt" --games 5 --pieces 500 --threads $T --label gate2 --out "$DATA/gate2.jsonl" | tail -2
fi
echo "all done"
