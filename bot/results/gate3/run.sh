#!/bin/sh
# Gate rules, round 3: as round 2, but t_vertical (a ready STSD / C-spin / imperial-cross
# slot) starts at +5 and is tuned. CEM's fitness is the real gate APP, so if the bonus
# makes the bot hoard the slot (as a hand-set +10 did), the tuner shrinks it. Resumable:
#   THREADS=14 sh run.sh          run or resume
#   powershell -File stop.ps1     stop now (the generation in progress reruns)
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
DATA="$DIR/../../../data/bot"
T=${THREADS:-14}

if [ ! -f "$DIR/start.txt" ]; then
    { grep -v "^# tuned" "$DIR/../gate2/gate2.txt"; echo "t_vertical   5.0000"; } > "$DIR/start.txt"
fi
echo "== CEM under gate rules (round 3), from gate2 + t_vertical 5"
"$BOT" tune --algo cem --mode gate --weights "$DIR/start.txt" --freeze default,b2b,b2b_level,b2b_next \
    --gens 30 --pop 32 --elite 8 --games 12 --pieces 300 --beam 24 --depth 4 --threads $T --save "$DIR/gate3.txt"

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
    heldout gate3 "$DIR/gate3.txt" $mode
done

echo "== replays: gate3 bot under gate rules"
if [ ! -f "$DATA/gate3.jsonl" ]; then
    "$BOT" play --mode gate --weights "$DIR/gate3.txt" --games 5 --pieces 500 --threads $T --label gate3 --out "$DATA/gate3.jsonl" | tail -2
fi
echo "all done"
