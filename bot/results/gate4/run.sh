#!/bin/sh
# Gate round 4: gate_b2b rules (each spin clear of a paid chain also earns its B2B bonus;
# plain clears break B2B) with clean, mostly aligned garbage in tuning and held-out:
# every 15 pieces 4 clean rows, the hole moving to an adjacent column with chance 0.2.
# From the gate3 weights, B2B weights unfrozen and seeded from the burst2 tune. 8 games
# per candidate (chains are less sparse now). Resumable:
#   THREADS=14 sh run.sh          run or resume
#   powershell -File stop.ps1     stop now (the generation in progress reruns)
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
DATA="$DIR/../../../data/bot"
T=${THREADS:-14}
G="15,4,0,0.2"

if [ ! -f "$DIR/start.txt" ]; then
    { grep -v -E "^# tuned|^(b2b|b2b_level|b2b_next) " "$DIR/../gate3/gate3.txt"; grep -E "^(b2b|b2b_level|b2b_next) " "$DIR/../burst2/burst2.txt"; } > "$DIR/start.txt"
fi
echo "== CEM under gate_b2b rules with garbage $G, from gate3 (+ burst2's B2B weights)"
"$BOT" tune --algo cem --mode gate_b2b --garbage $G --weights "$DIR/start.txt" --freeze default \
    --gens 30 --pop 32 --elite 8 --games 8 --pieces 300 --beam 24 --depth 4 --threads $T --save "$DIR/gate4.txt"

# heldout NAME WEIGHTS MODE GARBAGE(or -)
heldout() {
    name=$1; wf=$2; mode=$3; garb=$4
    tag="${mode}"; gflag=""
    if [ "$garb" != "-" ]; then tag="${mode}_garbage"; gflag="--garbage $garb"; fi
    out="$DIR/heldout_${name}_${tag}"
    if [ ! -f "$out.txt" ]; then
        "$BOT" play --mode "$mode" $gflag --weights "$wf" --games 20 --pieces 500 --threads $T --label "$name" --seq-out "$out.seq.csv" 2>/dev/null \
            | grep -E "^$name |clears per 100|^burst|template" > "$out.part" && mv "$out.part" "$out.txt"
    fi
    echo "-- $name, $tag"; cat "$out.txt"
}
heldout gate4 "$DIR/gate4.txt" gate_b2b $G
heldout gate3 "$DIR/../gate3/gate3.txt" gate_b2b $G
heldout gate4 "$DIR/gate4.txt" gate $G
heldout gate4 "$DIR/gate4.txt" burst2 $G
heldout gate4 "$DIR/gate4.txt" normal -

echo "== replays: gate4 bot under gate_b2b rules with garbage"
if [ ! -f "$DATA/gate4_garbage.jsonl" ]; then
    "$BOT" play --mode gate_b2b --garbage $G --weights "$DIR/gate4.txt" --games 5 --pieces 500 --threads $T --label gate4 --out "$DATA/gate4_garbage.jsonl" | tail -2
fi
echo "all done"
