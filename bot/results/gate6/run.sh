#!/bin/sh
# Gate round 6: round 5 with danger FIXED at round 4's value (-119.7). Only danger_relief
# (start +59.9, a 50% discount) and the other weights tune, so the tuner cannot make
# height costlier and offset it with relief: the discount is relief / 119.7. Same setup:
# gate_b2b, clean aligned garbage 15,4,0,0.2, 8 games per candidate. Resumable:
#   THREADS=14 sh run.sh          run or resume
#   powershell -File stop.ps1     stop now (the generation in progress reruns)
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
DATA="$DIR/../../../data/bot"
T=${THREADS:-14}
G="15,4,0,0.2"

[ -f "$DIR/start.txt" ] || cp "$DIR/../gate5/start.txt" "$DIR/start.txt"
echo "== CEM under gate_b2b rules with garbage $G, danger fixed, from round 5's start"
"$BOT" tune --algo cem --mode gate_b2b --garbage $G --weights "$DIR/start.txt" --freeze default --fix danger \
    --gens 30 --pop 32 --elite 8 --games 8 --pieces 300 --beam 24 --depth 4 --threads $T --save "$DIR/gate6.txt"

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
heldout gate6 "$DIR/gate6.txt" gate_b2b $G
heldout gate6 "$DIR/gate6.txt" gate $G
heldout gate6 "$DIR/gate6.txt" burst2 $G
heldout gate6 "$DIR/gate6.txt" normal -

echo "== replays: gate6 bot under gate_b2b rules with garbage"
if [ ! -f "$DATA/gate6_garbage.jsonl" ]; then
    "$BOT" play --mode gate_b2b --garbage $G --weights "$DIR/gate6.txt" --games 5 --pieces 500 --threads $T --label gate6 --out "$DATA/gate6_garbage.jsonl" | tail -2
fi
echo "all done"
