#!/bin/sh
# Gate round 5: as round 4 (gate_b2b, clean aligned garbage 15,4,0,0.2, 8 games), plus
# danger_relief -- the stack-height danger the T-spin chain would remove -- seeded at half
# of round 4's |danger| (a 50% discount on structure height), then tuned. Waits for round 4
# to finish, rebuilds the bot (round 4 holds the exe), then runs. Resumable:
#   THREADS=14 sh run.sh          run or resume
#   powershell -File stop.ps1     stop now (the generation in progress reruns)
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT="$DIR/../.."
BOT="$ROOT/target/release/bot.exe"
DATA="$DIR/../../../data/bot"
T=${THREADS:-14}
G="15,4,0,0.2"

until grep -q "all done" "$DIR/../gate4/run.out" 2>/dev/null; do sleep 30; done
if ! "$BOT" weights --weights "$DIR/../gate4/gate4.txt" 2>/dev/null | grep -q danger_relief; then
    (cd "$ROOT" && cargo build --release 2>&1 | tail -1)
fi
if [ ! -f "$DIR/start.txt" ]; then
    d=$(grep -E "^danger " "$DIR/../gate4/gate4.txt" | awk '{print ($2 < 0 ? -$2 : $2) / 2}')
    { grep -v "^# tuned" "$DIR/../gate4/gate4.txt"; echo "danger_relief $d"; } > "$DIR/start.txt"
fi
echo "== CEM under gate_b2b rules with garbage $G, from gate4 + danger_relief"
"$BOT" tune --algo cem --mode gate_b2b --garbage $G --weights "$DIR/start.txt" --freeze default \
    --gens 30 --pop 32 --elite 8 --games 8 --pieces 300 --beam 24 --depth 4 --threads $T --save "$DIR/gate5.txt"

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
heldout gate5 "$DIR/gate5.txt" gate_b2b $G
heldout gate5 "$DIR/gate5.txt" gate $G
heldout gate5 "$DIR/gate5.txt" burst2 $G
heldout gate5 "$DIR/gate5.txt" normal -

echo "== replays: gate5 bot under gate_b2b rules with garbage"
if [ ! -f "$DATA/gate5_garbage.jsonl" ]; then
    "$BOT" play --mode gate_b2b --garbage $G --weights "$DIR/gate5.txt" --games 5 --pieces 500 --threads $T --label gate5 --out "$DATA/gate5_garbage.jsonl" | tail -2
fi
echo "all done"
