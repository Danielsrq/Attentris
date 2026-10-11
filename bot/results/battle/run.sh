#!/bin/sh
# Tune for TetrisBattle's rules (--mode battle: a friend's RL bot's engine, exact parity
# checked): his kicks, T-spins only, TETR.IO attack with combos and B2B levels. Garbage
# as in his training: every 10 pieces a batch of 1-4 clean rows in a random column.
# From normal.txt (best untuned under these rules), combo weight seeded at +2.
# Resumable:
#   THREADS=14 sh run.sh          run or resume
#   powershell -File stop.ps1     stop now (the generation in progress reruns)
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
W="$DIR/../../weights"
T=${THREADS:-14}
G="10,1-4,0"

if [ ! -f "$DIR/start.txt" ]; then
    { grep -v "^#" "$W/normal.txt" | grep -v "^combo "; echo "combo        2.0000"; } > "$DIR/start.txt"
    sed -i '1i #! features normalised' "$DIR/start.txt"
fi
echo "== CEM under battle rules with garbage $G, from normal.txt"
"$BOT" tune --algo cem --mode battle --garbage $G --weights "$DIR/start.txt" --freeze default \
    --gens 30 --pop 32 --elite 8 --games 8 --pieces 300 --beam 24 --depth 4 --threads $T --save "$DIR/battle.txt"

# heldout NAME WEIGHTS GARBAGE(or -)
heldout() {
    name=$1; wf=$2; garb=$3
    tag="battle"; gflag=""
    if [ "$garb" != "-" ]; then tag="battle_garbage"; gflag="--garbage $garb"; fi
    out="$DIR/heldout_${name}_${tag}"
    if [ ! -f "$out.txt" ]; then
        "$BOT" play --mode battle $gflag --weights "$wf" --games 20 --pieces 500 --threads $T --label "$name" 2>/dev/null \
            | grep -E "^$name |clears per 100" > "$out.part" && mv "$out.part" "$out.txt"
    fi
    echo "-- $name, $tag"; cat "$out.txt"
}
for g in - $G; do
    heldout battle "$DIR/battle.txt" $g
    heldout normal_txt "$W/normal.txt" $g
done
echo "all done"
