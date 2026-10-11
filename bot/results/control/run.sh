#!/bin/sh
# Control for round 3: does CEM find t_vertical's value from 0, or did the +5 start do it?
# Two arms with today's code, identical but for t_vertical's starting value (0 / +5):
# gate rules, no garbage, from the gate2 weights, 12 games per candidate, 30 generations
# (round 3's setup). Then held-out for both, and a 1-D fitness slice over t_vertical
# 0..8 around each arm's final weights (tuning-strength search, 12 games) to see the
# landscape between 0 and the optimum. Resumable:
#   THREADS=14 sh run.sh          run or resume
#   powershell -File stop.ps1     stop now (the generation / batch in progress reruns)
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
T=${THREADS:-14}

for v in 0 5; do
    if [ ! -f "$DIR/start_$v.txt" ]; then
        { grep -v -E "^# tuned|^t_vertical " "$DIR/../gate2/gate2.txt"; echo "t_vertical   $v.0000"; } > "$DIR/start_$v.txt"
    fi
done
for v in 0 5; do
    echo "== CEM under gate rules, t_vertical starting at $v"
    "$BOT" tune --algo cem --mode gate --weights "$DIR/start_$v.txt" --freeze default,b2b,b2b_level,b2b_next \
        --gens 30 --pop 32 --elite 8 --games 12 --pieces 300 --beam 24 --depth 4 --threads $T --save "$DIR/ctrl$v.txt"
done

for v in 0 5; do
    out="$DIR/heldout_ctrl${v}_gate"
    if [ ! -f "$out.txt" ]; then
        "$BOT" play --mode gate --weights "$DIR/ctrl$v.txt" --games 20 --pieces 500 --threads $T --label "ctrl$v" 2>/dev/null \
            | grep -E "^ctrl$v |template" > "$out.part" && mv "$out.part" "$out.txt"
    fi
    echo "-- ctrl$v, gate"; cat "$out.txt"
done

for v in 0 5; do
    echo "== slice of t_vertical around ctrl$v"
    "$BOT" landscape slices --only t_vertical --values 0,1,2,3,4,5,6,7,8 --weights "$DIR/ctrl$v.txt" --tag "_ctrl$v" \
        --mode gate --games 12 --pieces 300 --beam 24 --depth 4 --threads $T --save "$DIR"
done
echo "all done"
