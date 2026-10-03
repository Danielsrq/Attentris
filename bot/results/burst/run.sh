#!/bin/sh
# Burst rule set: tune a bursty bot, compare with the normal-rules bot, Shapley values.
# Every step resumes (tune .state, held-out files, Shapley cache):
#   THREADS=14 sh run.sh          run or resume
#   powershell -File stop.ps1     stop now (the generation / batch in progress reruns)
#
# Feature mask (--freeze default): sz/lj/i spin slots, max_height, wounds, ready_repeat
# are pinned at 0 -- they carry nothing outside all-spin. 18 active weights incl. burst_armed.
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
W="$DIR/../../weights"
DATA="$DIR/../../../data/bot"
T=${THREADS:-14}
TUNE="--weights $W/normal.txt --freeze default --gens 30 --pop 32 --elite 8 --games 6 --pieces 300 --beam 24 --depth 4 --threads $T"

echo "== CEM under burst rules, from normal.txt (masked)"
"$BOT" tune --algo cem --mode burst $TUNE --save "$DIR/burst.txt"
echo "== control: CEM under normal rules, same mask and start"
"$BOT" tune --algo cem --mode normal $TUNE --save "$DIR/normal_frozen.txt"

# Held-out: 20 fresh games x 500 pieces at full search (seeds 1..20; tuning used 1000+),
# every weight set under both rule sets, with the per-piece attack sequence for the graphs.
heldout() {
    name=$1; wf=$2; mode=$3
    out="$DIR/heldout_${name}_${mode}"
    if [ ! -f "$out.txt" ]; then
        "$BOT" play --mode "$mode" --weights "$wf" --games 20 --pieces 500 --threads $T --label "$name" --seq-out "$out.seq.csv" 2>/dev/null \
            | grep -E "^$name |clears per 100|^burst" > "$out.part" && mv "$out.part" "$out.txt"
    fi
    cat "$out.txt"
}
for mode in burst normal; do
    echo "== held-out under $mode rules"
    heldout burst "$DIR/burst.txt" $mode
    heldout normal_frozen "$DIR/normal_frozen.txt" $mode
    heldout normal_txt "$W/normal.txt" $mode
done

# Shapley: baseline = the starting values everyone began from (standard.txt).
SH="--baseline $W/standard.txt --freeze default --perms 32 --games 12 --pieces 300 --beam 24 --depth 4 --threads $T"
echo "== Shapley: burst weights under burst rules"
"$BOT" shapley --mode burst --weights "$DIR/burst.txt" $SH --save "$DIR/shapley_burst"
echo "== Shapley: masked normal weights under normal rules"
"$BOT" shapley --mode normal --weights "$DIR/normal_frozen.txt" $SH --save "$DIR/shapley_normal"

echo "== replays: burst bot under burst rules"
if [ ! -f "$DATA/burst.jsonl" ]; then
    "$BOT" play --mode burst --weights "$DIR/burst.txt" --games 5 --pieces 500 --threads $T --label burst --out "$DATA/burst.jsonl" | tail -2
fi
echo "all done"
