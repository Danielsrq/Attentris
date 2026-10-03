#!/bin/sh
# burst2 rules (spin singles do not arm the doubling; window of 2 placements) with the
# vertical T-slot template (STSD / TST / C-spin, cap + lane) and t_followup in V.
# Every step resumes:
#   THREADS=14 sh run.sh          run or resume
#   powershell -File stop.ps1     stop now (the generation in progress reruns)
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
W="$DIR/../../weights"
DATA="$DIR/../../../data/bot"
T=${THREADS:-14}

# Start: the first-round burst tune, plus t_followup at 1.0 (one line of attack for a
# ready follow-up TSD) so the tuner explores around a meaningful value, not +-0.2 of 0.
if [ ! -f "$DIR/start.txt" ]; then
    { grep -v "^# tuned" "$DIR/../burst/burst.txt"; echo "t_followup   1.0000"; } > "$DIR/start.txt"
fi
echo "== CEM under burst2 rules (masked, 19 active weights)"
"$BOT" tune --algo cem --mode burst2 --weights "$DIR/start.txt" --freeze default --gens 30 --pop 32 --elite 8 \
    --games 6 --pieces 300 --beam 24 --depth 4 --threads $T --save "$DIR/burst2.txt"

heldout() {
    name=$1; wf=$2; mode=$3
    out="$DIR/heldout_${name}_${mode}"
    if [ ! -f "$out.txt" ]; then
        "$BOT" play --mode "$mode" --weights "$wf" --games 20 --pieces 500 --threads $T --label "$name" --seq-out "$out.seq.csv" 2>/dev/null \
            | grep -E "^$name |clears per 100|^burst" > "$out.part" && mv "$out.part" "$out.txt"
    fi
    cat "$out.txt"
}
echo "== held-out under burst2 rules"
heldout burst2 "$DIR/burst2.txt" burst2
heldout burst "$DIR/../burst/burst.txt" burst2
heldout normal_txt "$W/normal.txt" burst2
echo "== held-out under burst rules (the new weights)"
heldout burst2 "$DIR/burst2.txt" burst

echo "== replays: burst2 bot under burst2 rules"
if [ ! -f "$DATA/burst2.jsonl" ]; then
    "$BOT" play --mode burst2 --weights "$DIR/burst2.txt" --games 5 --pieces 500 --threads $T --label burst2 --out "$DATA/burst2.jsonl" | tail -2
fi
echo "all done"
