#!/bin/sh
# Held-out benchmark (normal rules, fibonacci B2B): 20 fresh games x 500 pieces, full
# search, seeds 1..20 (tuning used 1000+). Resumable: finished entries are kept.
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
W="$DIR/../../weights"
one() {
    name=$1
    wf=$2
    if [ ! -f "$DIR/heldout_$name.txt" ]; then
        "$BOT" play --mode normal --weights "$wf" --games 20 --pieces 500 --threads ${THREADS:-16} --label "$name" 2>/dev/null \
            | grep -E "^$name |clears per 100" > "$DIR/heldout_$name.part" && mv "$DIR/heldout_$name.part" "$DIR/heldout_$name.txt"
    fi
    cat "$DIR/heldout_$name.txt"
}
one start_standard "$W/standard.txt"
one tuned_normalised "$DIR/normalised.txt"
one tuned_raw "$DIR/raw.txt"
one allspin_slots "$W/allspin_slots.txt"
