#!/bin/sh
# Resumable B2B bonus cap sweep.
#
#   THREADS=10 sh sweep.sh      run or resume (default 10 threads)
#
# Each configuration's result is written to <name>.txt only once its run has finished
# (temp file, then rename), and a re-run skips every configuration that already has
# one -- so stopping costs at most the configuration in progress.
#
# To stop:
#   touch STOP                  finish the current configuration, then stop
#   powershell -File stop.ps1   stop immediately (the current configuration reruns)

DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
W="$DIR/../../weights"
T=${THREADS:-10}
rm -f "$DIR/STOP" "$DIR"/*.tmp "$DIR"/*.part   # leftovers from an interrupted run

run() {
    name=$1
    shift
    if [ -f "$DIR/$name.txt" ]; then
        echo "skip $name (done)"
        return
    fi
    if [ -f "$DIR/STOP" ]; then
        echo "STOP file found, stopping before $name"
        exit 0
    fi
    echo "run  $name"
    if "$BOT" play "$@" --threads "$T" --games 20 --pieces 500 --label "$name" > "$DIR/$name.tmp" 2>/dev/null; then
        grep -E "^$name |clears per 100" "$DIR/$name.tmp" > "$DIR/$name.part" && mv "$DIR/$name.part" "$DIR/$name.txt"
    fi
    rm -f "$DIR/$name.tmp" "$DIR/$name.part"
}

for cap in 0 1 2 3 4 5 6 none; do
    C=""
    [ "$cap" != none ] && C="--b2b-cap $cap"
    run "allspin_cap_$cap" --mode allspin --weights "$W/allspin_slots.txt" $C
done
for cap in 0 1 2 3 4 none; do
    C=""
    [ "$cap" != none ] && C="--b2b-cap $cap"
    run "standard_cap_$cap" --mode standard --weights "$W/standard.txt" $C
done
echo "sweep complete"
cat "$DIR"/*.txt
