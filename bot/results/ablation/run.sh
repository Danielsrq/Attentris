#!/bin/sh
# V vs search ablation, normal rules. Resumable: finished configurations are skipped.
#   THREADS=10 sh run.sh          run or resume
#   touch STOP                    stop after the current configuration
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
W="$DIR/../../weights"
T=${THREADS:-10}
rm -f "$DIR/STOP" "$DIR"/*.tmp "$DIR"/*.part

run() {
    name=$1
    shift
    [ -f "$DIR/$name.txt" ] && { echo "skip $name"; return; }
    [ -f "$DIR/STOP" ] && { echo "STOP found"; exit 0; }
    echo "run  $name"
    if "$BOT" play --mode normal --games 20 --pieces 300 --threads "$T" --label "$name" "$@" > "$DIR/$name.tmp" 2>/dev/null; then
        grep -E "^$name |clears per 100" "$DIR/$name.tmp" > "$DIR/$name.part" && mv "$DIR/$name.part" "$DIR/$name.txt"
    fi
    rm -f "$DIR/$name.tmp" "$DIR/$name.part"
}

for v in zero survival_only hand_set tuned; do
    if [ "$v" = tuned ]; then WF="$W/standard.txt"; else WF="$W/ablation/$v.txt"; fi
    run "${v}_d1" --weights "$WF" --depth 1 --no-chance
    run "${v}_d3" --weights "$WF" --depth 3 --no-chance
    run "${v}_d6" --weights "$WF" --depth 6
done
echo "ablation complete"
