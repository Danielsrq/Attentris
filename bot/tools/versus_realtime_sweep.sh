#!/bin/sh
# Where does Cold Clear 2 draw level with CEM-survival? Realtime (uncapped), CEM-survival d6 b64 at its own
# speed (~35 ms per piece) vs CC2 at THINK ms per move, 64 matches each, Cold Clear 1 rules.
# CC2 at 40 ms is known (54-10 for CEM, bot/results/versus/realtime/plain_vs_cc2_40.jsonl).
#   THINK="5 10 15 20 25 30" sh bot/tools/versus_realtime_sweep.sh >> bot/results/versus/realtime_sweep.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../attentris"; . "$ATT/plat.sh"
ROOT=$(cd "$DIR/../.." && pwd)
R="$ROOT/bot/results/versus/realtime"
REP="$ROOT/data/bot/versus/realtime"
mkdir -p "$R" "$REP"
PLAIN="cem:bot/results/survive/survive.txt:6:64"
for ms in ${THINK:-5 10 15 20 25 30}; do
    tag=plain_vs_cc2_$ms
    echo "== $tag $(date '+%H:%M')"
    (cd "$ROOT" && "$CPUPY" bot/tools/versus.py --realtime --a "$PLAIN" --b "cc2:$ms" --mode coldclear --games 64 --jobs 4 \
        --max-seconds 120 --out "$(wpath "$R/$tag.jsonl")" --record "$(wpath "$REP")" --record-first 2 --tag "$tag" | tail -3)
done
echo "sweep done"
