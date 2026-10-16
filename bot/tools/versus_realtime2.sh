#!/bin/sh
# Follow-ups to versus_realtime.sh (64 matches each, realtime, Cold Clear 1 rules):
#   cheaper blitz searches vs CC2 5 ms (the fast CC2 beat the d6 b64 blitz 61-3): d6 b16 and d4 b16, commit 6
#   garbage delay 333 ms (TETR.IO-like ~20 frames before incoming garbage can land): d6 b64 blitz vs CC2 40 ms / 5 ms
# Resumable.   sh bot/tools/versus_realtime2.sh > bot/results/versus/realtime_run2.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../attentris"; . "$ATT/plat.sh"
ROOT=$(cd "$DIR/../.." && pwd)
R="$ROOT/bot/results/versus/realtime"
REP="$ROOT/data/bot/versus/realtime"
mkdir -p "$R" "$REP"
SURV=bot/results/survive/survive.txt
run() {   # tag a b extra-flags...
    tag=$1; a=$2; b=$3; shift 3
    echo "== $tag $(date '+%H:%M')"
    (cd "$ROOT" && "$CPUPY" bot/tools/versus.py --realtime --a "$a" --b "$b" --mode coldclear --games 64 --jobs 4 \
        --max-seconds 120 --out "$(wpath "$R/$tag.jsonl")" --record "$(wpath "$REP")" --record-first 3 --tag "$tag" "$@" | tail -3)
}
run blitz_d6b16_vs_cc2_5 "cem:$SURV:6:16:commit6" cc2:5
run blitz_d4b16_vs_cc2_5 "cem:$SURV:4:16:commit6" cc2:5
run blitz_vs_cc2_40_delay333 "cem:$SURV:6:64:commit6" cc2:40 --garbage-delay-ms 333
run blitz_vs_cc2_5_delay333 "cem:$SURV:6:64:commit6" cc2:5 --garbage-delay-ms 333
echo "realtime follow-ups done"
