#!/bin/sh
# Is CEM's ~5 ms win over CC2 5 ms speed or quality? Realtime, 64 matches each, Cold Clear 1 rules:
#   --min-ms 6: every piece takes at least 6 ms of game time for both players (both naturally ~5.1-5.3 ms),
#   so both play at the same ~167 PPS and only the moves differ.
# Plus 6 recorded matches at natural speed (replays only) for each pairing.
#   sh bot/tools/versus_5ms_matched.sh > bot/results/versus/realtime_5ms_matched.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../attentris"; . "$ATT/plat.sh"
ROOT=$(cd "$DIR/../.." && pwd)
R="$ROOT/bot/results/versus/realtime"
REP="$ROOT/data/bot/versus/realtime"
mkdir -p "$R" "$REP"
SURV=bot/results/survive/survive.txt
for cfg in 5:12 4:16; do
    t=d$(echo "$cfg" | tr ':' 'b')
    echo "== ${t} vs cc2:5, both at least 6 ms per piece $(date '+%H:%M')"
    (cd "$ROOT" && "$CPUPY" bot/tools/versus.py --realtime --min-ms 6 --a "cem:$SURV:$cfg" --b cc2:5 --mode coldclear --games 64 --jobs 4 \
        --max-seconds 120 --out "$(wpath "$R/plain_${t}_vs_cc2_5_min6.jsonl")" --record "$(wpath "$REP")" --record-first 6 \
        --tag "plain_${t}_vs_cc2_5_min6" | tail -3)
    echo "== ${t} vs cc2:5, natural speed, 6 recorded matches $(date '+%H:%M')"
    (cd "$ROOT" && "$CPUPY" bot/tools/versus.py --realtime --a "cem:$SURV:$cfg" --b cc2:5 --mode coldclear --games 6 --jobs 4 \
        --max-seconds 120 --out "$(wpath "$R/plain_${t}_vs_cc2_5_rec.jsonl")" --record "$(wpath "$REP")" \
        --tag "plain_${t}_vs_cc2_5_rec" | tail -3)
done
echo "matched-speed test done"
