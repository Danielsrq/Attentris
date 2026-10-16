#!/bin/sh
# Blitz test (versus.py --realtime: uncapped speed, each player on its own clock). CEM-survival depth 6 beam 64
# plays its search plan's next 5 moves without searching (commit 6), so 6 pieces cost one ~35 ms search.
# Pairings, 64 matches each, Cold Clear 1 rules (Cold Clear 2's own table), game-time limit 120 s:
#   blitz vs CC2 40 ms   (about the CEM search time)
#   plain CEM-survival (commit 1) vs CC2 40 ms   (control: same engine without the blitz)
#   blitz vs CC2 10 ms / 5 ms   (CC2 playing fast too: it is an anytime search)
#   blitz vs plain CEM-survival   (same evaluator, only the blitz differs)
# The first 3 matches of each pairing are saved as replays (data/bot/versus/realtime/, versus_watch.py).
# Resumable (versus.py skips finished seeds).   sh bot/tools/versus_realtime.sh > versus_realtime.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../attentris"; . "$ATT/plat.sh"
ROOT=$(cd "$DIR/../.." && pwd)
R="$ROOT/bot/results/versus/realtime"
REP="$ROOT/data/bot/versus/realtime"
mkdir -p "$R" "$REP"
SURV=bot/results/survive/survive.txt
BLITZ="cem:$SURV:6:64:commit6"
PLAIN="cem:$SURV:6:64"
run() {   # tag a b
    tag=$1; a=$2; b=$3
    echo "== $tag $(date '+%H:%M')"
    (cd "$ROOT" && "$CPUPY" bot/tools/versus.py --realtime --a "$a" --b "$b" --mode coldclear --games 64 --jobs 4 \
        --max-seconds 120 --out "$(wpath "$R/$tag.jsonl")" --record "$(wpath "$REP")" --record-first 3 --tag "$tag" | tail -3)
}
run blitz_vs_cc2_40 "$BLITZ" cc2:40
run plain_vs_cc2_40 "$PLAIN" cc2:40
run blitz_vs_cc2_10 "$BLITZ" cc2:10
run blitz_vs_cc2_5 "$BLITZ" cc2:5
run blitz_vs_plain "$BLITZ" "$PLAIN"
echo "realtime tests done"
