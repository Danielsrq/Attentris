#!/bin/sh
# Replays for bot/tools/versus_watch.py: 3 matches each (coldclear rules, no 180, seeds 50001-50003, as the
# tournaments) of CEM vs CEM-survival, CEM vs Cold Clear 2 and CEM-survival vs Cold Clear 2, at the fixed time
# budget (Cold Clear 2 time-matched to its opponent, ~35-40 ms: the CEM bots' natural pace) and at 2 PPS
# (Cold Clear 2 500 ms per move; replays paced at 500 ms per piece). Into data/bot/versus/{fixed,2pps}.
#   sh bot/tools/versus_replays.sh
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT="$DIR/../.."
PY="$ROOT/external/tetrisbattle/.venv/Scripts/python.exe"
R="$ROOT/data/bot/versus"
WR=$(cygpath -m "$R")
CEM="cem:bot/weights/normal.txt:6:64"
SURV="cem:bot/results/survive/survive.txt:6:64"
mkdir -p "$R/fixed" "$R/2pps"
rec() {   # tag a b budget dir
    [ -f "$R/$5/$1.jsonl" ] && [ "$(grep -c '"seed"' "$R/$5/$1.jsonl")" -ge 3 ] && return
    "$PY" "$DIR/versus.py" --a "$2" --b "$3" --mode coldclear --games 3 --seed 50001 --pieces 1000 --jobs 3 \
        --out "$WR/$5/$1.jsonl" --record "$WR/$5" --tag "$1" --budget-ms $4 | tail -3
}
rec cem_vs_surv "$CEM" "$SURV" 0 fixed
rec cem_vs_cc2 "$CEM" cc2:match 0 fixed
rec surv_vs_cc2 "$SURV" cc2:match 0 fixed
rec cem_vs_surv "$CEM" "$SURV" 500 2pps
rec cem_vs_cc2 "$CEM" cc2:500 500 2pps
rec surv_vs_cc2 "$SURV" cc2:500 500 2pps
ls "$R/fixed" "$R/2pps"
