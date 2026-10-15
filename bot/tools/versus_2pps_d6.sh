#!/bin/sh
# As versus_2pps.sh, but AtTenTris searches depth 6 beam 20 (~470 ms/move, the whole preview) instead of
# depth 4 beam 32; only its pairings (vs Cold Clear 2 at 500 ms, CEM, CEM-survival).
# Versus tournament at a 500 ms/move budget (2 pieces per second; bot/tools/versus.py, one piece each per
# turn): Cold Clear 2 thinks 500 ms per move; the CEM bots keep their own search (depth 6 beam 64, ~40 ms,
# speed capped by the turn); AtTenTris searches with its Qs beam to fill the budget: depth 6 beam 20
# (measured 468 ms/move at full load; beam 24 = 552 ms). Rules as versus_tournament.sh (coldclear, then
# tetrio; SRS, T-spins only, no 180). 64 matches per pairing, seeds 50001+, up to 1000 pieces each,
# 4 matches at a time (two physical P-cores each). CEM vs CEM-survival is unchanged (same as the fixed-time
# tournament), so it is not rerun. Resumable.   sh bot/tools/versus_2pps_d6.sh > bot/results/versus/2pps_d6.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../attentris"; . "$ATT/plat.sh"
ROOT="$DIR/../.."
PY="$CPUPY"
R="$ROOT/bot/results/versus"
WR=$(wpath "$R")
OUT="$R/2pps_d6.txt"
mkdir -p "$R"
CEM="cem:bot/weights/normal.txt:6:64"
SURV="cem:bot/results/survive/survive.txt:6:64"
NN="nn:attentris/data/model_v51_dagger3_ft.pt:auto:6:20"
CC2="cc2:500"
run() {   # tag a b mode
    f="$WR/2pps_d6_$1_$4.jsonl"
    n=$(cat "$f" 2>/dev/null | grep -c '"seed"')
    if [ "$n" -lt 64 ]; then
        "$PY" "$DIR/versus.py" --a "$2" --b "$3" --mode $4 --games 64 --pieces 1000 --jobs 4 --out "$f" > "$R/2pps_d6_$1_$4.log" 2>&1
    fi
    echo "== $1 ($4)" | tee -a "$OUT"
    tail -3 "$R/2pps_d6_$1_$4.log" | tee -a "$OUT"
}
echo "== start $(date '+%H:%M')"
for m in coldclear tetrio; do
    run nn_vs_cc2 "$NN" "$CC2" $m
    run nn_vs_cem "$NN" "$CEM" $m
    run nn_vs_surv "$NN" "$SURV" $m
done
echo "2pps d6 done $(date '+%H:%M')"
