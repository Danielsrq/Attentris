#!/bin/sh
# Fixed-time versus tournament (bot/tools/versus.py): one piece each per turn, Cold Clear 2 thinking as
# long per move as its opponent (cc2:match). Rules Cold Clear is tuned for: SRS, T-spins only, no 180;
# coldclear (Cold Clear 1's attack table + guideline combos) and tetrio (TETR.IO table + combo multiplier).
# 64 matches per pairing and rule set, seeds 50001+, up to 1000 pieces per player, 4 matches at a time.
#   CEM       normal.txt, depth 6 beam 64 (the AtTenTris teacher)
#   SURV      CEM-survival, depth 6 beam 64
#   NN        AtTenTris v5.1-dagger3-ft, depth 1, auto style (danger on hot boards)
#   CC2       Cold Clear 2, time-matched
# Resumable.   sh bot/tools/versus_tournament.sh > bot/results/versus/tournament.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../attentris"; . "$ATT/plat.sh"
ROOT="$DIR/../.."
PY="$CPUPY"
R="$ROOT/bot/results/versus"
WR=$(wpath "$R")
OUT="$R/tournament.txt"
mkdir -p "$R"
CEM="cem:bot/weights/normal.txt:6:64"
SURV="cem:bot/results/survive/survive.txt:6:64"
NN="nn:attentris/data/model_v51_dagger3_ft.pt:auto"
CC2="cc2:match"
run() {   # tag a b mode
    f="$WR/$1_$4.jsonl"
    n=$(cat "$f" 2>/dev/null | grep -c '"seed"')
    if [ "$n" -lt 64 ]; then
        "$PY" "$DIR/versus.py" --a "$2" --b "$3" --mode $4 --games 64 --pieces 1000 --jobs 4 --out "$f" > "$R/$1_$4.log" 2>&1
    fi
    echo "== $1 ($4)" | tee -a "$OUT"
    tail -3 "$R/$1_$4.log" | tee -a "$OUT"
}
echo "== start $(date '+%H:%M')"
for m in coldclear tetrio; do
    run cem_vs_cc2 "$CEM" "$CC2" $m
    run surv_vs_cc2 "$SURV" "$CC2" $m
    run nn_vs_cc2 "$NN" "$CC2" $m
    run cem_vs_surv "$CEM" "$SURV" $m
    run nn_vs_cem "$NN" "$CEM" $m
    run nn_vs_surv "$NN" "$SURV" $m
done
echo "tournament done $(date '+%H:%M')"
