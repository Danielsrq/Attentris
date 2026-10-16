#!/bin/sh
# Plan labels for the multi-move variants (attentris-autoreg / attentris-plan): the teacher's
# planned line (depth 6: the move it plays and the 5 it expects to play next) for every
# training position, teacher games and DAgger alike, each searched by the teacher and beam
# that gave its move-1 label:
#   teacher games  raw (train + val), raw_1m (train1m)   normal.txt / gate4.txt, beam 64
#   DAgger 1       raw_dagger1                           normal.txt / gate4.txt, beam 64
#   DAgger 2       raw_dagger2_{tr,val}                  normal.txt, beam 128
#   DAgger 3       dagger3/raw_{tr,val} (danger)         CEM-survival, beam 128
#   DAgger 4       dagger4/raw_{tr,val}                  CEM-survival, beam 128
#   DAgger 5       dagger5all/raw_{tr,val}               CEM-survival; calm "normal" records
#                                                        normal.txt (plan_route.py), beam 128
# `bot label --plan`: one plain search per state (~46 ms at beam 64, ~78 ms at beam 128, one
# thread). One output file per input file, data/plan/<same path>/<file>, one record per input
# line ("i" = line). Written atomically, finished files are skipped: rerunning resumes.
# Progress every 30 s; plan_verify.py at the end (move 1 of the plan = the stored label?).
#   JOBS=64 sh attentris/plan_label.sh
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
BOT="$DIR/../bot/target/release/bot$EXE"
PY="$DIR/.venv/$VPY"
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 8)}
TW="normal=$(wpath "$DIR/../bot/weights/normal.txt"),gate4=$(wpath "$DIR/../bot/results/gate4/gate4.txt"),danger=$(wpath "$DIR/../bot/results/survive/survive.txt")"
export BOT PY TW DIR

# set (relative to data/)  beam  route
SETS="raw 64 -
raw_1m 64 -
raw_dagger1 64 -
raw_dagger2_tr 128 -
raw_dagger2_val 128 -
dagger3/raw_tr 128 -
dagger3/raw_val 128 -
dagger4/raw_tr 128 -
dagger4/raw_val 128 -
dagger5all/raw_tr 128 route
dagger5all/raw_val 128 route"

all=0
for s in $(echo "$SETS" | cut -d' ' -f1); do all=$((all + $(cat "$DIR/data/$s"/*.jsonl | wc -l))); done
echo "plan_label: $all states in $(echo "$SETS" | wc -l) sets, $JOBS jobs"
# one job list over every set, biggest files first (DAgger 1 is 2 files of ~12k states), so all
# jobs stay busy to the end: "OUT_DIR BEAM ROUTE FILE" per line
LIST="$DIR/data/plan/jobs.txt"
mkdir -p "$DIR/data/plan"
echo "$SETS" | while read -r set beam route; do
    mkdir -p "$DIR/data/plan/$set"
    rm -f "$DIR/data/plan/$set"/*.tmp
    for f in "$DIR/data/$set"/*.jsonl; do
        echo "$(wc -c < "$f") $DIR/data/plan/$set $beam $route $f"
    done
done | sort -rn | cut -d' ' -f2- > "$LIST"
total=$(wc -l < "$LIST")
done_files() { n=0; while read -r o b r f; do [ -f "$o/$(basename "$f")" ] && n=$((n + 1)); done < "$LIST"; echo $n; }
echo "$(done_files) / $total files done before"
t0=$(date +%s)
xargs -P "$JOBS" -L 1 sh "$DIR/plan_label_one.sh" < "$LIST" &
pid=$!
while kill -0 $pid 2>/dev/null; do
    sleep 30
    kill -0 $pid 2>/dev/null && echo "  $(done_files) / $total files, $(( $(date +%s) - t0 ))s"
done
wait $pid || { echo "plan_label: some files failed (rerun to resume)" >&2; exit 1; }
echo "plan_label: all sets done, $(( $(date +%s) - t0 ))s"
"$PY" "$DIR/plan_verify.py" $(echo "$SETS" | cut -d' ' -f1)
