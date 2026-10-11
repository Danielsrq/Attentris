#!/bin/sh
# DAgger round 3, DATA COLLECTION ONLY: the students' trouble positions under 80/20 garbage, for
# CEM-survival (bot/results/survive/survive.txt) to label later as a "danger" style.
#   1. play: 5.1-dagger, 5.1-dagger-ft, ToMe v4 at depth 1 (greedy), 100 games each with garbage
#      10,3-4,cheese0.2,0.2 on fresh seeds 34001+ (round 2: 30001-32100; tests: 5001+, 33001+);
#      every state dumped with the student's move entropy
#   2. check: can CEM-survival save a sample of the deaths (takeover 60 / 30 / 10 moves before)?
#   3. size: dagger_select.py on the dumps, trouble positions only (no random share)
# Resumable per step.
DIR=$(cd "$(dirname "$0")" && pwd)
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
G=10,3-4,cheese0.2,0.2
D="$DIR/data/dagger3"
WD=$(cygpath -m "$D")
SURV=$(cygpath -m "$DIR/../bot/results/survive/survive.txt")
mkdir -p "$D"
echo "== dagger3 collection started $(date)"
for m in v51_dagger v51_dagger_ft tome_v4; do
    [ -f "$D/$m.games" ] && grep -q "^AtTenTris" "$D/$m.games" && { echo "$m: $(grep "^AtTenTris" "$D/$m.games")"; continue; }
    rm -f "$D/$m.games" "$D/$m.jsonl"
    echo "$m: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --games 100 --depth 1 --seed 34001 --garbage $G \
        --out "$WD/$m.games" --dump "$WD/$m.jsonl" | grep "^AtTenTris")"
done
[ -f "$D/takeover.txt" ] || "$PYC" "$DIR/takeover.py" --deaths "v51_dagger:$WD/v51_dagger.games" "v51_dagger_ft:$WD/v51_dagger_ft.games" \
    "tome_v4:$WD/tome_v4.games" --garbage $G --teacher-weights "$SURV" --sample 8 --before 60,30,10 --jobs 16 > "$D/takeover.part" \
    && mv "$D/takeover.part" "$D/takeover.txt"
sed -n '/^summary/,$p' "$D/takeover.txt"
[ -f "$D/selected.jsonl" ] || "$PYC" "$DIR/dagger_select.py" --games "$WD/*.games" --states "$WD/*.jsonl" --out "$WD/selected.jsonl" --random 0
echo "dagger3 collection done $(date)"
