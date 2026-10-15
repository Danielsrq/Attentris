#!/bin/sh
# DAgger round 4, DATA COLLECTION + LABELLING ONLY (PLAN_2026-10-12.md step 4).
#   sh dagger4.sh collect MODEL...   each student plays 100 games at 80/20 (seeds 36001+) and 100 at
#                                    50/50 (seeds 37001+) at depth 1 with the danger tag, every state
#                                    dumped (fresh seeds: rounds 2-3 used 30001-35100, tests 5001+ / 33001+)
#   sh dagger4.sh label              select the trouble positions of every collected game (death /
#                                    danger / holes rising / unsure; no random share), label them with
#                                    CEM-survival at beam 128 (all moves), teacher "danger"; takeover check
# Resumable per student and step.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYC="$CPUPY"
BOT="$DIR/../bot/target/release/bot$EXE"
D="$DIR/data/dagger4"
WD=$(wpath "$D")
SURV="$DIR/../bot/results/survive/survive.txt"
OUT="$DIR/results_dagger4_collect.txt"
mkdir -p "$D/games"
cmd=$1; shift
if [ "$cmd" = collect ]; then
    for m in "$@"; do
        for g in "g20:36001:10,3-4,cheese0.2,0.2" "g50:37001:10,3-4,cheese0.5,0.2"; do
            tag=$(echo $g | cut -d: -f1); seed=$(echo $g | cut -d: -f2); garb=$(echo $g | cut -d: -f3)
            f="$D/games/${m}_$tag"
            [ -f "$f.games" ] && grep -q "^AtTenTris" "$f.games" && { echo "$m $tag: $(grep '^AtTenTris' "$f.games" | sed 's/.*: APP/APP/' | cut -c1-110)"; continue; }
            rm -f "$f.games" "$f.jsonl"
            echo "$m $tag: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --games 100 --depth 1 --style danger --seed $seed \
                --garbage $garb --out "$WD/games/${m}_$tag.games" --dump "$WD/games/${m}_$tag.jsonl" | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-140)" | tee -a "$OUT"
        done
    done
elif [ "$cmd" = label ]; then
    [ -f "$D/selected.jsonl" ] || "$PYC" "$DIR/dagger_select.py" --games "$WD/games/*.games" --states "$WD/games/*.jsonl" \
        --out "$WD/selected.jsonl" --random 0 | tee -a "$OUT"
    if [ ! -f "$D/labelled.jsonl" ]; then
        "$BOT" label --mode normal --weights "$SURV" --label danger --threads 16 --all-roots 1 --beam 128 \
            --out "$WD/labelled.jsonl.part" < "$D/selected.jsonl" > "$D/label.log" 2>&1 && mv "$D/labelled.jsonl.part" "$D/labelled.jsonl"
    fi
    echo "labelled: $(wc -l < "$D/labelled.jsonl") positions -> $D/labelled.jsonl" | tee -a "$OUT"
    # takeover check: can CEM-survival save a sample of each student's 80/20 deaths (30 / 10 moves before)?
    for f in "$D"/games/*_g20.games; do
        m=$(basename "$f" _g20.games)
        t="$D/takeover_$m.txt"
        [ -f "$t" ] || "$PYC" "$DIR/takeover.py" --deaths "$m:$(wpath "$f")" --style danger --garbage 10,3-4,cheese0.2,0.2 \
            --teacher-weights "$(wpath "$SURV")" --sample 8 --before 30,10 --no-alone --jobs 8 > "$t.part" 2> "$t.err" && mv "$t.part" "$t"
        echo "takeover, $m 80/20:" | tee -a "$OUT"; sed -n '/^summary/,$p' "$t" | tail -n +2 | tee -a "$OUT"
    done
    echo "dagger4 label done"
fi
