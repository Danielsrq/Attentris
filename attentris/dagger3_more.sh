#!/bin/sh
# DAgger round 3, more collection: 100 more games per student at 80/20 (seeds 34101+; 34001-34100
# done in dagger3.sh) and 100 per student at 50/50 clean/cheese (seeds 35001+); then the trouble
# positions of all of them (dagger_select.py, no random share), split by seed (every 10th seed held
# out), labelled by CEM-survival (bot/results/survive/survive.txt) at beam 128 with every move
# scored, as teacher "danger". Resumable per step.
DIR=$(cd "$(dirname "$0")" && pwd)
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
BOT="$DIR/../bot/target/release/bot.exe"
D="$DIR/data/dagger3"
WD=$(cygpath -m "$D")
mkdir -p "$D"
echo "== dagger3 more collection started $(date)"
for m in v51_dagger v51_dagger_ft tome_v4; do
    for kind in g20b g50; do
        tag="${m}_$kind"
        case $kind in
            g20b) extra="--seed 34101 --garbage 10,3-4,cheese0.2,0.2" ;;
            g50)  extra="--seed 35001 --garbage 10,3-4,cheese0.5,0.2" ;;
        esac
        [ -f "$D/$tag.games" ] && grep -q "^AtTenTris" "$D/$tag.games" && { echo "$tag: $(grep "^AtTenTris" "$D/$tag.games")"; continue; }
        rm -f "$D/$tag.games" "$D/$tag.jsonl"
        echo "$tag: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --games 100 --depth 1 $extra \
            --out "$WD/$tag.games" --dump "$WD/$tag.jsonl" | grep "^AtTenTris")"
    done
done
[ -f "$D/selected_all.jsonl" ] || "$PYC" "$DIR/dagger_select.py" --games "$WD/*.games" --states "$WD/*.jsonl" --out "$WD/selected_all.jsonl" --random 0
mkdir -p "$D/raw_tr" "$D/raw_val"
if [ ! -f "$D/labelled.jsonl" ]; then
    "$BOT" label --mode normal --weights "$DIR/../bot/results/survive/survive.txt" --label danger --threads 16 --all-roots 1 --beam 128 \
        --out "$WD/labelled.jsonl.part" < "$D/selected_all.jsonl" && mv "$D/labelled.jsonl.part" "$D/labelled.jsonl"
fi
echo "labelled: $(wc -l < "$D/labelled.jsonl") positions"
echo "dagger3 more done $(date)"
