#!/bin/sh
# DAgger round 2, DATA COLLECTION ONLY (no training): the students play, informative states are
# selected, the teacher labels them.
#   1. play: v5.1 and ToMe v4 at depth 1 on fresh seeds (30001+, not the test seeds 5001+ or the
#      dataset seeds), 100 games each of: greedy, sampled (temperature 1: more varied games), and
#      with garbage (10,3-4,cheese0.2,0.2: the high-stack boards where the models die); every state
#      dumped with the student's move entropy
#   2. select (dagger_select.py): states near deaths, danger boards, teacher-value drops, unsure
#      student moves, + 10% random; duplicates dropped
#   3. label: the normal teacher's search with a WIDER beam (128 instead of 64), every legal move
#      scored (--all-roots 1)
# Output: data/raw_dagger2/selected_labelled.jsonl (+ summary in dagger2.out). Resumable per step.
DIR=$(cd "$(dirname "$0")" && pwd)
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
BOT="$DIR/../bot/target/release/bot.exe"
D="$DIR/data/dagger2"
OUTD="$DIR/data/raw_dagger2"
mkdir -p "$D" "$OUTD"
until grep -q "v52 done" "$DIR/results_v52.txt" 2>/dev/null; do sleep 30; done
sleep 20
echo "== dagger2 collection started $(date)"
for m in v51 tome_v4; do
    for kind in greedy sampled garbage; do
        tag="${m}_${kind}"
        [ -f "$D/$tag.games" ] && grep -q "^AtTenTris" "$D/$tag.games" && continue
        rm -f "$D/$tag.games" "$D/$tag.jsonl"
        case $kind in
            greedy)  extra="--seed 30001" ;;
            sampled) extra="--seed 31001 --temp 1.0" ;;
            garbage) extra="--seed 32001 --garbage 10,3-4,cheese0.2,0.2" ;;
        esac
        echo "$tag: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --games 100 --depth 1 $extra --out "$D/$tag.games" --dump "$D/$tag.jsonl" | grep "^AtTenTris")"
    done
done
[ -f "$D/selected.jsonl" ] || "$PYC" "$DIR/dagger_select.py" --games "$D/*.games" --states "$D/*.jsonl" --out "$D/selected.jsonl"
if [ ! -f "$OUTD/selected_labelled.jsonl" ]; then
    "$BOT" label --mode normal --weights "$DIR/../bot/weights/normal.txt" --label normal --threads 16 --all-roots 1 --beam 128 \
        --out "$OUTD/selected_labelled.jsonl.part" < "$D/selected.jsonl" && mv "$OUTD/selected_labelled.jsonl.part" "$OUTD/selected_labelled.jsonl"
fi
echo "labelled: $(wc -l < "$OUTD/selected_labelled.jsonl") positions -> $OUTD/selected_labelled.jsonl"
echo "dagger2 collection done $(date)"
