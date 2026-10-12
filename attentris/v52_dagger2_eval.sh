#!/bin/sh
# The DAgger 2 stage of each lineage (v5.1 dagger-ft, v5.2-loop / v5.2-untied dagger-ft; normal style
# only -- they have no danger style), 64 games: depth 1 and depth 2 beam 4 without garbage, depth 1 at
# 80/20 and 50/50. Then one APP / VS table: both stages x the three architectures. Waits for v52_dagger.sh.
DIR=$(cd "$(dirname "$0")" && pwd)
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
D="$DIR/data/dagger3_eval"
WD=$(cygpath -m "$D")
OUT="$DIR/results_v52_dagger.txt"
until grep -q "v52 dagger done" "$DIR/v52_dagger.out" 2>/dev/null; do sleep 30; done
play() {   # model tag flags...
    m=$1; tag=$2; shift 2
    f="$D/${m}_normal_${tag}64.games"
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$m normal ${tag}64: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --games 64 --style normal "$@" --out "$(cygpath -m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-190)" | tee -a "$OUT"
}
for m in v51_dagger_ft v52_loop_dagger_ft v52_untied_dagger_ft; do
    play $m none --depth 1
    play $m d2 --depth 2 --beam 4
    play $m g20 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
    play $m g50 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.5,0.2
done
args=""
for m in v51_dagger_ft v52_loop_dagger_ft v52_untied_dagger_ft; do
    for t in none d2 g20 g50; do args="$args ${m}_${t}=$WD/${m}_normal_${t}64.games"; done
done
for m in v51_dagger3_ft v52_loop_dagger3_ft v52_untied_dagger3_ft; do
    for x in normal_none normal_d2 normal_g20 normal_g50 danger_g20 danger_g50; do args="$args ${m}_$x=$WD/${m}_$x.games"; done
done
echo "== APP / VS, DAgger 2 stage (normal style) and DAgger 3 stage (normal + danger)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "v52 dagger2 eval done"
