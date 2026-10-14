#!/bin/sh
# d=128 vs v5.1 base models (before any fine-tune), 64 games each, normal style: depth 2 beam 4 without
# garbage (play.py's default seeds), and depth 1 at 80/20 cheese (seeds 33001+, 10,3-4,cheese0.2,0.2).
# Resumable.   sh capacity_d128_tests.sh > capacity_d128_tests.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
D="$DIR/data/dagger3_eval"
OUT="$DIR/results_capacity.txt"
play() {   # name model tag flags...
    f="$D/$1_$3.games"; m=$2; t=$3; shift 3
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    echo "$t $(basename $m): $("$PYC" "$DIR/play.py" --model "$(cygpath -m "$DIR/data/$m")" --style normal --games 64 "$@" --out "$(cygpath -m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
for n in v51:model_v51.pt v51_d128:model_v51_d128.pt; do
    name=${n%%:*}; m=${n#*:}
    play ${name}_base $m d2 --depth 2 --beam 4
    play ${name}_base $m g20 --depth 1 --seed 33001 --garbage 10,3-4,cheese0.2,0.2
done
WD=$(cygpath -m "$D")
echo "== base models, d=64 (v5.1) vs d=128: depth 2 no garbage, depth 1 at 80/20 (64 games)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" v51_base_d2=$WD/v51_base_d2.games v51_d128_base_d2=$WD/v51_d128_base_d2.games \
    v51_base_g20=$WD/v51_base_g20.games v51_d128_base_g20=$WD/v51_d128_base_g20.games | tee -a "$OUT"
echo "d128 tests done"
