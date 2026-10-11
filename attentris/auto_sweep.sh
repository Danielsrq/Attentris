#!/bin/sh
# --style auto rules for dagger3-ft at depth 1: when to switch to the danger style. r12 = the first
# rule (stack >= 12 or holes >= 6); r10, r8 earlier; r8s = r8 with hysteresis (stays in danger until
# stack < 5 and no holes). No garbage (256 games: the APP cost), 80/20 and 50/50 (64 games, seeds
# 33001+, per-game logs for VS). Resumable per run.
DIR=$(cd "$(dirname "$0")" && pwd)
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
D="$DIR/data/dagger3_eval"
WD=$(cygpath -m "$D")
OUT="$DIR/results_auto_sweep.txt"
M="$DIR/data/model_v51_dagger3_ft.pt"
mkdir -p "$D"
for r in "r10:--auto-height 10 --auto-holes 4" "r8:--auto-height 8 --auto-holes 3" \
         "r8s:--auto-height 8 --auto-holes 3 --auto-exit-height 5 --auto-exit-holes 0" "r12:--auto-height 12 --auto-holes 6"; do
    name=${r%%:*}; flags=${r#*:}
    for g in "none:-" "g20:10,3-4,cheese0.2,0.2" "g50:10,3-4,cheese0.5,0.2"; do
        tag=${g%%:*}; garb=${g#*:}
        f="$D/auto_${name}_$tag.games"
        [ -f "$f" ] && grep -q "^AtTenTris" "$f" && continue
        rm -f "$f"
        if [ "$garb" = "-" ]; then gs="--games 256"; else gs="--games 64 --seed 33001 --garbage $garb"; fi
        echo "$name $tag: $("$PYC" "$DIR/play.py" --model "$M" --depth 1 --style auto $flags $gs --out "$(cygpath -m "$f")" \
            | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-230)" | tee -a "$OUT"
    done
done
args=""
for name in r12 r10 r8 r8s; do for tag in none g20 g50; do args="$args auto_${name}_$tag=$WD/auto_${name}_$tag.games"; done; done
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "auto sweep done"
