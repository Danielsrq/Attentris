#!/bin/sh
# --style auto, holes kept at >= 6 (T-spin setups' covered cells are not discounted in the Python
# hole count, so low hole thresholds trip on ordinary STSD / imperial building); only the height
# trigger moves: h10 (stack >= 10), h8 (>= 8), h8s (>= 8, stays in danger until stack < 5 and
# holes <= 2). Same tests as auto_sweep.sh, but 64 no-garbage games (a quick test). Waits for auto_sweep.sh (the CPU is shared).
DIR=$(cd "$(dirname "$0")" && pwd)
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
D="$DIR/data/dagger3_eval"
WD=$(cygpath -m "$D")
OUT="$DIR/results_auto_sweep.txt"
M="$DIR/data/model_v51_dagger3_ft.pt"
until grep -q "auto sweep done" "$DIR/auto_sweep.out" 2>/dev/null; do sleep 15; done
for r in "h10:--auto-height 10 --auto-holes 6" "h8:--auto-height 8 --auto-holes 6" \
         "h8s:--auto-height 8 --auto-holes 6 --auto-exit-height 5 --auto-exit-holes 2"; do
    name=${r%%:*}; flags=${r#*:}
    for g in "none:-" "g20:10,3-4,cheese0.2,0.2" "g50:10,3-4,cheese0.5,0.2"; do
        tag=${g%%:*}; garb=${g#*:}
        f="$D/auto_${name}_$tag.games"
        [ -f "$f" ] && grep -q "^AtTenTris" "$f" && continue
        rm -f "$f"
        if [ "$garb" = "-" ]; then gs="--games 64"; else gs="--games 64 --seed 33001 --garbage $garb"; fi
        echo "$name $tag: $("$PYC" "$DIR/play.py" --model "$M" --depth 1 --style auto $flags $gs --out "$(cygpath -m "$f")" \
            | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-230)" | tee -a "$OUT"
    done
done
args=""
for name in r12 r10 r8 r8s h10 h8 h8s; do for tag in none g20 g50; do args="$args auto_${name}_$tag=$WD/auto_${name}_$tag.games"; done; done
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "auto sweep 2 done"
