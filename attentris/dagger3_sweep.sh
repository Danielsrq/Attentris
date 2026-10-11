#!/bin/sh
# DAgger round 3, schedule sweep: the same 70/30 fine-tune as dagger3-ft (from dagger-ft, style table
# grown to 3) with a higher peak lr and/or more epochs: A 3e-4 x 6, B 6e-4 x 3, C 6e-4 x 6
# (dagger3-ft = 3e-4 x 3). Then at depth 1: danger style at 80/20 and 50/50 (64 games, seeds
# 33001+, per-game logs for VS), normal style with no garbage (256 games: APP / TSD / quads must hold).
DIR=$(cd "$(dirname "$0")" && pwd)
PYG="$DIR/.venv/Scripts/python.exe"
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
V51="$DIR/data/v51"
D="$DIR/data/dagger3_eval"
WD=$(cygpath -m "$D")
OUT="$DIR/results_dagger3_sweep.txt"
F=0.1712      # existing files per epoch: 70/30 with the 86,195 danger positions (dagger3_train.sh)
mkdir -p "$D"
for v in "A 3e-4 6" "B 6e-4 3" "C 6e-4 6"; do
    set -- $v; name=v51_dagger3_ft_$1; lr=$2; ep=$3
    log="$DIR/train_$name.out"
    if ! grep -q "^epoch $ep: val2" "$log" 2>/dev/null; then
        R=""; [ -f "$DIR/data/model_$name.pt" ] && R="--resume"
        "$PYG" "$DIR/train.py" --data "$V51" --batch 512 --device cuda --w-dense 0 --w-q1 0 --w-qs 1 --seed 0 --val2 val_danger.npz \
            --train train.npz@$F,dagger1.npz@$F,train1m.npz@$F,dagger2.npz@$F,danger.npz --epochs $ep --lr $lr --n-teachers 3 \
            --init "$DIR/data/model_v51_dagger_ft.pt" --out "$DIR/data/model_$name.pt" $R >> "$log" 2>&1
    fi
    echo "== $name (lr $lr, $ep epochs)" | tee -a "$OUT"
    grep -E "^epoch [0-9]+:" "$log" | sed 's/argmax Qs.*Qs [0-9.]* |/|/' | tee -a "$OUT"
done
for v in A B C; do
    name=v51_dagger3_ft_$v
    for g in "g20:10,3-4,cheese0.2,0.2" "g50:10,3-4,cheese0.5,0.2"; do
        tag=${g%%:*}; garb=${g#*:}
        f="$D/${name}_danger_$tag.games"
        [ -f "$f" ] && grep -q "^AtTenTris" "$f" || { rm -f "$f"; "$PYC" "$DIR/play.py" --model "$DIR/data/model_$name.pt" --games 64 --depth 1 \
            --seed 33001 --garbage $garb --style danger --out "$(cygpath -m "$f")" > /dev/null; }
    done
    echo "$name normal, no garbage: $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$name.pt" --games 256 --depth 1 | grep "^AtTenTris" | cut -c1-150)" | tee -a "$OUT"
done
args=""
for v in "" _A _B _C; do
    for tag in g20 g50; do args="$args dagger3-ft$v danger $tag=$WD/v51_dagger3_ft${v}_danger_$tag.games"; done
done
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "sweep done"
