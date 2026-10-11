#!/bin/sh
# DAgger round 3 follow-up: (1) the garbage play tests again with per-game logs (garbage cleared, for
# VS = (attack + garbage cleared) / pieces), dagger-ft vs dagger3-ft in style normal / danger / auto,
# 80/20 and 50/50, 64 games each on seeds 33001+ (the same games as results_dagger3_train.txt);
# (2) every death of dagger3-ft --style auto: can CEM-survival take over 30 / 10 moves before it and
# survive? Resumable per step.
DIR=$(cd "$(dirname "$0")" && pwd)
PYC="$DIR/../external/tetrisbattle/.venv/Scripts/python.exe"
D="$DIR/data/dagger3_eval"
WD=$(cygpath -m "$D")
mkdir -p "$D"
for g in "g20:10,3-4,cheese0.2,0.2" "g50:10,3-4,cheese0.5,0.2"; do
    tag=${g%%:*}; garb=${g#*:}
    for ms in "v51_dagger_ft:normal" "v51_dagger3_ft:normal" "v51_dagger3_ft:danger" "v51_dagger3_ft:auto"; do
        m=${ms%%:*}; s=${ms#*:}
        f="$D/${m}_${s}_$tag.games"
        [ -f "$f" ] && grep -q "^AtTenTris" "$f" && continue
        rm -f "$f"
        "$PYC" "$DIR/play.py" --model "$DIR/data/model_$m.pt" --games 64 --depth 1 --seed 33001 --garbage $garb --style $s \
            --out "$(cygpath -m "$f")" | grep "^AtTenTris" | cut -c1-120
    done
done
"$PYC" "$DIR/vs_score.py" \
    "dagger-ft, 80/20=$WD/v51_dagger_ft_normal_g20.games" "dagger3-ft normal, 80/20=$WD/v51_dagger3_ft_normal_g20.games" \
    "dagger3-ft danger, 80/20=$WD/v51_dagger3_ft_danger_g20.games" "dagger3-ft auto, 80/20=$WD/v51_dagger3_ft_auto_g20.games" \
    "dagger-ft, 50/50=$WD/v51_dagger_ft_normal_g50.games" "dagger3-ft normal, 50/50=$WD/v51_dagger3_ft_normal_g50.games" \
    "dagger3-ft danger, 50/50=$WD/v51_dagger3_ft_danger_g50.games" "dagger3-ft auto, 50/50=$WD/v51_dagger3_ft_auto_g50.games" \
    | tee "$DIR/results_dagger3_vs.txt"
SURV=$(cygpath -m "$DIR/../bot/results/survive/survive.txt")
for g in "g20:10,3-4,cheese0.2,0.2" "g50:10,3-4,cheese0.5,0.2"; do
    tag=${g%%:*}; garb=${g#*:}
    out="$D/takeover_auto_$tag.txt"
    [ -f "$out" ] || "$PYC" "$DIR/takeover.py" --deaths "v51_dagger3_ft:$WD/v51_dagger3_ft_auto_$tag.games" --style auto \
        --garbage $garb --teacher-weights "$SURV" --before 30,10 --no-alone --jobs 8 > "$out.part" 2> "$out.err" && mv "$out.part" "$out"
    echo "== takeover of dagger3-ft auto deaths, $tag" | tee -a "$DIR/results_dagger3_vs.txt"
    sed -n '/^summary/,$p' "$out" | tee -a "$DIR/results_dagger3_vs.txt"
    grep "DIED" "$out" | tee -a "$DIR/results_dagger3_vs.txt"
done
echo "dagger3 vs done"
