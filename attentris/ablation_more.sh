#!/bin/sh
# 64 more games per garbage setting on fresh seeds (53001+) for the d128 ablation models and their baseline,
# so every comparison has 128 paired games (with the 33001+ games of ablation_d128.sh / dagger5b.sh):
# depth 1 and depth 2 beam 4, normal / danger x 80/20, 50/50. A model whose training is still running
# (ablation_d128.sh) is waited for. Then ablation_stats.py: paired exact McNemar per setting and pooled.
# 4 models x 8 settings x 64 = 2048 games, ~40-60 min CPU. Resumable per game file:
#   sh attentris/ablation_more.sh > attentris/ablation_more.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYC="$CPUPY"
DE="$DIR/data/dagger3_eval"
OUT="$DIR/results_ablation_d128.txt"
MODELS=${MODELS:-"v51_d128_dagger5b_ft v51_d128_noflat_dagger5b_ft v51_d128_nooverlay_dagger5b_ft v51_d128_nosetup_dagger5b_ft"}
G20="--seed 53001 --garbage 10,3-4,cheese0.2,0.2"
G50="--seed 53001 --garbage 10,3-4,cheese0.5,0.2"
m() { wpath "$1"; }
play() {   # model style tag flags...
    f="$DE/$1_$2_$3_s53.games"; md=$1; s=$2; t=$3; shift 3
    [ -f "$f" ] && grep -q "^AtTenTris" "$f" && return
    rm -f "$f"
    echo "$md $s $t (seeds 53001+): $("$PYC" "$DIR/play.py" --model "$DIR/data/model_$md.pt" --style $s --games 64 "$@" --out "$(m "$f")" \
        | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-200)" | tee -a "$OUT"
}
echo "== 64 more games per garbage setting, seeds 53001+ ($MODELS) $(date '+%H:%M')" | tee -a "$OUT"
for md in $MODELS; do
    # wait for a model still in training (its last stage's log reports the final epoch)
    log="$DIR/train_$md.out"
    until grep -q "^epoch 3: train" "$log" 2>/dev/null; do sleep 60; done
    for s in normal danger; do
        play $md $s g20 --depth 1 $G20
        play $md $s g50 --depth 1 $G50
        play $md $s d2g20 --depth 2 --beam 4 $G20
        play $md $s d2g50 --depth 2 --beam 4 $G50
    done
done
"$PYC" "$DIR/ablation_stats.py" $MODELS | tee -a "$OUT"
echo "ablation more done"
