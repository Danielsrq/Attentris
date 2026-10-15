#!/bin/sh
# AtTenTris search profile, depth vs width: v5.1-dagger3-ft, auto style, single player under the versus
# conditions (coldclear rules, no 180, 80/20 cheese garbage 10,3-4,cheese0.2,0.2), 16 games x 500 pieces,
# seeds 33001+, 8 games at a time (one physical P-core each) so ms per move are comparable.
# Depth 1; depth 2 / 3 / 4 at beam 8 / 16 / 32; depth 6 at beam 8 / 16 / 20. Resumable.
#   sh nn_search_profile.sh > nn_search_profile.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYC="$CPUPY"
D="$DIR/data/nn_search_profile"
WD=$(wpath "$D")
OUT="$DIR/results_nn_search_profile.txt"
mkdir -p "$D"
M=$(wpath "$DIR/data/model_v51_dagger3_ft.pt")
echo "== start $(date '+%H:%M')"
args=""
for cfg in 1:1 2:8 2:16 2:32 3:8 3:16 3:32 4:8 4:16 4:32 6:8 6:16 6:20; do
    d=${cfg%:*}; b=${cfg#*:}
    tag="d${d}b$b"
    f="$D/$tag.games"
    if ! grep -q "^AtTenTris" "$f" 2>/dev/null; then
        rm -f "$f"
        echo "$tag: $("$PYC" "$DIR/play.py" --model "$M" --style auto --depth $d --beam $b --games 16 --jobs 8 --pieces 500 --seed 33001 \
            --mode coldclear --no-180 --garbage 10,3-4,cheese0.2,0.2 --out "$WD/$tag.games" | grep "^AtTenTris" | sed 's/.*: APP/APP/' | cut -c1-220)" | tee -a "$OUT"
    fi
    args="$args $tag=$WD/$tag.games"
done
echo "== APP / VS by search setting (16 games each, 80/20 garbage, coldclear rules, no 180)" | tee -a "$OUT"
"$PYC" "$DIR/vs_score.py" $args | tee -a "$OUT"
echo "profile done $(date '+%H:%M')"
