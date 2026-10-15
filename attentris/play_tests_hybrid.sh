#!/bin/sh
# Is "AtTenTris shortlist + teacher search" better than the teacher alone? Same code path for
# both (bot play-ext --shortlist; k = 99 = every move = the teacher itself), 64 games x 500
# pieces, seeds 5001-5064, normal rules, no garbage and stress. Then the shortlist with the
# beam widened (the search budget spent on fewer first moves).
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PY="$CPUPY"
OUT=${OUT:-$DIR/results_play_hybrid.txt}
MODEL=${MODEL:-$DIR/data/model_v3.pt}
GAMES=${GAMES:-64}
STRESS="10,3-4,cheese0.2,0.2"
: > "$OUT"
for g in "" "$STRESS"; do
    gflag=""; [ -n "$g" ] && gflag="--garbage $g"
    echo "== garbage: ${g:-none}" | tee -a "$OUT"
    for k in 99 10 5; do
        "$PY" "$DIR/play.py" --model "$MODEL" --games $GAMES --shortlist $k $gflag --out "$OUT.games" | grep "^AtTenTris" | tee -a "$OUT"
    done
done
echo "all done" | tee -a "$OUT"
