#!/bin/sh
# ToMe v4 blob viewer with both merge schedules (trained 0,0.17,0.17,0.17 and 0,0.17,0.25,0.25 at
# inference), the viewer's 48 positions + 24 random held-out ones, plus 30 states from the depth-1
# games at the heavier schedule (sample_states.py: 30 and 10 moves before each of the 9 deaths, 12
# random), labelled by the normal teacher (bot label) and packed. CPU only.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYG="$DIR/.venv/$VPY"
D=$(wpath "$DIR/data/tome_m25")
cat > "$D/play.json" <<'EOF'
[{"value": "0.582 (6/64 died) → 0.506 (9/64)", "label": "APP at depth 1: merge .17 · .17 · .17 (trained) → .17 · .25 · .25"},
 {"value": "0.720 (0/64) → 0.656 (4/64)", "label": "APP at depth 2 beam 4"}]
EOF
"$PYG" "$DIR/tome_viz.py" --model "$DIR/data/model_tome_v4.pt" --random 24 --src "$DIR/attn_v51.html" --out "$DIR/tome_v4_viz.html" \
    --merges "0,0.17,0.17,0.17;0,0.17,0.25,0.25" --games "$D/games.npz" --games-merge 0,0.17,0.25,0.25 \
    --games-notes "$D/picked_notes.json" --play "$D/play.json" --val-eval \
    --label "AtTenTris-ToMe v4 · 2 merging streams · sky + solid blocks + cells · 8 epochs"
