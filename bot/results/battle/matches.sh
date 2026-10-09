#!/bin/sh
# Head-to-head: his RL bot (best.pt) vs ours in his simulator (bot/tools/battle_match.py).
# 20 games x 500 pieces per player per configuration; results in match_<name>_<speed>.jsonl.
# Resumable: finished configurations are skipped.
DIR=$(cd "$(dirname "$0")" && pwd -W)
ROOT="$DIR/../../.."
TB="$ROOT/external/tetrisbattle"
PY="$TB/.venv/Scripts/python.exe"
EXE="$ROOT/bot/target/release/bot.exe"
JOBS=${JOBS:-8}
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$TB"

match() {
    name=$1; weights=$2; speed=$3; extra=$4
    tag=$(echo "$speed" | tr ':' '_')
    out="$DIR/match_${name}_${tag}"
    if [ ! -f "$out.txt" ]; then
        (cd "$TB" && "$PY" "$ROOT/bot/tools/battle_match.py" --games 20 --pieces 500 --speed "$speed" --jobs $JOBS \
            --exe "$EXE" --weights "$weights" --name "$name" --bot-args "$extra" --out "$out.jsonl") > "$out.part" 2>&1 \
            && mv "$out.part" "$out.txt"
    fi
    echo "== $name, $speed"; tail -4 "$out.txt"
}
match battle "$DIR/battle.txt" turn ""
match battle "$DIR/battle.txt" realtime ""
match battle "$DIR/battle.txt" cap:2 ""
match battle_d1 "$DIR/battle.txt" turn "--depth 1 --no-chance"
match battle_d1 "$DIR/battle.txt" realtime "--depth 1 --no-chance"
match normal_txt "$ROOT/bot/weights/normal.txt" turn ""
echo "all done"
