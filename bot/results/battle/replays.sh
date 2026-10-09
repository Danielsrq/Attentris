#!/bin/sh
# Re-run three head-to-head configurations with move recording, for bot/tools/battle_watch.py.
# (The match_*.jsonl files from matches.sh predate move recording; turn mode replays the same
# games, realtime ones differ slightly because thinking time is measured live.)
DIR=$(cd "$(dirname "$0")" && pwd -W)
ROOT="$DIR/../../.."
TB="$ROOT/external/tetrisbattle"
PY="$TB/.venv/Scripts/python.exe"
EXE="$ROOT/bot/target/release/bot.exe"
OUT="$DIR/replays"
mkdir -p "$OUT"
export SDL_VIDEODRIVER=dummy PYGAME_HIDE_SUPPORT_PROMPT=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH="$TB"

rec() {
    name=$1; weights=$2; speed=$3; extra=$4
    tag=$(echo "$speed" | tr ':' '_')
    out="$OUT/${name}_${tag}"
    if [ ! -f "$out.txt" ]; then
        (cd "$TB" && "$PY" "$ROOT/bot/tools/battle_match.py" --games 20 --pieces 500 --speed "$speed" --jobs 8 \
            --exe "$EXE" --weights "$weights" --name "$name" --bot-args "$extra" --out "$out.jsonl") > "$out.part" 2>&1 \
            && mv "$out.part" "$out.txt"
    fi
    echo "== $name, $speed"; tail -3 "$out.txt"
}
rec battle "$DIR/battle.txt" realtime ""
rec battle_d1 "$DIR/battle.txt" realtime "--depth 1 --no-chance"
rec normal_txt "$ROOT/bot/weights/normal.txt" turn ""
rec battle "$DIR/battle.txt" cap:2 ""
echo "all done"
