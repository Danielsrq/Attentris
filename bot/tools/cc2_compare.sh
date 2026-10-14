#!/bin/sh
# Quick test: Cold Clear 2 (unmodified, TBP, 1 worker thread) vs our CEM bots, solo games on the same
# seeds under Cold Clear 2's rules: --mode standard --no-180 (SRS, T-spins only, no 180), scored by our
# table. 64 games x 500 pieces per setting, 8 games at a time (one physical P-core each), no garbage /
# 80-20 / 50-50 cheese (10,3-4,cheese0.2|0.5,0.2), seeds 33001+.
#   Cold Clear 2 at fixed think times per move (requested 20 / 100 / 500 ms; the measured time is reported)
#   CEM (normal.txt) at depth 1 beam 1 / 2 x 8 / 4 x 32 / 6 x 64, measured ms per piece; CEM-survival d6 b64
#   under garbage
# Waits for fit_ablation.sh to finish (CPU-heavy play tests would distort the timings). Resumable.
#   sh bot/tools/cc2_compare.sh > bot/results/cc2/compare.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ROOT="$DIR/../.."
BOT="$ROOT/bot/target/release/bot.exe"
PY="$ROOT/attentris/.venv/Scripts/python.exe"
R="$ROOT/bot/results/cc2"
WR=$(cygpath -m "$R")
OUT="$R/compare.txt"
mkdir -p "$R"
while [ -d "$ROOT/attentris/.fit_ablation.lock" ]; do sleep 30; done
echo "== start $(date '+%H:%M')"
G20="10,3-4,cheese0.2,0.2"
G50="10,3-4,cheese0.5,0.2"
garb() { case $1 in none) echo "";; g20) echo "$G20";; g50) echo "$G50";; esac; }
cc2() {   # tag think_ms
    ga=$(garb $1); f="$WR/cc2_${2}ms_$1.jsonl"
    n=$(cat "$f" 2>/dev/null | grep -c '"done"')
    if [ "$n" -lt 64 ]; then
        "$PY" "$DIR/cc2_play.py" --games 64 --seed 33001 --think-ms $2 --jobs 8 --out "$f" ${ga:+--garbage $ga} | tail -1 | tee -a "$OUT"
    fi
}
cem() {   # tag name weights depth beam
    ga=$(garb $1); f="$WR/$2_$1.jsonl"
    if [ ! -f "$f" ]; then
        "$BOT" play --mode standard --no-180 --weights "$3" --depth $4 --beam $5 --games 64 --pieces 500 --seed 33001 \
            --threads 8 --label "$2" ${ga:+--garbage $ga} --game-log "$f.part" > "$R/$2_$1.log" 2>&1 && mv "$f.part" "$f"
        echo "$2 $1: $(grep -A1 '^config' "$R/$2_$1.log" | tail -1)" | tee -a "$OUT"
    fi
}
WN=$(cygpath -m "$ROOT/bot/weights/normal.txt")
WS=$(cygpath -m "$ROOT/bot/results/survive/survive.txt")
for t in none g20 g50; do
    cem $t cem_d1b1 "$WN" 1 1
    cem $t cem_d2b8 "$WN" 2 8
    cem $t cem_d4b32 "$WN" 4 32
    cem $t cem_d6b64 "$WN" 6 64
    [ $t != none ] && cem $t survive_d6b64 "$WS" 6 64
done
for t in none g20 g50; do
    cc2 $t 20
    cc2 $t 100
done
cc2 none 500
echo "== APP / VS (same seeds; ms per move: CEM from the logs' ms/pc, Cold Clear 2 measured think time)" | tee -a "$OUT"
args=""
for t in none g20 g50; do
    for n in cem_d1b1 cem_d2b8 cem_d4b32 cem_d6b64 survive_d6b64; do [ -f "$R/${n}_$t.jsonl" ] && args="$args ${n}_$t=$WR/${n}_$t.jsonl"; done
    for ms in 20 100 500; do [ -f "$R/cc2_${ms}ms_$t.jsonl" ] && args="$args cc2_${ms}ms_$t=$WR/cc2_${ms}ms_$t.jsonl"; done
done
"$PY" "$ROOT/attentris/vs_score.py" $args | tee -a "$OUT"
"$PY" - "$R" <<'EOF' | tee -a "$OUT"
import glob, json, os, sys
for f in sorted(glob.glob(os.path.join(sys.argv[1], "cc2_*.jsonl"))):
    g = [json.loads(l) for l in open(f)]
    n = len(g)
    print("%-22s think %.1f ms/move, %.0f nodes/move, illegal %d, no-move resigns %d, restarts %.1f/game" % (
        os.path.basename(f)[:-6], sum(r["cc2_think_ms"] for r in g) / n, sum(r["cc2_nodes_per_move"] for r in g) / n,
        sum(r.get("illegal", 0) for r in g), sum(r.get("cc2_no_move", 0) for r in g), sum(r["restarts"] for r in g) / n))
EOF
echo "cc2 compare done $(date '+%H:%M')"
