#!/bin/sh
# Relabel every position with a search score for EVERY legal move (bot label --all-roots 1):
# the normal search still picks the teacher's move; a second, root-diverse search keeps the
# best line of every first move. Original games -> data/raw_v3, DAgger-1 -> data/raw_dagger1_v3;
# then packed into data/v3/{train,val,dagger1}.npz. Resumable: finished files are skipped.
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYCPU="$CPUPY"
BOT="$DIR/../bot/target/release/bot$EXE"
W_normal="$DIR/../bot/weights/normal.txt"
W_gate4="$DIR/../bot/results/gate4/gate4.txt"
relabel() {
    src=$1; dst=$2
    mkdir -p "$dst"
    for f in "$src"/*.jsonl; do
        name=$(basename "$f")
        out="$dst/$name"
        [ -f "$out" ] && continue
        case $name in normal_*) w=$W_normal; t=normal ;; gate4_*) w=$W_gate4; t=gate4 ;; *) continue ;; esac
        "$BOT" label --mode normal --weights "$w" --label $t --threads 16 --all-roots 1 --out "$out.part" < "$f" 2>/dev/null \
            && mv "$out.part" "$out" && echo "relabelled $name"
    done
}
relabel "$DIR/data/raw" "$DIR/data/raw_v3"
relabel "$DIR/data/raw_dagger1" "$DIR/data/raw_dagger1_v3"
mkdir -p "$DIR/data/v3"
"$PYCPU" "$DIR/pack.py" --raw "$DIR/data/raw_v3" --out "$DIR/data/v3"
"$PYCPU" "$DIR/pack.py" --only "$DIR/data/raw_dagger1_v3" --name dagger1 --out "$DIR/data/v3"
echo "relabel done"
