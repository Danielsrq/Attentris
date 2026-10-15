#!/bin/sh
# One DAgger round (Ross, Gordon & Bagnell 2011): the student plays, the teacher labels the
# positions the student reached, and those labels join the training data.
#   1. collect: the current model plays GAMES games per style (normal / gate4) under the
#      training conditions (normal rules, garbage 15,4,0,0.2), every state dumped
#   2. label: each state gets the matching teacher's full search (bot label), written as
#      ordinary dataset records -- every legal move with its 1-ply and search scores
#   3. pack: data/<ROUND>.npz, train-only (validation stays the original teacher games)
#   ROUND=dagger1 MODEL=attentris/data/model.pt GAMES=150 sh attentris/dagger.sh
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
PYCPU="$CPUPY"
BOT="$DIR/../bot/target/release/bot$EXE"
ROUND=${ROUND:-dagger1}
MODEL=${MODEL:-$DIR/data/model.pt}
GAMES=${GAMES:-150}
G="15,4,0,0.2"
RAW="$DIR/data/raw_$ROUND"
mkdir -p "$RAW"
for style in normal gate4; do
    seed=$([ $style = normal ] && echo 7001 || echo 8001)
    st="$DIR/data/${ROUND}_states_$style.jsonl"
    [ -f "$st" ] || "$PYCPU" "$DIR/play.py" --model "$MODEL" --games $GAMES --seed $seed --depth 1 --style $style \
        --garbage $G --dump "$st"
    w=$([ $style = normal ] && echo "$DIR/../bot/weights/normal.txt" || echo "$DIR/../bot/results/gate4/gate4.txt")
    out="$RAW/${style}_dagger_s1.jsonl"
    [ -f "$out" ] || "$BOT" label --mode normal --weights "$w" --label $style --threads 16 --out "$out" < "$st"
done
"$PYCPU" "$DIR/pack.py" --only "$RAW" --name "$ROUND"
echo "round $ROUND done"
