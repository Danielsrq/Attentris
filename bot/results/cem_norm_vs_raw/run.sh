#!/bin/sh
# Controlled comparison: the same CEM tune twice under normal rules + fibonacci B2B,
# once with normalised features and once with raw features. Same start (standard.txt),
# same seeds, same settings; only the feature scaling differs.
#
#   THREADS=10 sh run.sh          run or resume (each tune resumes per generation)
#   powershell -File stop.ps1     stop now (the generation in progress reruns)
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
W="$DIR/../../weights"
T=${THREADS:-10}
COMMON="--mode normal --weights $W/standard.txt --gens 20 --pop 32 --elite 8 --games 6 --pieces 300 --beam 24 --depth 4 --threads $T"

echo "== normalised features"
"$BOT" tune $COMMON --save "$DIR/normalised.txt"
echo "== raw features"
"$BOT" tune $COMMON --raw-features --save "$DIR/raw.txt"
echo "comparison complete"
