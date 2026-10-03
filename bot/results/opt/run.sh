#!/bin/sh
# Landscape views + CEM vs CMA-ES, normal rules with fibonacci B2B. Every step resumes:
#   THREADS=14 sh run.sh          run or resume
#   powershell -File stop.ps1     stop now (the batch / generation in progress reruns)
DIR=$(cd "$(dirname "$0")" && pwd)
BOT="$DIR/../../target/release/bot.exe"
W="$DIR/../../weights"
CEMRUN="$DIR/../cem_norm_vs_raw"
T=${THREADS:-14}
# Fitness exactly as tuning measures it, but on 12 fresh seeded games for less noise.
LAND="--mode normal --games 12 --pieces 300 --beam 24 --depth 4 --threads $T --save $DIR"
TUNE="--mode normal --weights $W/standard.txt --gens 20 --pop 32 --elite 8 --games 6 --pieces 300 --beam 24 --depth 4 --threads $T"

echo "== view 1: slices around normal.txt"
"$BOT" landscape slices --weights "$W/normal.txt" $LAND
echo "== view 3: plane through start, CEM-normalised and CEM-raw results"
"$BOT" landscape plane --a "$W/standard.txt" --b "$CEMRUN/normalised.txt" --c "$CEMRUN/raw.txt" $LAND
echo "== CEM (logged)"
"$BOT" tune --algo cem $TUNE --save "$DIR/cem.txt"
echo "== CMA-ES (logged)"
"$BOT" tune --algo cma $TUNE --save "$DIR/cma.txt"
echo "== trajectories with fitness backgrounds"
"$BOT" landscape trajectory --run "$DIR/cem.txt" --grid 9 $LAND
"$BOT" landscape trajectory --run "$DIR/cma.txt" --grid 9 $LAND
echo "all done"
