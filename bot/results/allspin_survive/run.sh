#!/bin/sh
# CEM-allspins-survival: the CEM-survival recipe (results/survive/run.sh) under all-spin rules.
# Rules (--mode allspin): every spin counts as full (Minis promoted), non-T spins score like T-spins
# (2 / 4 / 6 + B2B), and two consecutive clears of the same type (piece + spin + lines; plain clears
# keyed by lines only) insert an unclearable wound line at the bottom for 10 clears.
# Garbage every 10 pieces of 3-4 rows, half the batches cheese; fitness VS = (attack + garbage cleared)
# over the full 500-piece budget. From allspin_slots.txt (tuned on clean boards, dies to cheese).
# No --freeze: `--freeze default` (as CEM-survival) zeroes the all-spin features (wounds, ready_repeat,
# the S/Z, L/J, I spin slots, eval::FROZEN_OUTSIDE_ALLSPIN), which carry the all-spin play.
# Then allspin.txt / allspin_slots.txt / allspin_survive.txt at beam 64 depth 6, 64 games each with no
# garbage, 80/20 and 50/50 clean/cheese on seeds 33001+ (the tune plays 1000-3950), and spin / B2B /
# survival stats (tools/spin_stats.py). Replays are gzipped to attentris/data/allspin_survive/.
# E = pieces between garbage batches (default 10, as CEM-survival; E=15 is milder): the tune writes
# allspin_survive.txt (E=10) or allspin_survive_e<E>.txt, the evals use the same E.
# Resumable (the tune per generation, the evals per file):
#   THREADS=32 sh bot/results/allspin_survive/run.sh > bot/results/allspin_survive/run.out 2>&1
#   E=15 THREADS=32 sh bot/results/allspin_survive/run.sh > bot/results/allspin_survive/run_e15.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../../attentris"; . "$ATT/plat.sh"
BOT="$DIR/../../target/release/bot$EXE"
W="$DIR/../../weights"
REP="$ATT/data/allspin_survive"
T=${THREADS:-32}
E=${E:-10}
G50="$E,3-4,cheese0.5,0.2"
G20="$E,3-4,cheese0.2,0.2"
SFX=""; [ "$E" != 10 ] && SFX="_e$E"
SW="$DIR/allspin_survive$SFX.txt"
mkdir -p "$REP"

echo "== CEM-allspins-survival: CEM under all-spin rules with garbage $G50, fitness VS, from allspin_slots.txt $(date '+%H:%M')"
"$BOT" tune --algo cem --mode allspin --garbage $G50 --fitness vs --weights "$(wpath "$W/allspin_slots.txt")" \
    --gens 30 --pop 32 --elite 8 --games 12 --pieces 500 --beam 24 --depth 4 --threads $T --save "$(wpath "$SW")"
[ -f "$SW" ] || { echo "tune did not write $SW"; exit 1; }

# eval NAME WEIGHTS TAG GARBAGE(or -)
eval_run() {
    name=$1; wf=$2; tag=$3; garb=$4
    log="$DIR/eval_${name}_${tag}.jsonl"; rep="$REP/${name}_${tag}.jsonl"
    [ -f "$log" ] && [ -f "$rep.gz" ] && return
    gflag=""; [ "$garb" != "-" ] && gflag="--garbage $garb"
    echo "$name $tag $(date '+%H:%M')"
    "$BOT" play --mode allspin $gflag --weights "$(wpath "$wf")" --games 64 --pieces 500 --seed 33001 --threads $T \
        --label "$name" --game-log "$(wpath "$log.part")" --out "$(wpath "$rep")" > "$log.out" 2>&1 \
        && mv "$log.part" "$log" && gzip -f "$rep" && sed -n 2p "$log.out" && rm -f "$log.out"
}
NAMES="allspin allspin_slots allspin_survive"; [ -n "$SFX" ] && NAMES="$NAMES allspin_survive$SFX"
wf_of() { case $1 in allspin|allspin_slots) echo "$W/$1.txt" ;; *) echo "$DIR/$1.txt" ;; esac; }
TAGS="none cheese20$SFX cheese50$SFX"
for tag in $TAGS; do
    case $tag in none) g=- ;; cheese20*) g=$G20 ;; cheese50*) g=$G50 ;; esac
    for name in $NAMES; do eval_run $name "$(wf_of $name)" $tag $g; done
done
args=""
for tag in $TAGS; do
    for name in $NAMES; do
        args="$args ${name}_$tag=$(wpath "$REP/${name}_$tag.jsonl.gz"),$(wpath "$DIR/eval_${name}_$tag.jsonl")"
    done
done
"$CPUPY" "$DIR/../../tools/spin_stats.py" $args | tee "$DIR/eval$SFX.txt"
echo "all done"
