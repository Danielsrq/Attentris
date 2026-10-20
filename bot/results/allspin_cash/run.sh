#!/bin/sh
# CEM-allspins-survival under the "singles build, doubles and triples cash" variant (--mode allspin_cash):
# all-spin rules (Minis promoted, repeat wounds), but a full spin single sends 1 and never earns the B2B
# bonus (it still extends the chain), a spin double sends 4 + bonus, a spin triple of any piece 9 + bonus.
# Tune: the CEM-survival recipe (fitness VS over the full 500 pieces, garbage every E=15 pieces of 3-4 rows,
# 50/50 clean/cheese), from allspin_survive_e15.txt (the all-spin survival weights; only the scoring changed).
# Evals under allspin_cash at beam 64 depth 6, 64 games: allspin_slots.txt and allspin_survive_e15.txt as
# they are (the search knows the new scoring, the weights don't) and the tuned allspin_cash_e15.txt; no
# garbage, 80/20 and 50/50 at E=15, seeds 33001+. Replays gzipped to attentris/data/allspin_cash/. Resumable:
#   THREADS=32 sh bot/results/allspin_cash/run.sh > bot/results/allspin_cash/run.out 2>&1
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR/../../../attentris"; . "$ATT/plat.sh"
BOT="$DIR/../../target/release/bot$EXE"
W="$DIR/../../weights"
REP="$ATT/data/allspin_cash"
MODE=allspin_cash
T=${THREADS:-32}
E=${E:-15}
G50="$E,3-4,cheese0.5,0.2"
G20="$E,3-4,cheese0.2,0.2"
SFX="_e$E"
SW="$DIR/allspin_cash$SFX.txt"
mkdir -p "$REP"

echo "== CEM-allspins-cash-survival: CEM under $MODE with garbage $G50, fitness VS, from allspin_survive_e15.txt $(date '+%H:%M')"
"$BOT" tune --algo cem --mode $MODE --garbage $G50 --fitness vs --weights "$(wpath "$DIR/../allspin_survive/allspin_survive_e15.txt")" \
    --gens 30 --pop 32 --elite 8 --games 12 --pieces 500 --beam 24 --depth 4 --threads $T --save "$(wpath "$SW")"
[ -f "$SW" ] || { echo "tune did not write $SW"; exit 1; }

# eval NAME WEIGHTS TAG GARBAGE(or -)
eval_run() {
    name=$1; wf=$2; tag=$3; garb=$4
    log="$DIR/eval_${name}_${tag}.jsonl"; rep="$REP/${name}_${tag}.jsonl"
    [ -f "$log" ] && [ -f "$rep.gz" ] && return
    gflag=""; [ "$garb" != "-" ] && gflag="--garbage $garb"
    echo "$name $tag $(date '+%H:%M')"
    "$BOT" play --mode $MODE $gflag --weights "$(wpath "$wf")" --games 64 --pieces 500 --seed 33001 --threads $T \
        --label "$name" --game-log "$(wpath "$log.part")" --out "$(wpath "$rep")" > "$log.out" 2>&1 \
        && mv "$log.part" "$log" && gzip -f "$rep" && sed -n 2p "$log.out" && rm -f "$log.out"
}
NAMES="allspin_slots allspin_survive_e15 allspin_cash$SFX"
wf_of() { case $1 in allspin_slots) echo "$W/$1.txt" ;; allspin_survive_e15) echo "$DIR/../allspin_survive/$1.txt" ;; *) echo "$DIR/$1.txt" ;; esac; }
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
