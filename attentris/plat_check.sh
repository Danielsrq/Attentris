#!/bin/sh
# Smoke test of the platform layer (Windows Git Bash or Linux): plat.sh variables, P-core pinning,
# the Rust bot, and 2 depth-1 games with the given model.   sh attentris/plat_check.sh [MODEL.pt]
DIR=$(cd "$(dirname "$0")" && pwd)
ATT="$DIR"; . "$ATT/plat.sh"
M=${1:-$DIR/data/model_v51_d128_dagger5b_ft.pt}
echo "EXE='$EXE' VPY=$VPY CPUPY=$CPUPY"
"$DIR/../bot/target/release/bot$EXE" cpus | tail -1
"$DIR/.venv/$VPY" -c "import sys; sys.path.insert(0, '$(wpath "$DIR")'); import pcores, torch; print('pcores:', pcores.pin()); print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"
"$CPUPY" "$DIR/play.py" --model "$(wpath "$M")" --games 2 --depth 1 --style normal | grep "^AtTenTris" | cut -c1-150
