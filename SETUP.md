# Attentris repo: push from this worktree, train on another machine (Windows or Ubuntu)

The **Attentris** GitHub repo is this worktree's branch (`worktree-burst`) published as `main`. The rest of the tetris checkout (`C:\Users\Admin\Documents\tetris`, its `main` branch and uncommitted files) is never pushed.
Git holds the code, results tables, weights (`bot/weights`, `bot/results`) and replays (`data/bot`), about 230 MB with nothing over 25 MB, so no LFS is needed. The Rust bot is part of it and is needed for training: tokenizer, labelling, play tests.
Git does not hold `attentris/data/` (models and datasets, about 11 GB), the venvs, build output or `external/`. The data goes to Google Drive as zips. Everything else is rebuilt.

## 1. Publish this worktree as Attentris (once, Windows)

Create an empty **private** repo `Attentris` on GitHub (no README or licence). Then in PowerShell, from this worktree (`C:\Users\Admin\Documents\tetris\.claude\worktrees\burst`):

```powershell
git remote add attentris https://github.com/<you>/Attentris.git
git config remote.attentris.push refs/heads/worktree-burst:refs/heads/main   # plain "git push" maps worktree-burst -> main
git push -u attentris worktree-burst:main
```

After that, in this worktree only: `git push` sends new commits, and `git pull` brings in commits made on the other machine.
The main checkout is unaffected; it has no upstream and pushes nothing unless you ask it to.
Push soon: this worktree lives under `.claude\worktrees\`, and once the commits are on GitHub they are safe even if the worktree folder is removed.

## 2. Back up the data to Google Drive (Windows)

`attentris/data_bundle.py` zips `attentris/data/` in three tiers:

| tier | what | size | needed for |
|---|---|---|---|
| `core` | `model_*.pt`, packed v5.1 datasets (`v51/`), small eval dirs | 1.4 GB | playing, fine-tuning, DAgger rounds |
| `labels` | labelled jsonl (`raw*`, `dagger2..5all`, state dumps) | 7.3 GB | re-tokenising / re-packing |
| `old` | v3 / v5 datasets, `structure_study`, `audit`, scaling-curve checkpoints | 2.3 GB | history only |

```powershell
.\attentris\.venv\Scripts\python attentris\data_bundle.py list
.\attentris\.venv\Scripts\python attentris\data_bundle.py pack --tier all --out bundle    # bundle\ is gitignored
.\attentris\.venv\Scripts\python attentris\data_bundle.py verify bundle\attentris_data_core.zip bundle\attentris_data_labels.zip bundle\attentris_data_old.zip
```

Upload the zips in `bundle\` to a Drive folder. The free tier's 15 GB holds them; `core` alone is enough to keep working.
After training new models on either machine, re-pack and re-upload `core`. Models are not in git.

## 3. Ubuntu machine

```bash
sudo apt install build-essential python3-venv git curl          # C linker for Rust, venv support
curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain 1.94.1 && . ~/.cargo/env

git clone https://github.com/<you>/Attentris.git && cd Attentris
(cd bot && cargo build --release)                               # bot/target/release/bot

python3 -m venv attentris/.venv
attentris/.venv/bin/python -m pip install -r attentris/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# data: download the zips from Drive into bundle/, then
attentris/.venv/bin/python attentris/data_bundle.py unpack "bundle/attentris_data_*.zip"

sh attentris/plat_check.sh      # platform, CPU pinning, torch + CUDA, 2 games with d128 dagger5b-ft
```

On Linux the run scripts (`sh attentris/<script>.sh`) work unchanged:
- `attentris/plat.sh` picks `bot` vs `bot.exe` and `.venv/bin/python` vs `.venv/Scripts/python.exe`.
- The play tests use the AtTenTris venv, since there is no TetrisBattle copy on that machine.
- **CPU pinning detects the processor.** On a hybrid Intel CPU (12th gen or later), Linux lists the P-cores in `/sys/devices/cpu_core/cpus`, and runs are pinned to them. On anything else (AMD, older Intel, a VM) every core counts as equal and nothing is restricted. `bot/target/release/bot cpus` shows what it found.
- **GPU training (any NVIDIA card):** `train.py` uses CUDA when torch sees a GPU (`--device cuda`, as the run scripts pass). Play tests and inference always run on the CPU (`map_location="cpu"`), so a play test never shares the GPU with a training run.
  - Needs the NVIDIA driver: `nvidia-smi` should list the card (`sudo ubuntu-drivers install` if not). The CUDA runtime ships inside the torch wheel; no CUDA toolkit is needed.
  - The cu126 wheel covers GTX 10-series up to RTX 40-series. For an RTX 50-series card, install from `.../whl/cu128` instead.
  - VRAM: d=128 runs use `--micro 2` to fit 8 GB. On a smaller card raise `--micro` (4, 8); the gradients are the same, it just takes longer. `attentris/vram_probe.py` measures it.
  - `sh attentris/plat_check.sh` prints the GPU name, VRAM and a test matmul.

The versus / Cold Clear 2 / TetrisBattle tools are Windows-side extras, not needed for training.
If they are wanted on Linux: `git clone https://github.com/MinusKelvin/cold-clear-2 external/cold-clear-2 && (cd external/cold-clear-2 && cargo build --release)`. The TetrisBattle copy (`external/tetrisbattle`, including `best.pt`) is the friend's and is not ours to publish.

## 4. A fresh Windows machine

Python 3.10, Rust (cargo 1.94) and Git. In PowerShell, start local programs with `.\`: without it, PowerShell reads `external\...` as a module name ("the module external could not be loaded"). The `.sh` scripts run in Git Bash.

```powershell
git clone https://github.com/<you>/Attentris.git; cd Attentris
cd bot; cargo build --release; cd ..
python -m venv attentris\.venv
.\attentris\.venv\Scripts\python -m pip install -r attentris\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
.\attentris\.venv\Scripts\python attentris\data_bundle.py unpack "bundle\attentris_data_*.zip"
```

Optional, for play / versus as on the original machine: copy the TetrisBattle folder to `external\tetrisbattle`, then:

```powershell
python -m venv external\tetrisbattle\.venv
.\external\tetrisbattle\.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\external\tetrisbattle\.venv\Scripts\python -m pip install numpy==2.2.6 pygame==2.6.1 gymnasium==1.4.0 pytest
git clone https://github.com/MinusKelvin/cold-clear-2 external\cold-clear-2
cd external\cold-clear-2; cargo build --release; cd ..\..
```

Check: `sh attentris/plat_check.sh` in Git Bash. It takes about 5 s for 2 games, and the d128 dagger5b-ft model scores about 0.74 APP over 256 games.
