# Moving the project to another machine

Git holds the code, results tables, weights (`bot/weights`, `bot/results`) and replays (`data/bot`), about 230 MB with nothing over 25 MB, so a plain GitHub repo works (no LFS).
Git does not hold `attentris/data/` (models and datasets, about 11 GB), the venvs, build output or `external/`. The data goes to Google Drive as zips. Everything else is rebuilt.

Commands are for PowerShell. Start local programs with `.\`: without it, PowerShell reads `external\...` as a module name ("the module external could not be loaded").

## 1. Push the code (once)

1. Create an empty **private** repo on GitHub (no README or licence, so the first push is clean).
2. From the main checkout (`C:\Users\Admin\Documents\tetris`, branch `main`):

```powershell
git merge --ff-only worktree-burst          # bring main up to date first
git remote add origin https://github.com/<you>/tetris.git
git push -u origin main
```

Only committed files are pushed. Untracked logs (`attentris\*.out`) and viewer pages (`attentris\*_viz.html`, `attn_*.html`) stay local; commit any you want to keep.
After that, `git push` each time you merge.

## 2. Back up the data to Google Drive

`attentris\data_bundle.py` zips `attentris\data\` in three tiers:

| tier | what | size | needed for |
|---|---|---|---|
| `core` | `model_*.pt`, packed v5.1 datasets (`v51\`), small eval dirs | 1.4 GB | playing, fine-tuning, DAgger rounds |
| `labels` | labelled jsonl (`raw*`, `dagger2..5all`, state dumps) | 7.3 GB | re-tokenising / re-packing |
| `old` | v3 / v5 datasets, `structure_study`, `audit`, scaling-curve checkpoints | 2.3 GB | history only |

```powershell
.\attentris\.venv\Scripts\python attentris\data_bundle.py list
.\attentris\.venv\Scripts\python attentris\data_bundle.py pack --tier all --out bundle    # bundle\ is gitignored
.\attentris\.venv\Scripts\python attentris\data_bundle.py verify bundle\attentris_data_core.zip bundle\attentris_data_labels.zip bundle\attentris_data_old.zip
```

Upload the zips in `bundle\` to a Drive folder. The free tier's 15 GB holds them; `core` alone is enough to keep working.
Re-pack and re-upload `core` after new models.
The friend's TetrisBattle copy (`external\tetrisbattle\`, including `best.pt`) is not ours to publish. Get it from them, or copy it over privately.

## 3. Set up the new machine

Windows. Needs Python 3.10, Rust (cargo 1.94) and Git; Godot 4 is only for the game UI.

```powershell
git clone https://github.com/<you>/tetris.git; cd tetris

# Rust bot (scripts and attentris\pcores.py call bot\target\release\bot.exe)
cd bot; cargo build --release; cd ..

# AtTenTris GPU venv (training)
python -m venv attentris\.venv
.\attentris\.venv\Scripts\python -m pip install -r attentris\requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126

# data: put the zips from Drive in bundle\, then (a glob works here)
.\attentris\.venv\Scripts\python attentris\data_bundle.py unpack "bundle\attentris_data_*.zip"

# TetrisBattle copy + its CPU venv (the play / versus scripts run play.py with this python:
# CPU torch, pygame for versus_watch.py). Copy the folder to external\tetrisbattle, then
python -m venv external\tetrisbattle\.venv
.\external\tetrisbattle\.venv\Scripts\python -m pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
.\external\tetrisbattle\.venv\Scripts\python -m pip install numpy==2.2.6 pygame==2.6.1 gymnasium==1.4.0 pytest

# Cold Clear 2, unmodified (versus / cc2 comparisons only)
git clone https://github.com/MinusKelvin/cold-clear-2 external\cold-clear-2
cd external\cold-clear-2; cargo build --release; cd ..\..
```

Check that the restore worked (4 games, about 5 s; APP about 0.78 on these 4, about 0.74 over 256 games):

```powershell
.\external\tetrisbattle\.venv\Scripts\python attentris\play.py --model attentris\data\model_v51_d128_dagger5b_ft.pt --style normal --games 4 --depth 1
```

The `.sh` scripts run in Git Bash (`sh attentris/d3_tests.sh`), where paths use forward slashes.
`.\bot\target\release\bot.exe cpus` should list the new CPU's performance cores; runs are pinned to them (`attentris\pcores.py`, `bot\src\cpu.rs`).
On a CPU without hybrid cores, pinning does nothing.
