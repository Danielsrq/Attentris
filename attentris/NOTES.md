# AtTenTris: handover notes (state on 2026-10-16)

Read this first in a new session. Machine setup and data sync are in `../SETUP.md`; the model and tokenizer in depth in `DESIGN.md`.
Results live in `results_*.txt` (one file per experiment, appended by the run scripts).

## What it is

AtTenTris is an explainable Tetris transformer. Its inputs are:
- **rule-based structure tokens** for the board: sky, clean wells, play rows, pits, piece-shaped slots (`fit_X`, open or covered), T-structures (TSD / TST / STSD / C-spin), setups (`setup_X`, one piece away), plus current / hold / preview piece tokens;
- a **FLAT-2D** attention bias (pairwise row / column span offsets and mask overlap, through a small MLP into every attention score);
- **every legal placement as a candidate token**, enumerated by the Rust move generator. The candidates cross-attend to the board tokens, and each gets a policy logit and a Qs value (the teacher's search score / 10).

So a move's cross-attention row, drawn on the board, is "what the model looked at for this move".
The network is distilled from search teachers (below), then improved with DAgger rounds on its own trouble positions.

**The user's goals:**
- explainability drives the project; the model that decides is the one explained;
- beat cms-RL (the friend's TetrisBattle RL bot);
- understand tokens, attention and FLAT-2D;
- learned tokenization (ToMe);
- better survival under garbage.

The user's stated main contributions: an explainable model that considers Tetris structures and 2D representations, and being small and efficient.

## Repo map

| Path | What |
|---|---|
| `attentris/tokenizer.py`, `model.py`, `train.py`, `play.py`, `pack.py` | tokenizer, model, training, play tests, jsonl -> npz packing |
| `attentris/model_tome.py`, `tome_*.py` | ToMe (learned token merging) variants, IoU / structure recovery |
| `attentris/*.sh` | one resumable run script per experiment; all source `plat.sh` (Windows / Linux) |
| `attentris/dagger_select.py`, `dagger5_route.py`, `split_by_seed.py` | DAgger: select trouble positions, route styles, train / val split by seed |
| `attentris/vs_score.py` | APP / VS tables from `.games` files |
| `attentris/data_bundle.py` | zip / sync `attentris/data` (gitignored, ~11 GB) via Google Drive |
| `bot/` (Rust) | game rules, CEM-tuned linear V + beam search, `bot dataset` / `label` / `expand` / `structures` / `play-ext` / `versus` / `tune` |
| `bot/weights/normal.txt` | teacher "CEM-depth6" (normal style); `bot/results/survive/survive.txt` = CEM-survival (danger style) |
| `bot/tools/` | versus harness (`versus.py`), Cold Clear 2 driver (`cc2_play.py`), replay viewer (`versus_watch.py`) |
| `data/bot/` | replays (`data/bot/versus/{fixed,2pps}` = 18 versus replays) |

## Architecture (current best: v5.1 at d = 128)

Details and the full diagram are in `DESIGN.md` ("Architecture as implemented").
- **Cells:** 200 cells, each `E_val + E_row + E_col`.
- **Region pooling:** one learned query per token kind over each token's cell mask.
- **Token embedding:** `+ E_kind + PE_box + W_extra·extras`.
- **Encoder:** 4 pre-norm self-attention blocks with the FLAT-2D bias computed once and shared, then 1 cross-attention block (candidates -> board tokens).
- **Heads:** policy and Qs (the Q1 head exists but has weight 0).
- **Style:** an embedding added to every candidate; 3 styles = `normal` / `gate4` / `danger`.
- **Sizes:** d = 64 has 274.5k params; **d = 128 has 1.06M params** (4 heads, 4 layers, FF 4d). Forward pass (pinned, 1 thread): d64 3.1 ms, d96 3.5 ms, d128 4.0 ms. A depth-1 move costs ~8 ms in total.
- **Tokenizer version v5:** open + covered piece slots, hang cells, `setup_X` tokens. `tok_version` is saved in every checkpoint.
- **Variants tried:**
  - v5.2: 2 cross passes, loop or untied; no gain.
  - ToMe v4.x (learned merges with streams): better structure recovery, weaker play.
  - Ablations: no setup tokens, no fit tokens, no T tokens (`results_setup_ablation.txt`, `results_fit_ablation.txt`).
  - Capacity: d64 / d96 / d128 (`results_capacity.txt`). d128 shows a val gap of 0.035, so memorisation is starting.

## Teachers and styles

| style tag | teacher | weights |
|---|---|---|
| `normal` | CEM-depth6 (CEM-tuned linear V, depth 6, beam 64; APP ~1.22) | `bot/weights/normal.txt` |
| `gate4` | "ambitious human" gate4 weights (base data only) | `bot/results/gate4/gate4.txt` |
| `danger` | CEM-survival (CEM tuned on VS fitness under 50/50 garbage) | `bot/results/survive/survive.txt` |

`play.py --style normal|gate4|danger|auto` picks the style row. `auto` uses danger on hot boards (stack >= 12 or >= 6 holes).
Labelling: `bot label --all-roots 1 --beam 128` gives a search score for every root move.

## Data (`attentris/data/v51/*.npz`, packed with tokenizer v5)

| file | what |
|---|---|
| `train.npz`, `train1m.npz`, `val.npz` | base teacher games: 1.09M positions (normal.txt + gate4), every move search-labelled |
| `dagger1.npz` | DAgger 1: v1's own states, matching teacher |
| `dagger2.npz`, `val_dagger2.npz` | DAgger 2 |
| `danger.npz`, `val_danger.npz` | DAgger 3: 95k trouble positions labelled by CEM-survival as `danger` (~75% hot / 25% calm, kept as a mix by the user's choice) |
| `dagger4.npz`, `val_dagger4.npz` | DAgger 4: 87k train, danger tag |
| `dagger5all.npz`, `val_dagger5all.npz` | DAgger 5 + extension, ~101k (see below) |

- **DAgger 5** was collected from d128 dagger3-ft: 400 + 400 games, depth 1, danger and normal styles, at 80/20 and 50/50; seeds 39001-44200.
  - Selection: `dagger_select.py`, 60-move death window, not lengthened ("more deaths, not longer windows").
  - All selected positions are labelled by CEM-survival as `danger`.
  - Routing (`dagger5_route.py`), normal-style games only: hot boards are also copied as `normal`; calm ones are labelled by normal.txt as `normal`.
- **Fine-tune recipe** (`dagger5b.sh`):
  - 3 epochs, lr 3e-4, batch 512, `--micro 2` at d128, `--w-qs 1 --w-dense 0 --w-q1 0 --n-teachers 3`, `--init` the previous model.
  - The mix is "70/30": the new data (danger + dagger4 + dagger5all) is used whole each epoch, while the old files (train, dagger1, train1m, dagger2) are drawn at a fraction `F = 7/3 · new / old` each (F = 0.6155 for dagger5b).
  - Val2 = the newest round's val file.
- Seeds are split by `split_by_seed.py`: every 10th seed is held out as val.

## Model lineage (checkpoints `attentris/data/model_<name>.pt`, with optimizer state: resumable)

`v51` (d64) -> `v51_dagger_ft` (DAgger 2) -> `v51_dagger3_ft`; the same lineage at `v51_d96*` and `v51_d128*`.
**Best: `v51_d128_dagger5b_ft`**: d128 base -> dagger_ft -> dagger3_ft -> dagger5b_ft (rounds 4 + 5 + extension).

Other checkpoints: `v51_d128_dagger5_ft`, ToMe `tome_v41*` / `tome_v42_{l4h,l4g,l6g}*`, `v52_{loop,untied}*`, ablations `v51_{nosetup,nofitx,not}*`.
`curve*.pt` are learning-curve checkpoints. train.py overwrites its `--out` file every 300 batches and at epoch end, atomically, and `--resume` continues from it.

## Benchmark protocol (how every model is tested)

`play.py` plays 500-piece games. The settings, all with fixed seeds so models are compared on identical games:

| name | flags | meaning |
|---|---|---|
| none | `--depth 1`, 256 games (headline) or 64 | no garbage; APP |
| 80/20 | `--seed 33001 --garbage 10,3-4,cheese0.2,0.2` | a garbage batch every 10 pieces, 3-4 rows, 20% of batches cheese (80% clean), 20% chance the hole column moves |
| 50/50 | `--seed 33001 --garbage 10,3-4,cheese0.5,0.2` | the same with half the batches cheese |
| d2 | `--depth 2 --beam 4`, 64 games | short search on no garbage |
| d3-d6 | `--depth N --beam 8` (`d3_tests.sh`, `MODEL=... DEPTH=N`) | only when the user asks (see rules) |

- Each garbage setting runs in both styles (`--style normal` and `--style danger`), 64 games each.
- **Metrics** (`vs_score.py`):
  - **APP** = attack per piece played;
  - **APPfull** = over the full 500-piece budget (a death scores 0 for the rest);
  - **VS / VSfull** = TETR.IO versus score without PPS, (lines sent + garbage cleared) / pieces;
  - **deaths** out of 64. Survival is strict: only 0/64 counts as surviving.
- Latency: ms/move from play.py (8 games at a time = one physical P-core each, for clean timing).
- The standard block is in `dagger5b.sh` (the `play()` helper): none 256 at depth 1, d2 64, then normal/danger × 80/20 and 50/50 at depth 1, then `vs_score.py`.

## Latest results (64 games unless noted)

**d128 lineage at depth 1** (deaths / 64; VSfull in brackets):

| model | APP none (256) | APP d2 | normal 80/20 | normal 50/50 | danger 80/20 | danger 50/50 |
|---|---|---|---|---|---|---|
| dagger3-ft | 0.724 | 0.827 | 32 | 55 | 9 | 43 |
| dagger5-ft | 0.739 | 0.856 | 27 | 59 | 6 | 24 |
| **dagger5b-ft** | **0.739** | 0.832 | **18** (0.921) | **44** (0.523) | **4** (1.049) | 25 (0.683) |
| teacher CEM-depth6 | 1.218 | | 18 | 33 | | |

The same dagger5b-ft numbers reproduce exactly on Linux (WSL Ubuntu 24.04).

**Depth sweep of d128 dagger5b-ft, beam 8** (deaths; danger 50/50 VSfull in brackets):

| depth | APP none | normal 80/20 | normal 50/50 | danger 80/20 | danger 50/50 | ms/move |
|---|---|---|---|---|---|---|
| 3 | 0.931 | 9 | 31 | 2 | 6 (0.918) | 127 |
| 4 | 0.995 | 15 | 23 | 0 | 2 (0.971) | 187 |
| 5 | 1.031 | 8 | 24 | 1 | 2 (0.995) | 240 |
| 6 | 1.044 | 11 | 20 | 1 | 3 (1.004) | 288 |

**Versus** (`bot/tools/versus.py`, 64 matches; coldclear and tetrio attack tables):
- Fixed time: CEM beats Cold Clear 2 43-21 / 39-25, CEM-survival beats CC2 56-8 / 56-8, NN depth 1 loses to CC2 9-55 / 12-52.
- At 2 PPS: NN d6 b20 vs CC2 17-47 / 23-41, and NN d4 b32 vs CC2 25-39 / 15-49.
- Cold Clear 2 is run unmodified over TBP (`bot/tools/cc2_play.py`, `--mode standard --no-180`).

## Working rules (from the user; keep them)

- **Commits:** plain messages, never a Claude co-author or attribution trailer.
- **Git:** the user pushes. GitHub `Danielsrq/Attentris` `main`; on the Windows machine that is the worktree branch `worktree-burst`, see SETUP.md.
- **Runs:**
  - Every run must be resumable and show progress.
  - Say how many games and roughly how long before starting a large run.
  - Never restart a reaped or killed job without the user's go-ahead.
- **CPU:** pin every compute run to the P-cores. Rust does it itself; Python calls `pcores.pin()`; `bot cpus` shows the detected cores. On a non-hybrid CPU nothing is restricted.
- **GPU:** only for AtTenTris training, and **never share it between two runs**. Play tests and inference run on the CPU.
- **Tests:**
  - Quick tests use **64 games per setting** (16 is too few to separate deaths); 256 only for headline no-garbage APP.
  - Routine play tests are depth 1, depth 1 with garbage, and depth 2 beam 4. **No depth 3+ tests** unless the user asks; they asked for the d3-d6 sweep and for 500 ms versus budgets.
- **Data:**
  - The danger DAgger data is a hot / calm mix by choice.
  - Plan targets (below) come from teacher games only.
- **Disk:** keep big scratch files off nearly full disks. On the Windows PC, C: is nearly full, so use D:.
- **Baselines:** use external bots' code as-is (CC2 unmodified) and match compute fairly (by time or depth / width).
- **Explainability first:** the explained model must be the one that decides, not a post-hoc evaluator.

## Next: the agreed plan for 2026-10-16 (not started)

Build two multi-move ("plan") variants of AtTenTris at **d = 128**, starting from `v51_d128_dagger5b_ft` (trained on base + DAgger 1-5). The question: does the network internalise search (several moves from one pass), and does predicting future moves improve move 1?

1. **attentris-autoreg (autoregressive decoder):**
   - Run the encoder once on board X.
   - Then one cross-attention step per move. Step k's queries are Rust's legal moves of position X + (k-1) moves, already in the data; the chosen previous move is added as a key / value token.
   - Always legal.
2. **attentris-plan (option C, coordinate head):**
   - One forward pass with step-k query tokens and a shared 1760-way coordinate head (hold 2 × rotation 4 × x 10 × y 22) with step embeddings.
   - A Rust legality check follows. If a move is illegal, fall back to a normal forward pass. The illegal rate is itself a metric.
3. **Plan length N** is a parameter: N = 2 first (end-to-end check), then N = 6 (current + 5 previews: the ceiling without chance).
4. **Data:**
   - Move-2..N targets come only from TEACHER-game consecutive positions (`train` + `train1m`), cut where garbage arrived (replay each move and compare boards).
   - DAgger positions' next move is the student's own, so their plan losses are masked.
   - The move-1 loss uses ALL data (teacher + DAgger 1-5), so nothing is forgotten.
5. **Metrics:**
   - plan consistency (predicted move k vs the move actually played later);
   - agreement with the teacher's next moves; teacher baseline = `game.rs` `plan_kept / plan_checks`;
   - the usual play tests for move 1;
   - later, viewer panels: step-1 / step-2 heatmaps, attention to the move-1 token, delta map.
6. **Estimates:**
   - ~1 working day to build both;
   - fine-tunes ~13-15 min (N = 2) and ~15-25 min (N = 6) on an 8 GB GPU;
   - optional from-scratch 8-epoch runs (~40-65 min each) vs the d128 base, as a control for the auxiliary-loss effect.
7. **Start with the sequence builder.** Don't relabel with stored teacher plans unless far-step targets prove too noisy (several CPU hours).

## Other open options (offered, not started; ask before starting any)

- NN dagger5b-ft at depth 5-6 in versus vs CC2 / CEM.
- The `auto` style test.
- DAgger 6, or oversampling normal-style hot positions.
- `fit_gap` check after the 62% main re-sampling.
- A full self-play CEM run (meta-level CEM, fitness = wins vs a frozen copy; smoke test in `bot/results/selfplay/`).
- MisaMino integration (needs a bridge).
- A guideline spawn rule for fairness vs CC2.
- A DAgger 4-only ablation.
- ToMe direction: merge network vs hybrid (the user is deciding).
