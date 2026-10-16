# AtTenTris: handover notes (state at the end of 2026-10-16, Ubuntu machine)

Read this first in a new session. The 2026-10-16 work (multi-move "plan" models, plan labels for all data,
CEM commit-K) is in "Multi-move (plan) models" below, with its results and the next tests for the Windows side. Machine setup and data sync are in `../SETUP.md`; the model and tokenizer in depth in `DESIGN.md`.
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
| `attentris/model_plan.py` | multi-move AtTenTris: variants `autoreg`, `plan`, `pieces` (subclass of the v5.1 model) |
| `attentris/plan_label.sh`, `plan_label_one.sh`, `plan_route.py`, `plan_verify.py` | the teacher's planned line (depth 6) for every training position (`bot label --plan`) |
| `attentris/seq_build.py`, `pack_plan.py` | consecutive teacher positions (cut at garbage); plan targets aligned with the packed sets |
| `attentris/plan_ft.sh`, `plan_scratch.sh`, `plan_modes.sh`, `plan_consistency_cem.py` | plan fine-tunes, from-scratch runs, play modes, the CEM plan-consistency table |
| `attentris/data_bundle.py` | zip / sync `attentris/data` (gitignored, ~11 GB) via Google Drive |
| `bot/` (Rust) | game rules, CEM-tuned linear V + beam search, `bot dataset` / `label` (`--plan`) / `expand` / `plan-cands` / `structures` / `play` (`--commit K`) / `play-ext` / `versus` / `tune` |
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
  - v5.2: 2 cross passes, untied (`--cross-layers 2`, +199k params, 1.26M at d128) or loop (`--cross-loops 2`, the same block twice, no new params). At d64: no gain. **At d128 through the full lineage, untied beats v5.1** (see "v5.2 at d = 128" under Latest results); loop is in between.
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

**Plan data (2026-10-16):**

| where | what | in the bundle |
|---|---|---|
| `data/plan/<raw dir>/*.jsonl` | the teacher's planned line (up to 6 moves) for all 1,537,655 positions, one record per raw line (`"i"`) | labels tier (372 MB) |
| `data/v51/<set>.plan.npz` | `plan_len`, `coord` (1760-way per step), `piece_step` / `piece_place` (pieces variant) | core |
| `data/v51/{train,train1m,val}.seq.npz` | consecutive teacher positions (`nxt`), the move actually played (`played`) | core |
| `data/v51plan/<set>.pc.npy` | autoreg: legal moves at plan steps 2-6 in the root's frame, 9.1 GB | **not bundled**: rebuild with `python attentris/pack_plan.py` (~4 min) |

- Plans come from the teacher and search settings that gave each set's move-1 labels: teacher games and DAgger 1 at beam 64 (`normal.txt` / `gate4.txt`); DAgger 2-5 at beam 128 (`normal.txt` for dagger2, CEM-survival for danger / dagger4 / dagger5all, except calm `normal`-tagged DAgger 5 records: `normal.txt`, `plan_route.py`).
- Check (`plan_verify.py`): plan move 1 = the stored label on 100% of DAgger positions; 95% on the teacher games. The in-game search knew the bag order, `bot label` does not. So move-1 targets stay the stored `chosen`, and plan steps 2.. are used only where plan move 1 = `chosen` (`plan_len`).
- Data quirk: `train.npz` / `val.npz` were packed from `raw_v3`, a relabel, so ~4% of their `chosen` are not the move played in the game (`seq_build.py` finds the played move by replay).
- 6-step plans are available for 99.3-100% of positions (train1m 94.7%).

## Model lineage (checkpoints `attentris/data/model_<name>.pt`, with optimizer state: resumable)

`v51` (d64) -> `v51_dagger_ft` (DAgger 2) -> `v51_dagger3_ft`; the same lineage at `v51_d96*` and `v51_d128*`.
**Best: `v51_d128_dagger5b_ft`**: d128 base -> dagger_ft -> dagger3_ft -> dagger5b_ft (rounds 4 + 5 + extension).

Other checkpoints: `v51_d128_dagger5_ft`, ToMe `tome_v41*` / `tome_v42_{l4h,l4g,l6g}*`, `v52_{loop,untied}*` (d64), `v52_d128_{untied,loop}{,_dagger_ft,_dagger3_ft,_dagger5b_ft}` (d128, full lineage), ablations `v51_{nosetup,nofitx,not}*`.
Plan models (2026-10-16, all d128; "-ft" = fine-tuned from `v51_d128_dagger5b_ft` with the dagger5b recipe, 3 epochs):
`v51_d128_ctrl_ft` (the control: the same 3 epochs, no plan loss), `v51_d128_autoreg2_ft`, `v51_d128_autoreg6_ft`, `v51_d128_plan2_ft`,
`v51_d128_pieces6_ft`, `v51_d128_pieces6L2_ft`, `v51_d128_pieces6L2fze10_ft`; `v51_d128_autoreg2` = trained from scratch with the `v51_d128` recipe.

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

**v5.2 at d = 128, full lineage** (`v52_d128.sh`, `results_v52_d128.txt`; dagger5b-ft of each, same recipe and seeds as v5.1; deaths / 64, VSfull in brackets):

| model | APP none (256) | APP d2 | normal 80/20 | normal 50/50 | danger 80/20 | danger 50/50 |
|---|---|---|---|---|---|---|
| v5.1 dagger5b-ft | 0.739 | 0.832 | 18 (0.921) | 44 (0.523) | 4 (1.049) | 25 (0.683) |
| **v5.2 untied dagger5b-ft** | **0.780** | **0.902** | 18 (0.970) | 48 (0.498) | 2 (1.088) | **14 (0.800)** |
| v5.2 loop dagger5b-ft | 0.759 | 0.887 | 19 (0.996) | 52 (0.504) | 0 (1.089) | 18 (0.751) |

Depth 2 beam 4 with garbage (deaths, VSfull):

| model | normal 80/20 | normal 50/50 | danger 80/20 | danger 50/50 |
|---|---|---|---|---|
| v5.1 dagger5b-ft | 10 (1.109) | 45 (0.593) | 2 (1.116) | 11 (0.848) |
| v5.2 untied | 10 (1.132) | 38 (0.650) | 0 (1.160) | 5 (0.902) |
| v5.2 loop | 14 (1.093) | 41 (0.613) | 4 (1.100) | 8 (0.882) |

- Val top1 is the same for all three (normal 0.649; val_dagger5all 0.492 for both v5.2 vs 0.483 for v5.1), but play differs. Untied wins every setting but normal 50/50 at depth 1. It is the best model so far on these tests. Not yet confirmed with 128-game re-tests, deeper search or versus.
- Caveats: untied has 19% more parameters, so part of the gain may be capacity (d128 already showed a val gap). Loop gets about half the gain with no new parameters, at the cost of a second cross pass. ms/move on this box (16 games in parallel) was ~42-44 at depth 1; compare latencies only on one machine.
- **Loop count at play time** (`v52_loops.sh`; loop model trained with 2 passes): APP none 1 / 2 / 3 / 4 passes = 0.488 / **0.759** / 0.615 / 0.393. Deaths at danger 50/50: 58 / 18 / 30 / 59. 2 passes is best everywhere. The block learned "exactly two passes", not an iterative refinement that improves with more passes (3 is the least bad off-design count, 4 collapses).

## Working rules (from the user; keep them)

- **Commits:** plain messages, never a Claude co-author or attribution trailer.
- **Git:** the user pushes. GitHub `Danielsrq/Attentris` `main`; on the Windows machine that is the worktree branch `worktree-burst`, see SETUP.md.
- **Runs:**
  - Every run must be resumable and show progress.
  - Say how many games and roughly how long before starting a large run.
  - Never restart a reaped or killed job without the user's go-ahead.
- **CPU:** pin every compute run to the P-cores. Rust does it itself; Python calls `pcores.pin()`; `bot cpus` shows the detected cores. On a non-hybrid CPU nothing is restricted (the Ubuntu box: AMD 3970X, 64 threads; train with `--threads 8`).
- **GPU:** only for AtTenTris training, and **never share it between two runs**. Play tests and inference run on the CPU. (Ubuntu: RTX 3080 Ti 12 GB; a d128 3-epoch fine-tune takes ~12 min, autoreg N = 6 ~27 min.) VRAM is activations, not parameters (2M params + Adam ~32 MB): at batch 512 `--micro 2` autoreg N = 6 peaks ~6.4 GB, the others less; on the 8 GB card use `--micro 4` for autoreg N = 6.
- **Tests:**
  - Quick tests use **64 games per setting** (16 is too few to separate deaths); 256 only for headline no-garbage APP.
  - Routine play tests are depth 1, depth 1 with garbage, and depth 2 beam 4. **No depth 3+ tests** unless the user asks; they asked for the d3-d6 sweep and for 500 ms versus budgets.
- **Data:**
  - The danger DAgger data is a hot / calm mix by choice.
  - Plan targets are the teacher's relabelled planned line, for teacher games and DAgger alike (2026-10-16).
- **Disk:** keep big scratch files off nearly full disks. On the Windows PC, C: is nearly full, so use D:.
- **Baselines:** use external bots' code as-is (CC2 unmodified) and match compute fairly (by time or depth / width).
- **Explainability first:** the explained model must be the one that decides, not a post-hoc evaluator.

## Multi-move (plan) models (2026-10-16)

**The question:** does AtTenTris internalise search (several moves from one pass), and does predicting future moves
improve move 1? The hope: commit the next 2-3 moves without a new decision, to save time.

**Short answer:**
- **Plans are partly learned:** given the true earlier moves, each later step is about as predictable as move 1 (~58%).
- **Not enough to commit:** committing to a network's plan costs ~30% APP. The CEM teacher loses almost nothing from committing (commit 2: -0.4%), so the idea is sound and the networks' plans are the bottleneck.
- **No play-strength gain:** plan training gives at most a small, not significant survival gain at depth 1, and none at depth 2.

### What was built

- **Plan labels** for all data (1.54M positions; `bot label --plan`, `plan_label.sh`; 44 min on 64 threads). See "Plan data" above. The original plan (trajectory targets, DAgger plan losses masked) was dropped: the teacher's own plan matches the moves played later only 75 / 61 / 48 / 37 / 25% at steps 2-6, so trajectory targets would teach "what it did after seeing more pieces". With relabelled plans, DAgger positions get plan targets too.
- **`model_plan.py`**, three variants, all loading `dagger5b-ft` exactly (move 1 identical at start, checked):
  - `autoreg`: the encoder once, then the SAME cross block once per step. Step k's queries are the legal moves at plan step k (Rust, `bot plan-cands`, mapped into the root's frame through earlier clears); earlier planned moves are extra key / value tokens. +896 params (step embeddings, a "previous move" vector). Always legal, but needs move generation at every step.
  - `plan`: N blank learned step tokens -> a new cross block -> self-attention among the steps -> a 1760-way coordinate head (hold x rotation x column x row). +626k params.
  - `pieces` (option A): one query per KNOWN piece (current, hold, preview 1-5), started from that piece's board token; cross + self; heads WHEN (step 1-6 / not placed) and WHERE (880). Hold needs no special case. +512k params; `--plan-layers L` = L rounds of (cross, self).
- **`train.py`**: `--plan-variant`, `--plan-steps N`, `--w-plan` (plan loss weight; mean over a position's steps), `--new-lr` (new parameters at their own peak lr), `--freeze-base` (train only the new branch; move 1 unchanged), `--plan-layers`, `--threads` (use 8 on the Ubuntu box: 64 OpenMP threads only spin).
- **`play.py`**: `--plan-check K` (the model's own K-step plan vs the moves it then plays), `--commit 2` (autoreg and pieces), `--depth 2 --autoreg-beam` (move 2 from the step-2 pass over the root encoding, children never encoded).
- **`bot play --commit K`** and versus `cem:W:D:B:commitK`: play the search plan's next K-1 moves without searching; garbage landing (scheduled or versus) cancels the rest.

### Results (64 games unless noted; deaths in brackets; VSfull for garbage settings, APP for "none")

**Plan models vs the start and the control** (control = dagger5b-ft + the same 3 epochs without plan loss):

| setting | dagger5b-ft | control | autoreg2-ft | autoreg6-ft | plan2-ft | pieces6-ft |
|---|---|---|---|---|---|---|
| depth 1, none (APP, 256) | 0.739 | 0.739 | 0.733 | 0.720 | 0.719 | 0.739 |
| normal 80/20 | 0.921 (18) | 0.918 (24) | 0.959 (19) | 0.846 (30) | 0.994 (15) | 0.937 (20) |
| normal 50/50 | 0.523 (44) | 0.487 (51) | 0.512 (48) | 0.541 (39) | 0.462 (53) | 0.567 (42) |
| danger 80/20 | 1.049 (4) | 1.046 (4) | 1.044 (3) | 1.012 (5) | 1.043 (3) | 1.037 (5) |
| danger 50/50 | 0.683 (25) | 0.775 (14) | 0.744 (16) | 0.704 (22) | 0.702 (26) | 0.729 (24) |
| depth 2, none (APP) | 0.832 | 0.832 | 0.864 | 0.839 | 0.805 | 0.852 |
| depth 2, normal 80/20 | 1.109 (10) | 1.132 (7) | 1.120 (11) | 1.087 (10) | | 1.148 (7) |
| depth 2, normal 50/50 | 0.593 (45) | 0.647 (38) | 0.713 (25) | 0.637 (42) | | 0.591 (42) |
| depth 2, danger 80/20 | 1.116 (2) | 1.109 (2) | 1.122 (0) | 1.130 (0) | | 1.129 (1) |
| depth 2, danger 50/50 | 0.848 (11) | 0.861 (11) | 0.892 (4) | 0.887 (6) | | 0.835 (12) |

APP and VS per piece are within about ±0.01 of dagger5b-ft everywhere: the VSfull differences are survival.

**128-game re-test, danger 50/50, fresh seeds 53001-53128** (the 64-game gaps above were partly noise):

| model | depth | deaths / 128 | APP | APPfull | VS | VSfull |
|---|---|---|---|---|---|---|
| dagger5b-ft | 1 | 48 | 0.516 | 0.436 | 0.840 | 0.709 |
| control | 1 | 40 | 0.520 | 0.445 | 0.844 | 0.724 |
| autoreg2-ft | 1 | 35 | 0.517 | 0.449 | 0.845 | 0.734 |
| dagger5b-ft | 2 | 13 | 0.572 | 0.546 | 0.905 | 0.863 |
| autoreg2-ft | 2 | 12 | 0.573 | 0.547 | 0.907 | 0.865 |

So: at depth 1 ~8 of the 13 fewer deaths come from the extra epochs, ~5 from the plan loss (neither significant); at depth 2 no difference.

**From scratch** (the `v51_d128` recipe, 8 epochs, + autoreg plan loss, w_plan 1.0; normal style):
`v51_d128` APP none 0.651 (256), d2 0.770, 80/20 62 deaths, 50/50 64; `v51_d128_autoreg2` 0.606, 0.775, 61, 64.
The plan loss cost 0.045 depth-1 APP from scratch (val step-2 top-1 0.626).

**Plan accuracy vs the teacher's plan (val, top-1 per step):**

| model | step 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| autoreg2-ft (teacher-forced) | (0.638) | 0.592 | | | | |
| autoreg6-ft (teacher-forced) | (0.639) | 0.593 | 0.582 | 0.571 | 0.560 | 0.609 |
| plan2-ft | 0.435 | 0.260 | | | | |
| pieces6-ft (all steps at once) | 0.582 | 0.338 | 0.219 | 0.166 | 0.129 | 0.104 |
| pieces6L2-ft (2 rounds of cross + self) | 0.612 | 0.353 | 0.238 | 0.172 | 0.137 | 0.108 |
| pieces6L2fze10-ft (backbone frozen, 10 epochs; plateau from epoch 8) | 0.625 | 0.379 | 0.260 | 0.186 | 0.149 | 0.117 |

Teacher-forced = given the teacher's true earlier moves; the pieces numbers are not (all steps from one pass), so they compound like free-running plans.

**Plan consistency** (step k of the plan made at X = the move played k-1 moves later; no garbage):

| | step 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| CEM teacher, step alone (20 val games, `plan_consistency_cem.py`) | 75.2% | 61.3% | 48.3% | 37.1% | 24.7% |
| CEM teacher, steps 1..k all | 73.7% | 53.8% | 37.8% | 24.6% | 13.8% |
| autoreg6-ft, step alone (32 games, `--plan-check 6`) | 63.8% | 41.0% | 25.5% | 15.5% | 9.5% |
| autoreg6-ft, steps 2..k all | 63.8% | 35.2% | 17.7% | 8.3% | 3.7% |
| autoreg2-ft, step alone (8 games) | 63.7% | 39.5% | 23.1% | 13.7% | 7.8% |
| untrained autoreg (dagger5b-ft weights), step 2 | 55.0% | | | | |

With garbage, autoreg6-ft is a little higher (step 2 64.6-66.4%, step 6 10.4-13.4%). The board-based match (order-free) is only ~1 point above the move-based one for the teacher.

**Using move 2** (autoreg2-ft, the same weights):

| mode | none (APP) | normal 80/20 | normal 50/50 | danger 80/20 | danger 50/50 | ms/move (8 games at a time) |
|---|---|---|---|---|---|---|
| depth 1 | 0.733 | 0.959 (19) | 0.512 (48) | 1.044 (3) | 0.744 (16) | 13.1 |
| commit 2 | 0.512 (256) | 0.511 (48) | 0.208 (63) | 0.837 (17) | 0.416 (48) | 13.4 |
| depth 2, move 2 from the autoreg step-2 pass | 0.766 | 0.983 (18) | 0.493 (49) | 1.070 (2) | 0.779 (16) | 50.1 |
| depth 2 (usual: children re-encoded) | 0.864 | 1.120 (11) | 0.713 (25) | 1.122 (0) | 0.892 (4) | 66.0 |

pieces6-ft commit 2: APP 0.511 (256); 80/20 0.494 (52), 50/50 0.210 (63); danger 0.629 (35), 0.331 (57). Only 19% of moves committed: its step 1 is not the policy's move in 55% of decisions, and 48% of its step-2 placements are illegal.
pieces6L2-ft (2 rounds of cross + self, 3 epochs): depth 1 APP 0.731 (256), 80/20 0.999 (17), 50/50 0.484 (51), danger 1.064 (3), 0.774 (16); depth 2 0.864, 1.044 (16), 0.669 (35), 1.110 (2), 0.861 (9). Commit 2: APP 0.494 (256); 80/20 0.335 APPfull (56), 50/50 0.124 (64); danger 0.455 (30), 0.198 (57); 21% of moves committed, step 1 not the policy's move in 53%, 43% of step-2 placements illegal. The second round did not make the plan playable.
pieces6L2fze10-ft (frozen backbone, 10 epochs: move 1 = dagger5b-ft exactly): commit 2 APP 0.481 (256); 80/20 0.361 APPfull (48), 50/50 0.145 (63); danger 0.465 (31), 0.165 (59); 22% committed, step 1 not the policy's move in 52%, 40% of step-2 placements illegal. More training (step 2 0.353 -> 0.379) did not help either: the parallel plan head plateaus far below what committing needs.
Commit 2 saved no time for the networks: the committed move still needs `bot expand` at X+1 (~6 ms with structure tokens), as much as the skipped encoder pass.

**CEM teacher commit K** (`bot play --commit K`, depth 6 beam 64; APP; time relative to commit 1):

| | commit 1 | commit 2 | commit 3 | commit 6 |
|---|---|---|---|---|
| normal.txt, none | 1.204 | 1.199 | 1.138 | 1.020 |
| normal.txt, 80/20 (APPfull) | 0.974 (18) | 0.915 (24) | 0.811 (26) | 0.749 (32) |
| normal.txt, 50/50 (APPfull) | 0.600 (33) | 0.589 (31) | 0.598 (24) | 0.478 (38) |
| CEM-survival, 80/20 | 1.049 (0) | 1.012 (0) | 0.990 (0) | 0.929 (0) |
| CEM-survival, 50/50 | 0.816 (0) | 0.793 (0) | 0.769 (0) | 0.714 (0) |
| think time | 100% | 50% | 33% | 17% |

A versus smoke test (4 matches, turn-based): CEM-survival commit 6 thought 23.9 ms/move vs 82 for the plain search (versus garbage cancels plans more often than every 10 pieces).

### Observations

- What decides commit is not how often a plan is kept but how costly the changes of mind are: the teacher's changes (25%) are near-ties, the networks' (36%) are real mistakes.
- Grounded queries win: real legal moves (autoreg) or real pieces (pieces) beat blank step tokens (plan); per-group learning rates (`--new-lr`) kept move 1 intact for pieces (plan2-ft, one lr, lost depth-1 APP 0.739 -> 0.719).
- Teacher-forced, every step is ~58% (like move 1); free-running plans compound and fall far below the teacher's.
- The step-2 "imagined child board" (autoreg beam) recovers only ~1/4 of real depth 2's gain: the network does not simulate a move as well as seeing its result.
- Multi-step training dilutes: autoreg6-ft is no better in play than autoreg2-ft, slightly worse at depth 1.
- Single-thread speed on the Ubuntu box (AMD 3970X, Zen 2) is below the Windows P-cores: CEM-depth6 44.7 ms/move; NN depth 1 13-16 ms (expand 6.0 + tokenize 1.9 + forward 4.8); NN depth 6 beam 8 444 ms. Compare latencies on one machine only.

### Realtime blitz (Windows, 2026-10-16)

`versus.py --realtime` gives each player its own clock, advanced by its measured think time per piece (uncapped speed), and applies the next lock in game time first.
- Garbage cancels the sender's own pending garbage first; the rest is queued for the opponent and lands after the receiver's next non-clearing placement, at most 8 rows.
- Cold Clear 2 gets its next position only when its own clock starts, so it gets no free background thinking.
- Scripts: `bot/tools/versus_realtime.sh` and `versus_realtime2.sh`; results in `bot/results/versus/realtime/`.
- Replays (first 3 matches per pairing): `data/bot/versus/realtime/`, viewed with `versus_watch.py`, which plays realtime replays at true game time.

64 matches each, Cold Clear 1 rules, CEM-survival = `survive.txt`, blitz = commit 6:

| A vs B | A wins - B wins | A PPS / B PPS | A APP / B APP | mean game time |
|---|---|---|---|---|
| blitz d6 b64 vs CC2 40 ms | **64 - 0** | 181 / 25 | 0.619 / 0.190 | 0.22 s |
| plain d6 b64 (commit 1) vs CC2 40 ms (control) | 54 - 10 | 27 / 25 | 0.874 / 0.796 | 2.31 s |
| plain d6 b64 vs CC2 40 ms, both at least 40 ms per piece (`--min-ms 40`, matched speed) | 47 - 17 | 23 / 24 | 0.888 / 0.809 | 3.29 s |
| blitz d6 b64 vs plain d6 b64 | **64 - 0** | 193 / 38 | 0.633 / 0.477 | 0.25 s |
| blitz d6 b64 vs CC2 10 ms | 40 - 24 | 116 / 96 | 0.725 / 0.736 | 0.44 s |
| blitz d6 b64 vs CC2 5 ms | 3 - **61** | 98 / 184 | 0.657 / 0.676 | 0.26 s |
| blitz d6 **b16** vs CC2 5 ms | **59 - 5** | 401 / 178 | 0.645 / 0.638 | 0.13 s |
| blitz d4 **b16** vs CC2 5 ms | **64 - 0** | 771 / 166 | 0.487 / 0.488 | 0.07 s |
| blitz d6 b64 vs CC2 40 ms, garbage delay 333 ms | 64 - 0 | 189 / 25 | 0.684 / 0.413 | 0.57 s |
| blitz d6 b64 vs CC2 5 ms, garbage delay 333 ms | 11 - **53** | 185 / 185 | 0.718 / 0.725 | 12.66 s |

Findings:
- **Uncapped, the faster bot wins** almost regardless of quality: matches are decided in well under a second of game time. Blitz turns CEM-survival's 54-10 into 64-0 against the same CC2.
- **It doesn't beat every bot:** CC2 is an anytime search and can play just as fast. At 5 ms per move (184 PPS) it beats the d6 b64 blitz 61-3, because its garbage cancels the blitz plans and forces new searches (98 PPS instead of 181).
- **A cheaper blitz search wins it back:** d6 b16 at 400 PPS goes 59-5, and d4 b16 at 771 PPS goes 64-0.
- **With a TETR.IO-like 333 ms garbage delay at equal speed (185 PPS each), quality decides again:** CC2 5 ms wins 53-11 in long matches.
- Caveat: the speeds are machine-dependent (Windows P-cores here), and "uncapped" means instant inputs (no DAS / ARR or animation costs).

### Next (Windows side and later)

1. **Versus blitz:** CEM-survival `cem:bot/results/survive/survive.txt:6:64:commit6` (and commit 2 / 3) vs CC2, (a) uncapped and (b) at a fixed PPS. Uncapped, ~6x the PPS at -12% APP should cancel most incoming garbage. Fixed PPS is the fair-time check.
2. **Equal-time check:** CEM commit 2 vs a plain search with half the budget (e.g. beam 32): commit only helps if it beats a cheaper search at equal time.
3. **pieces:** the parallel head plateaus (step 2 ~0.38; commit 2 APP ~0.48-0.51). Next idea: condition it on the ACTUAL move 1 (a move-1 token for the piece queries, as autoreg does), so step 2 is predicted given the move played, still as a coordinate (no move generation; a cheap Rust single-placement legality check). If step 2 reaches ~0.55-0.6 and commit 2 costs little, scale data (plan-only teacher labels: ~45 ms a position, 5M positions ~2 h on the Ubuntu box) and model size (d192 / d256).
4. A lean `bot expand` (legal moves only, no structure tokens) for any NN mode that needs move lists at later steps.
5. On a new machine: `cargo build --release` (new `bot label --plan`, `plan-cands`, `play --commit`), unpack the delta, `python attentris/pack_plan.py` if training autoreg (rebuilds `data/v51plan`).

## Other open options (offered, not started; ask before starting any)

- v5.2 untied d128 follow-ups: a 128-game re-test of danger / normal 50/50; a v5.1 d128 with the same parameter count (e.g. 5 self-attention layers) to separate capacity from the second cross pass; the depth sweep and versus with the untied model.
- NN dagger5b-ft at depth 5-6 in versus vs CC2 / CEM.
- The `auto` style test.
- DAgger 6, or oversampling normal-style hot positions.
- `fit_gap` check after the 62% main re-sampling.
- A full self-play CEM run (meta-level CEM, fitness = wins vs a frozen copy; smoke test in `bot/results/selfplay/`).
- MisaMino integration (needs a bridge).
- A guideline spawn rule for fairness vs CC2.
- A DAgger 4-only ablation.
- ToMe direction: merge network vs hybrid (the user is deciding).
