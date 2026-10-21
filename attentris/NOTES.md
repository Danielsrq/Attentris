# AtTenTris: handover notes (state at 2026-10-20, Ubuntu machine)

Read this first in a new session. The 2026-10-16 work (multi-move "plan" models, plan labels for all data,
CEM commit-K) is in "Multi-move (plan) models" below, with its results and the next tests for the Windows side.

**2026-10-20 (Ubuntu box), where to find it:**
- v5.2 (two cross passes) at d128 through the full lineage: untied beats v5.1 dagger5b-ft (APP 0.780 vs 0.739), the best model on the standard block; "Latest results", "v5.2 at d = 128".
- d128 ablations through the full lineage (FLAT-2D trained off, all overlay tokens off, setup tokens off) all three lose survival under garbage over 128 paired games (pooled McNemar p = 0.0002 / 3e-15 / 0.016; overlays matter most): "Latest results", "Ablations at d = 128".
- Token SHAP (which tokens drove a move) and a SHAP view on the attention page: "Explainability: token SHAP".
- All-spin: rule sets documented ("Teachers and styles", "Rule sets"); CEM-allspins-survival (survives only with garbage every 15 pieces); rule variants for game design (`allspin_s1`, `allspin_cash`): "Next plan", item 1. Machine setup and data sync are in `../SETUP.md`; the model and tokenizer in depth in `DESIGN.md`.
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
| `attentris/attn_viz.py`, `attn_template.html` | attention page (cross-attention per move, geometry vs content); SHAP view when the data has it |
| `attentris/token_shap.py`, `shap_page.py` | token SHAP per position (`--top N` moves, `--keep PAGE`, `--summary`); merge into an attention page |
| `attentris/ablation_d128.sh`, `ablation_more.sh`, `ablation_stats.py` | d128 ablation lineages (`VARIANTS=noflat nooverlay nosetup nofitx nofitt nopit`), 64 more games on seeds 53001+, paired McNemar tests |
| `attentris/group_shap.py` | Shapley values of token groups for VS / deaths at play time (one model, groups blanked with `play.py --drop-kinds`, wells as play rows with `--no-wells`, all coalitions) |
| `bot/results/allspin_survive/run.sh`, `bot/results/allspin_cash/run.sh` | CEM all-spin survival tunes and evals (portable); `bot/tools/spin_stats.py` = spin / B2B / wound stats from `bot play --out` replays |

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

### Rule sets (`bot --mode`; Godot `RuleSet.*`)

The teachers above and every AtTenTris model so far use **normal** rules. All-spin rules are a different game, with different attack, B2B and a penalty.

| | `normal` (default) | `allspin` | `allspin_nw` | super-allspin (proposed, not built) |
|---|---|---|---|---|
| which pieces can spin | all | all | all | all |
| non-T spin | Mini: 0 attack, but keeps B2B alive | **full**: scores like a T-spin, 2 / 4 / 6 for single / double / triple, + B2B | as allspin | as allspin |
| T-spin Mini | Mini (0 attack) | promoted to full | promoted | promoted |
| repeating the previous clear type | nothing | **wound**: an unclearable line is inserted at the bottom, pushing the stack up | nothing | **game over** (penalty = −∞) |

- **Clear type** (`rules.rs clear_type`): piece + spin + lines for spins (TSD → ZSD is *not* a repeat; ZSS → ZSS is), lines only for plain clears (any plain double → any plain double is a repeat).
- **Wound line:** counts down once per clear (any type) and after 10 clears (`wound_clears`) turns into an ordinary garbage line. Its hole follows the previous wound's column with probability 1 − `wound_messiness` (0.5). Incoming garbage enters beneath wounds. The search tracks `last_clear` and the wounds (`search.rs`), so the bot plans around repeats. Features `wounds`, `ready_repeat` and the S/Z, L/J, I spin-slot features exist for this mode.
- **B2B:** spins of any piece and quads extend the chain; plain singles / doubles / triples break it. Under all-spin almost every clear is a spin, so a chain lasts until a forced plain clear or death.
- **Do not tune all-spin weights with `--freeze default`**: it zeroes exactly the all-spin features (`eval::FROZEN_OUTSIDE_ALLSPIN`). It is right for normal-rules tunes only.
- Weights: `bot/weights/allspin.txt`, `allspin_slots.txt` (with spin-slot features; strongest on clean boards, both die to cheese), `bot/results/allspin_survive/allspin_survive.txt` (CEM-allspins-survival, 2026-10-20).
- Stats for these rules: `bot/tools/spin_stats.py` on `bot play --out` replays (+ `--game-log`): deaths, APP / APPfull / VSfull, spin share of clears, non-T share of spins, **B2B kept** (of the clears made while a chain was alive, the share that kept it), mean / max chain, repeats (wounds) per 100 pieces.

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

Other checkpoints: `v51_d128_dagger5_ft`, ToMe `tome_v41*` / `tome_v42_{l4h,l4g,l6g}*`, `v52_{loop,untied}*` (d64), `v52_d128_{untied,loop}{,_dagger_ft,_dagger3_ft,_dagger5b_ft}` (d128, full lineage), ablations `v51_{nosetup,nofitx,not}*` (d64) and `v51_d128_{noflat,nooverlay,nosetup,nofitx,nofitt,nopit}{,_dagger_ft,_dagger3_ft,_dagger5b_ft}` (d128, full lineage, 2026-10-20).
Strongest on the standard block since 2026-10-20: `v52_d128_untied_dagger5b_ft` (1.26M params; not yet re-tested at 128 games, deeper search or versus).
Bot weights from 2026-10-20 (in git): `bot/results/allspin_survive/allspin_survive_e15.txt` (all-spin, survives garbage every 15 pieces), `bot/results/allspin_cash/allspin_cash_e15.txt` (the `allspin_cash` variant).
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

**Ablations at d = 128, full lineage** (2026-10-20, `ablation_d128.sh`, `results_ablation_d128.txt`; each dagger5b-ft with the same recipes and seeds as v5.1 d128 dagger5b-ft). `noflat` = FLAT-2D bias trained off (`train.py --no-flat`: geometry only through the box / cell position embeddings, in the self- and cross-attention). `nooverlay` = every overlay token kind blanked (`--drop-kinds imperial,stsd,cspin,fractal,tsd,vertical,rect12,fit_,pit,setup_,hang`, ~6.9 tokens per position): left are sky / well / play / band and the piece tokens; the cells still feed the pooling.

| model | val normal / gate4 / dagger5all top-1 | APP none (256) | APP d2 | normal 80/20 | normal 50/50 | danger 80/20 | danger 50/50 |
|---|---|---|---|---|---|---|---|
| v5.1 dagger5b-ft | 0.645 / 0.629 / 0.483 | 0.739 | 0.832 | 18 (0.921) | 44 (0.523) | 4 (1.049) | 25 (0.683) |
| noflat | 0.640 / 0.626 / 0.481 | 0.724 | 0.835 | 30 (0.802) | 56 (0.372) | 2 (1.036) | 32 (0.633) |
| nooverlay | 0.635 / 0.633 / 0.483 | 0.729 | 0.855 | 34 (0.794) | 63 (0.323) | 11 (1.003) | 36 (0.594) |

Depth 2 beam 4 with garbage (deaths, VSfull): v5.1 10 (1.109) / 45 (0.593) / 2 (1.116) / 11 (0.848); noflat 17 (1.014) / 44 (0.596) / 1 (1.128) / 15 (0.792); nooverlay 24 (0.975) / 48 (0.559) / 1 (1.119) / 12 (0.799) (normal 80/20, normal 50/50, danger 80/20, danger 50/50).

- **Validation and clean-board APP hardly move** (top-1 within 0.01, APP none −0.010 to −0.015, d2 even +0.02); the cost shows **under garbage**: normal style 80/20 deaths 18 → 30 / 34, 50/50 44 → 56 / 63; danger 50/50 25 → 32 / 36. Both components matter for survival, the overlays slightly more.
- Top-1 is the wrong yardstick for these components (as for the heads): what they buy is in the hot, rare positions.
- Depth-2 search closes most of the gap at 50/50 and danger style, not at normal 80/20.
- Side effect: both train ~45% faster per epoch (131-136 s vs 237 s at dagger5b), noflat from no pair features, nooverlay from fewer tokens.

**128 paired games per garbage setting** (seeds 33001+ and 53001+, `ablation_more.sh`; `nosetup` = only the `setup_*` tokens blanked, val 0.635 / 0.629 / 0.487; `nofitx` (2026-10-21) = the non-T slots `fit_s,fit_z,fit_l,fit_j,fit_i` blanked together, the d64 fit ablation at d128, val 0.638 / 0.623 / 0.480; `nofitt` (2026-10-21) = only the open T recess `fit_t` blanked, the T structures stay, val 0.646 / 0.625 / 0.484; `nopit` (2026-10-21) = pit1-4 and rect12 blanked (the `pitrect` group of the group SHAP), val 0.636 / 0.628 / 0.487). Deaths / 128, exact McNemar p in brackets (`ablation_stats.py`, `ablation_stats_128.txt`); APP none over 256 games:

| setting | v5.1 dagger5b-ft | noflat | nooverlay | nosetup | nofitx | nofitt | nopit |
|---|---|---|---|---|---|---|---|
| APP none (256) | 0.739 | 0.724 | 0.729 | 0.722 | 0.715 | 0.728 | 0.754 |
| APP d2 (64) | 0.832 | 0.835 | 0.855 | 0.842 | 0.845 | 0.863 | 0.864 |
| normal 80/20 | 38 | 53 (0.08) | **69** (0.0001) | **65** (0.0007) | **67** (0.0002) | 47 (0.28) | 46 (0.34) |
| normal 50/50 | 92 | **112** (0.004) | **118** (<0.0001) | 104 (0.10) | **114** (0.0007) | **110** (0.006) | 98 (0.44) |
| danger 80/20 | 10 | 9 (1.0) | **25** (0.006) | 14 (0.50) | 17 (0.19) | 6 (0.42) | 8 (0.80) |
| danger 50/50 | 47 | 57 (0.19) | **70** (0.004) | 40 (0.40) | **65** (0.02) | 51 (0.69) | 43 (0.70) |
| normal d2 80/20 | 22 | 31 (0.21) | **45** (0.0008) | 17 (0.47) | 33 (0.10) | 34 (0.10) | 31 (0.20) |
| normal d2 50/50 | 77 | 83 (0.49) | **93** (0.03) | 86 (0.29) | **98** (0.003) | 82 (0.59) | 74 (0.80) |
| danger d2 80/20 | 5 | 5 (1.0) | 4 (1.0) | 4 (1.0) | 1 (0.22) | 2 (0.38) | 1 (0.13) |
| danger d2 50/50 | 18 | 25 (0.30) | 27 (0.20) | 22 (0.61) | 22 (0.58) | 19 (1.0) | 19 (1.0) |
| **pooled, 8 settings** (games only one side lost) | | 186 vs 120, **p = 0.0002** | 236 vs 94, **p = 3e-15** | 175 vs 132, **p = 0.016** | 201 vs 93, **p = 3e-10** | 170 vs 128, **p = 0.017** | 159 vs 148, p = 0.57 |

VSfull (128 games): normal 80/20 0.941 / 0.873 / 0.775 / 0.821 / 0.790 / 0.880 / 0.902; normal 50/50 0.516 / 0.389 / 0.352 / 0.479 / 0.370 / 0.425 / 0.478; danger 50/50 0.703 / 0.667 / 0.598 / 0.733 / 0.635 / 0.670 / 0.724.

- **Five of six components help survival** (pooled, game noise only). Ranking: overlays (all of them) ≫ non-T slots > FLAT-2D > setup tokens ≈ fit_t > pits (none: nopit 159 vs 148, p = 0.57, APP none even 0.754).
- **Overlays off** hurts in 6 of 8 settings individually, including danger style and depth 2: the rule tokens are the strongest single design element.
- **FLAT-2D off** hurts mainly in normal style at depth 1 (50/50 92 → 112); search and danger style mostly absorb it.
- **Setup tokens off** costs normal 80/20 (38 → 65) and is otherwise neutral, even slightly better at danger 50/50 (47 → 40, n.s.): setups matter to the attacking normal style, not to the survival style.
- **Non-T slots off** (`nofitx`) is the second-largest cost: significant in 4 of 8 settings, in both styles (normal 50/50 92 → 114, danger 50/50 47 → 65, normal d2 50/50 77 → 98), so most of the overlay cost is the slots. Matches token SHAP, where `fit_s/z/l/j/i` carry 13% of all |φ|, second after the play rows; S, Z, L, J each about equal, `fit_i` almost never present (4% of positions). Val and clean APP again barely move (APP none 0.715).
- **fit_t off** (`nofitt`; the idea: only a T fixes checkerboard parity, but a T recess is plain in the play-row cells): significant only at normal 50/50 (92 → 110); pooled p = 0.017 like the setup tokens; val and APP unchanged (none 0.728, d2 0.863). Token SHAP of `nofitt` on the same 300 positions (`data/shap/v51_d128_nofitt_dagger5b_ft.jsonl`): on the 18 T moves where the baseline sees a `fit_t` (4.6% of its |phi|), the move is the same in 13 and p(move) the same (0.42); the share does not move to the play rows (26.8% → 26.8%) but spreads mostly to the piece tokens (29.0% → 33.9%). Small n; the model replaces the token with little loss, not visibly by reading the rows.
- **Pits off** (`nopit`: pit1-4 and rect12) costs nothing: no setting significant, pooled 159 vs 148 (p = 0.57), danger 50/50 even 47 → 43, APP none 0.754 / d2 0.864 (the best of the ablations; within noise of the baseline). Yet the baseline leans on its pit tokens at play time (group SHAP −24 deaths): the open space above the surface is easy to read from the cells, so the label is redundant when trained without.
- Each model is one training run; a second training seed would test model-to-model noise (not done, the user's choice).

**Group SHAP at play time** (2026-10-21, `group_shap.py`; 6 players `group_shap6.out`, the first 5-player run `group_shap.out`; games in `data/group_shap/`). Players = 6 token groups: `fitx` (`fit_s/z/l/j/i`), `fitt`, `setup` (`setup_*`), `tstruct` (tsd, stsd, fractal, imperial, cspin, vertical), `pitrect` (pit1-4, rect12), `well`. One model, v5.1 d128 dagger5b-ft (trained with all tokens). An absent overlay group is blanked at inference (`play.py --drop-kinds`); an absent `well` is not blanked (its rows carry the cells): the clean zone is tokenized as play rows (`play.py --no-wells`, tokenizer `wells=False`; budget bands as usual), so its value is the label, the extras (length, hole column, covered) and the grouping. v(S) = mean VSfull or deaths over the 128 paired games of each depth-1 garbage setting (normal / danger x 80/20, 50/50); exact Shapley over all 64 coalitions (63 played, ~32k games, ~2 h CPU; all present = the baseline's own games, play is deterministic per seed), 90% bootstrap over the paired seeds. phi sums to v(all) − v(none): pooled over the 4 settings, all present 0.802 VSfull / 46.8 deaths per 128, none 0.055 / 128.0, so the phis sum to +0.747 and −81.3.

| group | phi VSfull [90%] | phi deaths [90%] | play-time leave-out: extra deaths | retrained ablation: extra deaths (mean of the 4 depth-1 settings) |
|---|---|---|---|---|
| fitx | +0.396 [0.387, 0.406] | −51.5 [−53.3, −49.6] | +65.5 | +19.0 (nofitx) |
| pitrect | +0.231 [0.224, 0.238] | −23.6 [−24.7, −22.5] | +42.0 | **+2.0** (nopit, pooled p = 0.57) |
| fitt | +0.046 [0.041, 0.051] | −1.7 [−2.8, −0.7] | +4.0 | +6.8 (nofitt) |
| well | +0.044 [0.039, 0.050] | −3.3 [−4.2, −2.3] | +11.3 | not trained |
| tstruct | +0.024 [0.019, 0.029] | −1.1 [−2.1, −0.0] | +3.5 | not trained at d128 |
| setup | +0.004 [−0.001, 0.010] | −0.1 [−1.1, +0.9] | +4.8 | +9.0 (nosetup) |
| (all overlays) | | | +81.0 | +23.8 (nooverlay) |

phi = the group's average marginal effect over all orders of adding the groups (its fair share of the total); leave-out = all present vs all but this group (not a Shapley value, does not sum to the total); retrained = a separate model trained without the group (the ablation table above).
- **The trained model leans on two groups:** the non-T slots and the pits. `fitx + pitrect` (wells on) gives 0.734 VSfull / 45.5 deaths (all six: 0.802 / 46.8): the rest add attack (+0.07 VSfull) but no survival. Either one alone collapses (fitx 0.433 / 85.5, pitrect 0.200 / 122.8): strongly complementary (interaction +0.127 VSfull, −28.8 deaths).
- **Play-time reliance is not necessity.** Retrained without pits, the model loses nothing (nopit: +2 deaths, n.s.; APP none 0.754) although this model's pit phi is −24 deaths; without the non-T slots it loses 19 of 65. The model reads the labels when it has them; trained without, it reads the cells. The non-T slots are the one group that carries information the cells give only at a cost.
- **The well label is modest:** −3.3 deaths, most at danger 50/50 (−6.8; normal 80/20 +0.4, n.s.), complementary with fitx (+0.044 VSfull). The model reads cheese holes mostly from the cells.
- **T structures and setups matter little at play time** in these garbage tests; their value is attack on clean boards (not measured here). Per-piece VS (attack per piece played, which removes the dead pieces but not all of survival) gives fitt and tstruct about twice their VSfull share.
- **fitx × fitt substitute** in VSfull (−0.028): both describe resting places for a piece.

## Explainability: token SHAP (2026-10-20)

Which tokens drove a move (Lundberg & Lee 2017, SHAP; Shapley values over the tokens of one position). `token_shap.py`:
- **Players:** the tokens of one position (board-side structure tokens and the 7 piece tokens). A coalition S keeps those tokens and blanks the rest as padding (as `--drop-kinds`); the cells and the candidates stay, so the empty coalition is "board cells + candidates only" (`model.py` `allow_empty`, analysis only: attention with no keys outputs 0).
- **Value:** log p(move | tokens in S), for the model's move, or with `--top N` for each of its N best moves plus the teacher's move (the same coalitions serve all moves: Shapley values are linear in the value).
- **Estimate:** permutation sampling in antithetic pairs; per move sum(phi) = log p(all tokens) - log p(no tokens), exactly. Noise at 64 permutations: two seeds correlate 0.84-0.96 per position; use 128 for single-position pictures. ~5-15 s per position on CPU.
- **Reading:** phi > 0 = the token makes this move more likely (a reason for it); phi < 0 = less likely (a reason against it, usually for a rival: find the moves where the same token is positive). Values are relative to "no tokens", and interactions are split evenly between the tokens involved.
- **Viewer:** `shap_page.py` adds the values to an attention page (`token_shap.py --keep PAGE --top 5`); `attn_template.html` has a "SHAP (toward this move)" view (board heat red / blue, piece panel, token list by |phi|). `attn_shap_v51_d128_dagger5b.html` = the 66 positions of the dagger5b-ft attention page, 128 permutations (published as an artifact, private).
- Example (position 6 of that page): `tsd` is −0.79 / −0.75 for both T-spin singles the model ranks first and second and +1.8 to +2.6 for the J-from-hold placements: the TSD token argues for keeping the slot rather than taking a single now.

**dagger5b-ft on 300 validation positions** (100 each from val, val_danger, val_dagger5all; 179 hot; `data/shap/v51_d128_dagger5b_ft.jsonl`, `token_shap.py --summary`). Share = the kind's part of a position's total |phi| (summed over its tokens, so many-token kinds score higher):
- play rows 0.29, the 5 previews 0.21, wells 0.10 (hot 0.12, calm 0.06), current 0.06, hold 0.05, sky 0.05, each slot kind (`fit_s/z/l/j`) and `pit2` ~0.05.
- The queue (current + hold + previews) carries about a third; previews push toward the chosen move 69% of the time (92% for quads).
- For clears the play rows mostly push against the chosen move (toward it 18% for quads, 31% for spin clears); for non-clearing moves they and the S/Z slots push toward it (+0.22, +0.20): row tokens favour building, clears are chosen despite them, driven by the queue and the sky.
- `tsd` is present in 68% of spin-clear positions with mean phi ~0: a T structure supports some moves and argues against others.

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

**CC2 think-time sweep vs plain CEM-survival d6 b64** (CEM at its own ~24-27 PPS; `bot/tools/versus_realtime_sweep.sh`, `THINK=...`):

| CC2 ms / move | 5 | 10 | 15 | 20 | 25 | 30 | 33 | 36 | 40 |
|---|---|---|---|---|---|---|---|---|---|
| CC2 PPS | 184 | 97 | 65 | 49 | 39 | 33 | 30 | 27 | 25 |
| CEM wins - CC2 wins | 0-64 | 0-64 | 2-62 | 5-59 | 12-52 | 25-39 | **30-34** | 47-17 | 54-10 |
| CC2 APP (CEM APP ~0.85-0.94 from 15 ms on) | 0.61 | 0.66 | 0.71 | 0.74 | 0.77 | 0.80 | 0.82 | 0.82 | 0.80 |

The two draw level at CC2 ≈ 33-34 ms per move: CC2 has to play ~25% faster (30 vs 24 PPS) to make up for its lower APP (0.82 vs 0.93).
CC2's quality saturates early: from 33 to 15 ms it loses only 0.11 APP while doubling its speed, which is why a fast CC2 wins uncapped. The crossover is steep: 0.39 at 30 ms, 0.73 at 36 ms.

**CEM-survival think time vs depth x beam** (`bot/tools/cem_budget_grid.sh`, our default rules, 16 games x 500 pieces, single-threaded, 8 games at a time; `bot/results/survive/budget/`):

| depth x beam | 1x1 | 2x4 | 3x8 | 3x16 | 4x8 | 6x4 | 5x8 | 4x16 | 5x12 | 6x8 | 6x16 | 6x64 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| ms / move | 0.1 | 0.5 | 1.6 | 3.2 | 2.4 | 3.3 | 3.1 | 4.7 | 4.7 | 6.7 | 11.9 | 34.6 |
| APP, no garbage | 0.27 | 0.51 | 0.68 | 0.77 | 0.88 | 0.85 | 0.98 | 0.94 | 0.99 | 1.01 | 1.04 | 1.15 |
| 50/50: APP (deaths / 16) | 0.26 (16) | 0.37 (16) | 0.53 (11) | 0.50 (11) | 0.58 (1) | 0.56 (2) | 0.60 (0) | 0.62 (0) | 0.64 (2) | 0.65 (1) | 0.72 (0) | 0.80 (0) |

Cost is ~linear in beam and in depth; at a fixed budget depth beats width (5x8 at 3.1 ms ≈ 4x16 at 4.7 ms; 6x4 > 3x16).

**At ~5 ms, CC2's rules** (`bot/tools/cem_vs_cc2_5ms_solo.sh`, `--mode standard --no-180`, 16 games): CEM d4b16 APP 0.768, d5b12 0.905, d6b64 1.117; CC2 5 ms 0.937. 50/50: CEM d5b12 VSfull 0.983 (0 deaths), d4b16 0.838 (3), d6b64 1.093 (0); CC2 5 ms 0.752 (10 deaths).
Caveat: in the solo harness CC2 keeps thinking while our game process works (measured 15.1 ms per move at `--think-ms 5`), so its solo numbers are flattering; the realtime versus harness gives it no free thinking.

**Realtime at equal ~5 ms** (64 matches): CEM-survival d4b16 vs CC2 5 ms **39-25** (188 vs 182 PPS, APP 0.704 vs 0.737); d5b12 vs CC2 5 ms **45-19** (197 vs 183 PPS, APP 0.765 vs 0.763, VS 1.097 vs 1.099). At equal time CEM is ahead; it lost the sweep above only because d6 b64 costs ~35 ms.
Speed or quality? With `--min-ms 6` (both clamped to at least 6 ms per piece; `bot/tools/versus_5ms_matched.sh`), d5b12 still wins **45-19** although slightly slower (157 vs 163 PPS: its occasional >6 ms searches cost pieces), so its edge is move quality (survival). d4b16 drops to 35-29 (within noise of even), so its natural-speed 39-25 was mostly speed.
Replays: `data/bot/versus/realtime/plain_d{4b16,5b12}_vs_cc2_5_{min6,rec}_s*.json` (6 each). `versus_watch.py` keys 5-9 = slow motion x0.05 / x0.02 / x0.01 / x0.005 / x0.002, and the status line shows the piece lead. Watch `plain_d5b12_vs_cc2_5_min6_s50003` (1.46 s, CEM wins 16 pieces behind).
From 2026-10-17 the versus lock line (`bot play-ext --versus`) also reports `hold` and `queue` (next 5 pieces), `versus.py --record` stores them, and the viewer draws a HOLD box and NEXT column per board. Replays with them: `plain_d{5b12,4b16}_vs_cc2_5_{rec,min6_rec}_s*.json`.

Findings:
- **Uncapped, the faster bot wins** almost regardless of quality: matches are decided in well under a second of game time. Blitz turns CEM-survival's 54-10 into 64-0 against the same CC2.
- **It doesn't beat every bot:** CC2 is an anytime search and can play just as fast. At 5 ms per move (184 PPS) it beats the d6 b64 blitz 61-3, because its garbage cancels the blitz plans and forces new searches (98 PPS instead of 181).
- **A cheaper blitz search wins it back:** d6 b16 at 400 PPS goes 59-5, and d4 b16 at 771 PPS goes 64-0.
- **With a TETR.IO-like 333 ms garbage delay at equal speed (185 PPS each), quality decides again:** CC2 5 ms wins 53-11 in long matches.
- Caveat: the speeds are machine-dependent (Windows P-cores here), and "uncapped" means instant inputs (no DAS / ARR or animation costs).

### Register tokens, frozen backbone (Windows, 2026-10-17)

`model.py --registers R`: R learned tokens appended to the board-side sequence (no mask, box or readout). They read and write in all 4 self-attention layers, and the candidates can attend to them in the cross block. Their attention bias is a learned constant per head: `reg_k` (any query to a register), `reg_q` (register to board token), `reg_c` (candidate to register). `train.py --registers R --reg-bias-init B --freeze-base --new-lr 1e-3` warm-starts from a checkpoint without registers and trains only those 4 tensors (~530 numbers at R = 4, d128). `test_registers.py` checks the start and the gradients. Script: `registers_frozen.sh` (`N=... BIAS=...`); results in `results_registers.txt`.

- **Bias start −8** (`v51_d128_reg4frz_ft`): the model starts as dagger5b-ft exactly (KL 2e-7), but the registers get ~0.05% attention and the gradient to the bias is too small. After 3 epochs it had only moved to −7.7, so validation was identical to dagger5b-ft to 3 decimals. Uninformative; play tests skipped.
- **Bias start −2** (`v51_d128_reg4frzb2_ft`): 5% of top-1 moves change at the start (KL 0.013). It recovers to dagger5b-ft's level within the first epoch, and the bias then **drifts shut** (−2 → −2.4 / −2.7): the frozen network learns to ignore the registers, not to use them.

| | val normal / gate4 top-1 | val2 top-1 | APP none (256) | APP d2 | normal 80/20 | normal 50/50 | danger 80/20 | danger 50/50 |
|---|---|---|---|---|---|---|---|---|
| dagger5b-ft | 0.645 / 0.629 | 0.483 | 0.739 | 0.832 | 0.921 (18) | 0.523 (44) | 1.049 (4) | 0.683 (25) |
| + 4 frozen-trained registers (bias −2) | 0.646 / 0.628 | 0.483 | 0.734 | 0.842 | 0.962 (16) | 0.495 (47) | 1.047 (5) | 0.669 (30) |

(VSfull, deaths / 64 in brackets.) All within noise: with a frozen backbone, registers neither help nor hurt. Frozen register training is ~40% faster per epoch (141 s vs 237 s).

**Steps 2-3** (`registers_steps.sh`; 50/50 dropped by the user's choice until a model looks promising):
- **Step 2** `v51_d128_reg4_ft`: dagger5b-ft + registers (bias −2), all weights train, 3 epochs, the dagger5b mix. Compared with `v51_d128_ctrl_ft`.
- **Step 3** `v51_d128_reg4`: from scratch with the `v51_d128` recipe (8 epochs, lr 1.5e-3), bias 0. Compared with `v51_d128`.

Usage check (`register_usage.py`, 5,000 val positions):
- attention share: total weight that board tokens (or candidates, in the cross block) put on the 4 registers; spread evenly it would be ~0.17;
- knock-out: no query may read the registers;
- mean ablation: register states overwritten after every layer by their average over the val positions, which removes only the position-specific content.

| model | attention on registers (self layers 1-4; cross) | knock-out Δ top-1 (normal / gate4) | mean ablation Δ top-1 |
|---|---|---|---|
| frozen, bias −2 | 0.012 / 0.008 / 0.005 / 0.007; 0.009 | 0.000 / 0.000 | +0.001 / 0.000 |
| step 2, unfrozen ft | 0.014 / 0.013 / 0.009 / 0.011; 0.017 | −0.002 / +0.001 | −0.001 / 0.000 |
| **step 3, from scratch** | **0.12 / 0.14 / 0.16 / 0.16; 0.19** | **−0.055 / −0.039** | **−0.028 / −0.029** |

Only the from-scratch model uses its registers. The warm-started ones keep them nearly shut (key bias ~−2.5) because their layers already work without them.

| | val normal / gate4 top-1 | APP none (256) | APP d2 | normal 80/20 | danger 80/20 |
|---|---|---|---|---|---|
| ctrl_ft | 0.645 / 0.626 | 0.739 | 0.832 | 0.918 (24) | 1.046 (4) |
| step 2 reg4_ft | 0.645 / 0.627 | 0.736 | 0.842 | 0.962 (20) | 1.055 (3) |
| v51_d128 (scratch) | 0.640 / 0.626 | 0.651 | 0.770 | 0.394 (62) | - |
| step 3 reg4 (scratch) | 0.643 / 0.628 | 0.645 | 0.786 | 0.366 (58) | - |

Strength is level in both pairs, within noise. From-scratch base models (no DAgger data) die at 80/20 either way.

**Through the DAgger lineage** (`registers_lineage.sh`: `v51_d128_reg4` → `_dagger_ft` → `_dagger3_ft` → `_dagger5b_ft`, the same recipes as dagger5b-ft; 20-25 min GPU):

| stage (val normal / gate4 top-1; val2 top-1) | without registers | with 4 registers |
|---|---|---|
| DAgger 2 | 0.642 / 0.624; 0.418 | 0.642 / 0.624; 0.413 |
| DAgger 3 | 0.639 / 0.628; 0.466 | 0.641 / 0.622; 0.466 |
| DAgger 5b | 0.645 / 0.629; 0.483 | 0.640 / 0.625; 0.486 |

| | APP none (256) | APP d2 | normal 80/20 | danger 80/20 |
|---|---|---|---|---|
| dagger5b-ft | 0.739 | 0.832 | 0.921 (18) | 1.049 (4) |
| `v51_d128_reg4_dagger5b_ft` | 0.733 | 0.867 | 0.912 (24) | 1.041 (5) |

On par with dagger5b-ft (d2 +0.035 is one 64-game run). The registers stay in use: attention 0.12-0.19 (0.19 in the last layer on DAgger 5 positions, above an average token's 0.15); knock-out −5.4 / −4.1 pts top-1 on val, −5.2 / −3.1 on val_dagger5all; mean ablation −2.7 / −2.4 on val, −1.0 / −0.9 on val_dagger5all. **This is the model for register probing and steering.**

**Linear probes on its registers** (`register_probe.py collect --source val|play`, `probe`; `probe_games.sh`; `results_probes.txt`; features in `data/probe/`):
- **Positions:**
  - 52,916 held-out validation positions (the 5 val files);
  - 44,500 positions from the model's own games: 128 at 80/20 + 128 at 50/50, normal style, every 2nd position; 149 deaths.
- **Probe:** logistic regression on one register (128 numbers), all 4 together, or the mean of the board tokens, per layer. It's fitted on 80% of games and scored by AUC on the other 20% (split by seed). Baseline: 6 board features (max / mean height, holes, bumpiness, deepest well, filled cells). Control: shuffled labels give AUC 0.50 / 0.50.
- **Labels for "own move":** the model's own top move, not the teacher's.

| target (play positions; val similar) | best single register | all 4 registers, layer 1 → 4 | board-token mean, L4 | board features |
|---|---|---|---|---|
| hot (height ≥ 12 or ≥ 6 holes) | 0.99 | 0.998 → 0.997 | 0.996 | 0.990 |
| T-structure token present | 0.96 (L1) → 0.999 (L2) | 0.993 → 1.000 | 1.000 | 0.853 |
| own move is a T-spin | **reg 3: 0.960 at L1** (others 0.82-0.87) | 0.977 → 0.983 | 0.951 | 0.719 |
| own move clears lines | 0.87 | 0.821 → 0.899 | 0.855 | 0.665 |
| own move sends attack | 0.93 | 0.896 → 0.957 | 0.895 | 0.680 |
| own move uses hold | 0.60 | 0.586 → 0.608 | 0.592 | 0.563 |
| dies within 10 moves (1.7%) | 0.96 | 0.952 → 0.959 | 0.957 | 0.949 |
| dies within 30 moves (5.2%) | 0.91 | 0.906 → 0.904 | 0.914 | 0.893 |

Reading:
- **Board state** (hot, height, holes) is in every register from layer 1, but it's just as readable from 6 simple board features: a summary, nothing deeper.
- **The model's own next move** (T-spin, clear, attack) is held more strongly in the registers than in the average board token, and grows with depth (attack 0.90 → 0.96). Register 3 is a candidate "T-spin register": 0.96 for the T-spin decision already at layer 1, the others 0.82.
- **Hold** is barely decodable anywhere (≤ 0.61). It's decided in the cross block (candidates compare hold vs no-hold), not on the board side.
- **No foresight of death:** dies-within-10/30 is decodable (0.95 / 0.90) but no better than the board features (0.95 / 0.89). The registers' "danger" is "the board looks bad", not a prediction beyond it. (The model never sees incoming garbage, which limits what it could foresee.)
- For steering, "hot" is a board-state direction. A T-spin / attack (intent) direction from register 3 is the more interesting one to test.

### Next plan (agreed 2026-10-20; 1a and the FLAT-2D / overlay / setup ablations done the same day)

**1. All-spin**
- **1a. CEM-allspins-survival: can an all-spin bot also survive garbage?** (done 2026-10-20, results below)
  - **Rules:** `--mode allspin`. Every spin counts as full and non-T spins score like T-spins (2 / 4 / 6); two consecutive clears of the same type insert a wound row.
  - **Start from** `bot/weights/allspin_slots.txt`. It and `allspin.txt` were tuned on clean boards and don't survive cheese (`bot/README.md`).
  - **Recipe:** as CEM-survival (`bot/results/survive/run.sh`): `bot tune --algo cem --mode allspin --garbage 10,3-4,cheese0.5,0.2 --fitness vs --gens 30 --pop 32 --elite 8 --games 12 --pieces 500 --beam 24 --depth 4`, **without `--freeze default`** (it would zero the all-spin features). ~11,500 games. Script: `bot/results/allspin_survive/run.sh` (portable; started 2026-10-20 on the Ubuntu box, 32 threads).
  - **Where to run:** the Ubuntu box (3970X, `--threads 32`-`64`) is fine. `run.sh` is Windows-only (`bot.exe`, `cygpath`): write a portable copy that sources `attentris/plat.sh`.
  - **Evaluate:** old vs new allspin weights at d6 b64, 64 games each: no garbage, 80/20, 50/50. Deaths, APP, VS, spin share, non-T share, B2B kept and chain length (`bot/tools/spin_stats.py`). The question is whether it still all-spins or just survives by playing safe.
  - **Kicks:** the bot (and Cold Clear 2, `external/cold-clear-2/src/movegen.rs`) uses guideline SRS for 90° turns plus an SRS+ table for 180° (`core/kick_tables.gd`, `bot/src/rules.rs kicks()`). TETR.IO's SRS+ differs mainly in the I piece's 90° kicks. That's minor for tuning, since the bot's game and move generator agree; swap in an authoritative TETR.IO table before playing on TETR.IO itself. "SRS+" isn't one fixed table: the perfect-tetris solver lists separate `srsPlus` / `srsTetrio` sets.
- **1a results (2026-10-20, Ubuntu box, `bot/results/allspin_survive/`, `eval.txt`, `eval_e15.txt`; d6 b64, 64 games, seeds 33001+; replays in `attentris/data/allspin_survive/`):**
  - **Garbage every 10 pieces is too harsh** for an all-spin bot. The tune from `allspin_slots.txt` stayed at fitness ~0.25 (CEM-survival under normal rules: ~0.8), and every all-spin weight set died 64/64 at 80/20 and 50/50 (85-126 pieces). The bots kept B2B (~0.99 kept) and kept spinning until they topped out.
  - **Garbage every 15 pieces (`E=15`, the user's choice) works:** `allspin_survive_e15.txt` reached fitness 1.38 (elite mean, gen 30) and survives:

    | weights (15-piece garbage) | none: died, APP | 80/20: died, APPfull, VSfull | 50/50: died, APPfull, VSfull | spin % of clears (80/20) | B2B kept / mean chain (80/20) | wounds / 100 pieces (80/20) |
    |---|---|---|---|---|---|---|
    | `allspin.txt` | 0, 2.703 | 64, 0.825, 0.878 | 64, 0.500, 0.524 | 93.0 | 0.991 / 46 | 1.66 |
    | `allspin_slots.txt` | 0, 2.714 | 64, 0.738, 0.780 | 64, 0.544, 0.569 | 94.6 | 0.999 / 61 | 1.51 |
    | `allspin_survive.txt` (tuned at E=10) | 0, 2.634 | 50, 1.688, 1.806 | 64, 0.947, 1.004 | 91.9 | 0.998 / 103 | 0.88 |
    | **`allspin_survive_e15.txt`** | 0, 2.201 | **1**, **1.879**, **2.112** | **4**, **1.695**, **1.923** | 79.6 | 0.916 / 11 | 0.85 |

  - The price: it still spins on ~80% of its clears (~78% of the spins non-T) but breaks B2B on purpose (B2B kept 0.92 / 0.88, chains ~8-11 clears instead of 40-100) and gives up ~0.5 APP on clean boards (2.20 vs 2.71). It digs with plain clears between spin chains rather than spinning through the garbage.
  - Its weights moved far: `danger` −0.16 → −30.7, `holes` −0.46 → −6.8, `wounds` −1.8 → −14.3, `b2b` 4.0 → 0.63, while `b2b_next` (keeping the next clear B2B-eligible) is 6.7.
  - Tune cost on the Ubuntu box: E=10 ~4 min (every game dies early), E=15 ~25 min, 32 threads.
- **All-spin weights in a normal-rules game** (`bot play --mode normal --believe allspin`, `bot/results/allspin_survive/believe/`, E=10): `allspin_slots` APP 2.714 → 0.286 (non-T spins become Minis worth 0; B2B still kept); `allspin_survive_e15` 2.201 → 0.499, but it keeps digging: 80/20 4 deaths, 50/50 13 (normal.txt: 18 / 33) at well under half the APP.
- **Rule variants for game design** (the user's: reward the harder spins). Per line, with B2B bonus b paid per clear, a spin double beats a spin single only when d > b + 2s, so with a flat per-clear bonus singles win at any realistic chain:
  - `--mode allspin_s1` (spin single base 1): the existing weights shift only a little (no garbage `allspin_slots` spin doubles 1.3 → 3.2 per 100 pieces, singles 39 → 35; `allspin_survive_e15` doubles 3.7 → 6.6). `bot/results/allspin_s1/`.
  - `--mode allspin_cash` ("singles build, doubles and triples cash"): spin single 1 and no B2B bonus (still extends the chain), spin double 4 + b, spin triple (any piece) 9 + b. Per line 1 / 2 + b/2 / 3 + b/3. Tune and evals: `bot/results/allspin_cash/run.sh` (from `allspin_survive_e15.txt`, E=15; tune fitness 1.14 → 1.18 elite mean). Under `allspin_cash`, E=15, d6 b64, 64 games (`eval_e15.txt`; per 100 pieces):

    | weights | none: died, APP | 80/20: died, VSfull | 50/50: died, VSfull | spin singles / doubles / triples (80/20) | B2B kept (80/20) |
    |---|---|---|---|---|---|
    | `allspin_slots` | 0, 1.478 | 64, 0.298 | 64, 0.257 | 10.0 / 14.6 / 0.25 | 0.996 |
    | `allspin_survive_e15` | 0, 1.297 | 0, 1.410 | 0, 1.289 | 8.8 / 12.5 / 0.71 | 0.795 |
    | **`allspin_cash_e15`** (tuned) | 0, **1.472** | 0, **1.653** | 1, **1.512** | 6.3 / 15.2 / 0.95 | 0.903 |

    The play flips to doubles at once (under `allspin`, singles were ~25-39 and doubles ~1-6 per 100 pieces); the tune adds ~0.2 APP / VS while keeping survival. Spin triples stay rare (~1 per 100 pieces) despite 9 + b: building 3-deep spin slots is hard for this eval (only slot features), so the next try would be a longer tune or a clean-board (attack) tune. APP is lower than under `allspin` (2.2) because singles no longer cash.
  - B2B: our Fibonacci table is the user's fit to TETR.IO's levels (`--b2b tetrio`); they differ by at most one level at a few chain lengths. The user keeps Fibonacci for now (rules can be swapped at inference).
- **1a'. If 1a works: super-allspin** (tentative, the user's idea): the same rules, but a repeated clear type ends the game at once instead of inserting a wound (a penalty of −∞). Needs a rules flag (e.g. `repeat_kills`) that makes a repeat a lock-out in both the game (`game.rs`) and the search (`search.rs`, `Outcome.lock_out`), a `--mode super_allspin`, then the same CEM-survival tune from the allspin-survival weights and the same evaluation. Expect it to trade attack for clear-type variety.
- **1b. If 1a works: all-spin data.** ~500k-1M positions from CEM-allspins plus ~500k from CEM-allspins-survival (`bot dataset` / `bot label --all-roots 1 --beam 128` with `--mode allspin`; smoke-test that spin flags and attack come through first). The split depends on 1a.
  - **Missing input:** the wound rule depends on the *previous* clear's type, which AtTenTris can't see. Add "last clear type" (and maybe "wound pending") as an input before training on all-spin data.
- **1c. Model:** either an all-spin model from scratch, or a 4th style tag (`allspin`) on dagger5b-ft: can one model switch between normal and all-spin play? Undecided.
  - **Per move, the two rule sets don't clash.** The candidates' spin / lines / attack inputs are computed under the active rules: an S-spin double arrives as mini, 0 attack under normal rules and as full, 4 attack under all-spin.
  - **For planning they do.** The value of building an S-spin slot, or of avoiding a repeated clear type, depends on the rules, but the board tokens are identical in both modes, and the style tag only reaches the candidates (never the board side). A mixed model needs a **rules input the board side sees**: a rules token, or a flag on every board token. It also needs the "last clear type" input.

**2. Ablations not done yet** (done: setup tokens, non-T slots, T structures + fit_t, all trained at d64; FLAT-2D at inference only; capacity, cross passes, ToMe, registers, plan heads, value targets, data size; on 2026-10-20 at d128 through the full lineage: FLAT-2D trained off, all overlays off, setup tokens off, non-T slots off, fit_t off, pits off (2026-10-21), see "Ablations at d = 128"):
- A second training seed for the d128 ablations and the baseline (the user skipped it for now): the paired tests cover game noise only.
- A plain **per-cell baseline** (200 cell tokens, no rule tokens, similar size), planned in DESIGN.md but never run. It's memory-heavy: attention and the FLAT-2D pair features grow with tokens², ~40× at 200 vs ~31 tokens. **20 row tokens** (DESIGN.md's other baseline) is the cheap structure-free alternative.
- The candidates' **attack input only** (keep lines and spin type): does the model learn the attack table (T-spin double ≫ double) from the Qs targets? The spin flag stays: whether a move is a spin depends on how the piece got there (the last action a rotation, plus the corner rule), not on its final cells. The same 4 cells can be a plain double or a T-spin double, so removing it would hide information the model can't recover.
- **Head masking with matched behaviour tests** (no training): the cross heads' roles are placement (head 2, −17.2 pts top-1 when off), queue (head 3), surroundings / holes (head 4) and general (head 1). Top-1 is the wrong measure for the specific roles, so pair each head with its behaviour:
  - head 3 off → does the model stop reacting to the previews? Counterfactual test: change the previews and count how often the move changes.
  - head 4 off → rougher stacks in play (bumpiness, holes, well depth).
  - head 2 off → expected to break everything.
  - Use mean-ablation (a head's average output) as well as zeroing it.
- **Preview length** (5 vs 1 vs 0 previews); **head count** (pruning curve first); the DAgger 4-only ablation.

### Next (Windows side and later)

1. **Versus blitz:** CEM-survival `cem:bot/results/survive/survive.txt:6:64:commit6` (and commit 2 / 3) vs CC2, (a) uncapped and (b) at a fixed PPS. Uncapped, ~6x the PPS at -12% APP should cancel most incoming garbage. Fixed PPS is the fair-time check.
2. **Equal-time check:** CEM commit 2 vs a plain search with half the budget (e.g. beam 32): commit only helps if it beats a cheaper search at equal time.
3. **pieces:** the parallel head plateaus (step 2 ~0.38; commit 2 APP ~0.48-0.51). Next idea: condition it on the ACTUAL move 1 (a move-1 token for the piece queries, as autoreg does), so step 2 is predicted given the move played, still as a coordinate (no move generation; a cheap Rust single-placement legality check). If step 2 reaches ~0.55-0.6 and commit 2 costs little, scale data (plan-only teacher labels: ~45 ms a position, 5M positions ~2 h on the Ubuntu box) and model size (d192 / d256).
4. A lean `bot expand` (legal moves only, no structure tokens) for any NN mode that needs move lists at later steps.
5. On a new machine: `cargo build --release` (new `bot label --plan`, `plan-cands`, `play --commit`; 2026-10-20: `--mode allspin_s1 | allspin_cash`), unpack the delta, `python attentris/pack_plan.py` if training autoreg (rebuilds `data/v51plan`).

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
