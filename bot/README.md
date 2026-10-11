# tetris bot (Rust)

Heuristic evaluation + beam search on a headless simulator that follows this
project's rules (`core/*.gd`). Output is JSONL in the replay viewer's schema.

```
cargo build --release
./target/release/bot bench  --no-allspin --games 20 --pieces 400          # does search help?
./target/release/bot play   --no-allspin --weights weights/standard.txt \
                            --games 5 --pieces 1000 --out ../data/bot/standard.jsonl
./target/release/bot pros   --no-allspin                                  # pro baseline, same metrics
./target/release/bot parity --data ../data/zenith_full.jsonl              # movegen vs real placements
./target/release/bot tune   --no-allspin --save weights/standard.txt      # CEM self-play tuning
cargo test --release
```

All Godot-viewer replays live in `data/bot/`, whatever produced them. The AtTenTris model runs
are the single combined `attentris_v51_tome_v4_d1.jsonl`: ten games (v5.1, ToMe v4, and the two
DAgger round 2 models, at depth 1 with two garbage runs), picked from the viewer's own run
dropdown. The per-game files were byte-identical slices of it and were dropped. The viewer has no file browser: it opens exactly the
path given, and its dropdown selects a run WITHIN that file, so a replay anywhere else is
simply invisible.

`bot/results/battle/` is not that. Those are match logs (`game`, `players`, `winner`,
`moves`) for the bot-vs-bot harness, not placement records, and the viewer cannot open them.

View a replay (one run per game in the dropdown):

```
godot --path .. res://replay.tscn -- --replay=res://data/bot/standard.jsonl
godot --headless --path .. -s res://tests/bot_replay_check.gd -- --replay=res://data/bot/standard.jsonl
```

## How it decides

    score(node) = attack actually sent on the path  +  V(board at the node)

* **Reward** is exact: `rules.rs` is `attack_table.gd`. The B2B bonus is
  floor(sqrt(chain)), as `b2b_bonus_for()` with `b2b_weight` 1.0; `--b2b flat` gives
  the old +1.
* **V** (`eval.rs`) is a weighted sum of neutral board measurements: holes, surface,
  height (with a quadratic danger term), rows one I away from a quad, T-slots and how
  full their rows are, B2B alive. The cavity of the best T-slot is not counted as a
  hole. Nothing forces a well or a spin; the weights come from `tune`.
* **Search** (`search.rs`): beam over current + 5 previews with hold as a branch,
  transposition dedupe, then one chance ply averaging the best reply over the pieces
  the 7-bag can still deal.
* **Moves** (`movegen.rs`): BFS over left/right/soft-drop/cw/ccw/180 with SRS+ kicks,
  so tucks and kicked spins are reachable. Spin status is decided per rotation, with
  the kick-5 Mini->full upgrade.

## All-spin mode

`--mode allspin` (Godot: `RuleSet.allspin()`, the third preset on the handling toggle):

* every spin is full (Minis promoted); non-T spins score like T-spins (2 / 4 / 6);
* two consecutive clears of the same type (piece + lines + spin; plain clears keyed by
  lines only) insert an unclearable wound line at the bottom, pushing the stack up. It
  counts down per clear and at 0 becomes an ordinary garbage line; its hole follows
  `wound_messiness` (default 0.5). Garbage enters beneath wounds;
* weights: `weights/allspin_slots.txt` (with the S/Z, L/J, I spin-slot features;
  replays in `data/bot/allspin_slots.jsonl`), or the earlier `weights/allspin.txt`.
  Both are tuned on clean boards and do not survive cheese.
* `bot slots [--piece Z --lines 3]` checks the spin-slot bitmask scan against the move
  generator and prints boards and masks as ASCII.
* `godot --headless --path .. -s res://tests/spin_path_check.gd -- --replay=<jsonl>`
  re-finds every spin clear with Godot's own kick tables.

```
./target/release/bot play --mode allspin --weights weights/allspin.txt --games 5 --pieces 1000 --out ../data/bot/allspin.jsonl
./target/release/bot pros --mode allspin
godot --headless --path .. -s res://tests/allspin_parity.gd -- --replay=<allspin jsonl without garbage>
```

## Piloting the Godot board

`bot plan` answers one decision for a running game, so the bot can fly the board live instead
of only producing replays. Live is necessary rather than nicer: the right-hand battler pushes
garbage back into the Tetris board from its own runtime RNG, so a recording desyncs the moment
a red breaches.

```
./target/release/bot serve --mode allspin --weights weights/allspin_slots.txt
            # {"ready":true,..} then one JSON reply per line of state on stdin
./target/release/bot plan  --json --mode allspin --weights weights/allspin_slots.txt                           --state-file <path>     # the same answer, one shot, for debugging
godot --headless --path .. --script res://tools/pilot_bench.gd -- --pieces 400 --seed 101
```

`serve` is what the game runs: the process is held open for the session, so the weights are
parsed once and a decision costs only its search. Spawning per decision was 95ms of a 112ms
round trip against a ~15ms search, which capped piloting at roughly 8 pieces per second.
Measured end to end through Godot, 200 piloted pieces: **12.6s including engine startup**, so
about 53ms a piece against 112ms+ before.

* The state holds `{board, queue, hold, b2b, wound_left, wound_hole, gate}`. Everything in
  it is reward state the board alone cannot carry, and each omission is silent: a missing
  `b2b` costs chain value, missing `wound_*` makes the bot plan into unclearable rows, and a
  missing `gate` cost the gate4 weights their ENTIRE output -- 2.6 APP in `bot play` against a
  flat 0.000 piloting Godot, because under gate rules `r.attack` comes only from a completed
  chain and a bot that believes the gate is shut never completes one.
* `--state-file` is the one-shot form of the same thing. A **file**, not a
  `--state` argument: the state is JSON, and a quoted JSON blob in a Windows process argument
  is mangled before this binary sees it. Both forms are accepted; the file is what Godot uses.
* `wound_*` travel because `--board` cannot express an unclearable row, and under all-spin
  rules they are real -- `RuleSet.allspin()` turns `repeat_penalty` on.
* Output is the replay schema's `plan`, so there is no second wire format. Godot acts on each
  step's **cells** and re-derives rotation and kick with its own tables
  (`solver/path_finder.gd`), so neither side trusts the other's coordinate conventions.
  Certified by `tests/spin_path_check.gd`: 2068 recorded spin placements, 0 not reproducible.
* Bad input is reported as `{"error": ...}` on stdout and the loop continues, never as a
  panic. A panic puts
  the reason on stderr, where a parent process reading stdout learns only "exit 101".

Measured in-game against this binary's own numbers, 400 pieces of all-spin:

| | APP | deaths |
|---|---|---|
| `bot play` | 2.582 | 0/8 |
| piloting Godot | 2.550 | 0 |

`BotPilot.PRESETS` is weight sets, and each one carries the rules it was tuned against as its
`--mode`. The mode is a property of the WEIGHTS, not of the running game: `allspin_slots.txt`
searches under all-spin even when the Godot game is scoring normal rules, because a reward
model the weights were fitted against is not separable from them.

Syncing the mode to the game's rules instead was tried and is fatal -- all-spin weights under
a normal-rules reward topped out every run of a three-seed, 500-piece test at around piece
300, against no deaths and 2.59 APP with a coherent mode. `pilot.set_belief()` overrides it
deliberately, in either direction, and says so on screen.

The weight sets, 250 piloted pieces each, on the rules each one was tuned against:

| bot | rules | APP in Godot | note |
|---|---|---|---|
| `allspin-slots` | allspin | 2.32 | strongest; all-spin pays for every spin |
| `allspin` | allspin | 2.30 | the earlier set, no spin-slot features |
| `normal` | normal | 0.97 | 5 quads; non-T spins are Minis |
| `standard` | guideline | 0.77 | 8 quads -- a quad-well bot, a different game |
| `gate4` | gate_b2b | 0.56 | tuned with garbage; gate pays only for chains |

`gate4` against 0.612 in `bot play` is the pairing working; it read 0.000 until `gate` was
added to the state.

`--mode allspin_nw` is all-spin scoring with `repeat_penalty` off: every spin still Full and
still paying like a T-spin, but a repeated clear type no longer inserts a wound line. Worth a
mode of its own because the obvious alternative is wrong -- wounds come from the repeat
penalty, not from all-spin, so dropping to `--mode normal` to lose them also makes every non-T
spin a Mini that sends nothing. Measured through Godot on one seed: **0.24 APP against 2.19**,
a ninefold collapse, because an all-spin bot's output is almost entirely non-T spin singles.

Within 2%, which is what says the rules pairing and the B2B chain are crossing the wire
intact. A large gap is the symptom to watch for; it has no other.

## Normal-rules weights

`weights/normal.txt`: tuned under normal rules with fibonacci B2B (CEM, 20 generations
from `standard.txt`). Held-out 1.26 APP vs 0.93 for `standard.txt`, no deaths; replays in
`data/bot/normal.jsonl`. The normalised-vs-raw feature comparison that produced it is in
`results/cem_norm_vs_raw/` (normalised converged faster early, both ended within noise).

Weight files are now written for normalised features (`#! features normalised`); older
files without the marker are raw and are converted on load. `bot cpus` shows the CPU
layout; the bot pins itself to the performance cores unless `--any-cores`.

## Rules switches

* default (`--mode normal`, Godot `RuleSet.normal()`, the game's default preset): every
  piece can spin, non-T spins are Minis; every Mini sends 0 (no B2B bonus either) but
  keeps B2B alive.

* default: `RuleSet.competitive()`: all-spin on (non-T immobile spins score 2/4/6).
* `--no-allspin`: T-spins only, as `RuleSet.guideline()`.
* `--garbage E,R,M`: R rows every E pieces, hole moves with probability M.
  `--cheese N`: N messy rows at the start.

Not modelled: lock-delay reset limits, gravity, combos (off in `attack_table.gd`).
