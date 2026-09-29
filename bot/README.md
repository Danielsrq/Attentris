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

## Rules switches

* default (`--mode normal`, Godot `RuleSet.normal()`, the game's default preset): every
  piece can spin, non-T spins are Minis; every Mini sends 0 (no B2B bonus either) but
  keeps B2B alive.

* default: `RuleSet.competitive()`: all-spin on (non-T immobile spins score 2/4/6).
* `--no-allspin`: T-spins only, as `RuleSet.guideline()`.
* `--garbage E,R,M`: R rows every E pieces, hole moves with probability M.
  `--cheese N`: N messy rows at the start.

Not modelled: lock-delay reset limits, gravity, combos (off in `attack_table.gd`).
