# The auto-battler

The right half of the screen. A Tetris quad or spin sends a fighter; fighters march, meet
and fight on their own. One lane, one unit type per side, nothing balanced.

**It takes no player input, and that is the whole point.** This replaced a tower defence
that failed playtesting for a reason worth keeping written down: Tetris already takes
nearly all of a player's attention, so a second board asking for placement decisions
competes for the rest and both halves get worse. Here the right half is a readout with
stakes, and its state is one legible thing — where the front line is — which can be read
peripherally without looking away from the stack.

## Where things live

| file | role |
|---|---|
| `core/battler_rules.gd` | every tunable, as an exported Resource, plus `spawns_unit()` |
| `core/unit_def.gd` | the two fighters: sheets, clips, frame counts, stats |
| `core/sprite_sheet.gd` | builds a `SpriteFrames` in code from an LPC grid |
| `battler/unit.gd` + `unit.tscn` | one fighter: state machine, detection, damage |
| `battler/arena.gd` | the right pane: draws it, spawns into the lane, nothing else |
| `tests/suite_battler.gd` | the pure groups plus four tree tests |

## The architecture reversal

The deleted tower defence kept collision in a pure headless sim and made nodes dumb. The
argument was that a central collision pass needs one authority, and `Area2D` would be a
second source of truth resolved a frame late.

**That reason does not apply here, so this is the project's first `Area2D` usage.** The
lane owns nothing, so there is no central pass — each unit detects for itself. And combat
is animation-timed: the hit should land on the frame the staff connects, which makes the
node that plays the animation the natural owner of when damage happens.

It is also the project's first `AnimatedSprite2D`, first `SpriteFrames` and first texture
pipeline. There was no precedent to copy.

## Seven things that fail silently

Collected because each one costs an hour and none of them produces an error.

**1. Area wiring.** Only the HURTBOX ever sits on a collision layer:

| | layer | mask | monitoring | monitorable |
|---|---|---|---|---|
| `Hurtbox` | own team | 0 | **no** | **yes** |
| `Hitbox` | 0 | enemy | **yes** | **no** |
| `Sight` | 0 | enemy | **yes** | **no** |

Making the hurtbox the only findable thing buys two properties free: disabling a corpse's
hurtbox makes it untargetable *and* stops it blocking in one operation, and friendly fire
is structurally impossible rather than a code check.

Get `monitoring`/`monitorable` backwards and **nothing reports an overlap** — no error, no
warning, units simply walk past each other forever. It reads exactly like a state-machine
bug. `BattlerUnit.wiring_errors()` checks it at startup and a test checks it with no tree.

**2. `is_instance_valid()` is the wrong liveness check.** A corpse stays a perfectly valid
instance for its whole death animation — 6 frames at 0.15 s is 0.9 seconds of attackers
swinging at a body while the next enemy walks past. The `alive` flag is set
**synchronously** when HP hits zero, before the deferred shape disable, and that flag is
what everything checks.

**3. A contact frame of 0 never fires.** `play()` on a clip already sitting at frame 0
emits no `frame_changed`, so the first swing of every fight would miss. Teal's thrust
contacts on 5 of 8, red's slash on 3 of 6, and a test asserts both are above zero.

**4. Damage must not be dealt from `frame_changed`.** That signal fires from the sprite's
own `_process`, outside the physics step, so querying overlaps there interleaves damage
with rendering non-deterministically. The handler sets a flag; `_physics_process` consumes
it.

**5. A `@tool` script that writes its own `position` dirties the scene.** The arena first
carried the pane offset on its transform, which looked tidier than offsetting every draw
call. But `setup()` runs at editor load too, so writing `position` **modified
`main.tscn`** — it came up dirty every time it was opened, and F5 saves modified scenes
before running, so pressing play had a save to do that nobody asked for. No other view node
in the project touches its transform; `BoardView` draws in absolute screen coordinates, and
the arena now does the same. The hazard the transform was dodging is real but belongs to
the *drawing*: offsetting only `_draw` while leaving entity positions pane-local is exactly
how the deleted tower defence put every entity on top of the Tetris board. Keeping drawing
and spawning in the same absolute space removes the mismatch instead of hiding it.

**6. A negative `z_index` on a child is an invisibility cloak.** `z_index` is relative to
the parent by default, and the arena paints an opaque fill in its own `_draw`. The y-jitter
that separates overlapping allies was fed straight into `z_index`, giving every unit that
landed in the **top half of the lane** a negative z — so it spawned, walked, fought and died
exactly as designed, behind a rectangle. Roughly half of all fighters were never seen, and
the symptom is "my T-spin didn't send a fighter", which looks like a spawn-rule bug and is
not. The jitter is now biased by half the play so the floor is `Z_FLOOR` (2), and `_die()`'s
decrement is floored at 1 rather than allowed to sink out of sight.

**7. An un-awaited coroutine suite reports success having run nothing.** `suite_battler`
needs physics frames, so its `run()` is a coroutine. If `run_tests.gd` called it plainly
the coroutine would suspend, the runner would sum the counters before it had run, and the
suite would report **"0 passed, 0 failed" with exit code 0**. The runner therefore does
`await suite.run()`, which is correct for sync suites too.

## The sprites

LPC Universal Spritesheet, 64×64 frames, 13 columns × 4 rows, rows being facings in the
order up, left, down, right. So teal walking right is row 3 and red walking left is row 1 —
no mirroring needed.

**Columns and rows are always derived from the texture size, never declared.** One rule
covers three irregularities: `hurt.png` is 832×64 so every facing collapses to row 0, and
the two oversize sheets **disagree** — teal's `thrust_oversize` is 8 columns, red's
`slash_oversize` is 6. A shared column constant would have been silently wrong for one
character.

**The oversize sheets are used at native size, and cannot be scaled down.** The 192px frame
is the 64px frame padded by exactly 64px on every side — same body, same height — so
scaling it to 64 would shrink the character to a third. Used as-is it costs nothing and
draws more of the weapon: teal's staff is 40px of art against 27px in the standard sheet.
Their overhang past the nominal frame differs, and only one needs room:

| sheet | overhang |
|---|---|
| `thrust_oversize` (teal) | none, any side |
| `slash_oversize` (red) | 49px left and right, 17px down |

So red's swing can reach ~11px outside a pane with 38px of slack, at the very edge. That is
cosmetic; clip the lane if it shows.

**Frame counts are declared, not detected** — detecting would mean decoding a PNG at
startup to recover a constant. They are pinned by a test that reads the source art and
checks the last declared frame has pixels and the next is blank, so a re-export from the LPC
generator cannot silently start playing blanks. Walk is 9 of 13 columns; playing 13 would
show four blank frames as a stutter every cycle.

**Nearest-neighbour filtering is not an import setting.** In Godot 4 it is a rendering
project setting plus a `CanvasItem` property, so `.png.import` is the obvious wrong place to
look and checking it costs a reimport. Both are set, and note the two enums disagree:
`default_texture_filter = 0` means Nearest, while `CanvasItem.texture_filter = 1` means
NEAREST.

## Headless cannot see the layout

`get_viewport_rect().size` with no window reports a square **1280x1280** against a `(0, 0)`
window, so the headless boot check was laying the lane out 280px below where it belongs —
and because nothing headless ever calls `_draw`, every boot check passed while the layout
was wrong. `main._layout_viewport()` and `replay._layout_viewport()` therefore fall back to
the project's configured window size for `DisplayServer.get_name() == "headless"` as well as
for the editor. A boot check that cannot see the layout is worth making see the real one.

## Pacing

| | |
|---|---|
| walk speed | 29.5 px/s — a 21.6 second crossing of the 638px pane |
| frame time | 0.15 s for every clip, retimeable live via `SpriteSheet.apply_speed` |
| red spawn | every 8 s ± 2 |
| HP / damage | 30 / 10, so three hits kill |
| hit flash | 0.2 s, alpha jittered 0.8–1.0 in 0.05 steps, not restartable |

**Walk speed is not a free parameter.** A 9-frame walk at 0.15 s is a 1.35 s cycle, and a
body that does not cover roughly one stride per cycle visibly slides or skates. A 64px frame
carries a ~32px body and an LPC stride covers about one to one and a half of those, so the
target is ~40px of ground per cycle — hence 40 / 1.35 = 29.5 px/s.

This is a live coupling, not a note: when `frame_seconds` went from 0.2 s to 0.15 s,
`walk_speed` moved from 22 to 29.5 with it. Leaving it at 22 would have cut the stride to
30px per cycle, which reads as the feet slipping on a treadmill.

## Why there are no invulnerability frames

There were, keyed per (attacker, target) rather than per victim, and the argument for that
keying was sound: per-*victim* immunity caps incoming damage at `damage / window` however
many attackers there are, so a stack of five would kill no faster than one and clearing more
lines would stop helping you push.

It was removed because it never did anything. Damage lands on an animation's **contact
frame**, not on a cooldown, so a fighter is already limited to one hit per swing — 1.2 s for
a teal thrust, 0.9 s for a red slash at 0.15 s a frame. A 0.3 s window inside a 0.9 s cadence
never once refused a hit it was meant to refuse. A guard that never fires is
indistinguishable from a guard that does not work, so it went rather than staying as
insurance.

**The cadence now comes from the art**, which is why red kills faster than teal: 6 slash
frames against 8 thrust frames is 2.7 s to kill versus 3.6 s, with no stat saying so. If
that asymmetry turns out to be wrong, the lever is the declared frame count, not a cooldown.

What replaced it is purely cosmetic: a 0.2 s flash, the sprite's alpha jittered between 0.8
and 1.0 in 0.05 steps. It is **not restartable** — a hit landing mid-flash is ignored —
because re-triggering on every hit leaves a unit attacked from several sides flickering
continuously, which reads as a rendering fault rather than as damage. It ends at full
opacity, and it ticks before the liveness check so a killing blow still flashes as the body
drops.

Allies now collide and cannot stack, which was the other half of the old argument. Several
attackers still reach one target, but they arrive from different sides rather than occupying
the same ground.

## How the halves touch

| direction | mechanism |
|---|---|
| Tetris → battler | `LINES_CLEARED` in `main._dispatch`: a quad or any spin sends a teal fighter. Singles and doubles send nothing, so the minigame pays out for exactly the clears worth learning |
| battler → Tetris | a red fighter reaching the left edge is banked as a breach; `main._physics_process` drains it and calls `Game.receive_garbage` |

The pane also carries a small readout — the title, a teal-vs-red living tally, and a line
telling you what sends a fighter when the field is empty. Without it the right half is 80px
of near-black on a near-black background, which does not read as a minigame waiting for
input; it reads as a half that failed to load.

The breach is **polled**, not signalled, matching the rest of `main`'s wiring, and it is
drained *after* the Tetris tick so a fighter spawned by a clear is on the field the same
frame. The resulting `GameEvent`s go back out through `_dispatch` rather than into an array
being iterated — the bug the deleted `_dispatch_defence` had to be fixed for.

`Game.receive_garbage` is reused unchanged, including both of its deaths: `garbage_bury`,
checked *before* pushing so the board is never left half-applied, and `garbage_block_out`.

## The editor keeps its own list of open files

Deleting the tower defence left `.godot/editor/editor_layout.cfg` still listing
`res://defence_game/scenes/enemy.tscn` in `open_scenes` **and as `current_scene`**, with
`view/defence_view.gd` in `open_scripts` and `selected_script`. The editor reopens that list
at startup, so it held a scene whose directory no longer existed — and F5 saves modified
scenes before running, which is where the "error saving" alert came from. It had nothing to
do with the battler.

`.godot/` is gitignored, so this is per-machine state that a `git rm` cannot clean and a
fresh clone never has. Worth knowing the next time a directory is deleted wholesale: the
quick fix is closing the phantom tab in the editor, and the file fix is pruning those keys.

## What is deliberately not done

- **Single target.** `_strike()` returns after the first hit; AoE is one line away.
- **Teal reaching the right edge does nothing.** No reward for winning the push yet.
- **No pooling.** The deleted defence view pooled because a freed node with a live id in an
  event loop is a bug class; there are no ids here, attackers hold direct references and
  check `alive`, so pooling would only import stale-state-on-reuse.
- **`switch_board` (TAB) is still declared but unread.** A zero-input minigame has no second
  board to switch to. `Game.advancing` and `gravity_scale` likewise have no writer any more;
  they are kept because a pause is obviously useful later, and their tests moved to
  `suite_board.gd`.
