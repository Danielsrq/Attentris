# The defence minigame

The right half of the screen. Enemies drift left across an 18×20 grid; the player builds
cannons that fire bouncing energy balls; enemies that reach the left wall are converted
into garbage rows pushed into the Tetris board.

That last part is the point. The two halves are one game because failure on the right
makes the left harder, and because Tetris line clears fire the cannons.

## Where things live

Pure logic in `core/`, mirroring the Tetris sim — `RefCounted`, no `Node`, no rendering,
so the whole loop is testable headlessly.

| file | role |
|---|---|
| `core/defence_rules.gd` | every tunable, as an exported `Resource` |
| `core/aim.gd` | the 16 directions, as an integer lattice |
| `core/tower_data.gd` | ASCII masks — the only place tower shapes are declared |
| `core/tower_set.gd` / `tower_def.gd` | the compiler and the compiled product |
| `core/muzzle.gd` | where a shot leaves the hull, precomputed 4×16 per tower |
| `core/defence_field.gd` | grid occupancy |
| `core/defence_tower.gd` / `_enemy.gd` / `_shot.gd` | the entities |
| `core/garbage_buffer.gd` | the 2 s window between a breach and the garbage |
| `core/build_cursor.gd` | placement intent, always kept legal |
| `core/defence_sim.gd` | owns all of it; `tick(delta) -> Array[DefenceEvent]` |
| `core/defence_event.gd` | what the view consumes |
| `view/defence_view.gd` | grid and ghost in `_draw()`, entities as pooled scenes |
| `defence_game/scenes/*.tscn` | the art-bearing nodes |
| `assets/defence/*.svg` | placeholders |

## Four decisions worth knowing before changing anything

**Direction is an integer, never a vector.** `Aim` stores an index 0–15 and reflects by
index arithmetic: `reflect_x(d) = (8 - d) & 15`, `reflect_y(d) = (16 - d) & 15`. So after
six bounces a shot is still exactly on the lattice and its speed is exactly unchanged,
because the magnitude is never recomputed. **Enemies therefore collide as axis-aligned
boxes**, however round the art is — a circle would reflect along the centre-to-centre
normal, produce an arbitrary angle, and make every bounce after it unplannable. That is
easy to "improve" and break.

**A bounce is a hit, so bounces are the damage budget.** One bounce = 10 damage, enemies
have 30 HP, so a kill costs three. Bounces spent on walls and towers are bounces not spent
on bodies, which is where the skill is. It also means `max_bounces` cannot go at or below
3 or a tower can never finish anything.

**Collision is a discrete per-tick overlap test and that is sound, not lazy.** At 240 px/s
and 60 Hz a shot moves 4 px against a 30 px cell, so tunnelling is impossible by a factor
of ten. The real hazard is the opposite: at 4 px/tick with a 6 px radius a shot stays
overlapping for three or four ticks, so without de-penetration it would spend its whole
budget on one wall. Pushing out to `surface ± (radius + ε)` is what makes the budget mean
anything. `DefenceRules.step_is_safe(cell)` encodes the limit, and `substeps` buys it back
if the speed is ever raised far enough to matter.

**No `Area2D`, no physics shapes.** The sim must own collision to be testable without a
scene, and a physics body would be a second source of truth that disagrees with it —
different shapes, and overlaps resolved on the physics step with results a frame stale,
which cannot give the exact contact a bounce needs the tick it happens. Entity scenes
carry `Sprite2D` and `AnimationPlayer` and nothing else.

## The departure from the project's grain

Everything else in this project is immediate-mode: `BoardView`'s own doc explains why
coloured squares are better as `draw_rect` than as nodes. The defence side is the first
place with instanced scenes and the first with textures, and that is deliberate — a
`_draw()` call cannot have a PNG swapped into it.

The dividing line is "will this ever be art?". The grid, gridlines, frame and build ghost
never will, so they stay in `_draw()`. Towers, enemies and shots will, so they are scenes.

Nodes are **pooled**, and not for speed — fifteen live shots would be fine instanced per
frame. Reuse removes the class of bug where a node is freed while an event loop still
holds its id, and `visible = false` plus reuse is less code than guarding against freed
instances. Sim ids are monotonic and never reused, so an id can never alias.

Entity lifecycle arrives as `DefenceEvent`s; positions are read straight off the sim each
frame. The other way round would mean an event per shot per tick.

Textures live on the `Sprite2D` nodes inside the scenes, not behind `@export`s on the
root. An export leaves the sprite you can see in the editor empty until runtime, which is
the opposite of what an art-swappable scene should feel like — open the scene, click the
sprite, change the texture. The tower's body is tiled per footprint cell, so its scene
carries one `Body/Tile` sprite that the script duplicates; anything you change on that
tile carries to every cell.

Node scripts resolve their children with `get_node_or_null` on first use rather than
`@onready`. A pooled node is set up immediately after `add_child`, and if its *parent* is
not in the tree yet then `_ready` has not run and every `@onready` is still null.

## How the halves touch

| direction | mechanism |
|---|---|
| Tetris → defence | `LINES_CLEARED` in `main._dispatch` calls `grant_free_shots(1)` — one extra shot per tower that does **not** reset any cooldown, so clearing lines can only add damage |
| defence → Tetris | `GARBAGE_FLUSHED` in `main._dispatch_defence` calls `Game.receive_garbage`, which pushes the rows and decides whether they kill |

Tetris ticks **first** within a frame, so a clear produced this tick grants its shot this
tick rather than next.

`Game.receive_garbage` exists because `Board.push_garbage` cannot kill you: it discards
whatever it shifts past row 0 and its own doc calls that "a top-out in real play", but
nothing detected it. Two deaths are named separately — `garbage_bury` (checked *before*
pushing, so the board is never left half-pushed) and `garbage_block_out`.

## Focus

TAB (`switch_board`) swaps which half the keys drive. **Tetris keeps simulating either
way** — it is handed a cleared `InputState` rather than being skipped, so gravity, lock
delay, ARE and spawning all keep running and the piece in play lands by itself. That is
the multitasking pressure the design is built around.

**Tetris FREEZES while the defence board has focus.** `Game.advancing = false` halts
everything that progresses on its own — gravity, the lock timer and the entry delay — and
the lock timer *resumes* rather than resetting, so the pause is not free lock delay.

This replaced an earlier design where gravity was merely scaled down and the multitasking
pressure was the point. Playtesting killed it: Tetris already takes nearly all of a
player's attention, so a second board competing for the rest made both halves worse.

Note the trap it walked into, because it is easy to walk back into. Scaling gravity stops
the FALL but not the LOCK — a piece already resting on the stack is governed by
`lock_delay_sec`, so with gravity alone it locks itself half a second after you look away.
That, not the falling, is what reads as "Tetris carried on without me".
`DefenceRules.unfocused_gravity_scale` survives as a knob but is no longer what pauses
anything.

## Pacing

The board is deliberately slow, and this is the number everything else follows from:

| | |
|---|---|
| enemy speed | 12 px/s — a 50 second crossing of the 600px field |
| spawn interval | 6 s — about eight bodies alive at once |
| fire interval | 3 s — raised less than the slowdown, so each body takes ~17 shots |
| projectile speed | 180 px/s, 4 s lifetime — about 720px of travel |

At 40 px/s the crossing was 15 seconds, which meant the player *had* to look at the right
half every fifteen seconds or leak. That is unaffordable next to Tetris. At 50 seconds the
board can be ignored for fifteen pieces and the cost is ground lost, not a loss.

Tests derive their tick budgets from these rules rather than hardcoding them — a 1200-tick
loop tuned to the old speed silently stopped reaching the wall when the board slowed.

Defence keys reuse the Tetris keys because focus is exclusive. They are separate InputMap
actions rather than one shared set because the vertical axis cannot be shared: in
`wasd_numpad`, `soft_drop` is **W** — semantically down, physically up — so a shared
action would invert the cursor between profiles. See `docs/input_map.md`.

## What is deliberately not done

- **Enemies pass through towers.** Collision would make towers walls and the game
  maze-building, which needs pathfinding and would let maze-building dominate
  ricochet-routing as the skill. Incremental, later.
- **Towers are free.** `max_towers` (10) stands in for cost so playtests still have a
  ceiling. `Resources.can_afford`/`spend` is the eventual gate and is already atomic.
- **Nothing is balanced.** Every number in `DefenceRules` is chosen to make the mechanics
  visible, not fair.
- **Enemy `y` never changes.** Lanes are spawn positions, not rails — the field is
  continuous and knockback would just work.
