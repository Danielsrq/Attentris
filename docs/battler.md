# The auto-battler

The right half of the screen. A Tetris quad or spin sends a fighter into an 8-body-deep lane;
fighters march, close on whatever comes into aggro, go around what blocks them, and fight.
One lane, one unit type per side, nothing balanced.

**It takes no player input, and that is the whole point.** This replaced a tower defence that
failed playtesting for a reason worth keeping written down: Tetris already takes nearly all of
a player's attention, so a second board asking for placement decisions competes for the rest
and both halves get worse.

A crowd with flanking is much busier to look at than the single file it started as, so while
it still never competes for the player's *hands*, it can compete for their *eyes*. That is
what the readout is for — a tally and a front-line tick, so the fight is optional viewing.

## Where things live

| file | role |
|---|---|
| `core/steering.gd` | pure static geometry: the swept step test, the escalation ladder, facing |
| `core/battle_field.gd` | the queryable world: lane shape, roster, nearest-enemy, does-it-fit |
| `core/battler_rules.gd` | every tunable, as an exported Resource, plus `spawns_unit()` |
| `core/unit_def.gd` | the two fighters: sheets, clips, frame counts, stats |
| `core/sprite_sheet.gd` | builds a `SpriteFrames` in code from an LPC grid |
| `battler/unit.gd` + `unit.tscn` | one fighter: state machine, targeting, damage, animation |
| `battler/arena.gd` | the right pane: draws it, spawns into the lane, sequences the units |
| `tests/suite_battler.gd` | the pure groups, plus the few that need a tree |

## Two architecture reversals, in opposite directions

The deleted tower defence kept collision in a pure headless sim and made nodes dumb, on the
argument that a central collision pass needs one authority.

**The single-file battler reversed that and used `Area2D`,** correctly: with the lane owning
nothing there was no central pass, each unit detected for itself, and combat was
animation-timed, so the node playing the animation was the natural owner of when a hit landed.

**The RTS rework reversed it back,** because the questions changed. Radial targeting needs
*ranked* candidates — nearest, and if that dies, next-nearest. Hysteresis needs a specific
remembered unit across frames, which is an identity and not an overlap. The step test needs a
swept segment against neighbour positions. `get_overlapping_areas()` answers none of those
well: it returns an unordered set, with no distances, one frame stale. So geometry moved to
`Steering` and `BattleField` as pure static functions, which is also what makes the hard parts
testable with no `SceneTree`.

The half of the original argument that survived — combat is animation-timed — never needed
`Area2D` at all. It is `frame_changed` setting a flag.

**One `Hitbox` area remains, inert.** A radius is the right model for single-target melee, but
a cone, a projectile, or a volume matching red's slash — whose art reaches 49px past its
nominal frame — is genuinely a *shape*, and that is the one job an area still does better than
arithmetic. It has no layer, no mask, no monitoring and a disabled shape until then.

## The scale

Everything derives from one quantum: the LPC body is ~32px wide inside its 64px frame.

| quantity | units | px |
|---|---|---|
| body diameter | 1 | 32 |
| lane depth | **8** | 256 (`ViewLayout.lane_h`) |
| aggro radius | 3 | 96 |
| attack reach | ~1 | 34 |

**Lane depth is the number of bodies that seals the lane,** which makes it the most
load-bearing constant here. At 5 deep, five units form an impenetrable wall and the front line
stops moving — a count reached constantly at these spawn rates, so stalemate would be the
default state rather than an event. 8 costs eight bodies and leaves lateral room for the
rotation search.

Its ratio to aggro decides what *kind* of flanking exists. 256 against 96 leaves no blind
corridor, so a flanker is always seen and a defender peels off to intercept: the flank is a
race decided by distance and numbers, not a gap to walk through.

Sprites are 64px against a 32px body, so units 32px apart visibly overlap. That is ordinary
2.5D, and is what the per-frame depth sort is for.

## Movement

Per unit, per step, in spawn order:

```
desired = toward the target, or the lane flow when there is none
for angle in [0, ±25, ±50, ±75, ±100, ±125, ±150]:   # committed side exhausted first
    if a step along it is free, take it
otherwise hold
```

**The ladder must exhaust one side before trying the other, and that was measured.** The
interleaved version (`0, ±25, ±50…`) was written first on the reasoning that a flat wall blocks
both sides equally. It does not: a wall of bodies has a *scalloped* clearance contour, and a
unit offset from its goal scores the two sides differently, so a far-side rung came up before
the committed side had escalated far enough to climb out, and each frame undid the last. A
unit pressed against eight bodies oscillated in one scallop for **400 frames**, closing 30px
of a 120px gap. Exhausting one side reaches the goal in **109**. Never crossing over at all is
worse still — **951 frames held, wedged** — and it also breaks the flat wall, which is the case
that actually occurs.

**The turn bias is an immutable per-unit constant,** alternated at spawn, not a side chosen at
each block. That is stronger than committing per-block and needs no state: a unit always
deflects the same way, which is what makes the search wall-following rather than greedy
hill-climbing. Alternating it means two units blocked by the same body split and go round
opposite sides, so a crowd splits instead of queueing.

**The blocked test is a swept circle, not a raycast.** A ray from the body centre passes
through gaps narrower than the body, so a unit would commit to a squeeze and wedge at the
opening. Point-to-segment distance against each body is exact for circles and refuses gaps the
body cannot fit.

**Lane bounds are just another obstacle.** Rejecting a step that leaves the lane rather than
clamping it makes the edge behave exactly like a body, so one rotation search handles walls
with no special case. The bounds are inset by the body radius *across* the flow and left wide
open *along* it — confining both axes would seal the lane and nothing could ever leave, so no
teal could retire and no red could breach.

**A closed pocket is a known limitation.** Neither ladder variant escapes a U. It is tolerable
because marching allies form lines, not pockets: a unit inside a crowd is *enclosed*, which is
a different state and is handled by holding still. The test asserts the invariant that makes it
safe rather than an escape that does not happen — no tunnelling into a body, no NaN, no leaving
the bounds.

## Targeting

Four rules, in order. No reachability test and no occupancy count: an unreachable target is
handled by the steering walking around it.

1. **An enemy within reach is the target.** Fight what you are touching. This is also what
   stops a unit orbiting a packed target while something stands right next to it.
2. Otherwise **keep the current target** while it is alive and within aggro × 1.3. Without that
   hysteresis, two near-equidistant enemies swap a unit back and forth forever.
3. Otherwise the **nearest living enemy in aggro**.
4. Otherwise **none** — attack-move along the lane flow.

## Five things that fail silently

Collected because each one costs an hour and none produces an error.

**1. A clip that never starts, for one team only.** Assigning a `SpriteFrames` leaves the
sprite's `animation` set to the first animation it contains, with nothing playing.
`SpriteSheet` builds facings in the order LEFT, RIGHT, UP, DOWN, so the first is always
`walk_1` — walk, facing left. **Red faces left.** So red's opening `_play(&"walk")` asked for
the name already sitting there, a name-only guard skipped the `play()` call, and red slid
across the lane on a frozen frame until its first fight changed the clip and unstuck it. Teal
wanted `walk_3` and never hit it. The guard now checks `is_playing()` too — but it must still
check the name, because `play()` on a running clip resets its frame counter and would restart
a swing before it ever reached its contact frame.

**2. `is_instance_valid()` is the wrong liveness check.** A corpse stays a perfectly valid
instance for its whole death animation. The `alive` flag is set **synchronously** when HP hits
zero, and it is now the *only* thing that makes a corpse stop being a target and stop holding
ground — every `BattleField` query filters on it.

**3. A contact frame of 0 never fires.** `play()` on a clip already at frame 0 emits no
`frame_changed`, so the first swing of every fight would miss. Teal's thrust contacts on 5 of
8, red's slash on 3 of 6, and a test asserts both are above zero.

**4. Damage must not be dealt from `frame_changed`.** That signal fires from the sprite's own
`_process`, outside the simulation step. The handler sets a flag; `advance()` consumes it, so
damage always lands on a step. For the same reason facing is set **once** on entering an
attack and then left alone: re-facing mid-swing restarts the clip, so a unit whose target
drifts across the diagonal would never reach its contact frame — landing no hits while looking
perfectly busy.

**5. A negative `z_index` on a child is an invisibility cloak.** `z_index` is relative to the
parent, and the arena paints an opaque pane fill. The y-jitter that once separated overlapping
allies was fed straight into `z_index`, so every unit in the top half of the lane spawned,
walked, fought and died behind a rectangle. The symptom is "my T-spin didn't send a fighter",
which looks like a spawn-rule bug. Depth now sorts from `y` every frame with a `Z_FLOOR` of 2,
and `_die()`'s decrement is floored at 1.

Two more found through this work that are not about the battler: a `@tool` script writing its
own `position` marks the edited scene modified, so F5 always had a save to do; and
`get_viewport_rect()` reports a square 1280×1280 against a `(0,0)` window under `--headless`,
which laid the lane out 280px low while every boot check passed, because nothing headless calls
`_draw`.

## The sprites

LPC Universal Spritesheet, 64×64 frames, 13 columns × 4 rows, rows being facings in the order
up, left, down, right. Facing is derived from velocity across all four rows; `hurt.png` is
single-row, and a corpse needs no facing.

**Columns and rows are always derived from the texture size, never declared.** One rule covers
three irregularities: `hurt.png` is 832×64 so every facing collapses to row 0, and the two
oversize sheets **disagree** — teal's `thrust_oversize` is 8 columns, red's `slash_oversize` is
6. A shared column constant would have been silently wrong for one character.

**The oversize sheets are used at native size and cannot be scaled down.** The 192px frame is
the 64px frame padded by exactly 64px on every side — same body, same height — so scaling it to
64 would shrink the character to a third. Their overhang differs, and only one needs room:

| sheet | overhang |
|---|---|
| `thrust_oversize` (teal) | none, any side |
| `slash_oversize` (red) | 49px left and right, 17px down |

**Frame counts are declared, not detected**, and are pinned by a test that reads the source art
and checks the last declared frame has pixels and the next is blank — so a re-export cannot
silently start playing blanks. Walk is 9 of 13 columns; playing 13 would stutter every cycle.

**Nearest-neighbour filtering is not an import setting.** In Godot 4 it is a project setting
plus a `CanvasItem` property, and the two enums disagree: `default_texture_filter = 0` means
Nearest, while `CanvasItem.texture_filter = 1` means NEAREST.

## Pacing

| | |
|---|---|
| walk speed | 29.5 px/s — a 21.6 second crossing of the 638px pane |
| frame time | 0.15 s, for walking and dying |
| **attack** | **1.0 s per swing**, divided across the clip's frames |
| red spawn | every 8 s ± 2 |
| HP / damage | 30 / 10, so three hits kill |
| hit flash | 0.2 s, alpha jittered 0.8–1.0 in 0.05 steps, not restartable |

**Attack speed is a stat, not a consequence of how many frames an artist drew.** Pacing the
swing at `frame_seconds` made it the latter: an 8-frame thrust took 1.2 s and a 6-frame slash
0.9 s, so red killed a third faster than teal for no reason anyone chose. The clip is now paced
to finish in `attack_sec` whatever its frame count, leaving the count as purely how smooth the
swing looks. It still sets *when* the hit connects — teal contacts 0.63 of the way through,
red 0.50.

**Walk speed is not a free parameter either.** A 9-frame walk at 0.15 s is a 1.35 s cycle, and
a body that does not cover roughly one stride per cycle visibly slides. The target is ~40px of
ground per cycle, hence 40 / 1.35 = 29.5. This is a live coupling: when `frame_seconds` went
from 0.2 s to 0.15 s, `walk_speed` moved from 22 with it.

## Why there are no invulnerability frames

There were, keyed per (attacker, target) rather than per victim, and that keying was right:
per-*victim* immunity caps incoming damage at `damage / window` however many attackers there
are, so a stack of five would kill no faster than one and clearing more lines would stop
helping you push.

It went because it never did anything. Damage lands on a contact frame, not a cooldown, so a
fighter is already limited to one hit per swing — now 1.0 s for everyone. A 0.3 s window inside
that never once refused a hit it was meant to refuse, and a guard that never fires is
indistinguishable from one that does not work.

What replaced it is purely cosmetic: a 0.2 s flash, alpha jittered between 0.8 and 1.0 in 0.05
steps. **Not restartable** — a hit landing mid-flash is ignored — because re-triggering on
every hit leaves a unit attacked from several sides flickering continuously, which reads as a
rendering fault rather than as damage. It ends at full opacity, and it ticks before the
liveness check so a killing blow still flashes as the body drops.

## How the halves touch

| direction | mechanism |
|---|---|
| Tetris → battler | `LINES_CLEARED` in `main._dispatch`: a quad or any spin sends a teal fighter, so the minigame pays out for exactly the clears worth learning |
| battler → Tetris | a red fighter reaching the near edge is banked as a breach; `main._physics_process` drains it into `Game.receive_garbage` |

The breach is **polled**, not signalled, matching the rest of `main`'s wiring, and drained
*after* the Tetris tick so a fighter spawned by a clear is on the field the same frame. The
resulting `GameEvent`s go back out through `_dispatch` rather than into an array being
iterated — the bug the deleted `_dispatch_defence` had to be fixed for.

**Garbage is the only difficulty curve.** Gravity is a fixed rate (`GravityMode.FIXED`) at every
level, because a rising fall rate is a second, invisible clock on top of the pressure the
minigame already applies, and the two together make a run end for reasons the player cannot
attribute. The level still rises and still feeds scoring; only `guideline()` keeps the curve,
since modelling official behaviour is that preset's whole purpose. Soft drop is only stable
because of this — it multiplies gravity, so under the curve it used to exceed a cell per tick
on its own past about level 6, whatever `sdf` was set to.

**A spawn is queued, never dropped.** Units cannot stack, so a busy entry edge is normal, and a
fighter that silently failed to appear because the edge was crowded would be indistinguishable
from a broken spawn rule. The readout shows the queue as `(+N)`; if that sticks rather than
draining, the lane is jammed and **lane depth** is the lever, not the steering.

## Letting the bot play the left half

The Rust bot in `bot/` is the brain for both hints and autoplay whenever it is built -- there
is no toggle, because there is no reason to choose the worse one. The Godot `BeamSolver` is the
quad-oriented hint search and cannot reach tucks or kicked spins, so it cannot see most of the
all-spin play this economy pays for. It stays as the **fallback**, so a clone with no Rust
toolchain still has hints and autoplay, and the overlay's `brain` row always names which one is
flying. Both answer `solve()` with `Array[Placement]`, so `_stage_placement` never changed.

**F6** draws its plan while you play; **F9** hands it the board; **B** cycles which weight
set; **F2** cycles the rules, and the bot's belief follows F2 automatically.

**Attack per piece is the wrong metric for this half.** `spawns_unit()` fires on a quad or any
spin, and a Mini is still a spin -- so a fighter is sent whether or not the clear pays. The
two come apart hard. Same `allspin_slots` weights, 200 piloted pieces, one seed:

| rules | bot believes | APP | fighters sent |
|---|---|---|---|
| allspin | allspin | 2.15 | 73 |
| allspin_nw | allspin_nw | 2.24 | **75** |
| normal | allspin (mismatched) | 0.24 | 73 |
| normal | normal | 0.84 | **51** |

Attack swings ninefold while spawn output barely moves -- until the bot *believes* normal
rules, at which point it stops spamming spins that no longer pay and sends a third fewer
fighters. So the lever on fighter output is what the bot believes, not what the game pays:
**keep the rules on `allspin` or `allspin_nw` if what you want is a busy lane.**

`cycle_bot` (B) steps through `BotPilot.PRESETS` and **brings the game's rules with it**, so
a weight set can never run under rules it was not tuned for. That pairing is not a nicety:
each set is tuned against one rule mode, and running it under another has no symptom except
playing worse than its own benchmarks. Output ranges from 2.32 APP (`allspin-slots`) down to
0.56 (`gate4`), which makes the toggle a difficulty dial as much as a bot picker.

The bot runs as a held-open child process (`bot serve`), started on first use and buried in
`main._exit_tree` -- a backend outliving the window is the failure mode worth guarding, and the
thread is joined before the kill so a blocking read cannot hang the join.

**It immediately showed the spawn economy is untuned.** The bot clears ~40 spins per 100
pieces, and `spawns_unit()` fires on a quad *or any spin*, so over 400 piloted pieces the lane
held **85 teal against 1 red, with 73 more queued** at the entry edge. The queue draining is
the signal to watch; it did not drain. Those numbers were set against a human, and the levers
are `red_interval`, `spawns_unit()`, and lane depth -- in that order.

## The editor keeps its own list of open files

Deleting the tower defence left `.godot/editor/editor_layout.cfg` still listing
`res://defence_game/scenes/enemy.tscn` in `open_scenes` **and as `current_scene`**. The editor
reopens that list at startup, so it held a scene whose directory no longer existed — and F5
saves modified scenes before running, which is where an "error saving" alert came from. It had
nothing to do with the battler.

`.godot/` is gitignored, so this is per-machine state a `git rm` cannot clean and a fresh clone
never has. Worth knowing the next time a directory is deleted wholesale.

## What is deliberately not done

- **Single target.** `_attack_step()` damages the chosen target only. AoE is a radius query on
  `BattleField`, not a shape.
- **No kill attribution.** `take_damage` takes an amount and nothing else.
- **Teal reaching the far edge does nothing.** No reward for winning the push yet.
- **No pathfinding.** In a straight lane the only obstacles are units, and when an enemy line
  is solid the correct behaviour is to fight it, not route around it. `flow_at(p, team)` and
  `centre_bounds` are the only two things that know the lane is a horizontal rectangle, so
  curved lanes are an implementation swap rather than a rewrite.
- **`switch_board` (TAB) is declared but unread.** A zero-input minigame has no second board.
  `Game.advancing` and `gravity_scale` likewise have no writer; they are kept because a pause
  is obviously useful later, and their tests live in `suite_board.gd`.
