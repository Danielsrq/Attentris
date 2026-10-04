class_name DefenceRules
extends Resource

## Every tunable of the defence minigame in one exported resource, so tuning is data
## rather than code -- the same bargain RuleSet makes for the Tetris half.
##
## None of these numbers are balanced. They are chosen to make the mechanics VISIBLE: a
## loop you can watch and judge. The one relationship that is not free to change is
## called out on `max_bounces`.

enum GarbageStyle { CLEAN, CHEESE }

@export_group("Grid")
## Columns and rows of the placement grid. Height x width in this order to match how the
## Tetris field is described (40x10). 18 rows of the playfield's 30px cell is 540px, which
## is exactly what remains of the right pane under the palette strip -- so these two
## numbers are what decide whether the board fits at all. See ViewLayout.defence_rect.
@export var rows: int = 18
@export var cols: int = 20

@export_group("Enemies")
## Pixels per second, leftward. 12 crosses the 600px field in 50 seconds.
##
## Slow on purpose, and the number this whole pacing pass turns on. Tetris takes nearly all
## of a player's attention, so a board that must be looked at every 15 seconds -- which 40
## px/s meant -- competes with it and both halves get worse. At 50 seconds you can ignore
## the right half for fifteen pieces and lose ground rather than lose.
##
## The DEFAULT a spawned body is given. Each DefenceEnemy carries its own `speed` from
## then on, so changing this affects the next spawn rather than retroactively speeding up
## everything already on the field -- and an enemy type can override it.
@export var enemy_speed: float = 12.0
@export var enemy_hp: int = 30
## Half-extents of an enemy's collision box: the distance from its centre to each edge,
## so (12, 12) is a 24x24 body. A BOX, not a circle, even when the art is round -- see the
## class doc on Aim for why a circle would break the 22.5 lattice.
@export var enemy_half_extents: Vector2 = Vector2(12.0, 12.0)
## Seconds between spawns. Scaled with the crossing time so the field still holds about
## eight bodies -- enough to be a situation, few enough to read in a glance, which is the
## only kind of looking the player can afford.
@export var spawn_interval: float = 6.0

@export_group("Projectiles")
@export var projectile_speed: float = 180.0
@export var projectile_radius: float = 6.0
## Seconds before a shot expires regardless of what it has hit. Raised with the slower
## shot so the travel budget stays about 720px -- a little over one field width, which is
## what keeps lifetime the binding limit in open space and bounces the binding limit in
## tight geometry.
@export var projectile_life: float = 4.0
## Bounces before a shot expires. One bounce is one hit, so this is also the damage
## budget, and that is why it cannot be small: at 10 damage against 30 HP a kill costs
## three hits, so anything at or under 3 can never finish an enemy once a single bounce
## has gone into a wall. 6 gives a ceiling of two kills and about one in practice.
##
## Bounces and lifetime deliberately bind in different places -- lifetime kills a shot
## drifting through open space, bounces kill one rattling inside tight geometry. If one of
## them never fires in play, that one is mistuned.
@export var max_bounces: int = 6
@export var hit_damage: int = 10
## Collision is a discrete overlap test once per tick, with no sweeping, which is sound
## only while a shot cannot cross an obstacle in one step. At 240px/s and 60Hz a step is
## 4px against a 30px cell, so there is a factor of ten in hand -- but raise the speed far
## enough and shots start passing through towers. `substeps` buys that back: the budget is
##     projectile_speed / 60 / substeps  <  cell - 2 * projectile_radius
## which at cell 30 and radius 6 means a step under 18px.
@export var substeps: int = 1

@export_group("Towers")
## Seconds between a tower's automatic shots. Raised less than the enemy slowdown, so each
## body now takes about 17 shots per crossing instead of 8: fewer, slower enemies each
## under sustained fire reads better at a glance than a stream of brief ones.
##
## A Tetris line clear grants one extra shot on top of this WITHOUT resetting the timer, so clearing lines can only ever add damage --
## if a clear reset the cooldown, clearing would feel like a penalty.
@export var fire_interval: float = 3.0
## Stands in for cost while towers are free. Without a ceiling a mechanical playtest fills
## all 90 slots, trivially wins, and teaches nothing about whether the loop has tension.
@export var max_towers: int = 10

@export_group("Garbage")
## Seconds the window stays open after the first enemy reaches the left wall. Every enemy
## that arrives while it is open adds a row, then the whole buffer is sent at once. The
## window doubles as the telegraph: the player watches bodies pile up against the wall and
## knows what is coming and roughly when.
@export var window_sec: float = 2.0
## Rows sent per flush. Overflow is carried, not dropped -- eleven breaches send 8 and
## keep 3. Matches TETR.IO's garbagecapmax, and without it one bad wave is simply fatal.
@export var flush_cap: int = 8
## Clean garbage shares one hole column across every row, which is effectively a free
## well. Cheese varies it and is the case that actually tests downstacking.
@export var style: GarbageStyle = GarbageStyle.CLEAN
## Chance per flush that the hole column steps one across. 0.1 is the value measured in
## real Zenith replays (`messiness_change`), so the drift is not invented.
@export_range(0.0, 1.0, 0.01) var hole_shift_chance: float = 0.1
## Carried overflow re-opens the window so it drains on its own. Without this, three rows
## carried from a bad wave sit forever if nothing ever breaches again.
@export var drain_carry: bool = true

@export_group("Focus")
## Passive gravity while the defence board has focus. 0 means stopped.
##
## Kept as a knob rather than hard-coded, but note it is no longer the thing that pauses
## Tetris: `Game.advancing` is, and it freezes the lock timer and the entry delay too.
## Scaling gravity alone left a grounded piece locking itself half a second after the
## player looked away, which reads as Tetris carrying on without them.
@export_range(0.0, 1.0, 0.05) var unfocused_gravity_scale: float = 0.0

@export_group("Meta")
@export var preset_name: String = "testing"

## The mechanical-testing preset: the export defaults, which are what every number above
## is documented as. Named rather than implied so a call site reads as a choice.
static func testing() -> DefenceRules:
	var r := DefenceRules.new()
	r.preset_name = "testing"
	return r

## Pixel size of the whole grid, at the playfield's cell size.
func pixel_size(cell: int) -> Vector2:
	return Vector2(float(cols * cell), float(rows * cell))

## Seconds a projectile step covers, honouring `substeps`.
func substep_delta(delta: float) -> float:
	return delta / float(maxi(1, substeps))

## True while collision can be a plain per-tick overlap test. False means a shot can cross
## an obstacle between two samples and `substeps` needs raising -- see the doc on
## `substeps` for the arithmetic.
func step_is_safe(cell: int) -> bool:
	var step := projectile_speed / 60.0 / float(maxi(1, substeps))
	return step < float(cell) - 2.0 * projectile_radius
