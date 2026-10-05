class_name BattlerRules
extends Resource

## Every tunable of the auto-battler, as an exported resource -- the same bargain RuleSet
## makes for Tetris.
##
## The design this serves needs ZERO player input, which is the whole reason it replaced a
## tower defence: Tetris takes nearly all of a player's attention, so anything asking for a
## placement decision competes with the stack and both halves get worse. Here a clear spawns
## a fighter and the fighters do the rest, so the right half is a readout with stakes and its
## state is one legible thing -- where the front line is.
##
## Nothing here is balanced. These numbers exist to make the mechanics visible.

@export_group("Spawning")
## Seconds between enemy spawns from the right edge.
@export var red_interval: float = 8.0
## Random spread on that interval, so the line does not pulse in lockstep.
@export var red_jitter: float = 2.0

@export_group("Movement")
## Pixels per second. NOT a free parameter: a 9-frame walk at 0.2s per frame is a 1.8s
## cycle, and a body that does not cover roughly one stride per cycle visibly slides or
## skates. A 64px frame carries a ~32px body, and an LPC stride covers about one to one and
## a half of those, so 17-26 px/s is the range that reads as grounded. 22 gives a 29 second
## crossing of the 638px pane -- comfortably neglectable, which is the point.
##
## If frame_seconds changes, this has to move with it.
@export var walk_speed: float = 22.0

@export_group("Space")
## Half a body, in pixels. The collision quantum, and the scale everything else derives from:
## the LPC body is ~32px wide inside its 64px frame, so one "unit" of distance is 32px.
##
## Note that "cannot stack" is a statement about BODIES, not frames. Two units 32px apart
## have heavily overlapping 64px sprites, which is ordinary 2.5D and is what the per-frame
## z-order from y is for.
@export var body_radius: float = 16.0
## How far a unit looks for something to fight, in pixels. 3 bodies, which at walk_speed is
## about 4.4 seconds of warning before contact.
##
## Its ratio to the lane DEPTH decides what kind of flanking exists. Shallower than the aggro
## radius and a unit on the centreline sees the entire lane, so there is no corridor to slip
## through and flanking is impossible. Deeper and there are true blind spots. At 256px deep
## against 96px of aggro there is no blind spot, so a flanker is always SEEN and a defender
## peels off to intercept -- flanking is a race decided by distance and numbers, which is the
## version that rewards having more units than the enemy can cover.
@export var aggro_px: float = 96.0
## Keep the current target until it leaves aggro by this factor. Pure hysteresis: without it
## two near-equidistant enemies swap a unit back and forth every frame forever.
@export var target_keep_factor: float = 1.3

@export_group("Steering")
## Degrees between rungs of the escalation ladder when a step is blocked.
@export var turn_step_deg: float = 25.0
## How far the ladder may deflect. PAST 90 on purpose: those angles move away from the goal,
## which is exactly what escapes a concave trap, because getting out of a pocket means first
## going further from the target.
@export var turn_max_deg: float = 150.0
## Smallest fraction of a full step still attempted at full deflection. Never zero -- a unit
## that stops moving while deflected freezes against the wall it is trying to walk around,
## which makes the whole escalation pointless.
@export_range(0.05, 1.0, 0.05) var min_step_scale: float = 0.35

@export_group("Animation")
## Seconds per frame for every clip. Retiming is a live knob (see SpriteSheet.apply_speed)
## rather than a rebuild, so this can be tuned while the game runs.
@export var frame_seconds: float = 0.2

@export_group("Garbage")
## How many rows one breach is worth.
@export var rows_per_breach: int = 1
## Chance the hole column steps one across between flushes. 0.1 is the value measured in
## real Zenith replays, so the drift is not invented.
@export_range(0.0, 1.0, 0.01) var hole_shift_chance: float = 0.1

@export_group("Meta")
@export var preset_name: String = "testing"

static func testing() -> BattlerRules:
	var r := BattlerRules.new()
	r.preset_name = "testing"
	return r

## Whether a Tetris clear is worth a fighter.
##
## A quad or ANY spin, so singles and doubles spawn nothing. That makes the minigame pay out
## for exactly the clears worth learning, and it is why ClearResult carries `spin` at all --
## its own class doc says it exists so the right-hand minigame can define its own economy
## off these fields.
##
## Static and tiny on purpose: a one-line rule is the kind someone edits, so it is worth
## being addressable by a test rather than buried in a dispatch block.
static func spawns_unit(r: ClearResult) -> bool:
	if r == null:
		return false
	return r.lines == 4 or r.spin != SpinDetect.Kind.NONE

## Seconds until the next enemy, jittered.
func next_red_delay(rng: RandomNumberGenerator) -> float:
	return maxf(0.5, red_interval + rng.randf_range(-red_jitter, red_jitter))
