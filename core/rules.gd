class_name RuleSet
extends Resource

## Every tunable rule in one exported resource, so tuning is data rather than code.

enum GravityMode { LEVEL_CURVE, FIXED }

@export_group("Handling")
## Delayed Auto Shift: hold time before the piece starts repeating, in ms.
@export var das_ms: float = 100.0
## Auto Repeat Rate: ms between repeats once DAS has charged. 0 = instant slide to wall.
@export var arr_ms: float = 0.0
## Soft Drop Factor: multiplier on gravity while soft drop is held. Because it multiplies
## GRAVITY rather than setting a rate, soft drop speeds up with the level: past roughly
## level 6 it exceeds a cell per tick on its own, whatever this is set to.
@export var sdf: float = 20.0
## DAS Cut Delay: DAS charge trimmed back to this after a rotation, in ms. 0 = off.
@export var dcd_ms: float = 0.0
## Entry delay between lock and the next spawn, in ms. 0 = off.
@export var are_ms: float = 0.0

@export_group("Lock")
@export var lock_delay_sec: float = 0.5
## Extended placement lockdown: moves and rotations reset the lock timer this many
## times before the piece locks regardless.
@export var max_move_resets: int = 15

@export_group("Gravity")
@export var gravity_mode: GravityMode = GravityMode.LEVEL_CURVE
@export var fixed_gravity_cells_per_sec: float = 1.0
@export var lines_per_level: int = 10

@export_group("Flags")
## Immobility-based spin detection for S, Z, L, J and I (the TETR.IO/Jstris rule).
@export var all_spin: bool = true
## Guideline 3-corner T-spin detection.
@export var t_spin: bool = true
@export var allow_180: bool = true
@export var infinite_hold: bool = false
## Kick offset index 4 is the 1x2 tuck; landing it upgrades a Mini to a full T-spin.
@export var mini_upgrade_on_kick5: bool = true
## Score non-T all-spins as Mini rather than full.
@export var non_t_spin_is_mini: bool = false

@export_group("All-spin")
## Every Mini is promoted to a full spin, so an S-spin double scores like a T-spin
## double and keeps B2B the same way.
@export var allspin_full: bool = false
## Two consecutive clears of the same type (piece + lines + spin; non-spin clears keyed
## by lines only) wound the bottom in-play row. See Board.add_wound.
@export var repeat_penalty: bool = false
## Clears a wound line stays out of play before it heals.
@export var wound_clears: int = 10

@export_group("Queue")
@export var next_count: int = 5
@export var first_piece_excludes: Array[String] = ["S", "Z", "O"]

@export_group("Meta")
@export var preset_name: String = "competitive"

## Modern competitive feel: instant wall slides, fast soft drop, all-spin on.
static func competitive() -> RuleSet:
	var r := RuleSet.new()
	# 60, not the guideline's 20. sdf multiplies gravity, so at level 1 this is one cell
	# per tick -- the field in 20 ticks instead of a full second, which is what soft drop
	# is for when you are placing by hand. Stated explicitly rather than left to the
	# export default so the deviation from guideline() is visible here.
	r.sdf = 60.0
	return r

## All-spin: competitive handling, every spin full, repeating a clear type wounds a row.
static func allspin() -> RuleSet:
	var r := RuleSet.new()
	r.preset_name = "allspin"
	r.all_spin = true
	r.allspin_full = true
	r.repeat_penalty = true
	return r

## Official-Tetris feel: slower repeat, T-spin only, no 180.
static func guideline() -> RuleSet:
	var r := RuleSet.new()
	r.preset_name = "guideline"
	r.das_ms = 133.0
	r.arr_ms = 33.0
	r.sdf = 20.0
	r.all_spin = false
	r.allow_180 = false
	return r

## Guideline level curve: seconds per row = (0.8 - (level-1) * 0.007) ^ (level-1).
## Level 1 is exactly 1.0 s per row, i.e. one cell per second.
func seconds_per_row(level: int) -> float:
	var base := 0.8 - float(level - 1) * 0.007
	if base < 0.001:
		base = 0.001
	return pow(base, float(level - 1))

func gravity_cells_per_tick(level: int, ticks_per_second: float) -> float:
	if gravity_mode == GravityMode.FIXED:
		return fixed_gravity_cells_per_sec / ticks_per_second
	return 1.0 / (seconds_per_row(level) * ticks_per_second)
