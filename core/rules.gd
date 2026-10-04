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
## by lines only) insert an unclearable wound line at the bottom. See Board.add_wound.
@export var repeat_penalty: bool = false
## Clears a wound line stays out of play before it heals.
@export var wound_clears: int = 10
## Chance a new wound line's hole moves from the previous wound's column. The hole only
## opens once the wound turns into garbage, so high messiness turns a stack of expired
## wound lines into cheese.
@export_range(0.0, 1.0, 0.05) var wound_messiness: float = 0.5

@export_group("Burst")
## Burst rules: a qualifying clear (ClearResult.qualifies_burst) placed straight after
## another one has its BASE attack doubled; the B2B bonus is added after, undoubled.
## A placement that clears nothing, a plain clear or a Mini single breaks it.
@export var burst_doubling: bool = false
## Base attack of a full T-spin triple (AttackTable.t_spin_attack[3]); 10 under burst.
@export var tst_attack: int = 6
## A qualifying clear is doubled when an arming clear happened within this many previous
## placements (1 = straight after).
@export var burst_window: int = 1
## Whether a full spin single arms the next clear (it can still be doubled itself).
@export var burst_tss_arms: bool = true
## Gate rules (a training rule set): only chains score. See gate_step.
@export var gate: bool = false
## gate_b2b: each spin clear of a paid gate chain also earns the B2B bonus of the chain
## count it reached; B2B runs, and plain clears break it, as in normal rules.
@export var gate_b2b: bool = false

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

## Normal rules (TETR.IO-style), the game's default: every piece can spin, but non-T
## spins always count as Minis. A Mini sends nothing (AttackTable) yet keeps B2B alive,
## like every spin that clears lines. T-spins keep the 3-corner full/Mini rule.
static func normal() -> RuleSet:
	var r := RuleSet.new()
	r.preset_name = "normal"
	r.all_spin = true
	r.non_t_spin_is_mini = true
	return r

## All-spin: competitive handling, every spin full, repeating a clear type wounds a row.
static func allspin() -> RuleSet:
	var r := RuleSet.new()
	r.preset_name = "allspin"
	r.all_spin = true
	r.allspin_full = true
	r.repeat_penalty = true
	return r

## Burst: normal rules, a T-spin triple worth 10, and back-to-back PLACEMENTS of
## qualifying clears (full T-spins, quads, Mini doubles/triples) double the second one.
static func burst() -> RuleSet:
	var r := RuleSet.normal()
	r.preset_name = "burst"
	r.burst_doubling = true
	r.tst_attack = 10
	return r

## Burst, sharpened and relaxed: spin singles no longer arm the doubling, and an arming
## clear arms the next TWO placements.
static func burst2() -> RuleSet:
	var r := RuleSet.burst()
	r.preset_name = "burst2"
	r.burst_window = 2
	r.burst_tss_arms = false
	return r

## One placement of the burst state. `left` = placements still armed. Returns
## [this clear's base is doubled, the new `left`]. Shared by Game and the parity check.
func burst_step(left: int, r: ClearResult) -> Array:
	if not burst_doubling:
		return [false, 0]
	var doubled := left > 0 and r.qualifies_burst()
	var arms := r.qualifies_burst() and (burst_tss_arms or not (r.spin == SpinDetect.Kind.FULL and r.lines == 1))
	return [doubled, burst_window if arms else maxi(0, left - 1)]

## Gate rules: normal rules, but a clear only scores as part of a chain. A full spin
## double or triple opens a gate; within `burst_window` placements, spin doubles/triples
## and quads join the chain and restart the window; anything else just uses up a
## placement. The chain pays the sum of its members (TSD 4, TST 10, quad 1) once it holds
## TWO spin clears: a quad keeps the gate open but cannot complete a chain. No B2B, no PC.
static func gate_rules() -> RuleSet:
	var r := RuleSet.normal()
	r.preset_name = "gate"
	r.gate = true
	r.burst_window = 2
	r.tst_attack = 10
	return r

## gate_b2b: the gate rules plus the B2B bonus on every spin clear of a paid chain, so
## digging with plain clears (which break B2B) costs future chain value.
static func gate_b2b_rules() -> RuleSet:
	var r := RuleSet.gate_rules()
	r.preset_name = "gate_b2b"
	r.gate_b2b = true
	return r

## Value of a clear that joins a gate chain (0 = it does not join).
func gate_value(r: ClearResult) -> int:
	if r.spin == SpinDetect.Kind.FULL and r.lines == 2:
		return 4
	if r.spin == SpinDetect.Kind.FULL and r.lines == 3:
		return tst_attack
	if r.lines == 4:
		return 1
	return 0

## One placement under gate rules. `st` = [placements left, spin clears in the chain,
## value held until it has two]. Returns [attack credited now, new state]. A chain's
## payout is certain once it holds two spin clears, so it is credited then, and each later
## member as it joins.
## `bonus`: the B2B bonus this clear earns, added to a spin clear's value under gate_b2b.
func gate_step(st: Array, r: ClearResult, bonus: int = 0) -> Array:
	var left: int = st[0]
	var spins: int = st[1]
	var held: int = st[2]
	var is_spin := 1 if r.spin == SpinDetect.Kind.FULL else 0
	var v := gate_value(r)
	if v > 0 and is_spin == 1:
		v += bonus
	if left > 0:
		if v > 0:
			var n := spins + is_spin
			if spins >= 2:
				return [v, [burst_window, n, 0]]
			if n >= 2:
				return [held + v, [burst_window, n, 0]]
			return [0, [burst_window, n, held + v]]
		if left == 1:
			return [0, [0, 0, 0]]
		return [0, [left - 1, spins, held]]
	if v > 0 and is_spin == 1:
		return [0, [burst_window, 1, v]]
	return [0, [0, 0, 0]]

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
