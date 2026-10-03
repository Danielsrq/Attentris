class_name AttackTable
extends Resource

## Lines of garbage a clear sends. This is the number the right-hand minigame will
## actually spend, so it lives in the sim next to ScoreTable rather than being
## derived in a view.
##
## Score and attack are deliberately separate currencies. Score rewards *volume*
## (a single is worth 100); attack rewards *efficiency* (a single is worth nothing).
## A player farming score and a player farming attack play differently, and the
## solver should be able to optimise either.

@export_group("Line clears")
## Guideline versus values. A single sends nothing -- that is the whole reason
## stacking for quads is worth the risk.
@export var lines_attack: Dictionary = {1: 0, 2: 1, 3: 2, 4: 4}

@export_group("Spins")
## Keyed by lines cleared. A T-Spin Double matching a quad at 4 is the usual
## trade-off: same damage, far fewer pieces.
@export var t_spin_attack: Dictionary = {0: 0, 1: 2, 2: 4, 3: 6}
## Any piece's Mini (a T-spin Mini, or a non-T spin under RuleSet.non_t_spin_is_mini).
## TETR.IO-style: a Mini sends nothing but, like every spin that clears, keeps B2B alive.
@export var t_spin_mini_attack: Dictionary = {0: 0, 1: 0, 2: 0}
## All-spin values for S/Z/L/J/I. Not a guideline concept, so these are a starting
## point rather than a standard.
@export var other_spin_attack: Dictionary = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}

@export_group("Bonuses")
## How the back-to-back bonus grows with the chain. `b2b` counts the difficult clears
## AFTER the first (ClearResult.b2b: -1 none, 0 first, >= 1 chain).
##
##   FIBONACCI (default): +1 at each chain that is a squared Fibonacci number, capped:
##       b2b     1-3   4-8   9-24   25-63   64-168   169-440   441+
##       bonus    +1    +2     +3      +4       +5        +6     +7
##   PRIME (power-up): +1 at each squared prime, capped; chains 1-3 earn nothing:
##       b2b     4-8   9-24   25-48   49-120   121-168   169-288   289+
##       bonus    +1     +2      +3       +4        +5        +6     +7
##   SQUARES (legacy): floor(sqrt(b2b) * b2b_weight), uncapped.
##   FLAT (original): +1 while a chain is live.
##
## Bounded growth matters: with every spin able to keep a chain alive, an uncapped bonus
## lets one chain run the whole game (a 380-chain paid +19 per clear under SQUARES).
enum B2bScaling { FIBONACCI, PRIME, SQUARES, FLAT }
const FIBONACCI_STEPS := [1, 4, 9, 25, 64, 169, 441]
const PRIME_STEPS := [4, 9, 25, 49, 121, 169, 289]
@export var b2b_scaling: B2bScaling = B2bScaling.FIBONACCI
## Only used by SQUARES.
@export var b2b_weight: float = 1.0
@export var perfect_clear_bonus: int = 10
## Combos are ignored for now; flip this on and fill the table when they matter.
@export var use_combo: bool = false
@export var combo_attack: PackedInt32Array = PackedInt32Array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5])

func attack_for(r: ClearResult) -> int:
	if r.lines <= 0:
		return 0
	var out := 0
	if r.spin == SpinDetect.Kind.MINI:
		# Every piece's Mini uses the Mini table.
		out = int(t_spin_mini_attack.get(r.lines, 0))
	elif r.spin == SpinDetect.Kind.FULL and r.piece_id == &"T":
		out = int(t_spin_attack.get(r.lines, 0))
	elif r.spin != SpinDetect.Kind.NONE:
		out = int(other_spin_attack.get(r.lines, 0))
	else:
		out = int(lines_attack.get(r.lines, 0))
	# Burst rules: the base doubles; the B2B bonus below is added after, undoubled.
	if r.burst:
		out *= 2

	# A Mini keeps the chain alive (is_difficult) but earns no bonus: it sends nothing.
	if r.is_difficult() and r.b2b_active() and r.spin != SpinDetect.Kind.MINI:
		out += b2b_bonus_for(r.b2b)
	if r.perfect_clear:
		out += perfect_clear_bonus
	if use_combo and r.combo >= 1:
		var i: int = mini(r.combo, combo_attack.size() - 1)
		out += combo_attack[i]
	return out

## Apply the rule set's attack-relevant values (the burst T-spin triple).
func configure(rules: RuleSet) -> void:
	t_spin_attack = t_spin_attack.duplicate()
	t_spin_attack[3] = rules.tst_attack

## Bonus for a chain of length `b2b`. Exposed separately so a solver can price a chain
## it is considering without building a ClearResult for it.
func b2b_bonus_for(b2b: int) -> int:
	if b2b < 1:
		return 0
	match b2b_scaling:
		B2bScaling.FIBONACCI:
			return _steps_reached(FIBONACCI_STEPS, b2b)
		B2bScaling.PRIME:
			return _steps_reached(PRIME_STEPS, b2b)
		B2bScaling.FLAT:
			return 1
		_:
			return int(floor(sqrt(float(b2b)) * b2b_weight))

static func _steps_reached(steps: Array, b2b: int) -> int:
	var n := 0
	for s in steps:
		if b2b >= s:
			n += 1
	return n
