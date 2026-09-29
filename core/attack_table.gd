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
@export var t_spin_mini_attack: Dictionary = {0: 0, 1: 1, 2: 2}
## All-spin values for S/Z/L/J/I. Not a guideline concept, so these are a starting
## point rather than a standard.
@export var other_spin_attack: Dictionary = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}

@export_group("Bonuses")
## Added to any difficult clear while a back-to-back chain is live.
@export var b2b_bonus: int = 1
@export var perfect_clear_bonus: int = 10
## Combos are ignored for now; flip this on and fill the table when they matter.
@export var use_combo: bool = false
@export var combo_attack: PackedInt32Array = PackedInt32Array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4, 5])

func attack_for(r: ClearResult) -> int:
	if r.lines <= 0:
		return 0
	var out := 0
	if r.spin != SpinDetect.Kind.NONE and r.piece_id == &"T":
		if r.spin == SpinDetect.Kind.FULL:
			out = int(t_spin_attack.get(r.lines, 0))
		else:
			out = int(t_spin_mini_attack.get(r.lines, 0))
	elif r.spin != SpinDetect.Kind.NONE:
		out = int(other_spin_attack.get(r.lines, 0))
	else:
		out = int(lines_attack.get(r.lines, 0))

	if r.is_difficult() and r.b2b_active():
		out += b2b_bonus
	if r.perfect_clear:
		out += perfect_clear_bonus
	if use_combo and r.combo >= 1:
		var i: int = mini(r.combo, combo_attack.size() - 1)
		out += combo_attack[i]
	return out
