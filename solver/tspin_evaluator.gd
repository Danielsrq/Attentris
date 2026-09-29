class_name TSpinEvaluator
extends BoardEvaluator

## Board scoring for a solver that BUILDS T-spin slots rather than avoiding the
## shapes they are made of.
##
## Kept as a separate class rather than a weight on the quad evaluator, because the
## two objectives genuinely disagree: a quad solver wants zero overhangs, a spin
## solver wants exactly one, deliberately placed. A single number tuned between them
## serves neither -- measured, the compromise lost to both extremes.
##
## The overhang question is resolved by INTENT, not by price:
##   * a hang belonging to a recognised slot  -> refunded, then rewarded
##   * any other covered cell                -> charged as a hole, same as always
## So there is no generic "overhang" weight to set, and nothing we want is scored
## with a negative number.

var last_slot: Dictionary = {}

func structural(board: Board, well: int, avail: Dictionary = {}, well_claims_i: bool = false) -> float:
	var total := super(board, well, avail, well_claims_i)
	var w := weights
	if w.w_slot_ready == 0.0 and w.w_slot_progress == 0.0:
		return total

	var slot := TSpinSlots.scan(board, _heights, well)
	last_slot = slot
	if not bool(slot["found"]):
		return total

	# The base pass charged w_holes for the hang's covered cells. A deliberate hang is
	# structure, so give that back -- but only because a slot was actually matched.
	total -= w.w_holes * float(slot["hang_cells"])

	# A slot is only worth anything if a T can reach it. Same conjunction the
	# dependency machinery uses: shape AND piece, never shape alone. Without this the
	# solver builds slots it cannot fill, and spends its T building them.
	var t_factor := _piece_factor(&"T", avail, w)

	if bool(slot["ready"]):
		total += w.w_slot_ready * t_factor
		if bool(slot["at_well"]):
			# The nub plugs the well, so both rows can actually complete. A slot
			# anywhere else stays blocked for as long as the well is open.
			total += w.w_slot_at_well * t_factor
	else:
		total += w.w_slot_progress * float(slot["fill"]) * t_factor
	return total

## 1.0 when the piece is in hold, tapering with queue distance, 0 when absent.
func _piece_factor(id: StringName, avail: Dictionary, w: EvalWeights) -> float:
	var list: Array = avail.get(id, [])
	if list.is_empty():
		return 0.0
	var dist: int = int(list[0])
	if dist <= 0:
		return 1.0
	return maxf(0.0, 1.0 - w.dep_relief_falloff * float(dist - 1))
