class_name TSpinSlots
extends RefCounted

## Finds T-spin Double slots by their NEGATIVE SPACE -- the empty cells a T would
## occupy -- rather than by the filled shape around them.
##
## The slot, with the nub column at `c`:
##
##     hang      . . #  . .        exactly one side covered, centre open for entry
##     pocket   . . .             (c-1, c, c+1) all empty  <- T's three cells
##     nub      # . #             only c empty              <- T's nub
##
## READY vs PARTIAL is the distinction that matters, and it is not about the local
## shape. A slot is only cashable if BOTH rows are otherwise full. That single check
## subsumes several rules that would otherwise need stating separately:
##
##   * two slots at the same height cancel each other -- neither row can complete
##   * a slot above an OPEN WELL is not cashable, because the well column is a gap
##     in both rows. Put the nub IN the well and the T plugs it itself, which is why
##     slots at the well behave differently from slots anywhere else.
##
## Exact match gates the payout; partial fill supplies the gradient, because a binary
## signal gives the search nothing to climb while it is building (the same lesson
## clean_well_rows taught).

## Scan result. One Dictionary per call rather than an object per slot, because this
## runs once per candidate placement.
static func scan(board: Board, heights: PackedInt32Array, well: int) -> Dictionary:
	var out := {
		"found": false,      ## a local slot shape exists anywhere
		"ready": false,      ## ...and both its rows are otherwise full
		"fill": 0.0,         ## best partial completeness of those rows, 0..1
		"nub_col": -1,
		"pocket_row": -1,
		"hang_cells": 0,      ## covered cells belonging to the hang
		"at_well": false,    ## the nub column IS the reserved well
	}
	var width := board.width
	var height := board.total_height
	var cells := board.cells

	for c in range(1, width - 1):
		# The nub rests on the surface of column c.
		var nub_row := height - heights[c] - 1
		var pocket_row := nub_row - 1
		if pocket_row < 1 or nub_row >= height:
			continue
		# Nub must be a 1-wide notch: empty at c, filled either side.
		if cells[nub_row * width + c] != 0:
			continue
		if cells[nub_row * width + c - 1] == 0 or cells[nub_row * width + c + 1] == 0:
			continue
		# Pocket: three empty cells.
		var prow := pocket_row * width
		if cells[prow + c - 1] != 0 or cells[prow + c] != 0 or cells[prow + c + 1] != 0:
			continue
		# Lip: exactly one side covered, centre open so the T can enter and rotate.
		var lrow := (pocket_row - 1) * width
		if cells[lrow + c] != 0:
			continue
		var left_lip := cells[lrow + c - 1] != 0
		var right_lip := cells[lrow + c + 1] != 0
		if left_lip == right_lip:
			continue   # neither (droppable, not a spin) or both (no entry)

		var lip_col: int = c - 1 if left_lip else c + 1
		var hang_cells := _lip_thickness(board, lip_col, pocket_row)
		var fill := _row_fill(board, pocket_row, nub_row, c)
		var ready: bool = fill >= 1.0

		if not out["found"] or fill > float(out["fill"]) or (ready and not bool(out["ready"])):
			out["found"] = true
			out["ready"] = ready
			out["fill"] = fill
			out["nub_col"] = c
			out["pocket_row"] = pocket_row
			out["hang_cells"] = hang_cells
			out["at_well"] = c == well
	return out

## How many covered cells sit under the hang. 1-high and 2-high hangs leave the same
## negative space below but different amounts of material above, and they differ in
## what is needed to rebuild the slot afterwards.
static func _lip_thickness(board: Board, lip_col: int, pocket_row: int) -> int:
	var n := 0
	var y := pocket_row
	while y >= 0 and board.get_cell(lip_col, y) == 0:
		n += 1
		y -= 1
	return maxi(1, n)

## Fraction of the cells the two rows still need, ignoring the ones the T supplies.
## 1.0 means the TSD would clear both rows right now.
static func _row_fill(board: Board, pocket_row: int, nub_row: int, c: int) -> float:
	var width := board.width
	var cells := board.cells
	var need := 0
	var have := 0
	var prow := pocket_row * width
	for x in width:
		if x >= c - 1 and x <= c + 1:
			continue          # the T fills these
		need += 1
		if cells[prow + x] != 0:
			have += 1
	var nrow := nub_row * width
	for x in width:
		if x == c:
			continue          # the nub fills this
		need += 1
		if cells[nrow + x] != 0:
			have += 1
	if need == 0:
		return 1.0
	return float(have) / float(need)

## The T placement that fills a found slot: rot 2 (nub down), box top-left one column
## left of the nub and one row above the pocket. Returns null if it does not fit.
static func placement_for(board: Board, def: PieceDef, slot: Dictionary) -> Placement:
	if not bool(slot["found"]):
		return null
	var pos := Vector2i(int(slot["nub_col"]) - 1, int(slot["pocket_row"]) - 1)
	var cells: Array = def.rotations[2]
	if not board.can_place(cells, pos):
		return null
	var p := Placement.new()
	p.def = def
	p.rot = 2
	p.pos = pos
	# Reached by rotating into the pocket, which is the whole point -- without this
	# the placement lands in a spin shape and registers as an ordinary clear.
	p.via_rotation = true
	p.kick_index = 0
	p.spin = SpinDetect.Kind.FULL
	for cc in cells:
		p.cells.append(pos + cc)
	return p
