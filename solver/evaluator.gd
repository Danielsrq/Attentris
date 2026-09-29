class_name BoardEvaluator
extends RefCounted

## Scores a board after a placement. Higher is better.
##
## Swap this class out (or just the weights) for a different objective -- the search
## only calls score(). A B2B/all-spin evaluator is a second implementation here, not
## a change to the search.
##
## Performance note: the obvious implementation makes one full-board pass per
## feature. At ~400 placements per solve that was the bulk of the runtime, so
## everything below is computed in a SINGLE pass over the columns, reading the
## PackedByteArray directly instead of going through get_cell().

var weights: EvalWeights

# Scratch, reused across calls so a solve does not allocate per placement.
var _heights := PackedInt32Array()
var _clean_rows: int = 0
var _well_fill: float = 0.0
var _max_height: int = 0
var _total_height: int = 40

func _init(w: EvalWeights = null) -> void:
	weights = w if w != null else EvalWeights.quad()

## `well` is the reserved column, `lines` the clear this placement produced,
## `landing_y` the row the piece came to rest on, and `b2b_before` the chain state
## going in (-1 none, 0 first difficult clear, >=1 chain live) -- matching Game.b2b.
func score(board: Board, well: int, lines: int, landing_y: int, b2b_before: int = -1) -> float:
	return structural(board, well, {}, false) + events(lines, landing_y, b2b_before, SpinDetect.Kind.NONE)

## Board state only: nothing about how the piece got there or what it cleared.
##
## Kept separate from events() because of HOW THE SEARCH COMBINES THEM. Summing a
## board score at every ply double-counts: a plan that clears early gets its lower,
## flatter board scored once per remaining ply, so deep searches drift toward early
## singles purely as an artefact of the accounting. Scoring only the CURRENT board
## plus accumulated per-placement events removes that bias.
##
## Also populates _heights, _max_height, _clean_rows and _well_fill, so events()
## must be called after it.
func structural(board: Board, well: int, avail: Dictionary = {}, well_claims_i: bool = false) -> float:
	var w := weights
	var width := board.width
	var height := board.total_height
	var cells := board.cells

	_total_height = height
	if _heights.size() != width:
		_heights.resize(width)

	var agg := 0
	var maxh := 0
	var stack_top := height     # topmost filled row across ALL columns, well included
	var col_trans := 0
	var par_a := 0
	var par_b := 0

	# --- single pass: heights, holes, covered depth, column transitions ---
	for x in width:
		var top := height          # first filled row, or `height` if empty
		var prev := 0
		var seen := false
		for y in range(height):
			var filled: int = 1 if cells[y * width + x] != 0 else 0
			if filled == 1 and not seen:
				seen = true
				top = y
			if filled == 1:
				# Checkerboard colour of this cell, for the parity feature. Folded
				# into the pass that is already reading every cell.
				if (x + y) & 1 == 0:
					par_a += 1
				else:
					par_b += 1
			if seen:
				if filled != prev:
					col_trans += 1
				prev = filled
		if seen and prev != 1:
			col_trans += 1   # floor counts as a transition
		var h := height - top
		_heights[x] = h
		stack_top = mini(stack_top, top)
		if x != well:
			agg += h
			maxh = maxi(maxh, h)

	# The flood fill only earns its ~50% cost when overhangs are priced differently
	# from holes -- which is a T-spin evaluator's concern, not a quad one. When both
	# cost the same the naive count is equivalent, so skip it.
	if w.w_overhang == w.w_holes:
		_naive_empties(board)
	else:
		_classify_empties(board, stack_top)
	var total := 0.0
	total += w.w_aggregate_height * float(agg)
	total += w.w_max_height * float(maxh)
	total += w.w_holes * float(_holes)
	total += w.w_overhang * float(_overhangs)
	total += w.w_covered_cells * float(_buried)
	total += w.w_col_transitions * float(col_trans)

	var bump := 0
	for x in range(width - 1):
		if x == well or x + 1 == well:
			continue
		bump += absi(_heights[x] - _heights[x + 1])
	total += w.w_bumpiness * float(bump)

	# Row transitions only over rows that contain anything. Empty rows above the
	# stack contribute a constant and scanning them is wasted work.
	total += w.w_row_transitions * float(_row_transitions(board, height - maxi(maxh, _heights[well])))

	# --- danger release ---
	# Above danger_height the well ban and the small-clear penalty mostly lift, so
	# the solver digs out instead of protecting a quad it may never get to take.
	# Without this, hard-reserving the well is a death sentence on an I drought.
	_max_height = maxh
	var relief := _relief()

	# --- dependencies and parity ---
	if w.w_i_dependency != 0.0 or w.w_deep_notch != 0.0 or w.w_cumulative_well != 0.0:
		_dep_used.clear()
		# The well's own claim comes first: if it is ready to cash, the next I is
		# already spoken for and cannot also fix a notch elsewhere.
		if well_claims_i:
			_dep_used[&"I"] = 1
		var notch2 := 0
		var cum_well := 0
		var dep_penalty := 0.0
		for x in width:
			if x == well:
				continue
			var lov: int = (_heights[x - 1] - _heights[x]) if x > 0 else 0x3FFFFFFF
			var rov: int = (_heights[x + 1] - _heights[x]) if x < width - 1 else 0x3FFFFFFF
			var d := mini(lov, rov)
			if d <= 0:
				continue
			cum_well += d
			if d < 2:
				# A 1-deep step is filled by almost any piece, including S and Z.
				# Not worth attributing.
				continue
			if d == 2:
				notch2 += 1
			var base: float = w.w_i_dependency if d >= 3 else w.w_deep_notch
			dep_penalty += base * _notch_relief(d, lov, rov, avail, w)
		total += dep_penalty
		total += w.w_cumulative_well * float(cum_well)

	_parity = absi(par_a - par_b)
	if w.w_parity != 0.0:
		total += w.w_parity * float(_parity)

	# --- well handling ---
	# Two signals, because one cannot do both jobs. `clean_rows` is exact but binary
	# per row: it stays 0 until the very last cell of a row lands, which leaves the
	# search nothing to climb while it is building. `fill` is the slope -- how full
	# those rows are on the way there. Using only the exact measure measurably
	# dropped the quad rate; using only a surface approximation was wrong on holes.
	_scan_well(board, well)
	total += w.w_well_depth * float(_clean_rows)
	total += w.w_well_progress * _well_fill
	if _clean_rows >= 4:
		total += w.w_quad_ready
	if _heights[well] > 0:
		total += w.w_well_blocked * relief * float(_heights[well])

	return total

## Per-placement terms: how high it landed, what it cleared, what that did to the
## chain. These ARE summed along the path, because each is a distinct event rather
## than a restatement of the same board.
func events(lines: int, landing_y: int, b2b_before: int, spin: int = SpinDetect.Kind.NONE,
		piece_id: StringName = &"", spare_i: int = 0) -> float:
	var w := weights
	var relief := _relief()
	var total := w.w_landing_height * float(_landing_depth(landing_y))

	if spin != SpinDetect.Kind.NONE and lines > 0:
		total += float(w.reward_spin_clear.get(lines, 0.0))
	elif lines >= 4:
		total += w.reward_quad
	elif lines > 0:
		total += w.penalty_small_clear * relief * float(lines)

	# An I spent without a quad. Nothing else fills a 1-wide 4-deep well, so this is
	# the one piece whose misuse costs a future clear rather than just a worse board.
	if w.w_i_wasted != 0.0 and piece_id == &"I" and lines < 4:
		var scale: float = w.i_wasted_relief if spare_i > 0 else 1.0
		if _max_height >= w.i_waste_height:
			# Tall stack: an I spent on survival is not an I wasted. Below half
			# height it is a future quad; above it, holding out for one tops you out.
			scale *= w.danger_relief
		total += w.w_i_wasted * scale

	# Only line clears move the chain; a placement that clears nothing leaves it
	# alone, exactly as Game does. "Difficult" matches Game: a quad, or any spin.
	if lines > 0:
		if lines >= 4 or spin != SpinDetect.Kind.NONE:
			if b2b_before >= 0:
				total += w.w_b2b_kept
		elif b2b_before >= 1:
			total += w.w_b2b_broken * relief
	return total

## How far column x sits below its shallower neighbour. Walls count as infinitely
## tall. This is the whole dependency measure: a notch is defined against its own
## neighbours, so no "base stack height" has to be found first.
##   >= 3  only a vertical I fits
##   == 2  I, or a vertical L/J/S/Z depending on which side is tall
##   == 1  almost anything
func notch_depth(width: int, x: int) -> int:
	var left: int = _heights[x - 1] if x > 0 else 0x7FFFFFFF
	var right: int = _heights[x + 1] if x < width - 1 else 0x7FFFFFFF
	return mini(left, right) - _heights[x]

## |white - black| over filled cells. Only T can change this, by exactly 2.
func parity_imbalance(board: Board, well: int) -> int:
	structural(board, well)
	return _parity
var _parity: int = 0

## Relief factor for one notch: 1.0 means "no piece available, pay in full".
##
## Which pieces can fill a 1-wide notch, derived from the shapes:
##   depth >= 3   only a vertical I. An L or J bar is 3 tall but its foot would
##                clash with the neighbour that makes the notch deep.
##   depth == 2   I, plus L if the LEFT neighbour is exactly 2 above (L's foot sits
##                top-left, resting on that neighbour), plus J if the RIGHT one is.
##                S and Z are offset by ONE column-step, so they fit depth-1 steps,
##                not depth-2 notches.
##
## Pieces are consumed greedily across notches, so two dependencies and one I means
## only one of them gets relief.
func _notch_relief(d: int, lov: int, rov: int, avail: Dictionary, w: EvalWeights) -> float:
	var best := -1
	var best_id: StringName = &""
	var dist := _nearest(&"I", avail)
	if dist >= 0:
		best = dist
		best_id = &"I"
	if d == 2:
		if lov == 2:
			var dl := _nearest(&"L", avail)
			if dl >= 0 and (best < 0 or dl < best):
				best = dl
				best_id = &"L"
		if rov == 2:
			var dj := _nearest(&"J", avail)
			if dj >= 0 and (best < 0 or dj < best):
				best = dj
				best_id = &"J"
	if best < 0:
		return 1.0
	_dep_used[best_id] = int(_dep_used.get(best_id, 0)) + 1
	if best == 0:
		return w.dep_relief_hold    # in hold: summonable at any time
	return minf(1.0, w.dep_relief_if_available + w.dep_relief_falloff * float(best - 1))

## Distance to the nearest unconsumed copy of `id`; -1 if none remain.
## Distance 0 means hold, 1 means the piece now in play, 2 the one after it.
func _nearest(id: StringName, avail: Dictionary) -> int:
	var list: Array = avail.get(id, [])
	var used: int = int(_dep_used.get(id, 0))
	if used >= list.size():
		return -1
	return int(list[used])

var _dep_used: Dictionary = {}

func _relief() -> float:
	return weights.danger_relief if _max_height >= weights.danger_height else 1.0

func _landing_depth(landing_y: int) -> int:
	return _total_height - landing_y

func _row_transitions(board: Board, from_row: int) -> int:
	var n := 0
	var width := board.width
	var cells := board.cells
	for y in range(maxi(0, from_row), board.total_height):
		var base := y * width
		var prev := 1  # wall counts as filled
		for x in width:
			var cur: int = 1 if cells[base + x] != 0 else 0
			if cur != prev:
				n += 1
			prev = cur
		if prev != 1:
			n += 1
	return n

## Rows that an I dropped into the well would ACTUALLY complete.
##
## This asks the exact question -- "is this row full except for the well?" -- rather
## than the surface-height approximation it replaces. That approximation compared the
## well against its two immediate neighbours' surface heights, which is wrong twice
## over:
##
##   1. Surface height ignores holes. A neighbour that is tall but hollow reported a
##      deep well when no row was anywhere near complete, so the solver believed a
##      quad was on offer, dropped the I, and cleared nothing.
##   2. Only the two adjacent columns were consulted. A quad needs all nine non-well
##      columns filled at those rows; a short or holed column anywhere else was
##      invisible.
##
## Both failure modes need holes to show up, which is why a clean board hid them.
func clean_well_rows(board: Board, well: int) -> int:
	_scan_well(board, well)
	return _clean_rows

## Fraction of the four rows above the well floor that is filled, ignoring the well.
## 0.0 on an empty board, 1.0 when a quad is ready.
func well_fill(board: Board, well: int) -> float:
	_scan_well(board, well)
	return _well_fill

## One pass computing both. Reads `_heights`, so score() must have run its column
## pass first.
func _scan_well(board: Board, well: int) -> void:
	var width := board.width
	var cells := board.cells
	# The well column's first filled row, computed locally rather than read from
	# _heights. The search asks "is a quad ready" BEFORE scoring the board, to work
	# out whether the next I is already spoken for, so this cannot depend on
	# structural() having run first.
	var well_top := board.total_height
	for wy in board.total_height:
		if cells[wy * width + well] != 0:
			well_top = wy
			break
	# Lowest empty row in the well is one above its first filled row.
	var y := well_top - 1
	var clean := 0
	var counting := true
	var filled := 0
	var slots := 0
	var rows := 0
	while y >= 0 and rows < 4:
		var base := y * width
		var complete := true
		for x in width:
			if x == well:
				continue
			slots += 1
			if cells[base + x] != 0:
				filled += 1
			else:
				complete = false
		if counting and complete:
			clean += 1
		else:
			counting = false
		rows += 1
		y -= 1
	_clean_rows = clean
	_well_fill = float(filled) / float(slots) if slots > 0 else 0.0

## Filled rows above the floor, per column. Public for the well picker and tests.
func column_heights(board: Board) -> PackedInt32Array:
	var out := PackedInt32Array()
	out.resize(board.width)
	for x in board.width:
		var h := 0
		for y in board.total_height:
			if board.get_cell(x, y) != 0:
				h = board.total_height - y
				break
		out[x] = h
	return out

## Every covered empty cell counted as a hole, no topology. Equivalent to the
## flood-fill classification whenever w_overhang == w_holes, and much cheaper.
func _naive_empties(board: Board) -> void:
	var width := board.width
	var height := board.total_height
	var cells := board.cells
	_holes = 0
	_overhangs = 0
	_buried = 0
	for x in width:
		var top := height - _heights[x]
		for y in range(top + 1, height):
			if cells[y * width + x] == 0:
				_holes += 1
				_buried += y - top

## Separates SEALED holes from mere overhangs by flood-filling the empty space from
## above the stack. An empty cell reached by the fill is connected to open air; one
## that is not is trapped.
##
## This matters because a T-spin slot is a covered cell that still reaches the
## surface sideways. Counting it as a hole makes the hole penalty fight every spin
## setup the solver might build. The topology is different, so the feature has to be.
##
## Only rows at or below `stack_top` are walked -- everything above the tallest
## column is trivially open and connected.
func _classify_empties(board: Board, stack_top: int) -> void:
	var width := board.width
	var height := board.total_height
	var cells := board.cells
	_holes = 0
	_overhangs = 0
	_buried = 0
	if stack_top >= height:
		return

	var from_row := maxi(0, stack_top - 1)
	var span := height - from_row
	var need := width * span
	if _open.size() != need:
		_open.resize(need)
	for i in need:
		_open[i] = 0

	# Seed: every empty cell on the first row, which sits above the whole stack.
	var stack: Array = []
	for x in width:
		if cells[from_row * width + x] == 0:
			_open[x] = 1
			stack.append(from_row * width + x)

	while not stack.is_empty():
		var idx: int = stack.pop_back()
		var y: int = idx / width
		var x: int = idx % width
		for d in [Vector2i(-1, 0), Vector2i(1, 0), Vector2i(0, -1), Vector2i(0, 1)]:
			var nx: int = x + d.x
			var ny: int = y + d.y
			if nx < 0 or nx >= width or ny < from_row or ny >= height:
				continue
			if cells[ny * width + nx] != 0:
				continue
			var oi: int = (ny - from_row) * width + nx
			if _open[oi] != 0:
				continue
			_open[oi] = 1
			stack.append(ny * width + nx)

	# Anything empty and covered is either an overhang (reached) or a hole (not).
	for x in width:
		var top := height - _heights[x]
		for y in range(top + 1, height):
			if cells[y * width + x] != 0:
				continue
			if _open[(y - from_row) * width + x] != 0:
				_overhangs += 1
			else:
				_holes += 1
				_buried += y - top

## True holes from the last structural() call.
func holes() -> int:
	return _holes

## Covered but connected cells -- overhangs, including T-spin slots.
func overhangs() -> int:
	return _overhangs

var _holes: int = 0
var _overhangs: int = 0
var _buried: int = 0
var _open := PackedByteArray()

## Emptiest edge column. Legacy: only used when weights.well_column is -1, which the
## quad preset no longer does, because a well that moves between solves never gets
## dug. Kept so the old behaviour stays reproducible.
static func pick_well(board: Board, heights: PackedInt32Array) -> int:
	return 0 if heights[0] <= heights[board.width - 1] else board.width - 1
