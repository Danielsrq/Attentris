class_name Board
extends RefCounted

## The playfield. A flat PackedByteArray with y = 0 at the TOP, increasing downward,
## so array indexing and screen space agree and the renderer never flips anything.
##
## Cell values are `PieceDef.index + 1`; 0 means empty. Storing the piece index
## rather than a plain occupied flag gives per-cell colour for free.
##
## `width`, `visible_height` and `buffer_height` are constructor arguments, so a
## 12-wide field or a taller buffer is a number change.

## Cell value for garbage. Deliberately outside the piece-index range so the view
## can colour it distinctly and the solver treats it as ordinary filled space.
const GARBAGE := 255

## Cell value for a wound line (all-spin repeat penalty): solid, never clearable,
## holding no piece. The row's real contents are parked in `wounds` until it heals.
const WOUND := 254

## Wound lines, oldest first. They occupy the bottom `wounds.size()` rows as a block,
## oldest at the TOP of the block next to the play area. A new wound takes the bottom
## in-play row and goes to the very bottom, pushing older wounds up, so the oldest is
## always the next to heal and heals straight back into play. Each entry:
##   {"left": clears until it heals, "cells": parked row, "owner": parked owners}
var wounds: Array = []

## Which PLACEMENT laid each cell, parallel to `cells`; 0 for empty and for garbage.
##
## `cells` records the piece TYPE, which is enough for colour but not for structure:
## two L pieces side by side are eight indistinguishable L cells, so "do these four
## make one whole L?" cannot be answered from type alone. An id per placement makes a
## piece identifiable for as long as it survives, and a piece that has lost cells to a
## line clear simply no longer has four.
var owner: PackedInt32Array

var width: int
## Phantom columns to the RIGHT of the playfield, for the T-spin assist. They are not
## part of the game: never cleared, never counted for a full row, invisible to the
## solver, the evaluator and parity, all of which iterate `width`. They exist only so a
## piece can occupy the space beside column width-1, which is what makes a spin into
## the last column possible at all -- ordinarily the piece would stick out of the field
## and can_place would reject it.
##
## Kept in their own array rather than widening `cells`, because `width` doubles as the
## row stride and the solver indexes cells[y * width + x] directly in its hot paths.
var ghost_width: int = 0
var ghost: PackedByteArray

## Phantom-column cell states. The default is WALL, so an unarmed assist behaves
## exactly like the field edge it replaces -- solid and invisible. Only ONE cell is
## ever OPEN, which is what makes this a pocket rather than a corridor: the first
## version left the column mostly empty and a piece could turn out of the field
## anywhere along it.
const GHOST_WALL := 0     ## solid, not drawn -- indistinguishable from the boundary
const GHOST_OPEN := 1     ## a piece may occupy this cell
const GHOST_SHOWN := 2    ## solid AND drawn -- the overhang and the support under it
var visible_height: int
var buffer_height: int
var total_height: int
var cells: PackedByteArray

func _init(w: int = 10, vis: int = 20, buf: int = 20) -> void:
	width = w
	visible_height = vis
	buffer_height = buf
	total_height = vis + buf
	cells = PackedByteArray()
	cells.resize(width * total_height)
	owner.resize(width * total_height)
	ghost.resize(0)
	clear()

func clear() -> void:
	cells.fill(0)
	owner.fill(0)
	wounds.clear()

## First row of the wound block, or total_height when there are none. Rows at or below
## this are out of play.
func wound_floor() -> int:
	return total_height - wounds.size()

## The bottom in-play row becomes a wound line at the bottom of the block for `clears`
## clears. Its cells are parked, not lost; the row turns solid.
func add_wound(clears: int) -> void:
	var y := wound_floor() - 1
	if y < 0:
		return
	var w := width
	var parked := cells.slice(y * w, (y + 1) * w)
	var parked_owner := owner.slice(y * w, (y + 1) * w)
	for x in w:
		cells[y * w + x] = WOUND
		owner[y * w + x] = 0
	wounds.append({"left": clears, "cells": parked, "owner": parked_owner})

## Called once per clear: every wound counts down, and the oldest heals at 0, its parked
## cells returning as the bottom in-play row. Returns how many healed.
func tick_wounds() -> int:
	for wd in wounds:
		wd["left"] = maxi(0, int(wd["left"]) - 1)
	var healed := 0
	while not wounds.is_empty() and int(wounds[0]["left"]) <= 0:
		var y := wound_floor()   # top of the block = the oldest
		var wd: Dictionary = wounds.pop_front()
		var parked: PackedByteArray = wd["cells"]
		var parked_owner: PackedInt32Array = wd["owner"]
		for x in width:
			cells[y * width + x] = parked[x]
			owner[y * width + x] = parked_owner[x]
		healed += 1
	return healed

func idx(x: int, y: int) -> int:
	return y * width + x

## A real cell of the playfield. Phantom columns are deliberately NOT in bounds: this
## is what idx(), get_cell() and set_cell() are gated on, and they index
## cells[y * width + x], so admitting x >= width here writes into the next row.
func in_bounds(x: int, y: int) -> bool:
	return x >= 0 and x < width and y >= 0 and y < total_height

## Inside the field OR a phantom column -- somewhere a piece may legally sit.
func in_play(x: int, y: int) -> bool:
	return x >= 0 and x < width + ghost_width and y >= 0 and y < total_height

func get_cell(x: int, y: int) -> int:
	if not in_bounds(x, y):
		return 0
	return cells[idx(x, y)]

func set_cell(x: int, y: int, v: int) -> void:
	if in_bounds(x, y):
		cells[idx(x, y)] = v

## Out of bounds counts as solid. This is what makes the T-spin corner test treat
## the walls, floor and ceiling of the field as filled, per the guideline.
func is_blocked(x: int, y: int) -> bool:
	if y < 0 or y >= total_height or x < 0:
		return true
	if x >= width:
		if x >= width + ghost_width:
			return true
		return ghost[y * ghost_width + (x - width)] != GHOST_OPEN
	return cells[y * width + x] != 0

## Hot path: called tens of thousands of times per solver run, so the bounds check
## and index maths are inlined rather than going through is_blocked/in_bounds/idx.
## Three function calls per cell was measurably most of the solver's runtime.
func can_place(piece_cells: Array, pos: Vector2i) -> bool:
	var px := pos.x
	var py := pos.y
	var w := width
	for c in piece_cells:
		var x: int = px + c.x
		var y: int = py + c.y
		if x < 0 or x >= w or y < 0 or y >= total_height:
			return false
		if cells[y * w + x] != 0:
			return false
	return true

## can_place, but phantom columns count as space a piece may occupy.
##
## ONLY the rotation path uses this. If movement and gravity used it too, a piece could
## simply be slid out of the field -- which is both wrong and, because the phantom
## column is not a real cell, how the first version of this crashed on lock. Restricting
## it to rotation is also what makes the assist a SPIN assist: the pocket can be entered
## by turning into it and no other way.
func can_place_in_play(piece_cells: Array, pos: Vector2i) -> bool:
	if ghost_width <= 0:
		return can_place(piece_cells, pos)
	var px := pos.x
	var py := pos.y
	var w := width
	var gw := ghost_width
	for c in piece_cells:
		var x: int = px + c.x
		var y: int = py + c.y
		if y < 0 or y >= total_height or x < 0 or x >= w + gw:
			return false
		if x < w:
			if cells[y * w + x] != 0:
				return false
		elif ghost[y * gw + (x - w)] != GHOST_OPEN:
			return false
	return true

## Full rows among a specific set of candidates. A placement can only complete rows
## it actually touched, so the solver checks 4 rows instead of scanning all 40.
func full_rows_among(candidate_rows: Array) -> Array:
	var out: Array = []
	var w := width
	var floor_y := wound_floor()
	for y in candidate_rows:
		# Wound lines are solid but out of play: never a clearable full row.
		if y < 0 or y >= floor_y:
			continue
		var full := true
		var base: int = y * w
		for x in w:
			if cells[base + x] == 0:
				full = false
				break
		if full:
			out.append(y)
	out.sort()
	return out

## Cells landing in a phantom column are DROPPED, not stored: set_cell is gated on
## in_bounds, which covers the playfield only. A piece that spun into the assist column
## therefore contributes just the cells that are actually on the board, and the row it
## completes is judged on the real columns alone.
func lock_cells(piece_cells: Array, pos: Vector2i, value: int, owner_id: int = 0) -> void:
	for c in piece_cells:
		var x: int = pos.x + c.x
		var y: int = pos.y + c.y
		if not in_bounds(x, y):
			continue
		cells[y * width + x] = value
		owner[y * width + x] = owner_id

func find_full_rows() -> Array:
	var out: Array = []
	for y in wound_floor():
		var full := true
		for x in width:
			if cells[idx(x, y)] == 0:
				full = false
				break
		if full:
			out.append(y)
	return out

## Remove the given rows and drop everything above them down. Rows need not be
## contiguous. Returns the number of rows removed.
func clear_rows(rows: Array) -> int:
	if rows.is_empty():
		return 0
	var doomed := {}
	for y in rows:
		doomed[y] = true
	var out := PackedByteArray()
	out.resize(cells.size())
	out.fill(0)
	var out_owner := PackedInt32Array()
	out_owner.resize(owner.size())
	out_owner.fill(0)
	var write := total_height - 1
	for y in range(total_height - 1, -1, -1):
		if doomed.has(y):
			continue
		for x in width:
			out[write * width + x] = cells[y * width + x]
			out_owner[write * width + x] = owner[y * width + x]
		write -= 1
	cells = out
	owner = out_owner
	return doomed.size()

## Exposed separately from clear_rows so the planned "destroy a row" powerup can
## call it directly instead of going through the piece-lock path.
func remove_row(y: int) -> void:
	clear_rows([y])

## Insert garbage rows at the bottom, shifting the stack up. `holes` lists the empty
## column for each row, bottom-most first. Content pushed above row 0 is lost, which
## in real play is a top-out.
##
## One hole column per row is the standard shape. Repeating the same column across
## rows makes "clean" garbage -- effectively a free well. Varying it makes "cheese",
## which is the case that actually tests downstacking.
##
## With wound lines present, garbage enters at the bottom of the PLAY area, above the
## wound block, so the block stays at the floor and every garbage row stays reachable.
func push_garbage(holes: Array, value: int = GARBAGE) -> void:
	var n := holes.size()
	if n <= 0:
		return
	var floor_y := wound_floor()
	for y in range(0, floor_y - n):
		for x in width:
			cells[y * width + x] = cells[(y + n) * width + x]
			owner[y * width + x] = owner[(y + n) * width + x]
	for i in n:
		var y := floor_y - 1 - i
		var hole: int = holes[i]
		for x in width:
			cells[y * width + x] = 0 if x == hole else value
			owner[y * width + x] = 0          # garbage belongs to no placement

func row_is_empty(y: int) -> bool:
	for x in width:
		if cells[idx(x, y)] != 0:
			return false
	return true

## No pieces in play. Wound lines hold no pieces, so they never block a perfect clear.
func is_empty() -> bool:
	for i in wound_floor() * width:
		if cells[i] != 0:
			return false
	return true

func duplicate_board() -> Board:
	var b := Board.new(width, visible_height, buffer_height)
	b.cells = cells.duplicate()
	b.owner = owner.duplicate()
	for wd in wounds:
		b.wounds.append({"left": wd["left"], "cells": wd["cells"].duplicate(), "owner": wd["owner"].duplicate()})
	return b
