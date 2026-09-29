@tool
extends Node2D

## Draws the playfield: gridlines, locked cells, the ghost and the active piece.
##
## Plain _draw() rather than a TileMapLayer -- crisp gridlines and a translucent
## ghost are a few lines here, and the cell size stays a single number.

var game: Game
var layout: ViewLayout
var show_grid: bool = true
## The replay viewer turns this off: its piece is already AT its final position,
## so a hard-drop ghost is meaningless, and on a spin placement it would point at
## wherever the piece would have fallen instead.
var show_ghost: bool = true
## Outlines the active piece. In live play the falling piece is obvious because it
## moves; in a replay it is drawn where it landed, flush against the stack, and
## without this it reads as part of the stack rather than as the piece being placed.
var outline_active: bool = false

const BG := Color(0.07, 0.07, 0.09)
const GRID := Color(1, 1, 1, 0.14)
const GRID_STRONG := Color(1, 1, 1, 0.28)
const FRAME := Color(0.85, 0.87, 0.95, 0.9)
const ACTIVE_EDGE := Color(1, 1, 1, 0.95)
## The phantom assist column. Deliberately washed out and drawn outside the frame so
## it reads as scaffolding rather than as part of the stack.
const GHOST_FILL := Color(0.42, 0.44, 0.48, 0.45)
const GHOST_EDGE := Color(1, 1, 1, 0.08)

## A completed structure is outlined briefly, because the payout is otherwise invisible:
## the cells look no different afterwards, and the player needs to see WHICH rectangle
## paid to learn to build another.
## The bot's forecast: the pieces it expects to play after this one, drawn faint and
## numbered in play order. Set by the replay viewer from a record's `plan`; empty for
## human replays. Each entry is {k: int, def: PieceDef, cells: Array[Vector2i]}, with
## cells already mapped onto the rows of the board as it stands now.
var plan: Array = []
var show_plan: bool = true
## Alpha fades with distance into the plan: the next piece is the most certain (it is
## what the bot will play unless a new preview changes its mind), later ones less so.
## Both ends stay well below the board's 1.0 so the forecast never reads as stack.
## Eased (quadratic), so the next piece stands clearly apart and the rest drop off
## fast: with five planned pieces, 0.50 / 0.28 / 0.17 / 0.09 / 0.06.
const PLAN_ALPHA_NEAR := 0.50
const PLAN_ALPHA_FAR := 0.06
const PLAN_LABEL := Color(1, 1, 1, 0.9)
## Rows the piece being placed is about to clear. The plan is drawn over today's rows,
## so a planned piece can straddle one of these; marking them makes that read as "this
## row disappears" rather than as a piece that does not match what gets played.
var clearing_rows: Array = []
const CLEARING_BAND := Color(1, 1, 1, 0.28)
const CLEARING_EDGE := Color(1, 1, 1, 0.7)

## Wound lines: a dark wash over the row and its counter in muted red.
const WOUND_TINT := Color(0.0, 0.0, 0.0, 0.35)
const WOUND_LABEL := Color(1.0, 0.55, 0.5, 0.95)

const STRUCT_EDGE := Color(1.0, 0.85, 0.35)
const STRUCT_SEC := 1.2
var _flashes: Array = []   ## [{rect: Rect2i, age: float}]

func setup(g: Game, l: ViewLayout) -> void:
	game = g
	layout = l
	queue_redraw()

func _draw() -> void:
	if game == null or layout == null:
		return
	var board := game.board
	draw_rect(Rect2(layout.field_origin, layout.field_size), BG)

	for vy in board.visible_height:
		var y := board.buffer_height + vy
		for x in board.width:
			var v := board.get_cell(x, y)
			if v != 0:
				_cell(layout.cell_pos(board, x, y), game.piece_set.color_for_value(v), 1.0)

	if board.wound_count() > 0:
		_wounds(board)

	if show_plan and not plan.is_empty():
		_plan(board)

	if game.active != null:
		var col := game.active.def.color
		if show_ghost:
			var ghost := game.ghost_pos()
			for c in game.active.cells():
				_piece_cell(board, ghost + c, col, 0.22)
		for c in game.active.cells():
			_piece_cell(board, game.active.pos + c, col, 1.0)
		if outline_active:
			_outline(board, game.active)

	if show_plan and not plan.is_empty():
		_clearing(board)

	if board.ghost_width > 0:
		_ghost(board)
	_structures(board)

	if show_grid:
		_grid(board)
	_frame(board)

## Marks a structure as just completed. The rect is in board rows and converted at draw
## time, so a line clear would leave it pointing at the wrong place -- hence
## drop_flashes(), which the dispatcher calls on a clear. Clears are dispatched before
## structures, so a structure completed BY the clear still gets its outline.
func flash_structure(rect: Rect2i) -> void:
	_flashes.append({"rect": rect, "age": 0.0})

func drop_flashes() -> void:
	_flashes.clear()

func _process(delta: float) -> void:
	if _flashes.is_empty():
		return
	for f in _flashes:
		f["age"] = float(f["age"]) + delta
	_flashes = _flashes.filter(func(f): return float(f["age"]) < STRUCT_SEC)
	queue_redraw()

func _structures(board: Board) -> void:
	var cs := float(layout.cell)
	for f in _flashes:
		var r: Rect2i = f["rect"]
		if r.position.y + r.size.y <= board.buffer_height:
			continue
		var a := 1.0 - float(f["age"]) / STRUCT_SEC
		var at := layout.cell_pos(board, r.position.x, r.position.y)
		var col := Color(STRUCT_EDGE.r, STRUCT_EDGE.g, STRUCT_EDGE.b, a)
		draw_rect(Rect2(at, Vector2(r.size.x * cs, r.size.y * cs)), col, false, 3.0)

## The field outline, detouring around the pocket when the assist is armed.
##
## A plain rectangle would draw a wall straight through the one cell a piece is allowed
## to turn into, which is exactly the wrong thing to tell the player. Stepping the
## outline out and back makes the opening read as part of the field.
func _frame(board: Board) -> void:
	var o := layout.field_origin
	var sz := layout.field_size
	var cs := float(layout.cell)
	var gap := _pocket_rows(board)
	if gap.is_empty():
		draw_rect(Rect2(o, sz), FRAME, false, 2.0)
		return
	var top: float = o.y + (gap[0] - board.buffer_height) * cs
	var bottom: float = top + cs
	# Anticlockwise from the top-left, bulging out over the pocket row.
	var pts := PackedVector2Array([
		o, Vector2(o.x + sz.x, o.y),
		Vector2(o.x + sz.x, top), Vector2(o.x + sz.x + cs, top),
		Vector2(o.x + sz.x + cs, bottom), Vector2(o.x + sz.x, bottom),
		Vector2(o.x + sz.x, o.y + sz.y), Vector2(o.x, o.y + sz.y), o,
	])
	draw_polyline(pts, FRAME, 2.0)

## Visible rows where the phantom column is open, i.e. the pocket.
func _pocket_rows(board: Board) -> Array:
	var out: Array = []
	if board.ghost_width <= 0:
		return out
	for vy in board.visible_height:
		var y := board.buffer_height + vy
		if board.ghost[y * board.ghost_width] == Board.GHOST_OPEN:
			out.append(y)
	return out

## The phantom column(s) right of the field.
func _ghost(board: Board) -> void:
	for gx in board.ghost_width:
		var col := board.width + gx
		for vy in board.visible_height:
			var y := board.buffer_height + vy
			# Only the overhang and its support are drawn. WALL cells are the field
			# boundary and OPEN is a hole -- drawing either, or any gridline out here,
			# would make the assist read as an eleventh column instead of a pocket.
			if board.ghost[y * board.ghost_width + gx] == Board.GHOST_SHOWN:
				_cell(layout.cell_pos(board, col, y), GHOST_FILL, 1.0)

## Traces the piece's silhouette: an edge is drawn only where the neighbouring cell
## is not part of the same piece, so the result is one outline rather than a grid.
func _outline(board: Board, piece: ActivePiece) -> void:
	var own := {}
	for c in piece.cells():
		own[piece.pos + c] = true
	var cs := float(layout.cell)
	for p in own:
		if p.y < board.buffer_height:
			continue
		var at := layout.cell_pos(board, p.x, p.y)
		for d in [Vector2i.LEFT, Vector2i.RIGHT, Vector2i.UP, Vector2i.DOWN]:
			if own.has(p + d):
				continue
			var from := at
			var to := at
			match d:
				Vector2i.LEFT:
					to += Vector2(0.0, cs)
				Vector2i.RIGHT:
					from += Vector2(cs, 0.0)
					to += Vector2(cs, cs)
				Vector2i.UP:
					to += Vector2(cs, 0.0)
				Vector2i.DOWN:
					from += Vector2(0.0, cs)
					to += Vector2(cs, cs)
			# Cells above the field would draw a lid across the top edge.
			if d == Vector2i.UP and p.y == board.buffer_height:
				continue
			draw_line(from, to, ACTIVE_EDGE, 2.0)

## Wound lines (all-spin repeat penalty): the cells are already drawn in the wound grey;
## this adds each line's clears-until-it-turns-to-garbage just left of the field.
func _wounds(board: Board) -> void:
	var cs := float(layout.cell)
	var font := ThemeDB.fallback_font
	var size := int(cs * 0.55)
	for y in range(board.buffer_height, board.total_height):
		if not board.is_wound(y):
			continue
		var at := layout.cell_pos(board, 0, y)
		draw_rect(Rect2(at, Vector2(board.width * cs, cs)), WOUND_TINT)
		draw_string(font, at + Vector2(-cs, cs * 0.5 + size * 0.36), str(board.wound_left[y]),
			HORIZONTAL_ALIGNMENT_CENTER, cs * 0.9, size, WOUND_LABEL)

func _clearing(board: Board) -> void:
	var cs := float(layout.cell)
	for y in clearing_rows:
		if int(y) < board.buffer_height:
			continue
		var at := layout.cell_pos(board, 0, int(y))
		var r := Rect2(at, Vector2(board.width * cs, cs))
		draw_rect(r, CLEARING_BAND)
		draw_line(r.position, r.position + Vector2(r.size.x, 0), CLEARING_EDGE, 1.0)
		draw_line(r.end - Vector2(r.size.x, 0), r.end, CLEARING_EDGE, 1.0)

## Faint planned pieces, each labelled with its place in the plan (1 = next piece).
func _plan(board: Board) -> void:
	var cs := float(layout.cell)
	var font := ThemeDB.fallback_font
	var size := int(cs * 0.55)
	var last := 1
	for step in plan:
		last = maxi(last, int(step["k"]))
	for step in plan:
		var def: PieceDef = step["def"]
		var cells: Array = step["cells"]
		var t := 0.0 if last <= 1 else float(int(step["k"]) - 1) / float(last - 1)
		var alpha := lerpf(PLAN_ALPHA_FAR, PLAN_ALPHA_NEAR, (1.0 - t) * (1.0 - t))
		var label := Color(PLAN_LABEL.r, PLAN_LABEL.g, PLAN_LABEL.b, PLAN_LABEL.a * lerpf(1.0, 0.55, t))
		var sum := Vector2.ZERO
		var shown := 0
		for c in cells:
			if c.y < board.buffer_height:
				continue
			_piece_cell(board, c, def.color, alpha)
			sum += Vector2(c)
			shown += 1
		if shown == 0:
			continue
		# Label the cell nearest the piece's centre, so it sits on the piece itself.
		var centre := sum / shown
		var best: Vector2i = cells[0]
		for c in cells:
			if c.y >= board.buffer_height and Vector2(c).distance_to(centre) < Vector2(best).distance_to(centre):
				best = c
		var at := layout.cell_pos(board, best.x, best.y)
		draw_string(font, at + Vector2(0, cs * 0.5 + size * 0.36), str(step["k"]),
			HORIZONTAL_ALIGNMENT_CENTER, cs, size, label)

func _piece_cell(board: Board, p: Vector2i, col: Color, alpha: float) -> void:
	if p.y < board.buffer_height:
		return  # inside the spawn buffer, above the visible field
	_cell(layout.cell_pos(board, p.x, p.y), col, alpha)

func _cell(at: Vector2, col: Color, alpha: float) -> void:
	var cs := float(layout.cell)
	var c := Color(col.r, col.g, col.b, col.a * alpha)
	draw_rect(Rect2(at + Vector2.ONE, Vector2(cs - 2.0, cs - 2.0)), c)
	# A lighter inner edge keeps adjacent same-colour cells readable.
	draw_rect(Rect2(at + Vector2.ONE, Vector2(cs - 2.0, cs - 2.0)),
		Color(1, 1, 1, 0.18 * alpha), false, 1.0)

func _grid(board: Board) -> void:
	var o := layout.field_origin
	var cs := float(layout.cell)
	for x in board.width + 1:
		var px := o.x + x * cs
		draw_line(Vector2(px, o.y), Vector2(px, o.y + layout.field_size.y), GRID, 1.0)
	for y in board.visible_height + 1:
		var py := o.y + y * cs
		# Every fourth row a touch brighter, so height is countable at a glance.
		var col := GRID_STRONG if y % 4 == 0 else GRID
		draw_line(Vector2(o.x, py), Vector2(o.x + layout.field_size.x, py), col, 1.0)
