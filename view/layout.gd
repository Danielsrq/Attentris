class_name ViewLayout
extends RefCounted

## Everything about where things sit on screen, derived from the board's dimensions.
## No pixel position is hardcoded, so a 12-wide field or a pentomino set needs no
## layout edits -- only `cell` changes what the sketch looks like.

var cell: int
var preview_cell: int
var margin: int = 20
var pad: int = 8
var title_h: int = 22
## Widest piece a preview box must hold, in cells. 5 leaves room for pentominoes.
var preview_span: int = 5

var field_origin: Vector2
var field_size: Vector2
var hold_rect: Rect2
var next_rect: Rect2
var stats_origin: Vector2
var right_pane: Rect2

func _init(board: Board, viewport: Vector2, cell_size: int = 30) -> void:
	cell = cell_size
	preview_cell = maxi(8, int(round(float(cell) * 0.7)))
	field_size = Vector2(board.width * cell, board.visible_height * cell)

	var box_w := preview_span * preview_cell + pad * 2
	var hold_x := float(margin)
	var field_x := hold_x + box_w + margin
	field_origin = Vector2(field_x, round((viewport.y - field_size.y) * 0.5))

	var slot_h := 3 * preview_cell + pad
	hold_rect = Rect2(Vector2(hold_x, field_origin.y), Vector2(box_w, title_h + slot_h + pad))
	stats_origin = Vector2(hold_x, hold_rect.end.y + margin + title_h)

	var next_x := field_x + field_size.x + margin
	next_rect = Rect2(Vector2(next_x, field_origin.y), Vector2(box_w, title_h + slot_h * 5 + pad))

	var pane_x := next_x + box_w + margin
	right_pane = Rect2(
		Vector2(pane_x, field_origin.y),
		Vector2(maxf(0.0, viewport.x - pane_x - margin), field_size.y))

## Top-left pixel of a visible-field cell. Rows above buffer_height are off-screen.
func cell_pos(board: Board, x: int, y: int) -> Vector2:
	return field_origin + Vector2(x * cell, (y - board.buffer_height) * cell)

## Offset that centres a piece's *occupied* cells (not its whole box) inside a slot,
## so an I and an O both look centred despite different box sizes.
func preview_origin(slot_rect: Rect2, d: PieceDef) -> Vector2:
	var lo := Vector2i(0x7FFFFFFF, 0x7FFFFFFF)
	var hi := Vector2i(-0x7FFFFFFF, -0x7FFFFFFF)
	for c in d.rotations[0]:
		lo = Vector2i(mini(lo.x, c.x), mini(lo.y, c.y))
		hi = Vector2i(maxi(hi.x, c.x), maxi(hi.y, c.y))
	var span := Vector2(float(hi.x - lo.x + 1), float(hi.y - lo.y + 1)) * preview_cell
	return slot_rect.position + (slot_rect.size - span) * 0.5 - Vector2(lo) * preview_cell
