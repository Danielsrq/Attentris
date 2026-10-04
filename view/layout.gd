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

## Height of the tower palette strip across the top of the right pane. The defence grid
## gets whatever is left, and at 18 rows of 30px that is exactly the 540 remaining -- so
## this number is also what decides whether the grid fits.
var palette_h: int = 60

var field_origin: Vector2
var field_size: Vector2
var hold_rect: Rect2
var next_rect: Rect2
var stats_origin: Vector2
## The whole right-hand column. Still used by replay.tscn, which has no defence board.
var right_pane: Rect2
## The two halves the defence game splits that column into.
var palette_rect: Rect2
var defence_rect: Rect2
## Resource bank and the earned log, as a full-width band under everything. They moved out
## of the pane when the defence board took it: the pane has 38px of slack beside a 600px
## grid, which is not a readout.
var economy_strip: Rect2

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

	palette_rect = Rect2(right_pane.position, Vector2(right_pane.size.x, float(palette_h)))
	defence_rect = Rect2(
		right_pane.position + Vector2(0.0, float(palette_h)),
		Vector2(right_pane.size.x, maxf(0.0, right_pane.size.y - float(palette_h))))

	# `pad` rather than `margin` below the field: the field is vertically centred, so there
	# are only 60px between its bottom edge and the window, and a full margin at both ends
	# would leave 20px of band.
	var strip_y := field_origin.y + field_size.y + float(pad)
	economy_strip = Rect2(
		Vector2(float(margin), strip_y),
		Vector2(maxf(0.0, viewport.x - float(margin) * 2.0),
			maxf(0.0, viewport.y - strip_y - float(pad))))

## Top-left pixel of a cols x rows defence grid, centred in `defence_rect`.
##
## The defence grid deliberately uses the SAME `cell` as the playfield -- that shared cell
## size is what makes the two halves read as one board rather than two games sharing a
## window. At 18 rows it fills `defence_rect` exactly in height, so only the horizontal
## slack is ever centred.
func defence_origin(cols: int, rows: int) -> Vector2:
	var span := Vector2(float(cols * cell), float(rows * cell))
	var slack := defence_rect.size - span
	# Clamped at zero, never halved into the negative. A pane NARROWER than the grid would
	# otherwise centre the board leftwards -- straight on top of the playfield, which is
	# exactly what it did at any window under about 1240px wide. Overflowing off the right
	# edge is ugly; overlapping the other game is wrong.
	return defence_rect.position + Vector2(maxf(0.0, slack.x), maxf(0.0, slack.y)) * 0.5

## True when the pane is actually wide enough for the grid. The defence view skips drawing
## when it is not, rather than painting over the Tetris board.
func defence_fits(cols: int, rows: int) -> bool:
	return defence_rect.size.x >= float(cols * cell) and defence_rect.size.y >= float(rows * cell)

## Top-left pixel of a defence cell, in the same spirit as cell_pos for the playfield.
func defence_cell_pos(origin: Vector2, x: int, y: int) -> Vector2:
	return origin + Vector2(float(x * cell), float(y * cell))

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
