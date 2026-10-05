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

## Depth of the battler lane, in pixels: 8 bodies at 32px.
##
## This number IS the number of bodies that seals the lane, which makes it the most
## load-bearing constant in the minigame. At 5 deep, five units form an impenetrable wall and
## the front line stops moving -- a count reached constantly at the current spawn rates, so a
## stalemate would be the default state rather than an event. 8 costs eight bodies to seal and
## leaves lateral room for the steering's rotation search to work in.
##
## Its ratio to BattlerRules.aggro_px also decides what KIND of flanking exists. 256 deep
## against 96 of aggro leaves no blind corridor, so a flanker is always seen and a defender
## peels off to intercept: the flank is a race decided by distance and numbers rather than a
## gap to walk through.
var lane_h: int = 256

var field_origin: Vector2
var field_size: Vector2
var hold_rect: Rect2
var next_rect: Rect2
var stats_origin: Vector2
## The whole right-hand column: the battler's home, and still used by replay.tscn.
var right_pane: Rect2
## One lane, vertically centred in the pane. One of about seven that would fit, so adding
## lanes later is a loop rather than a redesign.
var lane_rect: Rect2
## Resource bank and the earned log, as a full-width band under everything, where they
## moved when the pane was given to the minigame.
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

	lane_rect = Rect2(
		right_pane.position + Vector2(0.0, (right_pane.size.y - float(lane_h)) * 0.5),
		Vector2(right_pane.size.x, float(lane_h)))

	# `pad` rather than `margin` below the field: the field is vertically centred, so there
	# are only 60px between its bottom edge and the window, and a full margin at both ends
	# would leave 20px of band.
	var strip_y := field_origin.y + field_size.y + float(pad)
	economy_strip = Rect2(
		Vector2(float(margin), strip_y),
		Vector2(maxf(0.0, viewport.x - float(margin) * 2.0),
			maxf(0.0, viewport.y - strip_y - float(pad))))

## Mid-depth of the lane: the seed for a spawn, which then searches outward for room.
##
## Not a "floor". The lane is two-dimensional now, so a unit's y is a position it walks around
## in rather than a baseline it stands on, and there is no single line the feet sit at.
func lane_centre_y() -> float:
	return lane_rect.position.y + lane_rect.size.y * 0.5

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
