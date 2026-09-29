@tool
extends Node2D

## Playback readout for the replay viewer, drawn in the right-hand pane.
##
## Kept separate from Hud because it describes the VIEWER -- which run, how fast, where
## in it -- rather than the game, and because Hud is shared with live play.
##
## It also carries the PLACING box. In live play the current piece is obvious: it is
## the one moving. In a replay it is drawn where it landed, flush against the stack,
## so without somewhere to name it the eye reads the top of NEXT as the current piece.

const PANEL := Color(0.11, 0.11, 0.14)
const PANEL_EDGE := Color(1, 1, 1, 0.16)
const SLOT := Color(0.07, 0.07, 0.09)
const TEXT := Color(0.88, 0.90, 0.96)
const DIM := Color(0.55, 0.58, 0.66)
const ACCENT := Color(1.0, 0.85, 0.35)
const HELD := Color(0.55, 0.80, 1.0)
const LINE_H := 18.0
const SLOT_SIZE := Vector2(150.0, 86.0)

var layout: ViewLayout
var viewer: Node                    ## the replay scene root, read for playback state

var _font: Font

func setup(l: ViewLayout, v: Node) -> void:
	layout = l
	viewer = v
	_font = ThemeDB.fallback_font
	queue_redraw()

func _draw() -> void:
	if layout == null or viewer == null or _font == null:
		return
	var r := layout.right_pane
	draw_rect(r, PANEL)
	draw_rect(r, PANEL_EDGE, false, 1.0)
	var x := r.position.x + 12.0
	var y := r.position.y + 26.0
	_text(Vector2(x, y), "REPLAY", ACCENT, 15)

	_draw_placing(Rect2(Vector2(r.end.x - SLOT_SIZE.x - 12.0, r.position.y + 12.0), SLOT_SIZE))

	y += LINE_H * 1.6
	for row in viewer.info_rows():
		var label: String = row[0]
		if label == "":
			y += LINE_H * 0.6            # spacer
			continue
		var value: String = row[1]
		var col: Color = row[2] if row.size() > 2 else TEXT
		_text(Vector2(x, y), label, DIM, 12)
		_text(Vector2(x + 104.0, y), value, col, 12)
		y += LINE_H

	y += LINE_H * 0.4
	_text(Vector2(x, y), "KEYS", ACCENT, 12)
	y += LINE_H
	for k in [
		["space", "play / pause"],
		["← →", "step 1 placement"],
		[", .", "step 10"],
		["↑ ↓", "faster / slower"],
		["m", "pace: pps ↔ real time"],
		["[ ]", "previous / next run"],
		["r", "restart run"],
		["g", "grid"],
	]:
		_text(Vector2(x, y), k[0], DIM, 11)
		_text(Vector2(x + 58.0, y), k[1], TEXT, 11)
		y += LINE_H * 0.85

## The piece currently on the board, named and drawn so it is not mistaken for stack.
## When the player held, the piece that SPAWNED is different from the one being placed,
## and saying so is the whole story of that placement.
func _draw_placing(box: Rect2) -> void:
	var d: PieceDef = viewer.current_piece_def()
	_text(box.position + Vector2(0.0, 12.0), "PLACING", ACCENT, 12)
	var slot := Rect2(box.position + Vector2(0.0, 20.0), box.size - Vector2(0.0, 20.0))
	draw_rect(slot, SLOT)
	draw_rect(slot, PANEL_EDGE, false, 1.0)
	if d == null:
		return
	var o := layout.preview_origin(slot, d)
	var pc := float(layout.preview_cell)
	for c in d.rotations[0]:
		var at := o + Vector2(c) * pc
		draw_rect(Rect2(at + Vector2.ONE, Vector2(pc - 2.0, pc - 2.0)), d.color)
		draw_rect(Rect2(at + Vector2.ONE, Vector2(pc - 2.0, pc - 2.0)),
			Color(1, 1, 1, 0.25), false, 1.0)
	# Below the slot, not inside it: a long note clips the piece it describes.
	var note: String = viewer.placing_note()
	if note != "":
		_text(Vector2(slot.position.x, slot.end.y + 14.0), note, HELD, 11)

func _text(at: Vector2, text: String, col: Color, size: int) -> void:
	draw_string(_font, at, text, HORIZONTAL_ALIGNMENT_LEFT, -1, size, col)
