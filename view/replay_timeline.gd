@tool
extends Control

## Scrubber for the replay viewer, coloured by whether the board RECOVERS, so the
## desync estimate can be checked by eye rather than taken on trust.
##
##   green   the board is clean
##   teal    clean, but only because garbage inferred by tools/recover.mjs was replayed
##           here -- this stretch exists at all because the search reconstructed it
##   grey tick  where that garbage actually entered the board
##   amber   holes present, but the board still recovers from them later -- a cell
##           buried on purpose and reopened, which strong players do constantly
##   red     the episode that never recovers: from here the board no longer matches
##           the one the player could see
##
## A plain HSlider cannot show this, hence a custom control: it draws the bands and
## handles the dragging itself.

const TRACK_H := 16.0
const EMPTY := Color(0.20, 0.21, 0.25)
const GREEN := Color(0.35, 0.72, 0.42)
const AMBER := Color(0.85, 0.66, 0.22)
const RED := Color(0.80, 0.26, 0.26)
const TEAL := Color(0.25, 0.68, 0.72)
const HEAD := Color(1, 1, 1, 0.95)
const BREAK := Color(1.0, 0.45, 0.40)
## Deep grey, matching the garbage blocks on the field.
const GARBAGE := Color(0.38, 0.40, 0.45)
const EDGE := Color(1, 1, 1, 0.18)

const MARK_CLEAR := 0
const MARK_QUEUED := 1
const MARK_BROKEN := 2
const MARK_RECOVERED := 3

signal scrubbed(index: int)

var marks: PackedByteArray = PackedByteArray()
var index: int = 0
## First sample marked broken, or -1. Drawn as an explicit tick because the amber
## band before it is often only one or two samples wide -- about a pixel -- and would
## otherwise vanish against the red.
var break_at: int = -1
## Sample indices where inferred garbage entered. Drawn as ticks rather than a band:
## garbage arrives at an instant, and knowing WHEN matters more than knowing that the
## stretch after it was reconstructed.
var garbage_at: PackedInt32Array = PackedInt32Array()

func _init() -> void:
	custom_minimum_size = Vector2(120.0, TRACK_H + 8.0)
	size_flags_horizontal = Control.SIZE_EXPAND_FILL
	size_flags_vertical = Control.SIZE_SHRINK_CENTER
	# Never take focus, or this would swallow the arrow keys the viewer steps with.
	focus_mode = Control.FOCUS_NONE
	mouse_default_cursor_shape = Control.CURSOR_POINTING_HAND

func set_marks(m: PackedByteArray, first_broken: int = -1,
		entries: PackedInt32Array = PackedInt32Array()) -> void:
	marks = m
	break_at = first_broken
	garbage_at = entries
	queue_redraw()

func set_index(i: int) -> void:
	index = i
	queue_redraw()

func _colour(m: int) -> Color:
	match m:
		MARK_BROKEN:
			return RED
		MARK_QUEUED:
			return AMBER
		MARK_RECOVERED:
			return TEAL
		_:
			return GREEN

func _draw() -> void:
	var track := Rect2(Vector2(0.0, (size.y - TRACK_H) * 0.5), Vector2(size.x, TRACK_H))
	draw_rect(track, EMPTY)
	var n := marks.size()
	if n > 0 and size.x > 0.0:
		# Runs of one colour are drawn as single rects, so the bands stay crisp
		# instead of being stippled by per-sample rounding.
		var start := 0
		for i in range(1, n + 1):
			if i < n and marks[i] == marks[start]:
				continue
			var x0 := track.position.x + track.size.x * float(start) / float(n)
			var x1 := track.position.x + track.size.x * float(i) / float(n)
			draw_rect(Rect2(Vector2(x0, track.position.y),
				Vector2(maxf(1.0, x1 - x0), track.size.y)), _colour(marks[start]))
			start = i
	# Drawn under the break tick and the playhead, so neither is obscured.
	if n > 0:
		for g in garbage_at:
			var gx := track.position.x + track.size.x * float(g) / float(n)
			draw_line(Vector2(gx, track.position.y - 2.0),
				Vector2(gx, track.end.y + 2.0), GARBAGE, 3.0)
	if break_at >= 0 and n > 0:
		var bx := track.position.x + track.size.x * float(break_at) / float(n)
		draw_line(Vector2(bx, track.position.y - 4.0),
			Vector2(bx, track.end.y + 4.0), BREAK, 2.0)
	draw_rect(track, EDGE, false, 1.0)
	if n > 0:
		var hx := track.position.x + track.size.x * (float(index) + 0.5) / float(n)
		draw_line(Vector2(hx, track.position.y - 3.0),
			Vector2(hx, track.end.y + 3.0), HEAD, 2.0)

func _gui_input(event: InputEvent) -> void:
	if marks.is_empty() or size.x <= 0.0:
		return
	var at := -1.0
	var mb := event as InputEventMouseButton
	if mb != null and mb.pressed and mb.button_index == MOUSE_BUTTON_LEFT:
		at = mb.position.x
	var mm := event as InputEventMouseMotion
	if mm != null and (mm.button_mask & MOUSE_BUTTON_MASK_LEFT) != 0:
		at = mm.position.x
	if at < 0.0:
		return
	var i := int(floor(at / size.x * float(marks.size())))
	scrubbed.emit(clampi(i, 0, marks.size() - 1))
	accept_event()
