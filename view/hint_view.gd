@tool
extends Node2D

## Draws the solver's plan over the field as numbered outlines.
##
## Deliberately outline-only rather than filled: the existing ghost piece is a solid
## translucent shape showing where YOUR piece would land, and two overlapping
## translucent shapes would be unreadable. Outlines plus a step number read as a
## different kind of thing.
##
## Truncation matters. If a planned step clears lines, every later step was computed
## on a board whose rows have shifted down, so drawing them against the CURRENT board
## would place them wrong. The plan is therefore shown up to and including the first
## clearing step, and the overlay says it was cut short.

var game: Game
var layout: ViewLayout
var plan: Array = []
var enabled: bool = false

const OUTLINE_WIDTH := 2.0
const FIRST_STEP_ALPHA := 1.0
const LATER_STEP_ALPHA := 0.55

var _font: Font

func setup(g: Game, l: ViewLayout) -> void:
	game = g
	layout = l
	_font = ThemeDB.fallback_font
	plan = []
	queue_redraw()

func set_plan(p: Array) -> void:
	plan = p
	queue_redraw()

## Steps safe to draw against the current board: everything up to and including the
## first one that clears a line.
func visible_steps() -> Array:
	var out: Array = []
	for p in plan:
		out.append(p)
		if p.lines > 0:
			break
	return out

func truncated() -> bool:
	return visible_steps().size() < plan.size()

func _draw() -> void:
	if not enabled or game == null or layout == null or plan.is_empty():
		return
	var steps := visible_steps()
	for i in steps.size():
		_draw_step(steps[i], i)

func _draw_step(p: Placement, index: int) -> void:
	var board := game.board
	var cs := float(layout.cell)
	var alpha: float = FIRST_STEP_ALPHA if index == 0 else LATER_STEP_ALPHA
	var col := Color(p.def.color, alpha)

	var sum := Vector2.ZERO
	var drawn := 0
	for c in p.cells:
		if c.y < board.buffer_height:
			continue
		var at := layout.cell_pos(board, c.x, c.y)
		draw_rect(Rect2(at + Vector2.ONE, Vector2(cs - 2.0, cs - 2.0)),
			Color(p.def.color, alpha * 0.16))
		draw_rect(Rect2(at + Vector2.ONE, Vector2(cs - 2.0, cs - 2.0)),
			col, false, OUTLINE_WIDTH)
		sum += at + Vector2(cs, cs) * 0.5
		drawn += 1
	if drawn == 0:
		return

	# Step number at the centroid, on a disc so it stays legible over any colour.
	var centre := sum / float(drawn)
	var r := cs * 0.34
	draw_circle(centre, r, Color(0.05, 0.05, 0.07, 0.85 * alpha))
	draw_circle(centre, r, col, false, 1.5)
	var label := str(index + 1)
	draw_string(_font, centre + Vector2(-r * 0.55, r * 0.52), label,
		HORIZONTAL_ALIGNMENT_LEFT, -1, int(cs * 0.55), Color(1, 1, 1, alpha))

	# A step that needs a hold is not something a ghost can express on its own.
	if p.uses_hold:
		draw_string(_font, centre + Vector2(r + 3.0, r * 0.5), "H",
			HORIZONTAL_ALIGNMENT_LEFT, -1, int(cs * 0.45), Color(1.0, 0.85, 0.35, alpha))
