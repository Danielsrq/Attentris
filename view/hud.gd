@tool
extends Node2D

## Hold box, next queue, run stats, and a labelled placeholder for the second
## minigame so the two-pane framing is visible from day one.

var game: Game
var layout: ViewLayout

const PANEL := Color(0.11, 0.11, 0.14)
const PANEL_EDGE := Color(1, 1, 1, 0.16)
const TEXT := Color(0.88, 0.90, 0.96)
const TEXT_DIM := Color(0.55, 0.58, 0.66)
const ACCENT := Color(1.0, 0.85, 0.35)

## The named outcome stays on screen after it fires, at full brightness briefly and
## then dimmed, so it reads both as "that just happened" and as a standing record of
## the last thing worth naming.
const FLASH_SEC := 1.6
const FADE_SEC := 0.8
const RESTING_ALPHA := 0.55

var _font: Font
var _last: ClearResult = null
var _last_points: int = 0
var _age: float = 0.0
## Last structure payout, shown beside the resource readout so the player can see what
## the rectangle was worth and not just that one happened.
var _struct: Dictionary = {}
var _struct_age: float = 999.0

## Which page the right pane shows. Driven by main; see view/pane_tab.gd.
var pane_tab: int = PaneTab.GAMEPLAY
## The economy log: what each paying placement earned, newest last. This is the answer
## to "are structures actually firing?" -- the bank totals say a number went up, the log
## says which placement did it.
var _econ: Array = []
const ECON_MAX := 14

func setup(g: Game, l: ViewLayout) -> void:
	game = g
	layout = l
	_font = ThemeDB.fallback_font
	_last = null
	_last_points = 0
	_age = 0.0
	_struct = {}
	_struct_age = 999.0
	_econ.clear()
	queue_redraw()

## Called for every lock. Ordinary placements are ignored so the readout holds the
## last thing actually worth announcing rather than being wiped by the next piece.
func on_placement(r: ClearResult, points: int) -> void:
	if r == null or not r.is_named():
		return
	_last = r
	_last_points = points
	_age = 0.0

## Called once per completed structure, with the { group -> amount } it paid.
func on_structure(payout: Dictionary) -> void:
	_struct = payout
	_struct_age = 0.0

## Adds a line to the economy log. `piece_no` is the placement it belongs to, so a clear
## and a structure completed by the same piece line up.
func log_economy(piece_no: int, what: String, pay: Dictionary) -> void:
	_econ.append({"n": piece_no, "what": what, "pay": pay})
	while _econ.size() > ECON_MAX:
		_econ.pop_front()

func _process(delta: float) -> void:
	if Engine.is_editor_hint():
		return
	_age += delta
	_struct_age += delta

func _draw() -> void:
	if game == null or layout == null:
		return
	_draw_hold()
	_draw_next()
	_draw_stats()
	_draw_right_pane()
	if game.topped_out:
		_draw_top_out()

func _draw_hold() -> void:
	_panel(layout.hold_rect, "HOLD")
	var slot := Rect2(
		layout.hold_rect.position + Vector2(layout.pad, layout.title_h),
		Vector2(layout.hold_rect.size.x - layout.pad * 2, layout.hold_rect.size.y - layout.title_h - layout.pad))
	if game.hold_def != null:
		var dim := 0.4 if game.hold_used else 1.0
		_preview(slot, game.hold_def, dim)

func _draw_next() -> void:
	_panel(layout.next_rect, "NEXT")
	var slot_h := 3 * layout.preview_cell + layout.pad
	for i in game.next_queue.size():
		var slot := Rect2(
			layout.next_rect.position + Vector2(layout.pad, layout.title_h + slot_h * i),
			Vector2(layout.next_rect.size.x - layout.pad * 2, slot_h))
		_preview(slot, game.next_queue[i], 1.0 if i == 0 else 0.75)

func _preview(slot: Rect2, d: PieceDef, alpha: float) -> void:
	var o := layout.preview_origin(slot, d)
	var pc := float(layout.preview_cell)
	var col := Color(d.color.r, d.color.g, d.color.b, alpha)
	for c in d.rotations[0]:
		var at := o + Vector2(c) * pc
		draw_rect(Rect2(at + Vector2.ONE, Vector2(pc - 2.0, pc - 2.0)), col)

func _draw_stats() -> void:
	var lines := [
		["SCORE", str(game.score)],
		["ATTACK", str(game.attack_sent)],
		["LINES", str(game.lines)],
		["LEVEL", str(game.level)],
		["PIECES", str(game.pieces_placed)],
	]
	var y := layout.stats_origin.y
	for pair in lines:
		draw_string(_font, Vector2(layout.stats_origin.x, y), pair[0],
			HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_DIM)
		draw_string(_font, Vector2(layout.stats_origin.x, y + 20), pair[1],
			HORIZONTAL_ALIGNMENT_LEFT, -1, 20, TEXT)
		y += 46

	# Live chain state: the two numbers the resource economy will actually care
	# about. Deliberately separate from the placement readout below, which is a
	# record of one moment and can go stale.
	var chain := []
	if game.b2b >= 1:
		chain.append("B2B x%d" % game.b2b)
	if game.combo >= 1:
		chain.append("COMBO %d" % game.combo)
	if not chain.is_empty():
		draw_string(_font, Vector2(layout.stats_origin.x, y + 6), " / ".join(chain),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 14, ACCENT)
	y += 34
	_draw_last_placement(y)

## The outcome name, under the stats. Drawn on two lines because the left column is
## only as wide as the hold box and "T-Spin Mini Single" does not fit on one.
func _draw_last_placement(y: float) -> void:
	var x := layout.stats_origin.x
	draw_string(_font, Vector2(x, y), "LAST", HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_DIM)
	y += 22
	if _last == null:
		draw_string(_font, Vector2(x, y), "-", HORIZONTAL_ALIGNMENT_LEFT, -1, 18, TEXT_DIM)
		return

	var a := RESTING_ALPHA
	if _age < FLASH_SEC:
		a = 1.0
	elif _age < FLASH_SEC + FADE_SEC:
		a = lerpf(1.0, RESTING_ALPHA, (_age - FLASH_SEC) / FADE_SEC)

	var spin := _last.spin_name()
	if spin != "":
		draw_string(_font, Vector2(x, y), spin, HORIZONTAL_ALIGNMENT_LEFT, -1, 13,
			Color(ACCENT, a))
		y += 20
	draw_string(_font, Vector2(x, y), _last.count_name(), HORIZONTAL_ALIGNMENT_LEFT, -1, 20,
		Color(TEXT, a))
	y += 22
	if _last_points > 0:
		draw_string(_font, Vector2(x, y), "+%d" % _last_points, HORIZONTAL_ALIGNMENT_LEFT, -1, 14,
			Color(ACCENT, a * 0.9))
		y += 20
	var tags: Array = _last.tags()
	if not tags.is_empty():
		draw_string(_font, Vector2(x, y), " ".join(tags), HORIZONTAL_ALIGNMENT_LEFT, -1, 11,
			Color(TEXT_DIM, a))

func _draw_right_pane() -> void:
	var r := layout.right_pane
	if r.size.x <= 0.0 or pane_tab == PaneTab.OFF:
		return
	draw_rect(r, Color(0.09, 0.09, 0.12))
	draw_rect(r, Color(1, 1, 1, 0.10), false, 2.0)
	_draw_tabs(r)
	if pane_tab == PaneTab.DEBUG:
		return          # DebugOverlay draws the page itself, on top of this frame
	_draw_resources(r)
	_draw_econ(r)
	# Sits low in the pane so the debug overlay, which shares this space, does not
	# cover it. Centred across the pane's full width, not from its midpoint.
	var x := r.position.x
	draw_string(_font, Vector2(x, r.end.y - 52), "MINIGAME PANE",
		HORIZONTAL_ALIGNMENT_CENTER, r.size.x, 18, TEXT_DIM)
	draw_string(_font, Vector2(x, r.end.y - 28),
		"fed by the LINES_CLEARED event stream",
		HORIZONTAL_ALIGNMENT_CENTER, r.size.x, 13, Color(0.40, 0.42, 0.50))

func _draw_top_out() -> void:
	var r := Rect2(layout.field_origin, layout.field_size)
	draw_rect(r, Color(0, 0, 0, 0.66))
	draw_string(_font, r.position + Vector2(0, r.size.y * 0.5 - 10), "TOP OUT",
		HORIZONTAL_ALIGNMENT_CENTER, r.size.x, 28, Color(1, 0.4, 0.4))
	draw_string(_font, r.position + Vector2(0, r.size.y * 0.5 + 16),
		"%s  -  press R" % game.top_out_reason,
		HORIZONTAL_ALIGNMENT_CENTER, r.size.x, 14, TEXT)

func _panel(r: Rect2, title: String) -> void:
	draw_rect(r, PANEL)
	draw_rect(r, PANEL_EDGE, false, 1.0)
	draw_string(_font, r.position + Vector2(layout.pad, 15), title,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_DIM)

## The resource bank: one row per shape group, coloured by the piece that names it.
##
## The split matters more than the total, which is why the structures share is printed
## alongside rather than folded in. O is the case to watch: the clear economy can pay it
## at most 2 at a time, so a healthy O count means the player is building rectangles.
func _draw_resources(pane: Rect2) -> void:
	var x := pane.position.x + 14.0
	var y := pane.position.y + PaneTab.STRIP_H + 22.0
	draw_string(_font, Vector2(x, y), "RESOURCES", HORIZONTAL_ALIGNMENT_LEFT, -1, 14, TEXT_DIM)
	y += 10.0
	var sw := 12.0
	for g in ShapeGroups.all():
		if g == ShapeGroups.GARBAGE:
			continue          # nothing pays into it yet; a dug-garbage economy would
		y += 22.0
		var icon := ShapeGroups.icon_of(g)
		var col: Color = game.piece_set.get_def(icon).color if icon != &"" else TEXT_DIM
		draw_rect(Rect2(Vector2(x, y - sw + 2.0), Vector2(sw, sw)), col)
		draw_string(_font, Vector2(x + sw + 8.0, y), String(g),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 15, TEXT)
		draw_string(_font, Vector2(x + sw + 8.0, y), str(game.resources.amount(g)),
			HORIZONTAL_ALIGNMENT_RIGHT, 92.0, 15, TEXT)
		var built := game.resources.earned(Resources.STRUCTURES, g)
		if built > 0:
			draw_string(_font, Vector2(x + sw + 108.0, y), "+%d built" % built,
				HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_DIM)

	if _struct_age < FLASH_SEC + FADE_SEC and not _struct.is_empty():
		var a := 1.0 if _struct_age < FLASH_SEC else RESTING_ALPHA
		var parts: Array = []
		for g in ShapeGroups.all():
			if _struct.has(g):
				parts.append("%s+%d" % [String(g), int(_struct[g])])
		draw_string(_font, Vector2(x, y + 30.0), "STRUCTURE  %s" % " ".join(parts),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 15, Color(ACCENT.r, ACCENT.g, ACCENT.b, a))

## The tab strip. Drawn on every page, including the debug one, so the way back is
## always on screen -- an overlay you cannot find the exit from is how F1 used to feel.
func _draw_tabs(pane: Rect2) -> void:
	var x := pane.position.x + 12.0
	var y := pane.position.y + 8.0
	for t in PaneTab.pages():
		var tab: int = t
		var label := PaneTab.name_of(tab)
		var w := _font.get_string_size(label, HORIZONTAL_ALIGNMENT_LEFT, -1, 14).x + 20.0
		var box := Rect2(Vector2(x, y), Vector2(w, 20.0))
		var on: bool = tab == pane_tab
		if on:
			draw_rect(box, Color(ACCENT.r, ACCENT.g, ACCENT.b, 0.18))
			draw_rect(box, Color(ACCENT.r, ACCENT.g, ACCENT.b, 0.65), false, 1.0)
		else:
			draw_rect(box, Color(1, 1, 1, 0.06), false, 1.0)
		draw_string(_font, Vector2(x + 10.0, y + 15.0), label,
			HORIZONTAL_ALIGNMENT_LEFT, -1, 14, ACCENT if on else TEXT_DIM)
		x += w + 6.0
	draw_string(_font, Vector2(pane.end.x - 12.0 - 60.0, y + 15.0), "TAB",
		HORIZONTAL_ALIGNMENT_RIGHT, 60.0, 13, TEXT_DIM)

## The economy log: one line per placement that paid, newest last.
##
## Totals alone cannot answer "did that structure register?" -- the number moves whether
## the clear or the rectangle caused it. Naming the placement and the payout side by side
## does, and lines up a clear with a structure the same piece completed.
func _draw_econ(pane: Rect2) -> void:
	var x := pane.position.x + 14.0
	var y := pane.position.y + PaneTab.STRIP_H + 22.0 + 6 * 22.0 + 60.0
	draw_string(_font, Vector2(x, y), "EARNED", HORIZONTAL_ALIGNMENT_LEFT, -1, 14, TEXT_DIM)
	if _econ.is_empty():
		draw_string(_font, Vector2(x, y + 22.0), "nothing yet -- clear a line or tile a 4x3",
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT_DIM)
		return
	for i in _econ.size():
		var row: Dictionary = _econ[i]
		y += 20.0
		# Newest at full brightness, older fading, so the eye lands on what just happened.
		var a: float = 1.0 if i == _econ.size() - 1 else 0.45
		draw_string(_font, Vector2(x, y + 22.0), "#%d" % int(row["n"]),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(TEXT_DIM.r, TEXT_DIM.g, TEXT_DIM.b, a))
		var is_struct: bool = String(row["what"]) == "STRUCTURE"
		var col := ACCENT if is_struct else TEXT
		draw_string(_font, Vector2(x + 46.0, y + 22.0), String(row["what"]),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, Color(col.r, col.g, col.b, a))
		var pay: Dictionary = row["pay"]
		var parts: Array = []
		for g in ShapeGroups.all():
			if pay.has(g):
				parts.append("%s+%d" % [String(g), int(pay[g])])
		draw_string(_font, Vector2(x + 46.0, y + 22.0), " ".join(parts),
			HORIZONTAL_ALIGNMENT_RIGHT, pane.size.x - 60.0, 13,
			Color(col.r, col.g, col.b, a))
