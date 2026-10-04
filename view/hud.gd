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

## The economy log: what each paying placement earned, newest last. This is the answer
## to "are structures actually firing?" -- the bank totals say a number went up, the log
## says which placement did it.
var _econ: Array = []
const ECON_MAX := 14
## How many payouts the one-line log shows. The band is 44px tall, so this is width-bound
## rather than a choice about how much history is interesting.
const ECON_SHOWN := 4

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
	_draw_palette()
	_draw_economy()
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

## The tower palette strip across the top of the pane. Chrome only for now -- the tower
## list arrives with the build cursor. The pane below it belongs to the defence board,
## which draws its own grid, so the HUD no longer frames it: two scripts drawing one
## border is how the ghost-pocket outline went wrong.
func _draw_palette() -> void:
	var r := layout.palette_rect
	if r.size.x <= 0.0:
		return
	draw_rect(r, Color(0.09, 0.09, 0.12))
	draw_rect(r, Color(1, 1, 1, 0.10), false, 2.0)
	draw_string(_font, r.position + Vector2(14.0, 20.0), "TOWERS",
		HORIZONTAL_ALIGNMENT_LEFT, -1, 14, TEXT_DIM)
	draw_string(_font, r.position + Vector2(14.0, 42.0),
		"TAB switches board   WASD move   KP7/9 aim   KP4/6 turn   KP0 build",
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.40, 0.42, 0.50))

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

## Resource bank and the last few payouts, as one band under the whole screen.
##
## This used to be a tall column in the right pane, until the defence board took the pane
## and left 38px of slack beside a 600px grid. Laid out horizontally instead: the bank is
## five chips on the top line, the recent payouts a single line beneath. Both get smaller,
## and that is the right trade -- the bank is a number you glance at, and the log only has
## to answer "did that just pay?".
func _draw_economy() -> void:
	var r := layout.economy_strip
	if r.size.x <= 0.0 or r.size.y <= 0.0:
		return
	draw_rect(r, Color(0.09, 0.09, 0.12))
	draw_rect(r, Color(1, 1, 1, 0.10), false, 1.0)

	var top := r.position.y + 16.0
	var bottom := r.position.y + 34.0
	var x := r.position.x + 12.0
	draw_string(_font, Vector2(x, top), "BANK", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, TEXT_DIM)
	x += 44.0

	# One chip per group: swatch, name, count, and the structures share when there is one.
	# GARBAGE is skipped -- nothing pays into it yet; a dug-garbage economy would.
	const CHIP := 108.0
	var sw := 11.0
	for g in ShapeGroups.all():
		if g == ShapeGroups.GARBAGE:
			continue
		var icon := ShapeGroups.icon_of(g)
		var col: Color = game.piece_set.get_def(icon).color if icon != &"" else TEXT_DIM
		draw_rect(Rect2(Vector2(x, top - sw), Vector2(sw, sw)), col)
		draw_string(_font, Vector2(x + sw + 5.0, top), String(g),
			HORIZONTAL_ALIGNMENT_LEFT, -1, 13, TEXT)
		draw_string(_font, Vector2(x + sw + 5.0, top), str(game.resources.amount(g)),
			HORIZONTAL_ALIGNMENT_RIGHT, 58.0, 13, TEXT)
		var built := game.resources.earned(Resources.STRUCTURES, g)
		if built > 0:
			draw_string(_font, Vector2(x + sw + 68.0, top), "+%d" % built,
				HORIZONTAL_ALIGNMENT_LEFT, -1, 11, TEXT_DIM)
		x += CHIP

	# A completed structure stays named for a moment, at the right-hand end where there is
	# room, because the chip counts alone cannot say which rectangle paid.
	if _struct_age < FLASH_SEC + FADE_SEC and not _struct.is_empty():
		var a := 1.0 if _struct_age < FLASH_SEC else RESTING_ALPHA
		draw_string(_font, Vector2(r.position.x, top), "STRUCTURE %s" % _pay_text(_struct),
			HORIZONTAL_ALIGNMENT_RIGHT, r.size.x - 12.0, 13,
			Color(ACCENT.r, ACCENT.g, ACCENT.b, a))

	_draw_earned(r, bottom)

## The recent payouts, newest first, on one line.
##
## Totals alone cannot answer "did that structure register?" -- the number moves whether
## the clear or the rectangle caused it. Naming the placement and the payout together can,
## and lines a clear up with a structure the same piece completed.
func _draw_earned(r: Rect2, y: float) -> void:
	var x := r.position.x + 12.0
	draw_string(_font, Vector2(x, y), "EARNED", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, TEXT_DIM)
	x += 44.0
	if _econ.is_empty():
		draw_string(_font, Vector2(x, y), "nothing yet -- clear a line or tile a 4x3",
			HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.40, 0.42, 0.50))
		return
	# Newest first and bright, older dimmed, so the eye lands on what just happened.
	var shown := 0
	for i in range(_econ.size() - 1, -1, -1):
		if shown >= ECON_SHOWN or x > r.end.x - 160.0:
			break
		var row: Dictionary = _econ[i]
		var a: float = 1.0 if shown == 0 else 0.45
		var is_struct: bool = String(row["what"]) == "STRUCTURE"
		var col := ACCENT if is_struct else TEXT
		var text := "#%d %s %s" % [int(row["n"]), String(row["what"]), _pay_text(row["pay"])]
		draw_string(_font, Vector2(x, y), text, HORIZONTAL_ALIGNMENT_LEFT, -1, 12,
			Color(col.r, col.g, col.b, a))
		x += _font.get_string_size(text, HORIZONTAL_ALIGNMENT_LEFT, -1, 12).x + 18.0
		shown += 1

## "LJ+2 O+1" for a { group -> amount } payout, in ShapeGroups order so it reads the same
## way every time.
func _pay_text(pay: Dictionary) -> String:
	var parts: Array = []
	for g in ShapeGroups.all():
		if pay.has(g):
			parts.append("%s+%d" % [String(g), int(pay[g])])
	return " ".join(parts)
