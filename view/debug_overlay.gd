@tool
extends Node2D

## The verification surface, laid out in two columns inside the minigame pane.
##
## Left column is machine state: above all `kick` (the offset index the rotation
## used) and `spin`, which together prove the y-flip in KickTables is right, plus the
## live control bindings. Right column is the rolling history of named outcomes --
## the current one is in the HUD, on the left under the stats.

var game: Game
var layout: ViewLayout
var main: Node
var visible_overlay: bool = true

const TEXT := Color(0.75, 0.95, 0.80)
const LABEL := Color(0.45, 0.55, 0.50)
const HILITE := Color(1.0, 0.85, 0.35)
const ACCENT := Color(0.55, 0.85, 1.0)
const COLUMN_GAP := 300.0
const HISTORY_MAX := 20

## Set during _draw so main can hit-test mouse drags against it. Zero-size until the
## autoplay panel has been drawn at least once.
var slider_rect: Rect2 = Rect2()

var _font: Font
var _log: Array = []   ## named outcomes, newest last

func setup(g: Game, l: ViewLayout, m: Node) -> void:
	game = g
	layout = l
	main = m
	_font = ThemeDB.fallback_font
	_log.clear()
	queue_redraw()

func log_event(e: GameEvent) -> void:
	match e.type:
		GameEvent.Type.PIECE_LOCKED:
			# Named outcomes only -- an ordinary placement that neither spun nor
			# cleared describes as "" and is not worth a line.
			var r: ClearResult = e.data["result"]
			var outcome := r.describe()
			if outcome == "":
				return
			var tags: Array = r.tags()
			_push("%s%s%s%s" % [
				outcome,
				("   " + " ".join(tags)) if not tags.is_empty() else "",
				("   %datk" % r.attack) if r.attack > 0 else "",
				("   +%d" % e.data["points"]) if e.data["points"] > 0 else ""])
		GameEvent.Type.LEVEL_UP:
			_push("-- Level %d --" % e.data["level"])
		GameEvent.Type.TOP_OUT:
			_push("TOP OUT (%s)" % e.data["reason"])
		_:
			pass

func _push(s: String) -> void:
	_log.append(s)
	while _log.size() > HISTORY_MAX:
		_log.pop_front()

func _draw() -> void:
	if game == null or layout == null or not visible_overlay:
		return
	var x := layout.right_pane.position.x + 16.0
	if layout.right_pane.size.x <= 0.0:
		x = layout.field_origin.x + layout.field_size.x + 16.0
	# Draws OVER the right pane rather than beside it, which is why it is on a plain F1
	# toggle. Offset from the pane's own top now that there is no strip above it.
	var top := layout.right_pane.position.y + 14.0
	_draw_state_column(x, top)
	_draw_readout_column(x + COLUMN_GAP, top)

func _draw_state_column(x: float, y: float) -> void:
	y = _heading("SIM STATE", x, y)
	var p := game.active
	var rot_names := ["0", "R", "2", "L"]
	y = _row("piece", "%s" % (p.def.id if p != null else &"-"), x, y)
	y = _row("rot", rot_names[p.rot] if p != null else "-", x, y)
	y = _row("pos", "%s" % (p.pos if p != null else Vector2i.ZERO), x, y)
	y = _row("kick", "%d" % (p.last_kick_index if p != null else -1), x, y, true)
	y = _row("rot-was-last", "%s" % (p.last_action_was_rotation if p != null else false), x, y)
	y = _row("spin", SpinDetect.kind_name(game.last_spin), x, y, true)
	y = _row("lock", "%.2f / %.2f  resets %d" % [
		p.lock_timer if p != null else 0.0,
		game.rules.lock_delay_sec,
		p.move_resets if p != null else 0], x, y)
	y = _row("gravity", "%s  %.4f c/tick" % [
		"on" if game.gravity_enabled else "OFF", game.current_gravity()], x, y)
	if main != null and main.pilot != null:
		# Always shown, and highlighted when something is wrong. With no toggle left to read,
		# this row is the only thing that says which brain is flying -- and a mispaired bot
		# has no symptom of its own beyond playing worse than its own benchmarks.
		var trouble: String = main.pilot_status()
		var bot: Dictionary = main.pilot.preset()
		y = _row("brain", "rust %s / %s  %d nodes" % [
				bot["name"], main.pilot.mode, main.pilot.last_nodes]
			if trouble == "" else trouble, x, y, trouble != "")
	y = _row("das / arr", "%.0f / %.0f ms left" % [game.das_remaining(), game.arr_remaining()], x, y)
	y = _row("b2b / combo", "%d / %d" % [game.b2b, game.combo], x, y)
	y = _row("seed", str(game.seed_value), x, y)

	y += 12.0
	y = _heading("RULES  (%s)" % game.rules.preset_name, x, y)
	y = _row("das / arr / sdf", "%.0f / %.0f / %.0fx" % [
		game.rules.das_ms, game.rules.arr_ms, game.rules.sdf], x, y)
	y = _row("all-spin", "%s   t-spin %s   180 %s" % [
		game.rules.all_spin, game.rules.t_spin, game.rules.allow_180], x, y)
	y = _row("board", "%d x %d  (+%d buffer)" % [
		game.board.width, game.board.visible_height, game.board.buffer_height], x, y)
	y = _row("pieces", "%d in set" % game.piece_set.size(), x, y)

	y += 12.0
	if main != null and main.has_method("control_profile_name"):
		y = _heading("CONTROLS  (%s)" % main.control_profile_name(), x, y)
		# Live feedback: if a key never shows up here, the binding is wrong rather
		# than the sim ignoring it.
		var down: Array = main.held_actions()
		y = _row("down now", " ".join(down) if not down.is_empty() else "-", x, y, not down.is_empty())
	else:
		y = _heading("CONTROLS", x, y)
	if main != null and main.has_method("binding_lines"):
		for pair in main.binding_lines():
			if pair[0] == "":
				y += 4.0
				y = _line(pair[1], x, y, LABEL)
			else:
				y = _row(pair[0], pair[1], x, y)

func _draw_readout_column(x: float, y: float) -> void:
	if main != null and main.has_method("autoplay_status"):
		y = _draw_autoplay(x, y)
	if main != null and main.has_method("hint_status"):
		var h: Dictionary = main.hint_status()
		y = _heading("ASSIST  (H)", x, y)
		y = _row("hints", "ON" if h["enabled"] else "off", x, y, h["enabled"])
		if h["enabled"]:
			y = _row("plan", "%d of %d shown" % [h["shown"], h["plan_len"]], x, y)
			y = _row("search", "depth %d  beam %d" % [h["depth"], h["beam"]], x, y)
			# Flagged hot above 60ms: that is a four-frame hitch at spawn.
			y = _row("solve", "%.1f ms per spawn" % h["solve_ms"], x, y, h["solve_ms"] > 60.0)
			if h["truncated"]:
				y = _line("cut short: a step clears lines", x, y, LABEL)
		y += 12.0

	# The named outcome itself lives in the HUD, under the stats. This column is the
	# rolling record of everything that came before it.
	y = _heading("HISTORY", x, y)
	if _log.is_empty():
		y = _line("-", x, y, LABEL)
	for s in _log:
		y = _line(s, x, y, TEXT)

func _draw_autoplay(x: float, y: float) -> float:
	var a: Dictionary = main.autoplay_status()
	y = _heading("AUTOPLAY  (P)", x, y)
	y = _row("state", "PLAYING" if a["on"] else "off", x, y, a["on"])

	# Rate slider. Logarithmic, so 0.25 pps and 40 pps both get usable resolution.
	var w := 190.0
	var h := 8.0
	slider_rect = Rect2(Vector2(x, y + 4.0), Vector2(w, h))
	draw_rect(slider_rect, Color(1, 1, 1, 0.10))
	draw_rect(slider_rect, Color(1, 1, 1, 0.22), false, 1.0)
	var t: float = clampf(a["t"], 0.0, 1.0)
	draw_rect(Rect2(slider_rect.position, Vector2(w * t, h)), Color(HILITE, 0.55))
	var knob := Vector2(x + w * t, y + 4.0 + h * 0.5)
	draw_circle(knob, 6.0, HILITE)
	var rate: String = "uncapped" if a["uncapped"] else "%.2f pps" % a["pps"]
	draw_string(_font, Vector2(x + w + 12.0, y + 12.0), rate,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, HILITE)
	y += 26.0
	y = _line("drag, or [ / ]", x, y, LABEL)

	y += 4.0
	y = _row("pieces", "%d" % a["pieces"], x, y)
	y = _row("quads", "%d" % a["quads"], x, y)
	# The metric asked for: quads per piece dropped. Distinct from "efficiency",
	# which is reserved for score (or attack) per piece.
	# quad-eff ceiling is 0.100: a quad needs 40 cells and a piece supplies 4, so
	# ten pieces per quad is the theoretical best.
	y = _row("quad-eff", "%.3f of 0.100 max" % a["quad_eff"], x, y, true)
	y = _row("attack", "%d sent" % a["attack"], x, y)
	# The metric reserved for "true efficiency": damage per piece spent.
	y = _row("attack/piece", "%.3f" % a["attack_per_piece"], x, y, true)
	y = _row("score/piece", "%.1f" % a["score_per_piece"], x, y)
	return y + 12.0

func _heading(s: String, x: float, y: float) -> float:
	draw_string(_font, Vector2(x, y), s, HORIZONTAL_ALIGNMENT_LEFT, -1, 11, HILITE)
	return y + 16.0

func _row(label: String, value: String, x: float, y: float, hot: bool = false) -> float:
	draw_string(_font, Vector2(x, y), label, HORIZONTAL_ALIGNMENT_LEFT, -1, 12, LABEL)
	draw_string(_font, Vector2(x + 110.0, y), value, HORIZONTAL_ALIGNMENT_LEFT, -1, 12,
		HILITE if hot else TEXT)
	return y + 16.0

func _line(s: String, x: float, y: float, col: Color) -> float:
	draw_string(_font, Vector2(x, y), s, HORIZONTAL_ALIGNMENT_LEFT, -1, 12, col)
	return y + 15.0
