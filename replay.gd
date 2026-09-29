@tool
extends Node2D

## Viewer for the harvested TETR.IO Zenith dataset.
##
## Pros place 4+ pieces a second, which is unfollowable at native speed, so playback is
## paced two ways:
##
##   PPS       -- a fixed number of placements per second, regardless of how fast the
##                player actually was. Set it to 1 and every placement gets a full
##                second. This is the mode for reading a stack.
##   REAL_TIME -- the recorded gap between placements, divided by a speed multiplier.
##                Keeps the rhythm of the run, the hesitations and the bursts, which
##                PPS pacing flattens away. 0.25x is followable, 1.0x is what they did.
##
## Nothing is simulated. Each sample carries the board as it stood when the piece
## spawned, so a step is a render rather than a reconstruction, and stepping backwards
## is exact instead of an undo.

enum Pace { PPS, REAL_TIME }

const CELL := 30
const BOARD_WIDTH := 10
const BOARD_VISIBLE := 20
const BOARD_BUFFER := 20

const PPS_MIN := 0.25
const PPS_MAX := 20.0
const SPEED_MIN := 0.05
const SPEED_MAX := 8.0
## Floor on a single step's dwell, so a burst of near-simultaneous placements in
## real-time mode cannot flash past inside one frame.
const MIN_DWELL := 1.0 / 120.0

@export_file("*.jsonl") var dataset_path: String = ReplayLog.DEFAULT_PATH
@export var pace: Pace = Pace.PPS:
	set(v):
		pace = v
		if info != null:
			info.queue_redraw()
@export_range(PPS_MIN, PPS_MAX, 0.25) var pps: float = 1.5:
	set(v):
		pps = clampf(v, PPS_MIN, PPS_MAX)
		if info != null:
			info.queue_redraw()
@export_range(SPEED_MIN, SPEED_MAX, 0.05) var real_time_speed: float = 0.35:
	set(v):
		real_time_speed = clampf(v, SPEED_MIN, SPEED_MAX)
		if info != null:
			info.queue_redraw()
@export var play_on_start: bool = true
## Which run to open. Wrapped to what the dataset actually holds.
@export var run_index: int = 0:
	set(v):
		run_index = v
		# _open_run writes the wrapped value back, which re-enters this setter; the
		# guard stops that becoming infinite recursion.
		if is_inside_tree() and not _opening:
			_open_run(v)

var board_view: Node2D
var hud: Node2D
var info: Node2D
var controls: CanvasLayer

var piece_set: PieceSet
var board: Board
var game: Game
var layout: ViewLayout
var log: ReplayLog

var steps: Array = []
var index: int = 0
var playing: bool = false

var _accum: float = 0.0
var _lines_so_far: int = 0
var _attack_so_far: int = 0
var _load_ms: float = 0.0
var _sample_total: int = 0
var _opening: bool = false
## One byte per sample: 0 queue empty, 1 garbage waiting, 2 garbage has landed.
var _marks: PackedByteArray = PackedByteArray()
var _desync: int = -1
## Sample indices at which inferred garbage entered the board.
var _garbage_at: PackedInt32Array = PackedInt32Array()

func _ready() -> void:
	board_view = get_node_or_null("BoardView")
	hud = get_node_or_null("Hud")
	info = get_node_or_null("ReplayInfo")
	controls = get_node_or_null("ReplayControls")

	piece_set = PieceSet.tetrominoes()
	board = Board.new(BOARD_WIDTH, BOARD_VISIBLE, BOARD_BUFFER)
	# A real Game is built only so BoardView and Hud can draw from it unchanged. The
	# viewer writes its fields directly and never ticks it.
	game = Game.new(RuleSet.competitive(), piece_set, board, Scorer.new(ScoreTable.new()), 0)
	layout = ViewLayout.new(board, get_viewport_rect().size, CELL)

	if board_view != null:
		board_view.setup(game, layout)
		# The recorded piece is already at its landing cells, so a hard-drop ghost is
		# meaningless -- and being flush with the stack, it needs an outline to read
		# as the piece being placed rather than as part of the stack.
		board_view.show_ghost = false
		board_view.outline_active = true
	if hud != null:
		hud.setup(game, layout)
	if info != null:
		info.setup(layout, self)
	if controls != null:
		controls.setup(self)

	var t0 := Time.get_ticks_usec()
	log = ReplayLog.new(piece_set)
	if not log.load_file(dataset_path):
		push_error("ReplayLog: %s" % log.error)
		if info != null:
			info.queue_redraw()
		return
	_load_ms = float(Time.get_ticks_usec() - t0) / 1000.0
	_sample_total = log.total_samples()
	playing = play_on_start and not Engine.is_editor_hint()
	_open_run(run_index)

func _open_run(which: int) -> void:
	if log == null or log.run_count() == 0:
		return
	var wrapped := posmod(which, log.run_count())
	if wrapped != run_index:
		_opening = true
		run_index = wrapped
		_opening = false
	steps = log.steps_for(log.names[wrapped])
	_build_marks()
	_seek(0)

func _seek(to: int) -> void:
	if steps.is_empty():
		return
	index = clampi(to, 0, steps.size() - 1)
	_accum = 0.0
	# Lines and attack are accumulated rather than read off a sample: the dataset
	# records what each placement cleared, not a running total. Attack is scored
	# through our own AttackTable, so the number means what it means in our game
	# rather than in TETR.IO's.
	_lines_so_far = 0
	_attack_so_far = 0
	for i in index:
		_lines_so_far += int(steps[i].get("cleared", 0))
		var r := _result_for(steps[i])
		if r != null:
			_attack_so_far += game.attack_table.attack_for(r)
	_show()

func _show() -> void:
	if steps.is_empty():
		return
	var step: Dictionary = steps[index]
	log.apply(step, game)
	game.lines = _lines_so_far
	game.attack_sent = _attack_so_far
	game.last_result = _result_for(step)
	if game.last_result != null:
		game.last_result.attack = game.attack_table.attack_for(game.last_result)
	game.last_points = 0
	if hud != null:
		if game.last_result != null:
			hud.on_placement(game.last_result, 0)
		hud.queue_redraw()
	if board_view != null:
		board_view.queue_redraw()
	if info != null:
		info.queue_redraw()
	_refresh_controls()

## A ClearResult built from the recorded spin and line count, so the existing Hud can
## name the outcome ("T-Spin Double") from dataset facts rather than a re-derivation.
func _result_for(step: Dictionary) -> ClearResult:
	var lines := int(step.get("cleared", 0))
	if lines <= 0:
		return null
	var r := ClearResult.new()
	r.lines = lines
	r.rows = step.get("cleared_rows", [])
	var placed: Dictionary = step.get("place", {})
	r.piece_id = StringName(String(placed.get("type", "")).to_upper())
	match String(step.get("spin", "none")):
		"full":
			r.spin = SpinDetect.Kind.FULL
		"mini":
			r.spin = SpinDetect.Kind.MINI
		_:
			r.spin = SpinDetect.Kind.NONE
	return r

func _advance(by: int) -> bool:
	if steps.is_empty():
		return false
	var to := index + by
	if to >= steps.size():
		_seek(steps.size() - 1)
		playing = false
		return false
	_seek(maxi(0, to))
	return true

## Seconds to hold the current placement on screen.
func _dwell() -> float:
	if pace == Pace.PPS or steps.is_empty():
		return maxf(MIN_DWELL, 1.0 / maxf(PPS_MIN, pps))
	var here := int(steps[index].get("frame", 0))
	var next := here
	if index + 1 < steps.size():
		next = int(steps[index + 1].get("frame", here))
	var gap := float(maxi(1, next - here)) / Game.TICKS_PER_SECOND
	return maxf(MIN_DWELL, gap / maxf(SPEED_MIN, real_time_speed))

func _process(delta: float) -> void:
	if Engine.is_editor_hint() or not playing or steps.is_empty():
		return
	_accum += delta
	# Guarded so a very high pps cannot spin here indefinitely on a long frame.
	var guard := 0
	while playing and _accum >= _dwell() and guard < 240:
		_accum -= _dwell()
		guard += 1
		if not _advance(1):
			break

func _unhandled_input(event: InputEvent) -> void:
	if Engine.is_editor_hint():
		return
	var k := event as InputEventKey
	if k == null or not k.pressed or k.echo:
		return
	match k.keycode:
		KEY_SPACE:
			playing = not playing
			_accum = 0.0
		KEY_RIGHT, KEY_D:
			playing = false
			_advance(1)
		KEY_LEFT, KEY_A:
			playing = false
			_advance(-1)
		KEY_PERIOD:
			playing = false
			_advance(10)
		KEY_COMMA:
			playing = false
			_advance(-10)
		KEY_UP, KEY_W:
			_nudge_speed(1)
		KEY_DOWN, KEY_S:
			_nudge_speed(-1)
		KEY_M:
			pace = Pace.REAL_TIME if pace == Pace.PPS else Pace.PPS
		KEY_BRACKETRIGHT:
			_open_run(run_index + 1)
		KEY_BRACKETLEFT:
			_open_run(run_index - 1)
		KEY_R, KEY_HOME:
			_seek(0)
		KEY_END:
			_seek(steps.size() - 1)
		KEY_G:
			if board_view != null:
				board_view.show_grid = not board_view.show_grid
				board_view.queue_redraw()
		_:
			return
	if info != null:
		info.queue_redraw()

func _nudge_speed(dir: int) -> void:
	if pace == Pace.PPS:
		# Coarser steps once the rate is past the point of being readable anyway.
		var step := 0.25 if pps < 4.0 else 1.0
		pps = clampf(pps + step * dir, PPS_MIN, PPS_MAX)
	else:
		var step := 0.05 if real_time_speed < 0.5 else 0.25
		real_time_speed = clampf(real_time_speed + step * dir, SPEED_MIN, SPEED_MAX)

## --- control surface -------------------------------------------------------------
## Exposed so the on-screen controls do not have to know about the Pace enum or reach
## into playback state directly.

func is_pps() -> bool:
	return pace == Pace.PPS

func toggle_pace() -> void:
	pace = Pace.REAL_TIME if pace == Pace.PPS else Pace.PPS
	_refresh_controls()

func set_playing(v: bool) -> void:
	playing = v
	_accum = 0.0
	_refresh_controls()

## Min, max and step for whichever pacing the viewer is currently using.
func speed_range() -> Vector3:
	return Vector3(PPS_MIN, PPS_MAX, 0.25) if is_pps() 		else Vector3(SPEED_MIN, SPEED_MAX, 0.05)

func speed_value() -> float:
	return pps if is_pps() else real_time_speed

func set_speed_value(v: float) -> void:
	if is_pps():
		pps = v
	else:
		real_time_speed = v
	_refresh_controls()

func speed_text() -> String:
	return "%.2f pps" % pps if is_pps() else "%.2fx real time" % real_time_speed

## The pace the player was actually going at this point, for comparison.
func their_pps() -> float:
	if steps.is_empty() or index + 1 >= steps.size():
		return 0.0
	var gap := int(steps[index + 1].get("frame", 0)) - int(steps[index].get("frame", 0))
	return Game.TICKS_PER_SECOND / float(gap) if gap > 0 else 0.0

func seek_to(i: int) -> void:
	playing = false
	_seek(i)

func step_run(by: int) -> void:
	_open_run(run_index + by)

## Entries for the run dropdown: player, mods and the replay id, which is what you
## need to vet a specific reconstruction against the source replay.
func run_labels() -> Array:
	var out: Array = []
	if log == null:
		return out
	for n in log.names:
		var m: Dictionary = log.meta.get(n, {})
		var id := n
		var us := n.rfind("_")
		if us >= 0 and us + 1 < n.length():
			id = n.substr(us + 1)
		out.append("%s  %s  %s" % [String(m.get("user", "?")), String(m.get("mods", "?")), id])
	return out

func run_count() -> int:
	return log.run_count() if log != null else 0

## Opens a run by its position in run_labels(), for the dropdown.
func open_run_index(i: int) -> void:
	playing = false
	_open_run(i)

func run_title() -> String:
	if log == null or log.run_count() == 0:
		return "-"
	var n: String = log.names[run_index]
	var m: Dictionary = log.meta.get(n, {})
	return "%s  %s" % [String(m.get("user", "?")), String(m.get("mods", "?"))]

func _refresh_controls() -> void:
	if controls != null and controls.has_method("refresh"):
		controls.refresh()

## Classifies every sample for the coloured scrubber.
##
## The board is judged by whether it RECOVERS, not by how it looks at any instant.
## Strong players bury a cell on purpose and reopen it a few placements later, so a
## hole on its own means nothing -- across these runs the largest such episode still
## peaks at a median of 8 cells and always clears. What a lost garbage row produces is
## an episode that never clears, because the rows the player is digging toward are not
## on our board; those peak at a median of 41 and run to the topout.
##
## So the cut is found by walking back from the end to the last clean board. No count,
## height or deadline threshold is involved.
##
## This is an OPTIMISTIC bound. It finds where the damage became visible, not where it
## began: garbage could have landed earlier and left our board merely wrong rather than
## holed. It is the latest point that can be defended, not the earliest that is safe.
func _build_marks() -> void:
	_marks = PackedByteArray()
	_marks.resize(steps.size())
	_garbage_at = PackedInt32Array()
	_desync = -1
	if steps.is_empty():
		return
	# garbage_rows is cumulative, so it steps up exactly where garbage entered.
	var seen := 0
	for j in steps.size():
		var rows := int(steps[j].get("garbage_rows", 0))
		if rows > seen:
			_garbage_at.append(j)
			seen = rows
	var i := steps.size() - 1
	while i >= 0 and int(steps[i].get("sealed", 0)) > 0:
		i -= 1
	# i is the last clean board, so the unrecovered episode opens at i + 1.
	if i + 1 < steps.size():
		_desync = i + 1
	for j in steps.size():
		if _desync >= 0 and j >= _desync:
			_marks[j] = 2
		elif int(steps[j].get("sealed", 0)) > 0:
			_marks[j] = 1
		elif bool(steps[j].get("recovered", false)):
			# Clean, but only reachable because inferred garbage was replayed here.
			_marks[j] = 3
		else:
			_marks[j] = 0

func timeline_marks() -> PackedByteArray:
	return _marks

## Where inferred garbage entered, for the timeline's markers.
func garbage_marks() -> PackedInt32Array:
	return _garbage_at

## First placement at which recorded garbage landed, or -1 if it never did.
func desync_index() -> int:
	return _desync

## The piece on the board right now -- the one being placed, which after a hold swap
## is NOT the piece that spawned.
func current_piece_def() -> PieceDef:
	return game.active.def if game != null and game.active != null else null

## One line explaining where the placing piece came from, or "" when it just spawned.
func placing_note() -> String:
	if steps.is_empty():
		return ""
	var step: Dictionary = steps[index]
	if not bool(step.get("used_hold", false)):
		return ""
	var spawned := String(step.get("piece", "")).to_upper()
	# An empty hold slot swallows the spawned piece and serves the next one up.
	if step.get("hold", null) == null:
		return "%s spawned, held" % spawned
	return "from hold (%s held)" % spawned

## Rows for the ReplayInfo panel: [label, value] or [label, value, colour].
## An empty label is a spacer.
func info_rows() -> Array:
	if log == null:
		return [["status", "loading", Color(1.0, 0.85, 0.35)]]
	if steps.is_empty():
		return [["status", "no samples", Color(1.0, 0.55, 0.45)],
			["path", dataset_path.get_file()],
			["error", log.error if log.error != "" else "-"]]

	var step: Dictionary = steps[index]
	var run_name: String = log.names[run_index]
	var m: Dictionary = log.meta.get(run_name, {})
	var tier := String(step.get("tier", "?"))
	var pace_text := "%.2f pps" % pps if pace == Pace.PPS else "%.2fx real" % real_time_speed

	# What the player was actually doing at this point, for comparison with the pace.
	var actual := "-"
	if index + 1 < steps.size():
		var gap := int(steps[index + 1].get("frame", 0)) - int(step.get("frame", 0))
		if gap > 0:
			actual = "%.1f pps" % (Game.TICKS_PER_SECOND / float(gap))

	var placed: Dictionary = step.get("place", {})
	var piece_text := String(placed.get("type", "?")).to_upper()
	if bool(step.get("used_hold", false)):
		piece_text += "   (spawned %s)" % String(step.get("piece", "?")).to_upper()

	var rows: Array = [
		["player", String(m.get("user", "?"))],
		["mods", String(m.get("mods", "?"))],
		["run", "%d / %d" % [run_index + 1, log.run_count()]],
		["", ""],
		["placement", "%d / %d" % [index + 1, steps.size()]],
		["frame", str(int(step.get("frame", 0)))],
		["lines", str(_lines_so_far)],
		["tier", tier,
			Color(0.65, 0.85, 0.65) if tier == "strict" else Color(1.0, 0.85, 0.35)],
		# The evidence the cut was made on, so it can be judged rather than trusted.
		# Sealed holes are the signal that matters: deliberate upstacking is tall and
		# clean, a diverged reconstruction is hole-rich.
		["holes", str(int(step.get("sealed", 0))),
			Color(0.65, 0.85, 0.65) if int(step.get("sealed", 0)) == 0
			else Color(1.0, 0.55, 0.45)],
		["height", str(int(step.get("height", 0)))],
		# Garbage rows the recovery search inferred and replayed up to this point.
		# Zero means this stretch needed no reconstruction.
		["garbage", "%d rows%s" % [int(step.get("garbage_rows", 0)),
			"  (recovered)" if bool(step.get("recovered", false)) else ""],
			Color(0.35, 0.80, 0.85) if bool(step.get("recovered", false))
			else Color(0.55, 0.58, 0.66)],
		# Consecutive placements a sealed hole has survived. A hole covered on purpose
		# and reopened resets this; one that stays does not. Watching it climb is how
		# you see the reconstruction lose sync.
		["hole run", str(int(step.get("hole_run", 0))),
			Color(0.65, 0.85, 0.65) if int(step.get("hole_run", 0)) < 5
			else Color(1.0, 0.55, 0.45)],
		# Recorded incoming garbage still unpaid. Rises with each packet, falls by the
		# attack this placement produced; surplus attack goes up the tower rather than
		# being banked against later garbage.
		#
		# What matters is not the number but whether it still reaches ZERO. A strong
		# player drains this hundreds of times a run -- 382 times in 493 placements for
		# 5han. When it stops draining they have been overwhelmed, and that moment
		# agrees with the board's own verdict (see _build_marks) at r = 0.989 across
		# the 50 harvested runs, median two placements apart. Two signals with no
		# shared inputs, so each corroborates the other.
		["gb queue", "%d%s" % [int(step.get("q_depth", 0)),
			"   never drains again" if int(step.get("q_last_zero", -1)) >= 0
				and index > int(step.get("q_last_zero", -1)) else ""],
			Color(1.0, 0.35, 0.35) if int(step.get("q_last_zero", -1)) >= 0
				and index > int(step.get("q_last_zero", -1))
			else (Color(1.0, 0.85, 0.35) if int(step.get("q_depth", 0)) > 0
			else Color(0.65, 0.85, 0.65))],
		["", ""],
		["pace", pace_text, Color(1.0, 0.85, 0.35)],
		["their speed", actual],
		["state", "playing" if playing else "paused"],
		["", ""],
		["placing", piece_text],
	]
	if int(step.get("cleared", 0)) > 0:
		var res := _result_for(step)
		rows.append(["clear", res.describe() if res != null else "-",
			Color(0.65, 0.85, 0.65)])
	if _load_ms > 0.0:
		rows.append(["", ""])
		rows.append(["dataset", "%d samples, %.0f ms" % [_sample_total, _load_ms]])
	return rows
