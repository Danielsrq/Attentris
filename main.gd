@tool
extends Node2D

## Wires the pure simulation to input and the view, and ticks it at a fixed 60 Hz.
##
## Change CELL, BOARD_WIDTH or the piece set here; nothing downstream cares.
##
## This is a @tool script so the whole layout draws live in the editor viewport --
## see the "Editor preview" group in the inspector. The sim never ticks in the
## editor; it is built once, stuffed with representative state, and drawn.

const CELL := 30
const BOARD_WIDTH := 10
const BOARD_VISIBLE := 20
const BOARD_BUFFER := 20

## Starting control profile. See view/control_profiles.gd; F5 cycles at runtime.
const CONTROL_PROFILE := "wasd_numpad"

## Fixed so the editor preview looks the same every time the scene is opened.
const PREVIEW_SEED := 424242

var rules: RuleSet
var piece_set: PieceSet
var board: Board
var scorer: Scorer
var game: Game
var layout: ViewLayout

var reader := InputReader.new(CONTROL_PROFILE)
var input_state := InputState.new()
## Rule presets the handling toggle cycles through; "normal" is the default.
const RULE_PRESETS := ["normal", "burst", "burst2", "gate", "competitive", "guideline", "allspin"]
var rules_preset := 0

func _preset_rules() -> RuleSet:
	match RULE_PRESETS[rules_preset]:
		"normal": return RuleSet.normal()
		"guideline": return RuleSet.guideline()
		"allspin": return RuleSet.allspin()
		"burst": return RuleSet.burst()
		"burst2": return RuleSet.burst2()
		"gate": return RuleSet.gate_rules()
		_: return RuleSet.competitive()

## Right-pane page. GAMEPLAY is the default because the economy is the thing that needs
## watching now; the mechanics readout moved to being a page of the same pane rather
## than an overlay competing for it. TAB cycles, F1 blanks and restores.
var pane_tab: int = PaneTab.GAMEPLAY
var _pane_last: int = PaneTab.GAMEPLAY

## Assist mode. Search depth is tied to the number of steps shown, because a plan
## can only be as long as the search was deep.
## Beginner assist: a phantom column right of the field that builds the overhang a
## spin into the last column needs. Off by default -- it changes what is reachable.
var ghost_slot := GhostSlot.new()
const GHOST_COLUMNS := 1

var solver: BeamSolver
var hints_enabled := false
## Lookahead depth for the hint solver. Fixed: it was on F7/F8 and never touched,
## and a knob nobody turns is just a way for the two paths to drift apart.
var hint_steps := 2
var last_solve_ms := 0.0
const HINT_BEAM := 8

## Autoplay: the solver plays itself, rate-limited so it is watchable. Uncapped it
## is bounded only by solve time (~20 pps at depth 3), which is well past human.
var autoplay := false
## Placements per second. Mapped logarithmically from the slider, see pps_from_t().
var autoplay_pps := 1.0
const PPS_MIN := 0.25
const PPS_MAX := 60.0
## At or above this the interval is dropped entirely -- as fast as solves allow.
const PPS_UNCAPPED := 55.0
## Autoplay searches at the depth that measured best, independent of how many plan
## steps the hint overlay is showing.
const AUTOPLAY_DEPTH := 3
## Solve off the main thread. The sim is pure, but the game keeps mutating the board
## while a solve runs, so the worker gets a SNAPSHOT and the result is discarded if
## the board moved on while it was thinking.
var threaded_solve := true
var _solve_thread: Thread = null
var _solve_mutex := Mutex.new()
var _pending_plan: Array = []
var _plan_ready := false
var _solve_for_piece := -1
var _solve_started_usec := 0

var auto_pieces := 0
var auto_quads := 0
var auto_score_start := 0
var auto_attack_start := 0
var _auto_accum := 0.0
var _auto_staged := false

@export_group("Editor preview")
## Fills the editor viewport with a representative board, hold piece, next queue and
## placement readout, so layout and colour can be judged without pressing play.
@export var preview_in_editor: bool = true:
	set(value):
		preview_in_editor = value
		_refresh_editor_preview()
## Height of each column in the fake stack, cycled across the field's width.
@export var preview_stack: PackedInt32Array = PackedInt32Array([4, 6, 5, 8, 3, 7, 5, 9, 6, 4]):
	set(value):
		preview_stack = value
		_refresh_editor_preview()
## Tick this to rebuild after editing a view script. Immediately resets itself.
@export var preview_rebuild: bool = false:
	set(_value):
		preview_rebuild = false
		_refresh_editor_preview()

@onready var board_view: Node2D = $BoardView
@onready var hud: Node2D = $Hud
@onready var overlay: Node2D = $DebugOverlay
@onready var hints: Node2D = $HintView
@onready var audio: AudioBus = $Audio

func _ready() -> void:
	if not Engine.is_editor_hint():
		RenderingServer.set_default_clear_color(Color(0.04, 0.04, 0.06))
	_new_game(PREVIEW_SEED if Engine.is_editor_hint() else -1)

func _new_game(seed_v: int = -1) -> void:
	if seed_v < 0:
		seed_v = int(Time.get_ticks_usec()) & 0x7FFFFFFF
	rules = _preset_rules()
	piece_set = PieceSet.tetrominoes()
	board = Board.new(BOARD_WIDTH, BOARD_VISIBLE, BOARD_BUFFER)
	scorer = Scorer.new(ScoreTable.new())
	game = Game.new(rules, piece_set, board, scorer, seed_v)
	board.ghost_width = GHOST_COLUMNS
	board.ghost.resize(GHOST_COLUMNS * board.total_height)
	ghost_slot.update(board)
	layout = ViewLayout.new(board, get_viewport_rect().size, CELL)

	solver = BeamSolver.new(BoardEvaluator.new(EvalWeights.lookahead()))
	solver.beam_width = HINT_BEAM
	solver.depth = hint_steps

	board_view.setup(game, layout)
	board_view.drop_flashes()      # or a restart inherits the last run's outlines
	hints.setup(game, layout)
	hud.setup(game, layout)
	overlay.setup(game, layout, self)
	_set_pane_tab(pane_tab)        # setup() resets the HUD, so re-assert the page
	hints.enabled = hints_enabled
	auto_pieces = 0
	auto_quads = 0
	auto_score_start = 0
	auto_attack_start = 0
	_auto_staged = false
	_auto_accum = 0.0
	_dispatch(game.start())

	if Engine.is_editor_hint() and preview_in_editor:
		_build_preview()
	# After the preview, not before -- otherwise the plan is computed against an
	# empty board and then the preview fills it underneath.
	_resolve_hints()
	_redraw_views()

# ------------------------------------------------------------------------------
# Autoplay
# ------------------------------------------------------------------------------

## Two phases per piece so the result is readable at low rates: the piece is SNAPPED
## into its planned position immediately, sits there for the interval, then locks.
## Snapping and locking in the same frame would mean never seeing where it went.
##
## Deliberately bypasses Game.tick(): no gravity, no lock delay, no input. Planned
## placements are resting positions anyway, so gravity would be a no-op, but lock
## delay would fire at 0.5s and pre-empt any rate below 2 pps.
func _autoplay_step(delta: float) -> void:
	if game.topped_out or game.active == null:
		return
	if not _auto_staged:
		_stage_placement()
		_auto_accum = 0.0
		return
	_auto_accum += delta
	var interval := 0.0 if autoplay_pps >= PPS_UNCAPPED else 1.0 / autoplay_pps
	if _auto_accum < interval:
		return
	_auto_accum = 0.0
	var events: Array = []
	game.lock_piece(events)
	_dispatch(events)
	_auto_staged = false
	hints.set_plan([])      # the plan described the board before this lock

func _stage_placement() -> void:
	# With a threaded solve the plan arrives a few frames after the spawn, so this
	# waits for it rather than forcing a synchronous solve.
	if hints.plan.is_empty():
		if _solve_thread == null:
			_resolve_hints()
		return
	var plan: Array = hints.plan
	if plan.is_empty():
		return
	var p: Placement = plan[0]
	if p.uses_hold:
		var inp := InputState.new()
		inp.hold = true
		_dispatch(game.tick(inp))
		if game.active == null or game.active.def != p.def:
			return
	elif game.active.def != p.def:
		return
	if not game.board.can_place(p.def.rotations[p.rot], p.pos):
		return
	game.active.rot = p.rot
	game.active.pos = p.pos
	game.active.last_action_was_rotation = p.via_rotation
	game.active.last_kick_index = p.kick_index
	auto_pieces += 1
	if p.lines >= 4:
		auto_quads += 1
	_auto_staged = true

## Logarithmic, so the slider has useful resolution at both 0.5 pps and 40 pps.
static func pps_from_t(t: float) -> float:
	return PPS_MIN * pow(PPS_MAX / PPS_MIN, clampf(t, 0.0, 1.0))

static func t_from_pps(pps: float) -> float:
	return log(pps / PPS_MIN) / log(PPS_MAX / PPS_MIN)

func set_autoplay_t(t: float) -> void:
	autoplay_pps = pps_from_t(t)

func _physics_process(_delta: float) -> void:
	if Engine.is_editor_hint():
		return
	reader.poll()
	_collect_solve()
	_handle_meta_keys()
	if autoplay:
		_autoplay_step(_delta)
	else:
		reader.to_input_state(input_state)
		_dispatch(game.tick(input_state))
	if _needs_resolve:
		_resolve_hints()
	_redraw_views()

func _redraw_views() -> void:
	board_view.queue_redraw()
	hints.queue_redraw()
	hud.queue_redraw()
	overlay.queue_redraw()

var _needs_resolve := false

## Runs once per spawn, not per frame. Currently synchronous, so a deep plan costs a
## visible hitch at spawn -- the overlay reports the cost so it is not a mystery.
func _resolve_hints() -> void:
	_needs_resolve = false
	# Autoplay needs the plan even when the hint overlay is hidden.
	if not (hints_enabled or autoplay) or game == null or game.active == null or game.topped_out:
		hints.set_plan([])
		last_solve_ms = 0.0
		return
	solver.depth = AUTOPLAY_DEPTH if autoplay else hint_steps
	if threaded_solve and not Engine.is_editor_hint():
		_request_threaded_solve()
		return
	var queue: Array = [game.active.def]
	queue.append_array(game.next_queue)
	var t0 := Time.get_ticks_usec()
	var plan := solver.solve(game.board, queue, game.hold_def, Callable(), game.b2b)
	last_solve_ms = float(Time.get_ticks_usec() - t0) / 1000.0
	hints.set_plan(plan)

## Hand the worker a board it alone owns. Anything shared would race, because the
## game keeps locking pieces while the solve runs.
func _request_threaded_solve() -> void:
	if _solve_thread != null:
		return                      # one in flight; the next spawn re-requests
	var snapshot := game.board.duplicate_board()
	var queue: Array = [game.active.def]
	queue.append_array(game.next_queue)
	_solve_for_piece = game.pieces_placed
	_solve_started_usec = Time.get_ticks_usec()
	_solve_thread = Thread.new()
	_solve_thread.start(_solve_worker.bind(snapshot, queue, game.hold_def, game.b2b))

func _solve_worker(board: Board, queue: Array, hold: PieceDef, b2b: int) -> void:
	var plan := solver.solve(board, queue, hold, Callable(), b2b)
	_solve_mutex.lock()
	_pending_plan = plan
	_plan_ready = true
	_solve_mutex.unlock()

## Collect a finished solve. Called every tick; cheap when nothing is running.
func _collect_solve() -> void:
	if _solve_thread == null or _solve_thread.is_alive():
		return
	_solve_thread.wait_to_finish()
	_solve_thread = null
	_solve_mutex.lock()
	var plan: Array = _pending_plan
	var ready := _plan_ready
	_plan_ready = false
	_solve_mutex.unlock()
	last_solve_ms = float(Time.get_ticks_usec() - _solve_started_usec) / 1000.0
	if not ready:
		return
	# Stale: a piece locked while the worker was thinking, so the plan describes a
	# board that no longer exists.
	if game == null or game.pieces_placed != _solve_for_piece:
		_needs_resolve = true
		return
	hints.set_plan(plan)

func _exit_tree() -> void:
	if _solve_thread != null:
		_solve_thread.wait_to_finish()
		_solve_thread = null

func _handle_meta_keys() -> void:
	if reader.pressed("restart"):
		_new_game()
		return
	if reader.pressed("cycle_pane"):
		_set_pane_tab(PaneTab.next(pane_tab))
	if reader.pressed("toggle_debug"):
		# F1 keeps its old meaning -- blank the pane and bring it back -- which is now
		# just a shortcut to OFF and back to whichever page was up.
		_set_pane_tab(_pane_last if pane_tab == PaneTab.OFF else PaneTab.OFF)
	if reader.pressed("toggle_grid"):
		board_view.show_grid = not board_view.show_grid
	if reader.pressed("cycle_profile"):
		reader.next_profile()
	if reader.pressed("toggle_autoplay"):
		autoplay = not autoplay
		_auto_staged = false
		_auto_accum = 0.0
		if autoplay:
			# Show the plan while it plays -- watching without it is watching noise.
			hints_enabled = true
			hints.enabled = true
			auto_score_start = game.score
			auto_attack_start = game.attack_sent
	if reader.pressed("pps_down"):
		set_autoplay_t(t_from_pps(autoplay_pps) - 0.08)
	if reader.pressed("pps_up"):
		set_autoplay_t(t_from_pps(autoplay_pps) + 0.08)
	if reader.pressed("toggle_ghost"):
		ghost_slot.enabled = not ghost_slot.enabled
		ghost_slot.update(board)
	if reader.pressed("toggle_hints"):
		hints_enabled = not hints_enabled
		hints.enabled = hints_enabled
		_resolve_hints()
	if reader.pressed("toggle_gravity"):
		# Gravity off, soft drop still live: the easiest way to walk a piece into a
		# slot and check a kick by hand.
		game.gravity_enabled = not game.gravity_enabled
	if reader.pressed("toggle_handling"):
		rules_preset = (rules_preset + 1) % RULE_PRESETS.size()
		# Swap the ruleset in place so the run in progress is preserved.
		var keep_gravity := game.gravity_enabled
		game.rules = _preset_rules()
		rules = game.rules
		game.gravity_enabled = keep_gravity

## Single point of truth for the pane: the overlay is one PAGE of it, so its visibility
## is derived here rather than toggled independently.
func _set_pane_tab(tab: int) -> void:
	if tab != PaneTab.OFF:
		_pane_last = tab
	pane_tab = tab
	hud.pane_tab = tab
	overlay.visible_overlay = tab == PaneTab.DEBUG
	hud.queue_redraw()
	overlay.queue_redraw()

func _dispatch(events: Array) -> void:
	# This is the seam the right-hand minigame will hook into: it consumes the same
	# event stream rather than reaching into the sim.
	for e in events:
		overlay.log_event(e)
		if e.type == GameEvent.Type.PIECE_LOCKED:
			hud.on_placement(e.data["result"], e.data["points"])
			# The stack only changes on a lock, so that is the only time the assist
			# needs rebuilding. After a double clears two rows it re-forms two rows
			# lower, because the pattern is derived from the surface each time.
			ghost_slot.update(board)
		# Re-plan whenever a new piece is in play. A hold also spawns, so swapping
		# re-plans for free; ordinary movement does not invalidate a plan, because
		# the plan is about final positions rather than the route to them.
		# Clears move every row above them, which would leave a structure outline
		# pointing at the wrong cells. Dropping them here is safe: a structure
		# completed BY this clear is dispatched after it and re-flashes.
		if e.type == GameEvent.Type.LINES_CLEARED:
			board_view.drop_flashes()
		if e.type == GameEvent.Type.LINES_CLEARED:
			var r: ClearResult = e.data["result"]
			hud.log_economy(game.pieces_placed, r.describe().to_upper(),
				{r.resource_group(): r.resource_amount()})
		if e.type == GameEvent.Type.STRUCTURE_BUILT:
			board_view.flash_structure(e.data["rect"])
			hud.on_structure(e.data["payout"])
			hud.log_economy(game.pieces_placed, "STRUCTURE", e.data["payout"])
		if e.type == GameEvent.Type.PIECE_SPAWNED:
			_needs_resolve = true
		# The editor preview pushes synthetic events through here; it must stay silent.
		if not Engine.is_editor_hint():
			audio.handle_event(e)

func binding_lines() -> Array:
	return reader.binding_text()

func control_profile_name() -> String:
	return reader.profile_name

func held_actions() -> Array:
	return reader.held_actions()

## Mouse drag on the rate slider the overlay draws.
func _unhandled_input(event: InputEvent) -> void:
	if Engine.is_editor_hint() or overlay == null:
		return
	var r: Rect2 = overlay.slider_rect
	if r.size.x <= 0.0:
		return
	var pos := Vector2.ZERO
	if event is InputEventMouseButton:
		if event.button_index != MOUSE_BUTTON_LEFT or not event.pressed:
			return
		pos = event.position
	elif event is InputEventMouseMotion:
		if (event.button_mask & MOUSE_BUTTON_MASK_LEFT) == 0:
			return
		pos = event.position
	else:
		return
	if not r.grow(6.0).has_point(pos):
		return
	set_autoplay_t((pos.x - r.position.x) / r.size.x)
	get_viewport().set_input_as_handled()

## Autoplay state for the debug overlay.
func autoplay_status() -> Dictionary:
	var eff := float(auto_quads) / maxf(1.0, float(auto_pieces))
	return {
		"on": autoplay,
		"pps": autoplay_pps,
		"uncapped": autoplay_pps >= PPS_UNCAPPED,
		"t": t_from_pps(autoplay_pps),
		"pieces": auto_pieces,
		"quads": auto_quads,
		"quad_eff": eff,
		"score_per_piece": float(game.score - auto_score_start) / maxf(1.0, float(auto_pieces)),
		"attack": game.attack_sent - auto_attack_start,
		"attack_per_piece": float(game.attack_sent - auto_attack_start) / maxf(1.0, float(auto_pieces)),
	}

## Assist state for the debug overlay.
func hint_status() -> Dictionary:
	return {
		"enabled": hints_enabled,
		"steps": hint_steps,
		"depth": solver.depth if solver != null else 0,
		"beam": solver.beam_width if solver != null else 0,
		"solve_ms": last_solve_ms,
		"plan_len": hints.plan.size() if hints != null else 0,
		"shown": hints.visible_steps().size() if hints != null else 0,
		"truncated": hints.truncated() if hints != null else false,
	}

# ------------------------------------------------------------------------------
# Editor preview
# ------------------------------------------------------------------------------

func _refresh_editor_preview() -> void:
	if not Engine.is_editor_hint() or not is_node_ready():
		return
	_new_game(PREVIEW_SEED)

## Representative mid-game state. Every widget that can hold something holds
## something, so an empty one in the viewport means a real bug rather than an
## empty game. The numbers go through the real Scorer, not hand-written totals.
func _build_preview() -> void:
	var rng := RandomNumberGenerator.new()
	rng.seed = PREVIEW_SEED
	if not preview_stack.is_empty():
		for x in board.width:
			var h: int = preview_stack[x % preview_stack.size()]
			for i in h:
				board.set_cell(x, board.total_height - 1 - i, rng.randi_range(1, piece_set.size()))

	game.hold_def = piece_set.get_def(&"I")
	game.lines = 34
	game.level = 4
	game.pieces_placed = 91
	game.b2b = 2
	game.combo = 3

	# Widest case for the left column: a spin name, a count, points and two tags.
	var samples := [
		_preview_result(4, SpinDetect.Kind.NONE, &"I", 0, 0),
		_preview_result(1, SpinDetect.Kind.MINI, &"T", 1, 1),
		_preview_result(3, SpinDetect.Kind.FULL, &"T", 2, 2),
		_preview_result(2, SpinDetect.Kind.FULL, &"T", 2, 3),
	]
	game.score = 0
	for r in samples:
		var points: int = scorer.score_clear(r)
		game.score += points
		game.last_result = r
		game.last_points = points
		_dispatch([GameEvent.new(GameEvent.Type.PIECE_LOCKED, {
			"piece": r.piece_id, "spin": r.spin, "kick": 4,
			"cells": [], "result": r, "points": points,
		})])

func _preview_result(lines: int, spin: int, piece: StringName, b2b: int, combo: int) -> ClearResult:
	var r := ClearResult.new()
	r.lines = lines
	r.spin = spin
	r.piece_id = piece
	r.b2b = b2b
	r.combo = combo
	r.level = 4
	return r
