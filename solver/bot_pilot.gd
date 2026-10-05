class_name BotPilot
extends RefCounted

## Lets the Rust bot in `bot/` pilot the Tetris board.
##
## Deliberately shaped like `BeamSolver`: one `solve()` with the same arguments returning the
## same `Array[Placement]`. `main._solve_worker` can then call either without knowing which,
## and `_stage_placement` needs no change at all -- it already applies `rot`, `pos`,
## `via_rotation` and `kick_index` from whatever is in `hints.plan`.
##
## Why an external process rather than a GDScript port: the Godot solver is the quad-oriented
## hint search, and `solver/enumerator.gd` cannot even reach tucks or kicked spins, so it
## cannot play the all-spin game the minigame's economy is built around. The Rust search has
## spin-slot features and CEM-tuned weights, and costs 15ms of CPU per decision.
##
## The bot is held OPEN for the whole session -- `bot serve`, one request per line of stdin,
## one reply per line of stdout -- rather than spawned per decision. Spawning was 95ms of a
## 112ms round trip against a ~15ms search, which capped piloting at roughly 8 pieces per
## second; holding the process removes all of it, and the weights are parsed once instead of
## per piece.
##
## The cost is a lifecycle. The process is started on first use, restarted if it dies, and
## killed on shutdown; `main` owns that last part, because a backend outliving the game is the
## failure mode that leaves an orphan running after the window closes.
##
## Reads are BLOCKING and cannot be timed out -- FileAccess offers no such thing. A backend
## that hung rather than crashed would hang the solve thread with it, and the symptom would be
## that no new plan ever arrives. It has not happened; it is the one failure here with no
## guard.
##
## Neither engine trusts the other's coordinates: the bot returns the CELLS a piece should
## occupy, and PathFinder re-derives rotation and kick with Godot's own tables.

## Relative to the project root. Build output, so it is gitignored and may simply be absent.
const BINARY := "res://bot/target/release/bot.exe"
const BINARY_UNIX := "res://bot/target/release/bot"
var piece_set: PieceSet
## The rule mode handed to the bot. Must match the RuleSet the game is applying, or the bot
## searches for value the game will not pay -- see `main._pilot_rules_warning`.
var mode: String = "allspin"
var weights: String = "res://bot/weights/allspin_slots.txt"
## Mirrors BeamSolver's field so main can set it uniformly; the Rust search depth is fixed by
## its own beam config, so this is only carried, not used.
var depth: int = 6

var last_nodes: int = 0
var last_error: String = ""
## Placements the bot chose that Godot could not reach. Expected to stay 0 -- the measured
## rate over 2068 recorded spin placements is zero -- so any count here is a real finding
## rather than noise to tolerate.
var unreachable: int = 0

var _warned: bool = false
## The live backend: its stdio pipe and its process id, or null / -1.
var _io: FileAccess = null
var _pid: int = -1

## Absolute path to the binary, or "" when it has not been built.
static func binary_path() -> String:
	for p in [BINARY, BINARY_UNIX]:
		var abs := ProjectSettings.globalize_path(p)
		if FileAccess.file_exists(abs):
			return abs
	return ""

func available() -> bool:
	return binary_path() != ""

## Same signature as BeamSolver.solve, so either can be dropped into the worker.
##
## Returns [] on any failure, having set `last_error`. An empty plan makes autoplay wait
## rather than place something arbitrary, which is the right failure: a wrong placement is
## much harder to notice than a stalled one.
func solve(board: Board, queue: Array, hold: PieceDef, _progress: Callable, b2b: int) -> Array:
	if queue.is_empty() or piece_set == null:
		return []
	if not _ensure_started():
		return []
	# One line out, one line back. Blocking, and that is the point: this runs on main's solve
	# thread, so the wait costs a later plan rather than a dropped frame.
	_io.store_line(_state_json(board, queue, hold, b2b))
	_io.flush()
	var line := _io.get_line()
	if line.is_empty():
		# EOF: the backend died mid-session. Dropped rather than retried here, so one bad
		# state cannot become an infinite restart loop; the next solve starts it again.
		_stop()
		_fail("bot backend stopped -- it will be restarted on the next plan")
		return []
	var parsed: Variant = JSON.parse_string(line)
	if typeof(parsed) != TYPE_DICTIONARY:
		_fail("bot reply was not JSON: %s" % line.left(160))
		return []
	# The bot reports its own failures as JSON rather than dying, so a bad handover says what
	# was wrong instead of only that the pipe closed.
	if parsed.has("error"):
		_fail("bot rejected the state: %s" % parsed["error"])
		return []
	if not parsed.has("plan"):
		_fail("bot reply had no plan: %s" % line.left(160))
		return []
	return _to_plan(board, parsed)

# ------------------------------------------------------------------------------
# The backend's life
# ------------------------------------------------------------------------------

func running() -> bool:
	return _io != null and _pid > 0 and OS.is_process_running(_pid)

## Starts the backend if it is not up. Returns false having set `last_error` if it cannot be.
func _ensure_started() -> bool:
	if running():
		return true
	_stop()
	var exe := binary_path()
	if exe == "":
		_fail("bot binary not built -- run `cargo build --release` in bot/")
		return false
	var started: Dictionary = OS.execute_with_pipe(exe, [
		"serve", "--mode", mode, "--weights", ProjectSettings.globalize_path(weights),
	])
	if started.is_empty() or not started.has("stdio"):
		_fail("could not start the bot backend")
		return false
	_io = started["stdio"]
	_pid = int(started.get("pid", -1))
	# Waited for deliberately. The backend prints this only after loading its weights, so a
	# bad --weights path reports itself here instead of looking like a hang on the first plan.
	var hello := _io.get_line()
	if not hello.contains("ready"):
		_fail("bot backend did not come up: %s" % hello.left(160))
		_stop()
		return false
	_warned = false
	last_error = ""
	return true

## Closes the backend down. Safe to call when nothing is running.
##
## `quit` first so it can exit on its own and flush anything it owns; the kill is only the
## backstop for a process that has stopped reading its own stdin.
func shutdown() -> void:
	_stop()

func _stop() -> void:
	if _io != null:
		if _io.is_open():
			_io.store_line("quit")
			_io.flush()
			_io.close()
		_io = null
	if _pid > 0 and OS.is_process_running(_pid):
		OS.kill(_pid)
	_pid = -1

# ------------------------------------------------------------------------------
# The wire
# ------------------------------------------------------------------------------

## The live state, as one line of the backend's stdin.
##
## Only the occupied tail of the board is sent; the Rust side bottom-aligns what it gets. The
## wound arrays go with it because `--board` alone cannot express them, and under all-spin
## rules they are real: `RuleSet.allspin()` turns `repeat_penalty` on, so the board grows
## unclearable rows, and a bot that thought it could clear them would plan into a wall.
func _state_json(board: Board, queue: Array, hold: PieceDef, b2b: int) -> String:
	var top := 0
	while top < board.total_height and board.row_is_empty(top):
		top += 1
	var rows: Array = []
	var wound_left: Array = []
	var wound_hole: Array = []
	for y in range(top, board.total_height):
		var line := ""
		for x in board.width:
			line += "#" if board.is_blocked(x, y) else "."
		rows.append(line)
		wound_left.append(board.wound_left[y])
		wound_hole.append(board.wound_hole[y])
	var letters := ""
	for d in queue:
		letters += String(d.id).to_lower()
	return JSON.stringify({
		"board": rows,
		"queue": letters,
		"hold": null if hold == null else String(hold.id).to_lower(),
		"b2b": b2b,
		"wound_left": wound_left,
		"wound_hole": wound_hole,
	})

## The bot's plan as Placements.
##
## Step 0 is path-found into a real placement, because autoplay will apply it. The later steps
## are drawn only -- `hint_view` reads `cells` and `def`, and truncates the plan at the first
## clear anyway, since every step after one is computed on a board whose rows have moved. So
## they are built directly from the cells rather than path-found, which also avoids failing on
## a forecast that is not a legal resting position on TODAY's board.
func _to_plan(board: Board, parsed: Dictionary) -> Array:
	last_nodes = int(parsed.get("nodes", 0))
	var steps: Array = parsed["plan"]
	var out: Array = []
	for i in steps.size():
		var step: Dictionary = steps[i]
		var def: PieceDef = piece_set.get_def(StringName(String(step["type"]).to_upper()))
		if def == null:
			continue
		var want := PathFinder.cells_from(step["cells"])
		if i == 0:
			var p := PathFinder.find(board, def, want, _rules_for_spin())
			if p == null:
				unreachable += 1
				_fail("Godot could not reach the bot's placement (%s %s)" % [def.id, step["cells"]])
				return []
			p.uses_hold = bool(step.get("hold", false))
			out.append(p)
		else:
			out.append(_sketch(def, step, want))
	return out

## A draw-only placement: enough for the hint overlay, and honest about being no more.
func _sketch(def: PieceDef, step: Dictionary, want: Dictionary) -> Placement:
	var p := Placement.new()
	p.def = def
	p.cells = want.keys()
	p.uses_hold = bool(step.get("hold", false))
	# hint_view truncates at the first clearing step, so this has to be right even though
	# nothing else about a forecast step is used.
	p.lines = _lines_in(String(step.get("clear", "")))
	return p

static func _lines_in(clear_name: String) -> int:
	var n := clear_name.to_lower()
	if n.contains("quad") or n.contains("tetris"):
		return 4
	if n.contains("triple"):
		return 3
	if n.contains("double"):
		return 2
	if n.contains("single"):
		return 1
	return 0

## Spin detection needs a RuleSet. The mode the bot was told to use is the authority here, so
## the spin the pilot records is the one the bot was paid for.
func _rules_for_spin() -> RuleSet:
	return RuleSet.allspin() if mode == "allspin" else RuleSet.normal()

func _fail(msg: String) -> void:
	last_error = msg
	if not _warned:
		_warned = true
		push_warning("BotPilot: %s" % msg)
