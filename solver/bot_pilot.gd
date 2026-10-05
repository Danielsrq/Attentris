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
## The bot's rule mode for each of main.RULE_PRESETS.
##
## The mode is DERIVED from the rules the game is applying, never chosen separately. That is
## the whole guard against the one failure here with no symptom of its own: a bot searching
## under rules the game does not apply keeps playing well and quietly stops being paid for it.
##
## The two name sets already share seven spellings; only `guideline` differs, because the bot
## calls that one `standard`.
const MODE_FOR_RULES := {
	"normal": "normal",
	"burst": "burst",
	"burst2": "burst2",
	"gate": "gate",
	"gate_b2b": "gate_b2b",
	"competitive": "competitive",
	"guideline": "standard",
	"allspin": "allspin",
	"allspin_nw": "allspin_nw",
}

## The weight sets that are ready to fly. WEIGHTS only -- the rules are the other axis.
##
## `tuned_for` is the rules preset each set was tuned against, which is where the bot toggle
## jumps the game when you load it. It is a starting point, not a constraint: the rules stay
## free to change afterwards, and the bot's belief follows them.
const PRESETS := [
	{
		"name": "allspin-slots", "tuned_for": "allspin",
		"weights": "res://bot/weights/allspin_slots.txt",
		"note": "strongest all-spin, with the spin-slot features",
	},
	{
		"name": "allspin", "tuned_for": "allspin",
		"weights": "res://bot/weights/allspin.txt",
		"note": "the earlier all-spin set, without the spin-slot features",
	},
	{
		"name": "gate4", "tuned_for": "gate_b2b",
		"weights": "res://bot/results/gate4/gate4.txt",
		"note": "tuned WITH garbage, so the one that should survive the battler",
	},
	{
		"name": "normal", "tuned_for": "normal",
		"weights": "res://bot/weights/normal.txt",
		"note": "normal rules, 1.26 APP held out",
	},
	{
		"name": "standard", "tuned_for": "guideline",
		"weights": "res://bot/weights/standard.txt",
		"note": "no all-spin: quad wells and I-dependency, a different game",
	},
]

var piece_set: PieceSet
## Which of PRESETS is loaded.
var preset_index: int = 0
## The rules the bot SEARCHES under. A property of the bot, not of the game.
##
## Defaults to the rules its weight set was tuned against, and stays there when the game's
## rules change -- because weights and reward model are one package. `allspin_slots.txt` was
## tuned under all-spin, so it believes all-spin even while the environment scores normal
## rules, and that mismatch is the intended configuration rather than a fault.
##
## It is also the difference between working and dying. Measured at 500 pieces over three
## seeds: believing all-spin gives 2.59 APP and no deaths, while believing normal rules with
## the same all-spin weights topped out every single run at around piece 300 -- weights that
## value spin setups against a reward that will not pay for them.
var believe: String = "allspin"
## The rule mode handed to the bot, and the weights it searches with. The backend reads both
## ONCE at startup, so changing either restarts it -- which is why they go through setters
## rather than being assigned.
var mode: String = "allspin"
var weights: String = "res://bot/weights/allspin_slots.txt"
## How many steps of the returned plan to keep.
##
## It does NOT shorten the search -- the Rust beam depth is fixed by the backend's own config
## -- but it still has to be honoured, because this is what hint mode draws. BeamSolver's
## depth used to bound the plan's length, so without this F6 would go from two outlines over
## the board to six.
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

func preset() -> Dictionary:
	return PRESETS[clampi(preset_index, 0, PRESETS.size() - 1)]

## Loads a bot. The backend is stopped rather than reconfigured, because it reads its weights
## once at startup; the next solve brings it back up with the new ones.
## Loads a weight set AND the belief it was tuned with, together. They are one package: a
## reward model the weights were fitted against is not separable from the weights.
func use_preset(i: int) -> Dictionary:
	preset_index = posmod(i, PRESETS.size())
	var p := preset()
	weights = p["weights"]
	believe = String(p["tuned_for"])
	mode = MODE_FOR_RULES.get(believe, "allspin")
	_stop()
	_warned = false
	last_error = ""
	unreachable = 0
	return p

## Overrides what the bot believes. "" restores the belief its weight set was tuned for.
##
## The deliberate-mismatch control, in both directions: point an all-spin bot at normal rules
## to make it honest about the environment, or point any bot at rules it was not tuned for to
## see what that costs. Named rather than a bool so the overlay can say WHICH belief is held,
## because a belief that does not match the weights has no symptom of its own -- the bot keeps
## playing well and is quietly paid for a different game.
func set_belief(rules_preset_name: String) -> void:
	if rules_preset_name == "":
		believe = String(preset()["tuned_for"])
		_set_mode(MODE_FOR_RULES.get(believe, ""))
		return
	var want: String = MODE_FOR_RULES.get(rules_preset_name, "")
	if want == "":
		return
	believe = rules_preset_name
	_set_mode(want)

func _set_mode(want: String) -> void:
	if want == "" or want == mode:
		return
	mode = want
	_stop()

func next_preset() -> Dictionary:
	return use_preset(preset_index + 1)

## Same signature as BeamSolver.solve, so either can be dropped into the worker.
##
## Returns [] on any failure, having set `last_error`. An empty plan makes autoplay wait
## rather than place something arbitrary, which is the right failure: a wrong placement is
## much harder to notice than a stalled one.
## `gate_state` is set by the caller before each solve, because it is game state the board
## cannot carry: Game owns it, and under gate rules the search is blind without it.
var gate_state: Array = []

func solve(board: Board, queue: Array, hold: PieceDef, _progress: Callable, b2b: int) -> Array:
	if queue.is_empty() or piece_set == null:
		return []
	if not _ensure_started():
		return []
	# One line out, one line back. Blocking, and that is the point: this runs on main's solve
	# thread, so the wait costs a later plan rather than a dropped frame.
	_io.store_line(_state_json(board, queue, hold, b2b, gate_state))
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
func _state_json(board: Board, queue: Array, hold: PieceDef, b2b: int,
		gate: Array = []) -> String:
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
		# [placements left, spin clears in the chain, the opener's value]. Under gate rules
		# attack comes only from a completed chain, so a bot that cannot see the chain it is
		# already in never finishes one.
		"gate": gate,
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
	for i in mini(steps.size(), maxi(1, depth)):
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
