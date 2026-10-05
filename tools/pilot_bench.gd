extends SceneTree

## Runs the real game with the Rust bot flying it, headless, and reports what it did.
##
##   godot --headless --path . --script res://tools/pilot_bench.gd -- --pieces 300 --seed 7
##
## The point is to compare the pilot against the bot's OWN benchmarks. `bot play` reports
## 2.6-2.7 attack per piece and no deaths under all-spin rules; if this reports much less,
## something is being lost in the handover -- the rules pairing, the B2B chain, or placements
## Godot cannot reach -- and all three are silent failures that a suite cannot see, because
## they make the bot play worse rather than make anything error.
##
## It also reports the battler, since a bot that spins ~40 times per 100 pieces floods the
## lane and is the fastest way to see whether the minigame's spawn economy survives it.

var _main: Node
var _pieces := 300
var _frames := 0
var _max_frames := 120000
var _seed := 12345
var _bot := -1
var _restarted := false

var _args: Dictionary = {}

func _initialize() -> void:
	var args := _flags()
	_args = args
	_pieces = int(args.get("pieces", "300"))
	var seed_v := int(args.get("seed", "12345"))

	_main = load("res://main.tscn").instantiate()
	root.add_child(_main)
	# Synchronous solving: the worker thread is the right call in the game and only makes a
	# benchmark's piece count depend on how fast the machine is.
	_main.threaded_solve = false
	_main.hints_enabled = false
	var preset := String(args.get("rules", "allspin"))
	var idx: int = _main.RULE_PRESETS.find(preset)
	if idx < 0:
		push_error("unknown rules preset %s" % preset)
		quit(1)
		return
	_main.rules_preset = idx
	# --bot NAME picks one of BotPilot.PRESETS and brings the game's rules with it, so a run
	# cannot accidentally measure a bot under rules it was not tuned for.
	var want := String(args.get("bot", ""))
	if want != "":
		var found := -1
		for i in BotPilot.PRESETS.size():
			if String(BotPilot.PRESETS[i]["name"]) == want:
				found = i
		if found < 0:
			var names: Array = []
			for pr in BotPilot.PRESETS:
				names.append(pr["name"])
			push_error("unknown --bot %s (have %s)" % [want, ", ".join(names)])
			quit(1)
			return
		_bot = found
	_main.autoplay = true
	_main.autoplay_pps = _main.PPS_UNCAPPED
	# NOT restarted here. @onready vars are assigned by _ready, and _ready is deferred when
	# add_child runs during _initialize, so every view node is still null at this point --
	# calling _new_game now dies on board_view.setup(). The restart happens on the first
	# frame instead, by which time _ready has run.
	_seed = seed_v

	# The STATIC check: _main.pilot does not exist yet, because _ready is deferred when
	# add_child runs during _initialize.
	if not BotPilot.binary_path().is_empty():
		print("piloting %d pieces under '%s' rules" % [_pieces, preset])
	else:
		print("bot not built -- run: cargo build --release  (in bot/)")
		quit(1)

func _flags() -> Dictionary:
	var out := {}
	var argv := OS.get_cmdline_user_args()
	var i := 0
	while i < argv.size():
		var k := String(argv[i]).trim_prefix("--")
		if i + 1 < argv.size() and not String(argv[i + 1]).begins_with("--"):
			out[k] = argv[i + 1]
			i += 2
		else:
			out[k] = "1"
			i += 1
	return out

func _process(_delta: float) -> bool:
	if _main == null:
		return true
	_frames += 1
	if not _restarted:
		# Now that _ready has run, restart on the chosen seed under the chosen preset.
		_restarted = true
		# After _ready, so the pilot exists; the preset carries its own rules with it.
		if _bot >= 0:
			var p: Dictionary = _main.pilot.use_preset(_bot)
			# --rules, when given, wins: measuring a weight set under rules it was not tuned
			# for is a deliberate experiment, and the whole reason the axes are separate.
			if not _args.has("rules"):
				var r: int = _main.RULE_PRESETS.find(String(p["tuned_for"]))
				if r >= 0:
					_main.rules_preset = r
			print("weights '%s' under '%s' rules -- %s" % [
				p["name"], _main.RULE_PRESETS[_main.rules_preset], p["note"]])
		_main._new_game(_seed)
		var trouble: String = _main.pilot_status()
		if trouble != "":
			print("pairing: %s" % trouble)
		return false
	var game = _main.game
	if game == null:
		return false
	if game.topped_out or _main.auto_pieces >= _pieces or _frames > _max_frames:
		_report(game)
		return true
	return false

func _report(game) -> void:
	var placed: int = _main.auto_pieces
	var app := float(game.attack_sent) / float(maxi(1, placed))
	print("pieces %d   attack %d   APP %.3f   quads %d   died %s" % [
		placed, game.attack_sent, app, _main.auto_quads, game.topped_out])
	print("   b2b %d   level %d   wounds %d   unreachable placements %d" % [
		game.b2b, game.level, game.board.wound_count(), _main.pilot.unreachable])
	var arena = _main.battler
	if arena != null and arena.field != null:
		# SENT is the number that matters for the economy; alive is just the snapshot.
		print("   battler: sent %d teal / %d red  (%.2f teal per piece)  alive %d/%d  queued %d" % [
			arena.teal_sent, arena.red_sent,
			float(arena.teal_sent) / float(maxi(1, placed)),
			arena.alive_on(UnitDef.TEAL), arena.alive_on(UnitDef.RED), arena.pending()])
	if _main.pilot.last_error != "":
		print("   last pilot error: %s" % _main.pilot.last_error)
	# Freed explicitly: a tool that leaks its whole scene prints a wall of ObjectDB noise
	# after its own results, which makes a clean run look like a broken one.
	_main.queue_free()
	_main = null
