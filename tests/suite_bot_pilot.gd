extends TestHarness

## PathFinder and BotPilot, against real recorded bot output.
##
## The bot itself is not run here -- building it is a cargo step the suite must not depend on.
## What is tested is the part that can silently go wrong: turning the cells the bot names into
## a placement Godot can actually reach, with the spin credited. Both engines index row 0 at
## the top and both place a piece by a box offset, and nothing in either codebase enforces
## that; the records are the only thing that can.

const REPLAY := "res://data/bot/allspin_slots.jsonl"
## Normal rules, where a non-T spin is a MINI. The all-spin replay cannot exercise the
## Mini-versus-Full distinction at all, because all-spin promotes every Mini.
const REPLAY_NORMAL := "res://data/bot/normal.jsonl"

func run() -> void:
	test_path_finder_against_recorded_placements()
	test_best_spin_is_kept()
	test_pilot_wire_format()
	test_presets()
	test_allspin_without_wounds()
	test_pilot_degrades_without_a_binary()

## Reads the first `n` records that have a `place` with cells.
func _records(n: int, path: String = REPLAY) -> Array:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		return []
	var out: Array = []
	while out.size() < n and not f.eof_reached():
		var line := f.get_line().strip_edges()
		if line.is_empty():
			continue
		var v: Variant = JSON.parse_string(line)
		if typeof(v) == TYPE_DICTIONARY and v.has("place") and v.has("board"):
			out.append(v)
	return out

func _board_from(rows: Array) -> Board:
	var b := Board.new(10, 20, 20)
	for y in mini(rows.size(), b.total_height):
		var row: String = rows[y]
		for x in mini(row.length(), b.width):
			if row[x] != ".":
				b.set_cell(x, y, 1)
	return b

func test_path_finder_against_recorded_placements() -> void:
	group("PathFinder: the bot's cells resolve to placements Godot can reach")

	var recs := _records(120)
	check(not recs.is_empty(), "the recorded all-spin replay is readable")
	if recs.is_empty():
		return

	var ps := PieceSet.tetrominoes()
	var rules := RuleSet.allspin()
	var reached := 0
	var missed := 0
	var cells_wrong := 0
	var spin_disagrees := 0
	for v in recs:
		var place: Dictionary = v["place"]
		var def: PieceDef = ps.get_def(StringName(String(place["type"]).to_upper()))
		var b := _board_from(v["board"])
		var want := PathFinder.cells_from(place["cells"])
		var p := PathFinder.find(b, def, want, rules)
		if p == null:
			missed += 1
			continue
		reached += 1
		# The placement must stand on exactly the cells the bot named. This is the origin
		# convention -- the one thing a bare (x, y) handover would have got silently wrong.
		if PathFinder.cells_at(p.def, p.rot, p.pos) != want:
			cells_wrong += 1
		# And a recorded spin must still be a spin here, found with Godot's own kick tables.
		# Godot may credit one the record called none (a better path to the same cells), but
		# losing a recorded spin would mean the pilot silently plays for less attack.
		if String(place.get("spin", "none")) != "none" and p.spin == SpinDetect.Kind.NONE:
			spin_disagrees += 1

	eq(missed, 0, "every recorded placement is reachable under Godot's kick tables")
	eq(cells_wrong, 0, "   and lands on exactly the cells the bot named")
	eq(spin_disagrees, 0, "   with every recorded spin still registering as one")
	check(reached >= 100, "a meaningful number of placements were checked (%d)" % reached)

	# A placement nobody can reach must come back null rather than a wrong guess.
	var sealed := Board.new(10, 20, 20)
	for x in sealed.width:
		for y in range(sealed.buffer_height, sealed.total_height):
			sealed.set_cell(x, y, 1)
	var nowhere := PathFinder.find(sealed, ps.get_def(&"T"),
		{Vector2i(0, 0): true, Vector2i(1, 0): true, Vector2i(2, 0): true, Vector2i(1, 1): true},
		rules)
	eq(nowhere, null, "an unreachable target returns null, not a near miss")

func test_best_spin_is_kept() -> void:
	group("PathFinder: the best spin, not the first one found")

	# The same cells can be reached by different kicks that score differently, because the
	# Mini-to-Full upgrade turns on the kick index. Taking the FIRST spin found silently
	# downgraded Full spins to Minis -- invisible under all-spin, where every spin is Full
	# already, and catastrophic under gate rules, where gate_value() pays 4 for a Full spin
	# double and nothing whatsoever for a Mini. It zeroed the gate4 bot's entire output.
	var recs := _records(400, REPLAY_NORMAL)
	check(not recs.is_empty(), "the recorded normal-rules replay is readable")
	if recs.is_empty():
		return

	var ps := PieceSet.tetrominoes()
	var rules := RuleSet.normal()
	var full_seen := 0
	var downgraded := 0
	for v in recs:
		var place: Dictionary = v["place"]
		if String(place.get("spin", "none")) != "full":
			continue
		full_seen += 1
		var def: PieceDef = ps.get_def(StringName(String(place["type"]).to_upper()))
		var b := _board_from(v["board"])
		var p := PathFinder.find(b, def, PathFinder.cells_from(place["cells"]), rules)
		if p == null or p.spin != SpinDetect.Kind.FULL:
			downgraded += 1

	check(full_seen >= 20, "the replay contains full spins to check (%d)" % full_seen)
	eq(downgraded, 0, "a recorded FULL spin is never resolved as a Mini or lost")

	eq(PathFinder._spin_rank(SpinDetect.Kind.FULL), 2, "Full outranks")
	eq(PathFinder._spin_rank(SpinDetect.Kind.MINI), 1, "   Mini, which outranks")
	eq(PathFinder._spin_rank(SpinDetect.Kind.NONE), 0, "   no spin at all")

func test_pilot_wire_format() -> void:
	group("BotPilot: the state it sends and the plan it reads back")

	var pilot := BotPilot.new()
	pilot.piece_set = PieceSet.tetrominoes()
	var ps := pilot.piece_set
	var b := Board.new(10, 20, 20)
	# One wound row and one ordinary row, so both survive the trip. Wounds matter because
	# RuleSet.allspin() turns repeat_penalty on, and --board alone cannot express them.
	var bottom := b.total_height - 1
	for x in 9:
		b.set_cell(x, bottom, Board.GARBAGE)
	b.wound_left[bottom] = 2
	b.wound_hole[bottom] = 9

	var queue: Array = [ps.get_def(&"T"), ps.get_def(&"I")]
	var raw: String = pilot._state_json(b, queue, ps.get_def(&"O"), 4)
	var sent: Dictionary = JSON.parse_string(raw)
	eq(sent["queue"], "ti", "the queue goes as lower-case letters, active piece first")
	eq(sent["hold"], "o", "   with hold alongside it")
	# int(): JSON has one number type, so a round-trip turns 4 into 4.0.
	eq(int(sent["b2b"]), 4, "and the B2B chain goes too -- the search is blind to it otherwise")
	eq(sent["board"].size(), 1, "only the occupied tail of the board is sent")
	eq(String(sent["board"][0]), "#########.", "   with blocked cells as #")
	eq(sent["wound_left"].size(), 1, "the wound arrays line up with the rows sent")
	eq(int(sent["wound_left"][0]), 2, "   carrying the countdown")
	eq(int(sent["wound_hole"][0]), 9, "   and the hole it will leave")

	# Reading a plan back: step 0 becomes a real placement, later steps are draw-only.
	var reply := {
		"use_hold": false,
		"nodes": 1234,
		"plan": [
			{"k": 0, "type": "i", "hold": false, "clear": "",
				"cells": [[0, 39], [1, 39], [2, 39], [3, 39]]},
			{"k": 1, "type": "t", "hold": true, "clear": "T-Spin Double",
				"cells": [[5, 38], [4, 39], [5, 39], [6, 39]]},
		],
	}
	var empty := Board.new(10, 20, 20)
	var plan: Array = pilot._to_plan(empty, reply)
	eq(plan.size(), 2, "both steps come back")
	eq(pilot.last_nodes, 1234, "and the node count is carried for the overlay")
	var first: Placement = plan[0]
	eq(first.def.id, &"I", "step 0 is the piece the bot named")
	eq(PathFinder.cells_at(first.def, first.rot, first.pos),
		PathFinder.cells_from(reply["plan"][0]["cells"]),
		"   path-found onto exactly those cells")
	eq(plan[1].lines, 2, "a forecast step's clear name gives the line count hint_view truncates on")
	eq(plan[1].cells.size(), 4, "   and its cells, which is all the overlay draws")

	# An unreachable step 0 must empty the plan, so autoplay waits instead of placing
	# something arbitrary -- a stalled bot is far easier to notice than a wrong one.
	var bad := {"plan": [{"k": 0, "type": "i", "hold": false, "clear": "",
		"cells": [[0, 0], [1, 0], [2, 0], [3, 0]]}]}
	var none: Array = pilot._to_plan(empty, bad)
	eq(none.size(), 0, "a step Godot cannot reach empties the plan rather than guessing")
	eq(pilot.unreachable, 1, "   and is counted, because the measured rate is zero")

func test_presets() -> void:
	group("BotPilot: every bot on the toggle is real and correctly paired")

	# Godot's rules presets, so the pairing can be checked without instantiating main.
	var godot_presets := ["normal", "burst", "burst2", "gate", "gate_b2b", "competitive",
		"guideline", "allspin", "allspin_nw"]
	# The modes bot/src/main.rs rules_for() accepts. A mode outside this list makes the
	# backend panic on startup, which reads as "the bot would not come up".
	var bot_modes := ["normal", "burst", "burst2", "gate", "gate_b2b", "standard",
		"competitive", "allspin", "allspin_nw"]

	check(BotPilot.PRESETS.size() >= 2, "there is more than one bot to choose between")
	var names := {}
	for p in BotPilot.PRESETS:
		var name := String(p["name"])
		check(not names.has(name), "'%s' appears once" % name)
		names[name] = true
		check(FileAccess.file_exists(p["weights"]),
			"'%s' weights exist at %s" % [name, p["weights"]])
		check(String(p["tuned_for"]) in godot_presets,
			"'%s' names a real Godot preset to start on (%s)" % [name, p["tuned_for"]])
		check(not String(p.get("note", "")).is_empty(),
			"'%s' says what it is for" % name)

	# Cycling wraps, and reloads mode and weights together -- a half-applied switch would
	# leave the bot searching with one bot's weights under another's rules.
	var pilot := BotPilot.new()
	var first: Dictionary = pilot.use_preset(0)
	eq(pilot.weights, first["weights"], "loading a weight set sets its weights")

	# Every rules preset must map to a mode the bot accepts, or a belief set to it silently
	# leaves the bot searching the previous rules.
	for g in godot_presets:
		check(BotPilot.MODE_FOR_RULES.has(g), "'%s' rules have a bot mode" % g)
		check(String(BotPilot.MODE_FOR_RULES.get(g, "")) in bot_modes,
			"   and the bot accepts it (%s)" % BotPilot.MODE_FOR_RULES.get(g, ""))
	eq(BotPilot.MODE_FOR_RULES.size(), godot_presets.size(),
		"the map covers every rules preset and invents none")

	# Belief belongs to the WEIGHTS, not to the game. A reward model the weights were fitted
	# against is not separable from them, and getting this backwards is fatal rather than
	# suboptimal: all-spin weights believing normal rules topped out every run of a
	# three-seed, 500-piece test, against no deaths and 2.59 APP believing all-spin.
	for i in BotPilot.PRESETS.size():
		var loaded: Dictionary = pilot.use_preset(i)
		eq(pilot.believe, String(loaded["tuned_for"]),
			"'%s' believes the rules it was tuned for" % loaded["name"])
		eq(pilot.mode, String(BotPilot.MODE_FOR_RULES[pilot.believe]),
			"   and its mode follows that belief")

	# Overriding it is the experiment, in either direction, and clearing restores the pairing.
	pilot.use_preset(0)
	var tuned: String = String(pilot.preset()["tuned_for"])
	pilot.set_belief("normal")
	eq(pilot.believe, "normal", "a belief can be pointed at the real environment")
	eq(pilot.mode, "normal", "   and the mode goes with it")
	pilot.set_belief("")
	eq(pilot.believe, tuned, "clearing it restores what the weights were tuned for")
	eq(pilot.mode, String(BotPilot.MODE_FOR_RULES[tuned]), "   mode included")
	pilot.set_belief("not_a_preset")
	eq(pilot.believe, tuned, "an unknown belief is refused rather than silently applied")

func test_allspin_without_wounds() -> void:
	group("RuleSet: all-spin scoring without the wound lines")

	# Wounds come from repeat_penalty, not from all-spin. The obvious way to lose them --
	# dropping to normal rules -- also makes every non-T spin a Mini that sends nothing, which
	# measured 0.24 APP against 2.19 for an all-spin-tuned bot. This preset is the narrow
	# change: the same scoring, minus the unclearable rows.
	var full := RuleSet.allspin()
	var nw := RuleSet.allspin_no_wounds()

	eq(nw.preset_name, "allspin_nw", "it is its own preset")
	check(full.repeat_penalty, "all-spin penalises a repeated clear type")
	check(not nw.repeat_penalty, "   and this variant does not -- the one difference")

	# Everything that makes it ALL-SPIN has to survive, or it is just normal rules again.
	eq(nw.all_spin, full.all_spin, "every piece can still spin")
	eq(nw.allspin_full, full.allspin_full, "   and every spin is still Full, not a Mini")
	eq(nw.non_t_spin_is_mini, full.non_t_spin_is_mini,
		"   so a non-T spin still pays, which is what normal rules would have taken away")

	# And the board really stops growing wounds. Driven through Board directly, because the
	# live wound count ticks down into ordinary garbage and so proves nothing after the fact.
	var b := Board.new(10, 20, 20)
	b.add_wound(10, 4)
	eq(b.wound_count(), 1, "a wound added directly is a wound")
	check(b.is_wound(b.total_height - 1), "   and it sits at the bottom")

func test_pilot_degrades_without_a_binary() -> void:
	group("BotPilot: an unbuilt bot is a message, not a mystery")

	var pilot := BotPilot.new()
	pilot.piece_set = PieceSet.tetrominoes()
	var queue: Array = [pilot.piece_set.get_def(&"T")]
	# Whether the binary exists depends on whether cargo has been run, so both outcomes are
	# legitimate -- what must not happen is a silent empty plan with no explanation.
	if pilot.available():
		check(not BotPilot.binary_path().is_empty(), "a built bot reports its path")
	else:
		var plan: Array = pilot.solve(Board.new(10, 20, 20), queue, null, Callable(), -1)
		eq(plan.size(), 0, "an unbuilt bot plans nothing")
		check(pilot.last_error.contains("cargo build"),
			"   and says exactly what to run")
