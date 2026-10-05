extends TestHarness

## PathFinder and BotPilot, against real recorded bot output.
##
## The bot itself is not run here -- building it is a cargo step the suite must not depend on.
## What is tested is the part that can silently go wrong: turning the cells the bot names into
## a placement Godot can actually reach, with the spin credited. Both engines index row 0 at
## the top and both place a piece by a box offset, and nothing in either codebase enforces
## that; the records are the only thing that can.

const REPLAY := "res://data/bot/allspin_slots.jsonl"

func run() -> void:
	test_path_finder_against_recorded_placements()
	test_pilot_wire_format()
	test_pilot_degrades_without_a_binary()

## Reads the first `n` records that have a `place` with cells.
func _records(n: int) -> Array:
	var f := FileAccess.open(REPLAY, FileAccess.READ)
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
