extends SceneTree

## Rust <-> Godot parity for the all-spin rules.
##
##   godot --headless --path . -s res://tests/allspin_parity.gd -- --replay=<bot allspin jsonl>
##
## Replays every placement of a bot run (made with `bot play --mode allspin`) through
## GODOT's Board, wound lines, ClearResult.clear_type() and AttackTable, starting from an
## empty board, and checks after each placement that Godot arrives at exactly the board
## the Rust engine recorded next: every cell, each wound line's row, counter and hole, the
## lines cleared, the repeat flag and the attack. Spin labels are taken from the record
## (spin detection itself is covered by the move-generator parity against real play).

func _init() -> void:
	var path := ""
	# --rules=normal: no repeat penalty (normal rules); default: all-spin.
	var penalty := true
	# --rules=burst / burst2: normal rules plus the burst doubling and TST 10.
	var burst_rules: RuleSet = null
	for arg in OS.get_cmdline_user_args():
		if arg == "--rules=normal" or arg == "--rules=burst" or arg == "--rules=burst2" or arg == "--rules=gate" or arg == "--rules=gate_b2b":
			penalty = false
		if arg == "--rules=gate":
			burst_rules = RuleSet.gate_rules()
		if arg == "--rules=gate_b2b":
			burst_rules = RuleSet.gate_b2b_rules()
		if arg == "--rules=burst":
			burst_rules = RuleSet.burst()
		if arg == "--rules=burst2":
			burst_rules = RuleSet.burst2()
		if arg.begins_with("--replay="):
			path = arg.substr("--replay=".length())
	var burst := burst_rules != null
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		print("FAIL cannot open ", path)
		quit(1)
		return
	var runs := {}
	var order: Array = []
	for line in f.get_as_text().split("\n", false):
		var r = JSON.parse_string(line)
		if typeof(r) != TYPE_DICTIONARY:
			continue
		var name: String = r["replay"]
		if not runs.has(name):
			runs[name] = []
			order.append(name)
		runs[name].append(r)

	var ps := PieceSet.tetrominoes()
	var table := AttackTable.new()
	table.configure(burst_rules if burst else RuleSet.normal())
	var bad := 0
	var totals := {"placements": 0, "repeats": 0, "healed": 0, "attack": 0, "bursts": 0}
	for name in order:
		var steps: Array = runs[name]
		var b := Board.new(10, 20, 20)
		var b2b := -1
		var last_type := ""
		var armed := 0
		var gate_st: Array = [0, 0, 0]
		var gate_credit := 0
		for i in steps.size():
			var step: Dictionary = steps[i]
			var why := _compare(b, step)
			if why != "":
				bad += 1
				if bad <= 8:
					print("board mismatch %s before placement %d: %s" % [name, i + 1, why])
				break
			var place: Dictionary = step["place"]
			var id := StringName(String(place["type"]).to_upper())
			var v: int = ps.get_def(id).index + 1
			var touched := {}
			for c in place["cells"]:
				b.set_cell(int(c[0]), int(c[1]), v)
				touched[int(c[1])] = true
			var full := b.full_rows_among(touched.keys())
			b.clear_rows(full)
			var r := ClearResult.new()
			r.lines = full.size()
			r.piece_id = id
			match String(step.get("spin", "none")):
				"full": r.spin = SpinDetect.Kind.FULL
				"mini": r.spin = SpinDetect.Kind.MINI
				_: r.spin = SpinDetect.Kind.NONE
			r.perfect_clear = r.lines > 0 and b.is_empty()
			if r.lines > 0:
				b2b = b2b + 1 if r.is_difficult() else -1
			r.b2b = b2b
			if burst and burst_rules.gate:
				var bonus := table.b2b_bonus_for(r.b2b) if burst_rules.gate_b2b else 0
				var gs := burst_rules.gate_step(gate_st, r, bonus)
				gate_credit = gs[0]
				gate_st = gs[1]
				r.burst = gate_credit > 0
			elif burst:
				var bs := burst_rules.burst_step(armed, r)
				r.burst = bs[0]
				armed = bs[1]
			if penalty and r.lines > 0:
				r.healed = b.tick_wounds()
				var t := r.clear_type()
				r.repeat = t == last_type
				if r.repeat:
					# The hole is chosen by messiness in the engine; take the recorded one.
					b.add_wound(10, int(step.get("wound_hole", 0)))
				last_type = t
			var atk := gate_credit if burst and burst_rules.gate else table.attack_for(r)
			var problems: Array = []
			if r.lines != int(step["cleared"]):
				problems.append("lines %d vs %d" % [r.lines, int(step["cleared"])])
			if r.repeat != bool(step.get("repeat", false)):
				problems.append("repeat %s vs %s" % [r.repeat, step.get("repeat")])
			if r.burst != bool(step.get("burst", false)):
				problems.append("burst %s vs %s" % [r.burst, step.get("burst")])
			if atk != int(step["attack"]):
				problems.append("attack %d vs %d" % [atk, int(step["attack"])])
			if not problems.is_empty():
				bad += 1
				if bad <= 8:
					print("%s placement %d: %s" % [name, i + 1, ", ".join(problems)])
			totals["placements"] += 1
			totals["repeats"] += int(r.repeat)
			totals["healed"] += r.healed
			totals["attack"] += atk
			totals["bursts"] += int(r.burst)
		print("%s: %d placements checked" % [name, steps.size()])
	print("placements %d, repeats %d, wound lines healed %d, bursts %d, attack %d -- problems %d" % [
		totals["placements"], totals["repeats"], totals["healed"], totals["bursts"], totals["attack"], bad])
	quit(0 if bad == 0 else 1)

## "" when Godot's board matches the record's spawn-time board and wound list.
func _compare(b: Board, step: Dictionary) -> String:
	var rows: Array = step["board"]
	for y in rows.size():
		var row: String = rows[y]
		for x in row.length():
			var ch := row[x]
			var v := b.get_cell(x, y)
			if ch == "." and v != 0:
				return "cell (%d,%d) filled in Godot, empty in Rust" % [x, y]
			if ch == "W" and v != Board.WOUND:
				return "cell (%d,%d) is a wound in Rust, not in Godot" % [x, y]
			if ch != "." and ch != "W" and (v == 0 or v == Board.WOUND):
				return "cell (%d,%d) is a piece in Rust, %d in Godot" % [x, y, v]
	var want: Array = step.get("wounds", [])
	if want.size() != b.wound_count():
		return "%d wounds in Rust, %d in Godot" % [want.size(), b.wound_count()]
	for wd in want:
		var y := int(wd["row"])
		if int(wd["left"]) != b.wound_left[y]:
			return "wound on row %d has %d left in Rust, %d in Godot" % [y, int(wd["left"]), b.wound_left[y]]
		if int(wd["hole"]) != b.wound_hole[y]:
			return "wound on row %d has hole %d in Rust, %d in Godot" % [y, int(wd["hole"]), b.wound_hole[y]]
	return ""
