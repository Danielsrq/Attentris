extends SceneTree

## Independent check of the Rust move generator using GODOT's rules code.
## For every spin clear in a bot replay: rebuild the board, BFS from Godot's spawn
## position using only Board.can_place and KickTables.get_kicks (first fitting kick
## wins, as Game.try_rotate), and require a rotation that lands grounded on exactly the
## recorded cells with SpinDetect.detect() reporting a spin.
## Then, for one chosen placement, replay the found input path through a real Game
## (try_rotate / soft drop / lock_piece) and print what the game itself reports.

var ps := PieceSet.tetrominoes()
var rules := RuleSet.allspin()

func _init() -> void:
	var path := ""
	var run := "allspin_s101"
	var show := 179
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--replay="):
			path = arg.substr(9)
		elif arg.begins_with("--run="):
			run = arg.substr(6)
		elif arg.begins_with("--show="):
			show = int(arg.substr(7))
	var f := FileAccess.open(path, FileAccess.READ)
	var checked := 0
	var bad := 0
	for line in f.get_as_text().split("\n", false):
		var r = JSON.parse_string(line)
		if typeof(r) != TYPE_DICTIONARY or String(r.get("spin", "none")) == "none" or int(r["cleared"]) == 0:
			continue
		var b := _board(r)
		var def: PieceDef = ps.get_def(StringName(String(r["place"]["type"]).to_upper()))
		var want := {}
		for c in r["place"]["cells"]:
			want[Vector2i(int(c[0]), int(c[1]))] = true
		var found := _bfs(b, def, want)
		checked += 1
		if found.is_empty():
			bad += 1
			if bad <= 5:
				print("NOT REPRODUCED: %s placement %d %s" % [r["replay"], int(r["n"]) + 1, r["clear"]])
		elif String(r["replay"]) == run and int(r["n"]) + 1 == show:
			_demo(r, b, def, found)
	print("spin clears checked with Godot's kick tables: %d, not reproducible: %d" % [checked, bad])
	quit(0 if bad == 0 else 1)

func _board(r: Dictionary) -> Board:
	var b := Board.new(10, 20, 20)
	var rows: Array = r["board"]
	for y in rows.size():
		for x in 10:
			var ch: String = rows[y][x]
			if ch == "W":
				b.set_cell(x, y, Board.WOUND)
			elif ch != ".":
				b.set_cell(x, y, 1)
	# Register the wound lines too, or their solid rows would read as full lines.
	for wd in r.get("wounds", []):
		b.wound_left[int(wd["row"])] = int(wd["left"])
		b.wound_hole[int(wd["row"])] = int(wd.get("hole", 0))
	return b

func _cells_at(def: PieceDef, rot: int, pos: Vector2i) -> Dictionary:
	var out := {}
	for c in def.rotations[rot]:
		out[pos + c] = true
	return out

## BFS over (pos, rot). Returns the action list ending in the spin rotation, or [].
func _bfs(b: Board, def: PieceDef, want: Dictionary) -> Array:
	var spawn := Vector2i((b.width - def.box) / 2, b.buffer_height - def.first_filled_row)
	if not b.can_place(def.rotations[0], spawn):
		return []
	var start := [spawn, 0]
	var parent := {start: null}
	var queue: Array = [start]
	var moves := {"left": Vector2i(-1, 0), "right": Vector2i(1, 0), "down": Vector2i(0, 1)}
	var turns := {"cw": 1, "ccw": 3, "180": 2}
	while not queue.is_empty():
		var s: Array = queue.pop_front()
		var pos: Vector2i = s[0]
		var rot: int = s[1]
		for name in moves:
			var np: Vector2i = pos + moves[name]
			var ns := [np, rot]
			if b.can_place(def.rotations[rot], np) and not parent.has(ns):
				parent[ns] = [s, name]
				queue.append(ns)
		for name in turns:
			var to: int = (rot + turns[name]) % 4
			var kicks := KickTables.get_kicks(def.kick_class, rot, to)
			for i in kicks.size():
				var np: Vector2i = pos + kicks[i]
				if not b.can_place(def.rotations[to], np):
					continue
				var ns := [np, to]
				# A spin: grounded right after this rotation, on the recorded cells.
				if not b.can_place(def.rotations[to], np + Vector2i(0, 1)) and _cells_at(def, to, np) == want:
					var p := ActivePiece.new()
					p.def = def
					p.rot = to
					p.pos = np
					p.last_action_was_rotation = true
					p.last_kick_index = i
					if SpinDetect.detect(b, p, rules) != SpinDetect.Kind.NONE:
						var actions: Array = ["%s (kick %d)" % [name, i]]
						var cur = s
						while parent[cur] != null:
							actions.push_front(parent[cur][1])
							cur = parent[cur][0]
						return actions
				if not parent.has(ns):
					parent[ns] = [s, name]
					queue.append(ns)
				break
	return []

## Plays the path through a real Game and reports what lock_piece says.
func _demo(r: Dictionary, b: Board, def: PieceDef, actions: Array) -> void:
	print("placement %d (%s): Godot BFS path from spawn:" % [int(r["n"]) + 1, r["clear"]])
	# Collapse repeated moves for readability.
	var out: Array = []
	for a in actions:
		if not out.is_empty() and out[-1][0] == a:
			out[-1][1] += 1
		else:
			out.append([a, 1])
	print("   " + ", ".join(out.map(func(e): return e[0] if e[1] == 1 else "%s x%d" % [e[0], e[1]])))
	var g := Game.new(rules, ps, b, Scorer.new(ScoreTable.new()), 1)
	var p := ActivePiece.new()
	p.def = def
	p.rot = 0
	p.pos = Vector2i((b.width - def.box) / 2, b.buffer_height - def.first_filled_row)
	g.active = p
	# Chain as it stood before this placement: the record holds the value after it.
	g.b2b = int(r["b2b"]) - 1 if int(r["b2b"]) >= 1 else -1
	for a in actions:
		var name: String = String(a).split(" ")[0]
		match name:
			"left": g._try_move(Vector2i(-1, 0))
			"right": g._try_move(Vector2i(1, 0))
			"down": g._try_move(Vector2i(0, 1), false)
			"cw": g.try_rotate((g.active.rot + 1) % 4)
			"ccw": g.try_rotate((g.active.rot + 3) % 4)
			"180": g.try_rotate((g.active.rot + 2) % 4)
	print("   after the path: rot %d box %s, last kick index %d" % [g.active.rot, g.active.pos, g.active.last_kick_index])
	var ev: Array = []
	g.lock_piece(ev)
	print("   Godot Game.lock_piece reports: %s, %d line(s), attack %d (bot recorded %s, attack %d)" % [
		g.last_result.describe(), g.last_result.lines, g.last_result.attack, r["clear"], int(r["attack"])])
