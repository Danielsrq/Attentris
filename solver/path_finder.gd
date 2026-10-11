class_name PathFinder
extends RefCounted

## Finds a legal way to stand a piece on a given set of cells, using Godot's OWN kick tables.
##
## This is what lets the Rust bot pilot the board without either engine having to trust the
## other's coordinate conventions. The bot says which CELLS the piece should occupy; the
## rotation, the position and the kick index are re-derived here, against the rules the game
## actually applies. So whatever this returns is reachable by construction, and a placement the
## bot can see but Godot cannot reach is reported rather than quietly played as something else.
##
## Measured before it was relied on: `tests/spin_path_check.gd` runs this against every spin
## clear in `data/bot/allspin_slots.jsonl` -- 2068 placements, 0 not reproducible.
##
## Lifted out of that test so the pilot and the check share one implementation. Two copies of
## a BFS over kick tables would eventually disagree, and the one under test would be the one
## nobody was using.

## Cells a piece would occupy at a rotation and position, as a set.
static func cells_at(def: PieceDef, rot: int, pos: Vector2i) -> Dictionary:
	var out := {}
	for c in def.rotations[rot]:
		out[pos + c] = true
	return out

## Cells from a replay/plan record's `[[x, y], ...]`, as the same kind of set.
static func cells_from(raw: Array) -> Dictionary:
	var out := {}
	for c in raw:
		out[Vector2i(int(c[0]), int(c[1]))] = true
	return out

## The placement standing on `want`, or null if no legal path reaches it.
##
## Prefers a path whose LAST action is a rotation registering the BEST spin available. That
## preference is not cosmetic, and neither is the word "best":
##
## `Game` only consults `SpinDetect` when `last_action_was_rotation` is set, so a placement
## that omits it is silently un-spinnable whatever shape it lands in. And the same cells can
## be reached by different kicks that score differently -- the Mini-to-Full upgrade turns on
## the kick index -- so taking the FIRST spin found rather than the best quietly downgrades
## Full spins to Minis.
##
## That cost real attack and was invisible under all-spin rules, where every spin is Full
## already. Under normal-based rules it halved the `normal` bot's output, and under gate rules
## it zeroed the `gate4` bot's entirely, because gate_value() pays for a Full spin double and
## nothing at all for a Mini. The Rust move generator keeps the best spin per cell set for
## exactly this reason; this now matches it.
## `out_actions`, when given, is filled with the input names leading to the returned
## placement -- which is how `tests/spin_path_check.gd` can replay the path through a real
## Game and show it. Carried here rather than left to the caller so there is ONE BFS over the
## kick tables: two copies would drift, and the one under test would be the one nobody used.
static func find(b: Board, def: PieceDef, want: Dictionary, rules: RuleSet,
		out_actions: Array = []) -> Placement:
	var spawn := Vector2i((b.width - def.box) / 2, b.buffer_height - def.first_filled_row)
	if not b.can_place(def.rotations[0], spawn):
		return null
	var best: Placement = null                       # best so far, by spin rank
	var best_path: Array = []
	var best_rank := -1
	var start := [spawn, 0]
	var seen := {start: true}
	# state -> [previous state, the input that got here]; null for the spawn.
	var parent := {start: null}
	var queue: Array = [start]
	# Down, not hard-drop: a tuck needs the piece to travel sideways under an overhang, so the
	# walk has to be cell by cell.
	var moves := {&"left": Vector2i(-1, 0), &"right": Vector2i(1, 0), &"down": Vector2i(0, 1)}
	var turns := {&"cw": 1, &"ccw": 3, &"180": 2}
	while not queue.is_empty():
		var s: Array = queue.pop_front()
		var pos: Vector2i = s[0]
		var rot: int = s[1]
		if best == null and cells_at(def, rot, pos) == want and _grounded(b, def, rot, pos):
			# A move-last path can never register a spin, so it ranks lowest.
			best = _placement(b, def, rot, pos, false, -1, rules)
			best_rank = 0
			best_path = _path_to(parent, s)
		for m in moves:
			var np: Vector2i = pos + moves[m]
			var ns := [np, rot]
			if not seen.has(ns) and b.can_place(def.rotations[rot], np):
				seen[ns] = true
				parent[ns] = [s, m]
				queue.append(ns)
		for t in turns:
			var to: int = (rot + int(turns[t])) % 4
			var kicks := KickTables.get_kicks(def.kick_class, rot, to)
			for i in kicks.size():
				var np: Vector2i = pos + kicks[i]
				if not b.can_place(def.rotations[to], np):
					continue
				# Checked BEFORE the visited test and before the break: the same resting state
				# can be reached by a move with no spin or by a kick with one, so a state
				# already seen can still be the first to arrive here by rotation.
				if cells_at(def, to, np) == want and _grounded(b, def, to, np):
					var p := _placement(b, def, to, np, true, i, rules)
					var rank := _spin_rank(p.spin)
					if rank > best_rank:
						best = p
						best_rank = rank
						best_path = _path_to(parent, s)
						best_path.append("%s (kick %d)" % [t, i])
					# Only a FULL spin is worth stopping for; anything less may still be
					# beaten by another kick to the same cells.
					if rank >= 2:
						out_actions.assign(best_path)
						return best
				var ns2 := [np, to]
				if not seen.has(ns2):
					seen[ns2] = true
					parent[ns2] = [s, t]
					queue.append(ns2)
				# First fitting kick wins, exactly as Game.try_rotate resolves it.
				break
	if best != null:
		out_actions.assign(best_path)
	return best

## Full beats Mini beats no spin. The ordering is what makes "best" mean anything, and it is
## the whole difference between a T-spin double paying 4 under gate rules and paying 0.
static func _spin_rank(spin: int) -> int:
	if spin == SpinDetect.Kind.FULL:
		return 2
	if spin == SpinDetect.Kind.MINI:
		return 1
	return 0

## The inputs leading to a state, oldest first.
static func _path_to(parent: Dictionary, state: Array) -> Array:
	var out: Array = []
	var cur: Variant = state
	while parent.get(cur) != null:
		var step: Array = parent[cur]
		out.push_front(step[1])
		cur = step[0]
	return out

static func _grounded(b: Board, def: PieceDef, rot: int, pos: Vector2i) -> bool:
	return not b.can_place(def.rotations[rot], pos + Vector2i(0, 1))

static func _placement(b: Board, def: PieceDef, rot: int, pos: Vector2i,
		by_rotation: bool, kick: int, rules: RuleSet) -> Placement:
	var p := Placement.new()
	p.def = def
	p.rot = rot
	p.pos = pos
	p.via_rotation = by_rotation
	p.kick_index = kick
	p.cells = []
	for c in def.rotations[rot]:
		p.cells.append(pos + c)
	p.spin = SpinDetect.Kind.NONE
	if by_rotation:
		var probe := ActivePiece.new()
		probe.def = def
		probe.rot = rot
		probe.pos = pos
		probe.last_action_was_rotation = true
		probe.last_kick_index = kick
		p.spin = SpinDetect.detect(b, probe, rules)
	p.lines = _lines_cleared(b, def, rot, pos)
	return p

## How many rows this placement would complete. On a COPY, because the caller is mid-search
## and the live board must not move under it.
static func _lines_cleared(b: Board, def: PieceDef, rot: int, pos: Vector2i) -> int:
	var probe := b.duplicate_board()
	probe.lock_cells(def.rotations[rot], pos, 1)
	var rows: Array = []
	for c in def.rotations[rot]:
		var y: int = pos.y + c.y
		if not rows.has(y):
			rows.append(y)
	return probe.full_rows_among(rows).size()
