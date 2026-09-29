class_name PlacementEnumerator
extends RefCounted

## Every placement a player could actually reach by "rotate at the top, slide across,
## drop" -- the straight-drop move set. No tucks, no spins.
##
## The reachability walk is the part that is easy to skip and wrong to skip: a
## placement is only legal if the piece can travel horizontally from its spawn column
## to the target column AT SPAWN HEIGHT. With a tall tower in the way it cannot, and
## enumerating it anyway means the hint points at a placement the player is physically
## unable to make -- precisely when the stack is high and the hint matters most.

## All reachable placements for `def` on `board`. `spawn` is the piece's spawn
## position (Game.spawn_pos_for), which is where reachability is measured from.
static func placements_for(board: Board, def: PieceDef, spawn: Vector2i, uses_hold: bool = false) -> Array:
	var out: Array = []
	var seen := {}
	for rot in 4:
		var cells: Array = def.rotations[rot]
		var lo := 0x7FFFFFFF
		var hi := -0x7FFFFFFF
		for c in cells:
			lo = mini(lo, c.x)
			hi = maxi(hi, c.x)
		# Rotating at spawn has to be possible before anything else.
		if not board.can_place(cells, Vector2i(spawn.x, spawn.y)):
			continue
		for x in range(-lo, board.width - hi):
			if not _can_reach_column(board, cells, spawn, x):
				continue
			var rest := _drop(board, cells, Vector2i(x, spawn.y))
			if rest.y < 0:
				continue
			# Two rotation states can settle on the same cells (I, S, Z, O); keep one.
			var key := _key(cells, rest)
			if seen.has(key):
				continue
			seen[key] = true
			var p := Placement.new()
			p.def = def
			p.rot = rot
			p.pos = rest
			p.uses_hold = uses_hold
			for c in cells:
				p.cells.append(rest + c)
			out.append(p)
	return out

## Horizontal walk at spawn height, one column at a time. Cheap -- at most `width`
## collision checks -- and it is what rules out placements behind a tall stack.
static func _can_reach_column(board: Board, cells: Array, spawn: Vector2i, target_x: int) -> bool:
	var step := 1 if target_x > spawn.x else -1
	var x := spawn.x
	while x != target_x:
		x += step
		if not board.can_place(cells, Vector2i(x, spawn.y)):
			return false
	return true

static func _drop(board: Board, cells: Array, from: Vector2i) -> Vector2i:
	if not board.can_place(cells, from):
		return Vector2i(from.x, -1)
	var p := from
	while board.can_place(cells, p + Vector2i(0, 1)):
		p.y += 1
	return p

static func _key(cells: Array, pos: Vector2i) -> String:
	var parts: Array = []
	for c in cells:
		parts.append("%d,%d" % [pos.x + c.x, pos.y + c.y])
	parts.sort()
	return ";".join(parts)
