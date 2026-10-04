class_name Muzzle
extends RefCounted

## Where a shot leaves a tower: the point at which the aim ray exits the footprint hull.
##
## Not the cannon cell's centre. A cannon with body cells in front of it would spawn the
## shot inside solid geometry, and forbidding that would mean banning interior cannons
## across four rotations and sixteen directions -- which is exactly the authoring freedom
## the mask is supposed to give. Letting the cannon cell define the ray ORIGIN and sliding
## the spawn point out to the hull makes every mask legal.
##
## Solved once per (rotation, direction) at compile time and cached on the TowerDef, so
## firing is a table lookup rather than a ray walk. 4 x 16 = 64 entries per tower.
##
## A note on a rule that turned out to be impossible: "the cannon is walled in, so the
## tower cannot fire" cannot happen. A finite solid axis-aligned footprint cannot enclose
## a cell such that a straight ray never escapes -- every ray leaves a bounded set. The
## three real failures are a mask with no cannon (a compile-time assert in TowerSet), a
## muzzle that lands outside the FIELD, which needs the board and so is checked at fire
## time by `clears_field`, and the step cap below, which exists only so a malformed or
## disjoint mask returns rather than hangs.
##
## Offsets are in CELL units, relative to the footprint's top-left. Pixels are a view
## concern, so nothing here needs to know the cell size.

## Walk limit. A ray crosses at most one cell boundary per step and the hull is `box`
## across, so twice that plus slack can only be reached by a mask that is not solid.
static func step_cap(box: int) -> int:
	return box * 2 + 2

## Solves one (cells, cannon, direction). Returns {ok: bool, offset: Vector2}.
##
## A DDA in cell space: step boundary to boundary, and the first boundary whose far side
## is NOT part of the footprint is the exit. Walking cell by cell rather than marching in
## small increments makes this exact, and handles a concave mask for free -- a U-shaped
## tower firing from inside the notch exits through the notch.
static func solve(cells: Array, cannon: Vector2i, dir: int, box: int) -> Dictionary:
	var occupied := {}
	for c in cells:
		occupied[c] = true
	var v := Aim.vector(dir)
	# The cell CENTRE, so the first step can never begin exactly on a boundary -- which is
	# what would otherwise give a zero-length step and stall the walk.
	var pos := Vector2(cannon) + Vector2(0.5, 0.5)
	var cell := cannon
	for _i in step_cap(box):
		var tx := INF
		if absf(v.x) > 1e-9:
			var nx := float(cell.x + 1) if v.x > 0.0 else float(cell.x)
			tx = (nx - pos.x) / v.x
		var ty := INF
		if absf(v.y) > 1e-9:
			var ny := float(cell.y + 1) if v.y > 0.0 else float(cell.y)
			ty = (ny - pos.y) / v.y
		var t := minf(tx, ty)
		if t >= INF or t < 0.0:
			return {"ok": false, "offset": Vector2.ZERO}
		pos += v * t
		var step := Vector2i.ZERO
		if absf(tx - ty) < 1e-9:
			# An exact corner: both boundaries at once, so the walk moves diagonally.
			step = Vector2i(1 if v.x > 0.0 else -1, 1 if v.y > 0.0 else -1)
		elif tx < ty:
			step = Vector2i(1 if v.x > 0.0 else -1, 0)
		else:
			step = Vector2i(0, 1 if v.y > 0.0 else -1)
		var nxt := cell + step
		if not occupied.has(nxt):
			return {"ok": true, "offset": pos}
		cell = nxt
	return {"ok": false, "offset": Vector2.ZERO}

## The whole 64-entry table for a def, in the layout TowerDef.muzzles expects.
static func table(def: TowerDef) -> Array:
	var out: Array = []
	out.resize(4 * Aim.COUNT)
	for rot in 4:
		for dir in Aim.COUNT:
			out[rot * Aim.COUNT + dir] = solve(def.cells(rot), def.cannon(rot), dir, def.box)
	return out
