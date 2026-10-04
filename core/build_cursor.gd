class_name BuildCursor
extends RefCounted

## Where the player is about to build, and facing which way. Pure intent: it holds no
## towers and changes nothing, so the placement rules can be tested without a scene and
## the view can draw a ghost from it without asking the sim anything.
##
## `origin` is the top-left cell of the footprint, matching ActivePiece.pos meaning the
## bounding box corner -- the two grids speak one language.
##
## The cursor is kept legal at all times: moving clamps so the footprint stays on the
## grid, and so does TURNING, because a rotation can transpose a 3x2 extent into 2x3 and
## push it off the right edge. A cursor that can sit somewhere a tower cannot go is a
## cursor that lies.

var origin: Vector2i = Vector2i.ZERO
var rot: int = 0
var aim: int = 0
## Index into TowerSet.defs.
var selected: int = 0

var _towers: TowerSet
var _field: DefenceField

func _init(ts: TowerSet, f: DefenceField) -> void:
	_towers = ts
	_field = f
	_clamp()

func def() -> TowerDef:
	if _towers.defs.is_empty():
		return null
	return _towers.defs[clampi(selected, 0, _towers.defs.size() - 1)]

func move(dx: int, dy: int) -> void:
	origin += Vector2i(dx, dy)
	_clamp()

## Turns the footprint. Clamps afterwards, because the extent may have transposed.
func turn_footprint(steps: int) -> void:
	rot = (rot + steps) & 3
	_clamp()

func turn_aim(steps: int) -> void:
	aim = Aim.turn(aim, steps)

func select(i: int) -> void:
	if _towers.defs.is_empty():
		return
	selected = clampi(i, 0, _towers.defs.size() - 1)
	_clamp()

func select_next(step: int = 1) -> void:
	if _towers.defs.is_empty():
		return
	selected = (selected + step) % _towers.defs.size()
	if selected < 0:
		selected += _towers.defs.size()
	_clamp()

## The cells the ghost covers, in grid coordinates.
func cells() -> Array[Vector2i]:
	var out: Array[Vector2i] = []
	var d := def()
	if d == null:
		return out
	for c in d.cells(rot):
		out.append(origin + c)
	return out

## Whether a tower could actually be built here right now. The view colours the ghost by
## this, and the sim re-checks it -- the cursor is intent, not permission.
func can_build(sim: DefenceSim) -> bool:
	var d := def()
	if d == null:
		return false
	return sim.can_build(d, rot, origin)

func _clamp() -> void:
	var d := def()
	if d == null:
		return
	origin = _field.clamp_origin(origin, d.extent(rot))
