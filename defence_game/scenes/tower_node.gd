extends Node2D

## The visible half of a DefenceTower. Deliberately dumb: it is told where to be and what
## to look like, and decides nothing. All authority is in DefenceSim.
##
## Two sprite layers, because there are two independent rotations. The BODY turns in four
## 90-degree steps with the footprint; the BARREL turns in sixteen 22.5-degree steps with
## the aim. One sprite could not express both, which is why swapping in art means
## supplying two textures rather than one.
##
## Body cells are duplicated from the `Body/Tile` sprite in tower.tscn rather than
## authored one per cell, because a footprint is 4 cells for one tower and 5 for another
## and changes shape as it turns. The texture stays on that sprite, where you can see it:
## an export on the root would leave the visible node empty until runtime.

var cell_px: int = 30

## Resolved on first use rather than with @onready. A pooled node is set up immediately
## after add_child, and if its PARENT is not in the tree yet then _ready has not run and
## every @onready is still null -- which is silent, because assigning to null just errors
## at the point of use. Looking them up on demand works either way.
var _body: Node2D
var _barrel: Sprite2D
var _muzzle: Marker2D
var _mount: Sprite2D

var _tiles: Array[Sprite2D] = []

func _resolve() -> void:
	if _body != null:
		return
	_body = get_node_or_null("Body")
	_barrel = get_node_or_null("Barrel")
	_muzzle = get_node_or_null("Muzzle")
	_mount = get_node_or_null("Mount")

func setup(px: int) -> void:
	_resolve()
	cell_px = px
	if _barrel != null and _barrel.texture != null:
		# Pivot at the left edge, vertically centred on the texture, so rotating the node
		# turns the barrel instead of swinging it around the tower.
		_barrel.centered = false
		_barrel.offset = Vector2(0.0, -_barrel.texture.get_height() * 0.5)
	if _mount != null:
		_mount.centered = false
		_scale_to_cell(_mount)

## Called on build and whenever the footprint or aim changes.
func apply(tower: DefenceTower) -> void:
	_resolve()
	position = Vector2(tower.origin) * float(cell_px)
	_rebuild_body(tower)
	var m: Dictionary = tower.def.muzzle(tower.rot, tower.aim)
	var local: Vector2 = Vector2(m["offset"]) * float(cell_px)
	_muzzle.position = local
	_barrel.position = Vector2(tower.def.cannon(tower.rot)) * float(cell_px) \
		+ Vector2(float(cell_px), float(cell_px)) * 0.5
	# Negative because Aim counts counter-clockwise on screen while rotation counts
	# clockwise. Getting this wrong mirrors every shot's apparent origin.
	_barrel.rotation = -float(tower.aim) * Aim.STEP
	_barrel.modulate = tower.def.color
	# On the cannon cell, so turning the block visibly moves something even when the
	# footprint is square and the body does not change at all.
	if _mount != null:
		_mount.position = Vector2(tower.def.cannon(tower.rot)) * float(cell_px)

## Lays one body sprite per footprint cell.
##
## Built from the `Body/Tile` sprite in the scene rather than from code-created Sprite2Ds,
## so the texture lives on a node you can see and swap. The tile is also the template:
## extra cells are duplicates of it, which means anything you change on it -- texture,
## modulate, shader -- carries to every cell for free.
func _rebuild_body(tower: DefenceTower) -> void:
	if _body == null:
		return
	if _tiles.is_empty():
		var first: Sprite2D = _body.get_node_or_null("Tile")
		if first == null:
			return
		first.centered = false
		_tiles.append(first)
	var cells: Array[Vector2i] = tower.def.cells(tower.rot)
	while _tiles.size() < cells.size():
		var s: Sprite2D = _tiles[0].duplicate()
		_body.add_child(s)
		_tiles.append(s)
	for i in _tiles.size():
		var s: Sprite2D = _tiles[i]
		s.visible = i < cells.size()
		if not s.visible:
			continue
		s.position = Vector2(cells[i]) * float(cell_px)
		s.modulate = tower.def.color
		_scale_to_cell(s)

## Scales a one-cell sprite to the playfield's cell size, so art authored at any
## resolution lands on the grid.
func _scale_to_cell(s: Sprite2D) -> void:
	if s.texture == null:
		return
	var tex := s.texture.get_size()
	if tex.x > 0.0 and tex.y > 0.0:
		s.scale = Vector2(float(cell_px), float(cell_px)) / tex
