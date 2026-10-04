class_name DefenceTower
extends RefCounted

## A tower standing on the grid. The def holds the shape; this holds where it is, which
## way it faces, and when it may fire again.
##
## `rot` turns the footprint in 4 steps and `aim` turns the barrel in 16, independently.
## They are different rotations of different things: the footprint decides which cells are
## occupied and where the cannon sits, the aim decides where the shot goes. One value
## could not express both, which is also why the node has a separate Barrel sprite.

var id: int = 0
var def: TowerDef
## Top-left cell of the footprint. Matches ActivePiece.pos meaning the bounding box corner,
## so the two grids speak one language.
var origin: Vector2i = Vector2i.ZERO
var rot: int = 0
var aim: int = 0
var cooldown: float = 0.0
## Shots granted by line clears, spent before the cooldown is consulted.
var free_shots: int = 0

func _init(i: int, d: TowerDef, at: Vector2i, r: int = 0, a: int = 0) -> void:
	id = i
	def = d
	origin = at
	rot = r & 3
	aim = a & 15

## The footprint's cells in GRID coordinates. Typed so callers can index them as
## Vector2i without a cast.
func cells() -> Array[Vector2i]:
	var out: Array[Vector2i] = []
	for c in def.cells(rot):
		out.append(origin + c)
	return out

func extent() -> Vector2i:
	return def.extent(rot)

## Where the next shot appears, in pixels relative to the field's top-left.
func muzzle_pos(cell_px: int) -> Vector2:
	var m: Dictionary = def.muzzle(rot, aim)
	return (Vector2(origin) + m["offset"]) * float(cell_px)

func can_fire() -> bool:
	return bool(def.muzzle(rot, aim)["ok"]) and (free_shots > 0 or cooldown <= 0.0)

## Spends a shot. Free shots go first, and spending one does NOT touch the cooldown --
## a line clear can only ever add damage, never delay the next automatic shot.
func spend_shot(interval: float) -> String:
	if free_shots > 0:
		free_shots -= 1
		return "line_clear"
	cooldown = interval
	return "cooldown"
