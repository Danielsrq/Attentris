class_name DefenceEnemy
extends RefCounted

## One enemy, drifting left. Position is continuous in pixels relative to the field's
## top-left; `lane` is only where it started, because nothing moves y yet.
##
## Collides as an axis-aligned BOX however round the art is. A circle would reflect along
## the centre-to-centre normal and produce an arbitrary angle, leaving the 22.5 lattice on
## the first bounce and making every ricochet after it unplannable -- see Aim.

var id: int = 0
var pos: Vector2 = Vector2.ZERO
var hp: int = 0
var lane: int = 0
## Pixels per second, leftward. Carried per body rather than read from DefenceRules each
## tick, so a fast scout and a slow brute need no change to the sim -- the rule is only the
## default a type starts from. Same reasoning as `half_extents`, which was already per-body
## for exactly this reason.
var speed: float = 40.0
## HALF-EXTENTS: the distance from `pos` to each edge, so (12, 12) is a 24x24 box
## centred on the enemy. Centre-plus-half-extents rather than a Rect2 because that makes
## box_min/box_max one subtraction and one addition, and keeps the box symmetric about the
## position the sim actually moves.
var half_extents: Vector2 = Vector2(12.0, 12.0)

func _init(i: int, at: Vector2, hit_points: int, in_lane: int, h: Vector2,
		spd: float = 40.0) -> void:
	id = i
	pos = at
	hp = hit_points
	lane = in_lane
	half_extents = h
	speed = spd

## How far this body moves in `delta`. On the entity so the sim never has to ask the rules
## what a particular enemy's speed is.
func step(delta: float) -> float:
	return speed * delta

func box_min() -> Vector2:
	return pos - half_extents

func box_max() -> Vector2:
	return pos + half_extents

func is_dead() -> bool:
	return hp <= 0
