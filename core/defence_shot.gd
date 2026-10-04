class_name DefenceShot
extends RefCounted

## An energy ball. Direction is an Aim index, never a vector, so reflection is integer
## arithmetic and the shot stays exactly on the lattice -- see Aim.
##
## `bounces` is the damage budget as well as the range limit: one bounce is one hit, so a
## shot that spends its bounces on walls has nothing left for bodies. That is the whole
## skill of the thing, and it is why `max_bounces` has to exceed the three hits a kill
## costs.

var id: int = 0
var pos: Vector2 = Vector2.ZERO
var dir: int = 0
## Cached velocity, rewritten only on a bounce. The one place a velocity is ever written.
var vel: Vector2 = Vector2.ZERO
var bounces: int = 0
var age: float = 0.0
var source: int = 0
## True until the shot has cleared EVERY tower footprint. Not just its own: with the
## muzzle pushed out to the hull, an adjacent tower could still contain the spawn point,
## and a shot that bounced off its neighbour the instant it appeared would be unreadable.
## The consequence, accepted deliberately: a cannon buried in a cluster tunnels out of the
## whole cluster rather than rattling inside it.
var ignore_towers: bool = true

func _init(i: int, at: Vector2, direction: int, speed: float, from: int) -> void:
	id = i
	pos = at
	dir = direction & 15
	vel = Aim.velocity(dir, speed)
	source = from

## Turns the shot and recomputes the cached velocity. Speed is passed back in rather than
## stored so it can never drift from the rules.
func set_dir(d: int, speed: float) -> void:
	dir = d & 15
	vel = Aim.velocity(dir, speed)
