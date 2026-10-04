class_name Aim
extends RefCounted

## The sixteen directions a cannon can face, as an integer index 0..15.
##
## 0 is +x, and each step is 22.5 degrees counter-clockwise ON SCREEN. Board space is
## y-down, like Board, so the stored vectors carry a negated y and direction 4 points up.
##
## A direction is an INDEX, never a Vector2. That is the whole design, and it is what
## makes bouncing exact: reflecting off an axis-aligned surface flips one component, and
## on this lattice that is integer arithmetic on the index --
##
##     reflect_x(d) = (8 - d) & 15      a vertical surface: left wall, tower side
##     reflect_y(d) = (16 - d) & 15     a horizontal surface: floor, ceiling, tower top
##     reverse(d)   = (d + 8) & 15      an exact corner, both at once
##
## So after six bounces the direction is still exactly one of the sixteen, and the speed
## is exactly unchanged because the magnitude is never recomputed. Only POSITION
## accumulates float error, and at 4px per tick over a 3s lifetime that is sub-pixel.
##
## Reflecting off a CIRCLE would use the centre-to-centre normal and produce an arbitrary
## angle, which leaves the lattice immediately and makes every later bounce unplannable.
## That is why enemies collide as axis-aligned boxes however round their art looks.

const COUNT := 16
const STEP := TAU / float(COUNT)

## Unit vectors per direction, built once at class load rather than transcribed. Exact by
## construction, and no trig at call time -- a projectile only asks on a bounce anyway,
## because it caches its velocity until the next one.
static var _dirs: PackedVector2Array = PackedVector2Array()

static func _static_init() -> void:
	_dirs.resize(COUNT)
	for d in COUNT:
		# Negative angle because y grows downward: this turns counter-clockwise on screen.
		_dirs[d] = Vector2.RIGHT.rotated(-float(d) * STEP)

## The unit vector for a direction. Any integer is accepted and wrapped, so callers never
## have to normalise before asking.
static func vector(dir: int) -> Vector2:
	return _dirs[dir & 15]

## Velocity for a direction at a given speed. The only place a projectile's velocity is
## ever written, which is why it cannot drift.
static func velocity(dir: int, speed: float) -> Vector2:
	return _dirs[dir & 15] * speed

## Off a vertical surface: x flips, y is unchanged.
static func reflect_x(dir: int) -> int:
	return (8 - dir) & 15

## Off a horizontal surface: y flips, x is unchanged.
static func reflect_y(dir: int) -> int:
	return (16 - dir) & 15

## Straight back the way it came, for an exact corner hit.
static func reverse(dir: int) -> int:
	return (dir + 8) & 15

## Turn by `steps`, positive counter-clockwise. What the aim keys do.
static func turn(dir: int, steps: int) -> int:
	return (dir + steps) & 15

## Degrees, for the overlay. Whole numbers for the eight cardinals and diagonals, and a
## half for the rest, so "67.5" is correct rather than rounded.
static func label(dir: int) -> String:
	var deg := float(dir & 15) * 22.5
	return "%.0f" % deg if is_equal_approx(deg, roundf(deg)) else "%.1f" % deg
