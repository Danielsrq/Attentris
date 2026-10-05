class_name Steering
extends RefCounted

## Local navigation, as pure static geometry: no node, no physics server, no scene tree.
##
## A deliberate return to how the Tetris half is built. The hard parts of RTS-ish movement
## are all geometry -- refusing a gap narrower than the body, escaping a concave trap, not
## vibrating against a wall -- and geometry is both easy to get subtly wrong and cheap to
## test exhaustively. None of it needs an Area2D.
##
## The movement rule in full:
##
##   desired = toward the target, or along the lane flow when there is no target
##   for each angle in the escalation ladder (committed side first):
##       if a step along it is free, take it
##   otherwise hold
##
## Four details carry the whole thing, and each of them is load-bearing. They are documented
## on the function that implements them.

## How far off `desired` a step may land before it counts as having gone AROUND something.
## Tight, because it only has to exclude float noise on an unobstructed step.
const STRAIGHT_DOT := 0.999

# ------------------------------------------------------------------------------
# The blocked test
# ------------------------------------------------------------------------------

## Whether a body of `radius` can travel from `from` to `to` touching nothing and staying
## inside `bounds`.
##
## A SWEPT CIRCLE, not a raycast. A ray cast from the body's CENTRE passes happily between
## two obstacles whose gap is narrower than the body, so a unit would commit to a squeeze and
## then wedge at the opening -- which looks like a pathing bug and is really a test that
## measured the wrong thing. Taking the distance from each obstacle centre to the swept
## SEGMENT is exact for circles and refuses gaps the body cannot fit through.
##
## `bounds` is the region a body CENTRE may occupy, already inset by the caller. Rejecting a
## step that would leave it, rather than clamping the step, is what makes the lane edge
## behave exactly like another body: the same rotation search then slides along the wall with
## no special case for walls.
static func step_free(from: Vector2, to: Vector2, radius: float,
		obstacles: PackedVector2Array, obstacle_radius: float, bounds: Rect2) -> bool:
	if not bounds.has_point(to):
		return false
	var reach := radius + obstacle_radius
	var reach_sq := reach * reach
	for o in obstacles:
		if _segment_distance_squared(from, to, o) < reach_sq:
			return false
	return true

# ------------------------------------------------------------------------------
# The escalation ladder
# ------------------------------------------------------------------------------

## Straight ahead, then the committed side EXHAUSTED from shallow to steep, and only then the
## other side.
##
## Exhausting one side is the whole of wall-following, and interleaving the two sides breaks
## it. That is not a judgement call -- the interleaved version was written first and measured:
## a unit pressed against a line of eight bodies 28px apart oscillated in one scallop of the
## wall for 400 frames and closed 30px of a 120px gap, because the wall's clearance contour is
## wavy and the unit's offset from its goal makes the two sides score differently. A rung on
## the far side came up before the committed side had escalated far enough to climb out, so
## each frame undid the last. Exhausting the committed side first makes the unit always deflect
## FURTHER the same way, which is what walks it to the end of the wall.
##
## Shallow before steep on that side, so it takes the gentlest deflection that actually works
## rather than swinging wide.
##
## The far side is still reachable, after every rung of the committed side has failed, so a
## unit whose chosen side is genuinely walled in is never trapped by its own commitment.
##
## The ladder runs PAST 90 degrees on purpose. Those angles move away from the goal, which is
## precisely what is needed to escape a concave trap: getting out of a pocket means first
## going further from the target.
static func angles(step_deg: float, max_deg: float, bias: int) -> PackedFloat32Array:
	var out := PackedFloat32Array([0.0])
	if step_deg <= 0.0:
		return out
	var side := 1.0 if bias >= 0 else -1.0
	for sign_of in [side, -side]:
		var a := step_deg
		while a <= max_deg + 0.001:
			out.append(a * sign_of)
			a += step_deg
	return out

## How much of the full step length to attempt at a given deflection.
##
## Never zero. A unit at full deflection still has to creep, or it freezes against the wall
## it is trying to walk around and the escalation was pointless. `cos` gives a natural taper
## -- a nearly-straight step moves nearly the full distance -- and `min_scale` is the floor
## that keeps a sideways or backward step alive.
static func step_scale(angle_deg: float, min_scale: float) -> float:
	return maxf(min_scale, cos(deg_to_rad(angle_deg)))

## The displacement to actually take this frame, or ZERO when fully enclosed.
##
## `distance` is the full step length for the frame (speed * delta). `bias` is the committed
## turn side, -1 or +1; see `deflected` for why it has to be committed at all.
static func choose_step(from: Vector2, desired: Vector2, distance: float, radius: float,
		obstacles: PackedVector2Array, obstacle_radius: float, bounds: Rect2,
		bias: int, step_deg: float, max_deg: float, min_scale: float) -> Vector2:
	if distance <= 0.0 or desired.is_zero_approx():
		return Vector2.ZERO
	var dir := desired.normalized()
	for angle in angles(step_deg, max_deg, bias):
		var step := dir.rotated(deg_to_rad(angle)) * distance * step_scale(angle, min_scale)
		if step_free(from, from + step, radius, obstacles, obstacle_radius, bounds):
			return step
	return Vector2.ZERO

## Whether a chosen step had to go AROUND something rather than straight at the goal.
##
## This is what drives turn-side commitment, and commitment is not a polish detail. On a
## head-on block the two sides score identically, so an uncommitted choice flips every frame
## and the unit vibrates in place instead of moving.
##
## It also does something less obvious and more important. Picking the locally-better
## rotation is hill-climbing on distance-to-target, and hill-climbing gets trapped: against a
## wall four or five bodies wide, NO small rotation gets you nearer, because escaping means
## going temporarily further away. Holding one side converts the algorithm from greedy
## gradient descent into WALL-FOLLOWING, which escapes any finite obstacle. It is the
## difference between working and pressing into the back of your own line forever.
static func deflected(step: Vector2, desired: Vector2) -> bool:
	if step.is_zero_approx() or desired.is_zero_approx():
		return true
	return step.normalized().dot(desired.normalized()) < STRAIGHT_DOT

# ------------------------------------------------------------------------------
# Drawing direction
# ------------------------------------------------------------------------------

## Which LPC row a velocity should be drawn facing.
##
## Dominant axis wins. The diagonal tie breaks to the HORIZONTAL because a unit crossing the
## lane reads better walking sideways than facing the camera, and because the lane's long
## axis is the one a player is tracking.
static func facing_from(v: Vector2, fallback: int) -> int:
	if v.is_zero_approx():
		return fallback
	if absf(v.x) >= absf(v.y):
		return SpriteSheet.RIGHT if v.x > 0.0 else SpriteSheet.LEFT
	return SpriteSheet.DOWN if v.y > 0.0 else SpriteSheet.UP

# ------------------------------------------------------------------------------

## Squared distance from point `p` to segment `a`-`b`. Squared throughout so the swept test
## never takes a square root.
static func _segment_distance_squared(a: Vector2, b: Vector2, p: Vector2) -> float:
	var ab := b - a
	var len_sq := ab.length_squared()
	if len_sq <= 0.0000001:
		return a.distance_squared_to(p)
	var t := clampf((p - a).dot(ab) / len_sq, 0.0, 1.0)
	return (a + ab * t).distance_squared_to(p)
