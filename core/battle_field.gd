class_name BattleField
extends RefCounted

## The world the units move in: where the lane is, which way it flows, who is alive, and
## whether a given step fits. Every method is a question; none of them is a decision.
##
## It does NOT own fights. The arena sequences movement because collision-aware stepping
## needs a defined order, and this answers the geometry questions, but which enemy to attack
## and how much damage lands stay with the unit. That split is what keeps a unit's behaviour
## readable in one file.
##
## It holds unit references but only ever reads three things from each: `alive`, `position`
## and `team`. That is the whole contract, so every query here is testable with a three-line
## stand-in and no scene tree -- which is the point of having this class at all rather than
## leaving the queries on the arena Node2D.
##
## `units` is in SPAWN order, and that is the order the arena steps them in. Sequential
## movement gives the first mover the contested space; that is harmless, but it has to be
## deterministic, and spawn order is stable where tree order is only incidentally so.

## The lane in screen pixels.
var lane: Rect2
## Every unit shares one body size; there is one unit type per side.
var body_radius: float = 16.0
## The region a body CENTRE may occupy.
##
## Inset by the body radius ACROSS the flow, so a body never half-leaves the lane, and left
## wide open ALONG it, because a unit has to be able to walk out of the far end -- a teal
## retiring at the right edge, a red breaching at the left. Confining both axes would seal
## the lane and nothing could ever leave.
##
## Computed here rather than inside Steering so that Steering stays a dumb geometric
## predicate and every piece of lane-SHAPE knowledge lives in this one class.
var centre_bounds: Rect2
var units: Array = []

func _init(lane_rect: Rect2 = Rect2(), radius: float = 16.0) -> void:
	lane = lane_rect
	body_radius = radius
	_recompute_bounds()

func _recompute_bounds() -> void:
	# Generous along the flow: wide enough that x never binds, so the only edges that can
	# reject a step are the ones that should.
	var slack := maxf(lane.size.x, 1024.0)
	centre_bounds = Rect2(
		Vector2(lane.position.x - slack, lane.position.y + body_radius),
		Vector2(lane.size.x + slack * 2.0, maxf(0.0, lane.size.y - body_radius * 2.0)))

# ------------------------------------------------------------------------------
# The roster
# ------------------------------------------------------------------------------

func track(u: Object) -> void:
	if u != null and not units.has(u):
		units.append(u)

func forget(u: Object) -> void:
	units.erase(u)

## Living units, in spawn order. Corpses are excluded everywhere: a dying unit is not a
## target, and it does not block -- it is playing an animation on its way out, and having it
## hold ground would stop a line eroding when it should.
func living() -> Array:
	var out: Array = []
	for u in units:
		if u != null and is_instance_valid(u) and u.alive:
			out.append(u)
	return out

func living_count(team: int = -1) -> int:
	var n := 0
	for u in living():
		if team < 0 or u.team == team:
			n += 1
	return n

# ------------------------------------------------------------------------------
# Geometry
# ------------------------------------------------------------------------------

## Every living body except `ignore`, as bare positions.
##
## Rebuilt per caller rather than snapshotted once per frame. At the tens of units this
## design produces that is a few thousand Vector2 copies a frame, which is free, and it
## cannot go stale or mis-index the way a shared snapshot plus a skip-index can. If the unit
## count ever reaches the hundreds, snapshotting is the obvious first optimisation.
func obstacles(ignore: Object = null) -> PackedVector2Array:
	var out := PackedVector2Array()
	for u in units:
		if u == null or u == ignore or not is_instance_valid(u) or not u.alive:
			continue
		out.append(u.position)
	return out

## Whether a body could stand at `p` without touching anyone or leaving the lane.
##
## A zero-length step, so it reuses exactly the geometry movement uses. Two separate notions
## of "fits" would eventually disagree, and the spawn placement that depends on this one has
## to agree with the stepping that happens immediately afterwards.
func has_room(p: Vector2, radius: float, ignore: Object = null) -> bool:
	return Steering.step_free(p, p, radius, obstacles(ignore), body_radius, centre_bounds)

## The displacement a unit should take this frame. Gathers the obstacles and defers to
## Steering, so callers never assemble geometry themselves.
func step(from: Vector2, desired: Vector2, distance: float, bias: int, ignore: Object,
		step_deg: float, max_deg: float, min_scale: float) -> Vector2:
	return Steering.choose_step(from, desired, distance, body_radius,
		obstacles(ignore), body_radius, centre_bounds, bias, step_deg, max_deg, min_scale)

# ------------------------------------------------------------------------------
# The lane's shape
# ------------------------------------------------------------------------------

## The attack-move direction for a team at a point: where to walk with nothing to fight.
##
## Constant today, because the lane is straight. It takes a POSITION anyway, because that is
## the signature a curved lane needs -- a tangent along a path, or a sample from a flow
## field -- and this plus `centre_bounds` are the only two things in the whole design that
## know the lane is a horizontal rectangle. Making them an implementation swap now costs one
## unused parameter and saves a rewrite later.
func flow_at(_p: Vector2, team: int) -> Vector2:
	return Vector2.LEFT if team == UnitDef.RED else Vector2.RIGHT

## Where a team's units enter, and where the opposing team breaches.
func spawn_x(team: int) -> float:
	return lane.end.x if team == UnitDef.RED else lane.position.x

## Whether a unit has walked off its own far end.
func past_far_end(p: Vector2, team: int) -> bool:
	return p.x <= lane.position.x if team == UnitDef.RED else p.x >= lane.end.x

# ------------------------------------------------------------------------------
# Targeting
# ------------------------------------------------------------------------------

## The nearest living enemy within `radius`, or null.
##
## Ties resolve to the earlier spawn, which keeps a crowded frame deterministic.
func nearest_enemy(from: Vector2, team: int, radius: float, ignore: Object = null) -> Object:
	var best: Object = null
	var best_sq := radius * radius
	for u in units:
		if u == null or u == ignore or not is_instance_valid(u) or not u.alive:
			continue
		if u.team == team:
			continue
		var d := from.distance_squared_to(u.position)
		if d < best_sq:
			best_sq = d
			best = u
	return best
