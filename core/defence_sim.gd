class_name DefenceSim
extends RefCounted

## The defence minigame. Pure logic: no Node, no Area2D, no rendering, so the whole loop
## is testable headlessly and the view is a projection of this rather than a second
## authority.
##
## Collision is analytic and owned here on purpose. Area2D would be a SECOND source of
## truth that disagrees with this one -- different shapes, and overlaps resolved on the
## physics step with results a frame stale, which cannot give the exact contact point a
## bounce needs the same tick it happens.
##
## Everything is in PIXELS relative to the field's top-left. `cell_px` is the one number
## shared with the view, passed in rather than stored in DefenceRules so it cannot drift
## from ViewLayout.cell.
##
## Tick order is fixed and load-bearing:
##   enemies move -> shots move (substepped, resolving walls, towers, bodies) ->
##   ignore_towers clears -> ages and culls -> garbage window -> spawn -> fire
## Shots move after enemies so a body that walked into a resting shot is hit this tick
## rather than next. Firing is last so a shot never moves on the tick it is created.

const EPS := 0.001

var rules: DefenceRules
var cell_px: int
var field: DefenceField
var towers: TowerSet
var garbage: GarbageBuffer
var rng := RandomNumberGenerator.new()

## Typed, so `for t in placed` yields a DefenceTower rather than a Variant. That is not
## cosmetic: an untyped element makes every `:=` off it un-inferable, and the compiler
## stops checking the calls entirely.
var placed: Array[DefenceTower] = []
var enemies: Array[DefenceEnemy] = []
var shots: Array[DefenceShot] = []

var kills: int = 0
var breaches: int = 0
var shots_fired: int = 0

var _spawn_left: float = 0.0
var _next_id: int = 0

func _init(r: DefenceRules, ts: TowerSet, px: int, seed_v: int = 0) -> void:
	rules = r
	towers = ts
	cell_px = px
	field = DefenceField.new(r.cols, r.rows)
	garbage = GarbageBuffer.new(r, seed_v)
	rng.seed = seed_v
	_spawn_left = r.spawn_interval

## Monotonic ids, never reused, so a view's id -> node map cannot alias a freed entity.
func _id() -> int:
	_next_id += 1
	return _next_id

# ------------------------------------------------------------------------------
# Building
# ------------------------------------------------------------------------------

## Whether a tower may be built here. Cost is not consulted yet: `Resources.can_afford`
## is the eventual gate and towers are free during mechanical testing, so the only limits
## are the grid and `max_towers` -- a count ceiling standing in for a price, without which
## a playtest fills all 90 slots and teaches nothing.
func can_build(def: TowerDef, rot: int, origin: Vector2i) -> bool:
	if placed.size() >= rules.max_towers:
		return false
	return field.can_place(def, rot, origin)

func build(def: TowerDef, rot: int, origin: Vector2i, aim: int, events: Array) -> DefenceTower:
	if not can_build(def, rot, origin):
		return null
	var t := DefenceTower.new(_id(), def, origin, rot, aim)
	placed.append(t)
	field.place(t)
	events.append(DefenceEvent.new(DefenceEvent.Type.TOWER_BUILT, {"id": t.id, "tower": t}))
	return t

func tower_at(cell: Vector2i) -> DefenceTower:
	var id := field.at(cell.x, cell.y)
	if id <= 0:
		return null
	for t in placed:
		if t.id == id:
			return t
	return null

func remove_tower(t: DefenceTower, events: Array) -> void:
	if t == null:
		return
	field.remove(t)
	placed.erase(t)
	events.append(DefenceEvent.new(DefenceEvent.Type.TOWER_REMOVED, {"id": t.id}))

## One free shot per tower, from a Tetris line clear. Does not touch any cooldown.
func grant_free_shots(n: int = 1) -> void:
	for t in placed:
		t.free_shots += n

# ------------------------------------------------------------------------------
# Tick
# ------------------------------------------------------------------------------

func tick(delta: float) -> Array:
	var events: Array = []
	_move_enemies(delta, events)
	_move_shots(delta, events)
	_age_shots(delta, events)
	_tick_garbage(delta, events)
	_spawn(delta, events)
	_fire(delta, events)
	return events

func _move_enemies(delta: float, events: Array) -> void:
	for i in range(enemies.size() - 1, -1, -1):
		var e: DefenceEnemy = enemies[i]
		# The body's own speed, not the rule's: that is what lets enemy types differ later
		# without touching this loop.
		e.pos.x -= e.step(delta)
		# Breach when the centre reaches the wall. The body is then despawned and converted
		# into a garbage row -- failing here is what makes the Tetris half harder.
		if e.pos.x <= 0.0:
			events.append(DefenceEvent.new(DefenceEvent.Type.ENEMY_BREACHED,
				{"id": e.id, "pos": e.pos}))
			enemies.remove_at(i)
			breaches += 1
			garbage.touch()

func _move_shots(delta: float, events: Array) -> void:
	var n := maxi(1, rules.substeps)
	var sub := delta / float(n)
	for _s in n:
		for i in range(shots.size() - 1, -1, -1):
			var sh: DefenceShot = shots[i]
			sh.pos += sh.vel * sub
			# One bounce per substep at most, so a corner cannot double-charge the budget.
			var surface := _resolve(sh, events)
			if surface != "":
				sh.bounces += 1
				events.append(DefenceEvent.new(DefenceEvent.Type.SHOT_BOUNCED,
					{"id": sh.id, "surface": surface, "pos": sh.pos}))
			if sh.ignore_towers and not _in_any_tower(sh):
				sh.ignore_towers = false

## Resolves the first surface this shot is overlapping, flipping its direction and pushing
## it clear. Returns the surface name, or "" for a clean step.
##
## Pushing out is not cosmetic. At 4px per tick against a 6px radius a shot stays
## overlapping for three or four ticks, so without de-penetration it would spend its whole
## bounce budget on one wall. Clearing by radius + EPS means the next step cannot
## re-penetrate, because 4px of travel cannot cross 6px of clearance.
func _resolve(sh: DefenceShot, events: Array) -> String:
	var r := rules.projectile_radius
	var size := field.pixel_size(cell_px)

	# Walls. Both axes can hit at once in a corner, which is an exact reversal.
	var hit_x := false
	var hit_y := false
	if sh.pos.x - r < 0.0:
		sh.pos.x = r + EPS
		hit_x = true
	elif sh.pos.x + r > size.x:
		sh.pos.x = size.x - r - EPS
		hit_x = true
	if sh.pos.y - r < 0.0:
		sh.pos.y = r + EPS
		hit_y = true
	elif sh.pos.y + r > size.y:
		sh.pos.y = size.y - r - EPS
		hit_y = true
	if hit_x or hit_y:
		var d := sh.dir
		if hit_x and hit_y:
			d = Aim.reverse(d)
		elif hit_x:
			d = Aim.reflect_x(d)
		else:
			d = Aim.reflect_y(d)
		sh.set_dir(d, rules.projectile_speed)
		return "wall"

	# Enemies before towers: a body standing inside a tower footprint should be hit, not
	# shielded by it. Enemies pass through towers for now, so that overlap is possible.
	for e in enemies:
		var h := _box_hit(sh.pos, r, e.box_min(), e.box_max())
		if not bool(h["hit"]):
			continue
		_apply_bounce(sh, h)
		e.hp -= rules.hit_damage
		events.append(DefenceEvent.new(DefenceEvent.Type.ENEMY_HIT,
			{"id": e.id, "damage": rules.hit_damage, "hp_left": e.hp, "pos": e.pos}))
		if e.is_dead():
			events.append(DefenceEvent.new(DefenceEvent.Type.ENEMY_KILLED,
				{"id": e.id, "pos": e.pos}))
			enemies.erase(e)
			kills += 1
		return "enemy"

	if sh.ignore_towers:
		return ""
	for t in placed:
		for c in t.cells():
			var box := field.cell_box(c.x, c.y, cell_px)
			var h2 := _box_hit(sh.pos, r, box.position, box.end)
			if not bool(h2["hit"]):
				continue
			_apply_bounce(sh, h2)
			return "tower"
	return ""

func _apply_bounce(sh: DefenceShot, h: Dictionary) -> void:
	sh.pos += h["push"]
	var axis: int = h["axis"]
	var d := sh.dir
	if axis == 2:
		d = Aim.reverse(d)
	elif axis == 0:
		d = Aim.reflect_x(d)
	else:
		d = Aim.reflect_y(d)
	sh.set_dir(d, rules.projectile_speed)

## Circle against an axis-aligned box, treating the circle as a box of half-extent r.
##
## Returns {hit, axis, push}: axis 0 flips x, 1 flips y, 2 is an exact corner. The axis
## chosen is the one with the SHALLOWER overlap, which is the face the shot came through;
## picking the deeper one would reflect it back the way it was already going.
static func _box_hit(c: Vector2, r: float, lo: Vector2, hi: Vector2) -> Dictionary:
	var dx := minf(c.x + r, hi.x) - maxf(c.x - r, lo.x)
	var dy := minf(c.y + r, hi.y) - maxf(c.y - r, lo.y)
	if dx <= 0.0 or dy <= 0.0:
		return {"hit": false}
	var centre := (lo + hi) * 0.5
	var push := Vector2.ZERO
	var axis := 0
	if absf(dx - dy) < EPS:
		axis = 2
		push = Vector2(
			(dx + EPS) * (1.0 if c.x >= centre.x else -1.0),
			(dy + EPS) * (1.0 if c.y >= centre.y else -1.0))
	elif dx < dy:
		axis = 0
		push = Vector2((dx + EPS) * (1.0 if c.x >= centre.x else -1.0), 0.0)
	else:
		axis = 1
		push = Vector2(0.0, (dy + EPS) * (1.0 if c.y >= centre.y else -1.0))
	return {"hit": true, "axis": axis, "push": push}

func _in_any_tower(sh: DefenceShot) -> bool:
	var r := rules.projectile_radius
	for t in placed:
		for c in t.cells():
			var box := field.cell_box(c.x, c.y, cell_px)
			if bool(_box_hit(sh.pos, r, box.position, box.end)["hit"]):
				return true
	return false

func _age_shots(delta: float, events: Array) -> void:
	for i in range(shots.size() - 1, -1, -1):
		var sh: DefenceShot = shots[i]
		sh.age += delta
		# Bounces checked first so a shot that spent its last bounce reports that rather
		# than "life", which is the more useful of the two for tuning.
		if sh.bounces >= rules.max_bounces:
			events.append(DefenceEvent.new(DefenceEvent.Type.SHOT_EXPIRED,
				{"id": sh.id, "reason": "bounces"}))
			shots.remove_at(i)
		elif sh.age >= rules.projectile_life:
			events.append(DefenceEvent.new(DefenceEvent.Type.SHOT_EXPIRED,
				{"id": sh.id, "reason": "life"}))
			shots.remove_at(i)

func _tick_garbage(delta: float, events: Array) -> void:
	var rows := garbage.tick(delta)
	if rows > 0:
		events.append(DefenceEvent.new(DefenceEvent.Type.GARBAGE_FLUSHED,
			{"rows": rows, "holes": garbage.holes_for(rows)}))

func _spawn(delta: float, events: Array) -> void:
	if rules.spawn_interval <= 0.0:
		return
	_spawn_left -= delta
	if _spawn_left > 0.0:
		return
	_spawn_left += rules.spawn_interval
	var lane := rng.randi_range(0, rules.rows - 1)
	var size := field.pixel_size(cell_px)
	var e := DefenceEnemy.new(_id(),
		Vector2(size.x - rules.enemy_half_extents.x, (float(lane) + 0.5) * float(cell_px)),
		rules.enemy_hp, lane, rules.enemy_half_extents, rules.enemy_speed)
	enemies.append(e)
	events.append(DefenceEvent.new(DefenceEvent.Type.ENEMY_SPAWNED, {"id": e.id, "enemy": e}))

func _fire(delta: float, events: Array) -> void:
	for t in placed:
		if t.cooldown > 0.0:
			t.cooldown -= delta
		if not t.can_fire():
			continue
		var cause := t.spend_shot(rules.fire_interval)
		var sh := DefenceShot.new(_id(), t.muzzle_pos(cell_px), t.aim,
			rules.projectile_speed, t.id)
		shots.append(sh)
		shots_fired += 1
		events.append(DefenceEvent.new(DefenceEvent.Type.SHOT_FIRED,
			{"id": sh.id, "shot": sh, "tower_id": t.id, "cause": cause}))
