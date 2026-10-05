@tool
extends Node2D

## The right pane: draws it, spawns into the lane, and sequences the units. Nothing else.
##
## It SEQUENCES movement but does not arbitrate fights. Collision-aware stepping needs a
## defined order -- whoever steps first takes the contested space -- and a node cannot choose
## when it runs relative to its siblings, so the walk has to be driven from here. Which enemy
## to attack and how much damage lands stay inside the unit, and the geometry stays in
## BattleField and Steering.
##
## In `battler/` rather than `view/` on purpose: everything in `view/` draws a pure sim held
## elsewhere, and that is still true of every file there. Here the state IS the nodes, so a
## file in `view/` would be a view only by position in the tree.
##
## Draws and spawns in ABSOLUTE screen coordinates, exactly like BoardView, and never touches
## its own transform.
##
## It carried the pane offset on `position` at first, which looked tidier and was wrong for a
## reason that only shows up in the editor: this is a @tool script, so writing `position` at
## editor load MODIFIES the edited scene. main.tscn then came up dirty every time it was
## opened, and F5 -- which saves modified scenes before running -- had a save to do that the
## user never asked for. No other view node in the project writes its transform.
##
## The hazard the transform was avoiding is real but belongs to the DRAWING: offsetting only
## _draw while leaving entity positions pane-local is how the deleted tower defence put every
## entity over the Tetris board. Keeping both in the same absolute space removes the mismatch
## instead of hiding it behind a transform.
##
## It owns the whole PANE, not just the lane. The lane is 256px of a 600px pane, and a node
## that owned only the lane could only draw the lane -- which on a dark background reads as a
## right half that failed to load.

const UNIT := preload("res://battler/unit.tscn")

const PANE_BG := Color(0.07, 0.08, 0.10)
const PANE_EDGE := Color(0.85, 0.87, 0.95, 0.55)
## The lane reads as a lit band inside the pane rather than as the pane itself, so the right
## half is obviously present even with nothing on the field.
const LANE_BG := Color(0.13, 0.15, 0.19)
const LANE_EDGE := Color(0.85, 0.87, 0.95, 0.22)
const TEXT := Color(0.74, 0.78, 0.86)
const TEXT_DIM := Color(0.55, 0.58, 0.66)
const TEAL := Color(0.35, 0.80, 0.78)
const RED := Color(0.88, 0.42, 0.42)
const FRONT := Color(1.0, 0.92, 0.55, 0.65)
const TITLE := "BATTLER"

## Lowest z_index a living unit may take.
##
## z_index is RELATIVE to the parent by default, and this node paints an opaque pane fill in
## its own _draw. So a unit on a negative z draws BEHIND that fill and is simply never seen --
## it spawns, walks, fights and dies correctly and invisibly, which reads as a broken spawn
## rule. 2 leaves room for _die()'s decrement to still land above the fill.
const Z_FLOOR := 2

var rules: BattlerRules
var layout: ViewLayout
## The queryable world. Owns the roster and all the geometry; see BattleField.
var field: BattleField
var rng := RandomNumberGenerator.new()

## Breaches banked since the last poll. main drains this rather than being signalled, which
## keeps the no-signals rule that the rest of main's wiring follows.
var _breaches: int = 0
var _hole: int = 4
var _red_left: float = 0.0
var _font: Font
## Teams queued to enter, oldest first. See _drain_pending.
var _pending: Array[int] = []
## Spawns so far, only to alternate the turn bias.
var _spawned: int = 0

func setup(r: BattlerRules, l: ViewLayout, seed_v: int = 0) -> void:
	rules = r
	layout = l
	_font = ThemeDB.fallback_font
	rng.seed = seed_v
	_hole = rng.randi_range(0, 9)
	_breaches = 0
	_pending.clear()
	_spawned = 0
	for c in get_children():
		c.queue_free()
	# Absolute, and the transform is deliberately left alone -- see the class doc.
	field = BattleField.new(l.lane_rect, r.body_radius)
	_red_left = r.next_red_delay(rng)
	queue_redraw()

# ------------------------------------------------------------------------------
# Spawning
# ------------------------------------------------------------------------------

## A Tetris clear worth a fighter. Called from main._dispatch.
func spawn_teal() -> void:
	_pending.append(UnitDef.TEAL)

func spawn_red() -> void:
	_pending.append(UnitDef.RED)

## Places everything queued that can find room, and KEEPS the rest for next frame.
##
## Queueing rather than dropping is the whole point. Units cannot stack any more, so a busy
## entry edge is normal, and a fighter that silently failed to appear because the edge was
## crowded would be indistinguishable from a broken spawn rule -- the same class of bug as a
## unit spawning behind the pane fill: correct in the simulation, absent on screen.
func _drain_pending() -> void:
	if _pending.is_empty():
		return
	var still: Array[int] = []
	for team in _pending:
		if not _try_place(team):
			still.append(team)
	_pending = still

func _try_place(team: int) -> bool:
	# The editor preview pushes synthetic clears through the same dispatch, and instancing
	# Area2Ds into the editor's physics world is not something a preview should do.
	if Engine.is_editor_hint() or rules == null or field == null:
		return true
	var x := field.spawn_x(team)
	for y in _entry_depths():
		var at := Vector2(x, y)
		if field.has_room(at, rules.body_radius):
			_instance(UnitDef.of(team), at)
			return true
	return false

## Candidate entry depths: one per body across the lane, starting from a random one so
## arrivals spread across the lane instead of always stacking from the same edge.
func _entry_depths() -> PackedFloat32Array:
	var out := PackedFloat32Array()
	var step := rules.body_radius * 2.0
	var usable := field.centre_bounds
	var slots := maxi(1, int(usable.size.y / step))
	var first := rng.randi_range(0, slots - 1)
	for i in slots:
		out.append(usable.position.y + step * (0.5 + float((first + i) % slots)))
	return out

func _instance(d: UnitDef, at: Vector2) -> void:
	var u: BattlerUnit = UNIT.instantiate()
	add_child(u)
	# Alternating turn bias, so two units blocked by the same body go round opposite sides and
	# a crowd splits rather than funnelling into one queue.
	_spawned += 1
	u.setup(d, rules, 1 if _spawned % 2 == 0 else -1)
	u.position = at
	field.track(u)

# ------------------------------------------------------------------------------
# The tick
# ------------------------------------------------------------------------------

func _physics_process(delta: float) -> void:
	if Engine.is_editor_hint() or rules == null or field == null:
		return
	_red_left -= delta
	if _red_left <= 0.0:
		_red_left = rules.next_red_delay(rng)
		spawn_red()
	_drain_pending()
	# In spawn order, on a snapshot: a unit can free itself part-way through a pass.
	for u in field.units.duplicate():
		if u != null and is_instance_valid(u):
			u.advance(delta, field)
	_sweep()
	_prune()
	_depth_sort()

## Retires anything that walked off its far end. A red reaching the near edge is a breach and
## is banked for main to turn into garbage; a teal reaching the far edge just leaves.
func _sweep() -> void:
	for u in field.units.duplicate():
		if u == null or not is_instance_valid(u) or not u.alive:
			continue
		if not field.past_far_end(u.position, u.team):
			continue
		if u.team == UnitDef.RED:
			_breaches += 1
		field.forget(u)
		u.queue_free()

## Drops freed units from the roster. A corpse stays on it deliberately -- it is filtered out
## of every query by `alive` -- but once queue_free has run the reference is dangling, and
## over a long run the array would only grow.
func _prune() -> void:
	var keep: Array = []
	for u in field.units:
		if u != null and is_instance_valid(u):
			keep.append(u)
	if keep.size() != field.units.size():
		field.units = keep

## Lower on the screen draws in front: the standard 2.5D read. Recomputed every frame rather
## than fixed at spawn, because a unit's depth is now something it walks around in.
func _depth_sort() -> void:
	var top := field.lane.position.y
	for u in field.units:
		if u == null or not is_instance_valid(u) or not u.alive:
			continue
		u.z_index = Z_FLOOR + int(u.position.y - top)

# ------------------------------------------------------------------------------
# What main reads
# ------------------------------------------------------------------------------

## Hands over the banked breaches and clears the counter.
func take_breaches() -> int:
	var n := _breaches
	_breaches = 0
	return n

## One hole column per row, walking at most one column per flush. 0.1 is the messiness
## measured in real Zenith replays, and holes[0] is the BOTTOM-most row for push_garbage --
## irrelevant while every entry matches, but it has to stay right.
func take_holes(rows: int) -> Array:
	if rng.randf() < rules.hole_shift_chance:
		_hole = clampi(_hole + (1 if rng.randf() < 0.5 else -1), 0, 9)
	var out: Array = []
	for _i in rows:
		out.append(_hole)
	return out

func living() -> int:
	return field.living_count() if field != null else 0

## Living units on one side. Corpses are excluded, same as everywhere else, so the readout
## agrees with what the units themselves consider a target.
func alive_on(team: int) -> int:
	return field.living_count(team) if field != null else 0

## Units waiting for room at the entry edge. Drawn, because a queue that never drains would
## otherwise be invisible.
func pending() -> int:
	return _pending.size()

# ------------------------------------------------------------------------------
# Drawing: the pane and the lane inside it. Units are nodes and draw themselves.
# ------------------------------------------------------------------------------

func _draw() -> void:
	if layout == null:
		return
	var pane := layout.right_pane
	if pane.size.x <= 0.0:
		return
	draw_rect(pane, PANE_BG)
	draw_rect(pane, PANE_EDGE, false, 1.0)
	draw_string(_font, pane.position + Vector2(8.0, 16.0), TITLE,
		HORIZONTAL_ALIGNMENT_LEFT, -1, 12, TEXT_DIM)

	var lane := layout.lane_rect
	draw_rect(lane, LANE_BG)
	draw_rect(lane, LANE_EDGE, false, 1.0)

	if Engine.is_editor_hint() or field == null:
		return
	# The readout. The whole premise of a zero-input minigame is that its state can be read
	# peripherally, and a crowd with flanking is much busier to look at than a single file --
	# so the front line has to be legible without watching the fight.
	var teal := alive_on(UnitDef.TEAL)
	var red := alive_on(UnitDef.RED)
	_draw_front_line(lane, teal, red)
	var tally := "%d  vs  %d" % [teal, red]
	if pending() > 0:
		tally = "%s  (+%d)" % [tally, pending()]
	draw_string(_font, pane.position + Vector2(pane.size.x - 128.0, 16.0), tally,
		HORIZONTAL_ALIGNMENT_RIGHT, 120.0, 12, TEXT)
	draw_rect(Rect2(pane.position + Vector2(pane.size.x - 140.0, 7.0), Vector2(7.0, 7.0)), TEAL)
	draw_rect(Rect2(pane.position + Vector2(pane.size.x - 8.0, 7.0), Vector2(7.0, 7.0)), RED)
	if teal + red == 0:
		draw_string(_font, Vector2(pane.position.x, lane.position.y - 10.0),
			"clear a quad or spin to send a fighter",
			HORIZONTAL_ALIGNMENT_CENTER, pane.size.x, 12, TEXT_DIM)

## A tick where the two sides meet: midway between the furthest-advanced of each.
##
## One mark, so "who is winning" is a glance rather than a count. With only one side present
## it marks that side's spearhead instead, which is still the thing worth knowing.
func _draw_front_line(lane: Rect2, teal: int, red: int) -> void:
	if teal + red == 0:
		return
	var tip_teal := -INF
	var tip_red := INF
	for u in field.living():
		if u.team == UnitDef.TEAL:
			tip_teal = maxf(tip_teal, u.position.x)
		else:
			tip_red = minf(tip_red, u.position.x)
	var x := 0.0
	if teal > 0 and red > 0:
		x = (tip_teal + tip_red) * 0.5
	elif teal > 0:
		x = tip_teal
	else:
		x = tip_red
	x = clampf(x, lane.position.x, lane.end.x)
	draw_line(Vector2(x, lane.position.y - 6.0), Vector2(x, lane.end.y + 6.0), FRONT, 1.0)
