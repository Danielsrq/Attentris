@tool
extends Node2D

## The defence board on screen: the grid and the build ghost in `_draw()`, and the towers,
## enemies and shots as instanced scenes.
##
## That split is deliberate and is the one place this project departs from its own grain.
## Everything else here is immediate-mode -- BoardView's own doc explains why coloured
## squares are better off as `draw_rect`. But a `_draw()` call cannot have a PNG swapped
## into it, so anything that will carry art has to be a node. The dividing line is simply
## "will this ever be art?": the grid, gridlines, frame and ghost never will, so they stay
## drawn; towers, enemies and shots will, so they are scenes with Sprite2D children.
##
## Entity nodes are POOLED. Not for speed -- fifteen live shots would be fine instanced
## per frame -- but because reuse removes the whole class of bug where a node is freed
## while an event loop still holds its id. `visible = false` and reuse is less code than
## guarding against freed instances.
##
## Lifecycle arrives as DefenceEvents; positions are read off the sim every frame. The
## other way round would mean an event per shot per tick, or diffing dictionaries looking
## for what appeared.

const TOWER_SCENE := preload("res://defence_game/scenes/tower.tscn")
const ENEMY_SCENE := preload("res://defence_game/scenes/enemy.tscn")
const SHOT_SCENE := preload("res://defence_game/scenes/shot.tscn")

const BG := Color(0.07, 0.08, 0.10)
const GRID := Color(1, 1, 1, 0.07)
const GRID_STRONG := Color(1, 1, 1, 0.16)
const FRAME := Color(0.85, 0.87, 0.95, 0.9)
const GHOST_OK := Color(0.55, 0.85, 1.0, 0.35)
const GHOST_BAD := Color(1.0, 0.45, 0.45, 0.30)
const GHOST_EDGE := Color(1, 1, 1, 0.55)
const AIM_LINE := Color(1.0, 0.85, 0.35, 0.8)
## The cannon cell, called out separately from the rest of the footprint. On a square mask
## the body is identical in all four rotations, so without this the only thing that moves
## when you turn the block is a dot on the hull -- and turning appears to do nothing.
const GHOST_CANNON := Color(1.0, 0.85, 0.35, 0.55)
const TEXT_DIM := Color(0.55, 0.58, 0.66)

## Every fourth column is brighter, so distance is countable at a glance -- the same trick
## BoardView uses on rows.
const GRID_EMPHASIS := 4

var sim: DefenceSim
var cursor: BuildCursor
var layout: ViewLayout
## Drawn only while the defence board has focus, so the player can see which half they are
## driving without a label.
var focused: bool = false
## Set by main when the InputMap is missing actions this board needs. Drawn over the grid,
## because a defence board that silently ignores every key is otherwise indistinguishable
## from a broken one -- the keys are declared in project.godot, not in code, so "nothing
## happens" is a configuration state rather than a bug.
var input_warning: String = ""

var _font: Font
var _towers: Dictionary = {}     ## sim id -> Node2D
var _enemies: Dictionary = {}
var _shots: Dictionary = {}
var _free_towers: Array[Node2D] = []
var _free_enemies: Array[Node2D] = []
var _free_shots: Array[Node2D] = []

func setup(s: DefenceSim, c: BuildCursor, l: ViewLayout) -> void:
	sim = s
	cursor = c
	layout = l
	_font = ThemeDB.fallback_font
	# The NODE carries the board's offset, so everything below is in field-local pixels --
	# the same coordinates DefenceSim works in. Entity nodes are children, so they inherit
	# it; drawing happens in the same local space. Offsetting only the drawing, as this did
	# at first, put the grid in the pane and every tower, enemy and shot over the Tetris
	# board instead.
	position = l.defence_origin(s.rules.cols, s.rules.rows)
	_release_all()
	queue_redraw()

# ------------------------------------------------------------------------------
# Events and per-frame sync
# ------------------------------------------------------------------------------

## Consumes one tick's DefenceEvents. Creation and release only -- movement is `sync`.
func handle_events(events: Array) -> void:
	for e in events:
		match e.type:
			DefenceEvent.Type.TOWER_BUILT:
				var n := _acquire(TOWER_SCENE, _free_towers)
				n.setup(layout.cell)
				n.apply(e.data["tower"])
				_towers[int(e.data["id"])] = n
			DefenceEvent.Type.TOWER_REMOVED:
				_release(_towers, int(e.data["id"]), _free_towers)
			DefenceEvent.Type.ENEMY_SPAWNED:
				var n2 := _acquire(ENEMY_SCENE, _free_enemies)
				n2.setup(e.data["enemy"])
				_enemies[int(e.data["id"])] = n2
			DefenceEvent.Type.ENEMY_KILLED:
				_release(_enemies, int(e.data["id"]), _free_enemies)
			DefenceEvent.Type.ENEMY_BREACHED:
				# The stub plays if the scene supplies an animation; either way the node
				# comes back to the pool, because the sim has already despawned the body.
				var hit: Node2D = _enemies.get(int(e.data["id"]), null)
				if hit != null:
					hit.play_breach()
				_release(_enemies, int(e.data["id"]), _free_enemies)
			DefenceEvent.Type.SHOT_FIRED:
				var n3 := _acquire(SHOT_SCENE, _free_shots)
				n3.setup()
				_shots[int(e.data["id"])] = n3
			DefenceEvent.Type.SHOT_EXPIRED:
				_release(_shots, int(e.data["id"]), _free_shots)

## Copies this frame's positions out of the sim. Called every frame, after handle_events.
func sync() -> void:
	if sim == null:
		return
	for t in sim.placed:
		var n: Node2D = _towers.get(t.id, null)
		if n != null:
			n.apply(t)
	for e in sim.enemies:
		var n2: Node2D = _enemies.get(e.id, null)
		if n2 != null:
			n2.apply(e)
	for s in sim.shots:
		var n3: Node2D = _shots.get(s.id, null)
		if n3 != null:
			n3.apply(s)
	queue_redraw()

func _acquire(scene: PackedScene, pool: Array[Node2D]) -> Node2D:
	var n: Node2D
	if pool.is_empty():
		n = scene.instantiate()
		add_child(n)
	else:
		n = pool.pop_back()
	n.visible = true
	return n

func _release(live: Dictionary, id: int, pool: Array[Node2D]) -> void:
	var n: Node2D = live.get(id, null)
	if n == null:
		return
	live.erase(id)
	n.visible = false
	pool.append(n)

func _release_all() -> void:
	for d in [_towers, _enemies, _shots]:
		for id in (d as Dictionary).keys():
			var n: Node2D = d[id]
			n.visible = false
	_towers.clear()
	_enemies.clear()
	_shots.clear()
	# Everything already instanced goes back to its pool rather than being freed, so a
	# restart does not churn the scene tree.
	_free_towers.clear()
	_free_enemies.clear()
	_free_shots.clear()
	for c in get_children():
		if c is Node2D:
			c.visible = false
			match c.name.get_slice("@", 0):
				"Tower": _free_towers.append(c)
				"Enemy": _free_enemies.append(c)
				"Shot": _free_shots.append(c)

# ------------------------------------------------------------------------------
# Drawing: grid, frame, ghost
# ------------------------------------------------------------------------------

func _draw() -> void:
	if sim == null or layout == null:
		return
	var size := sim.field.pixel_size(layout.cell)
	if size.x <= 0.0:
		return
	if not layout.defence_fits(sim.rules.cols, sim.rules.rows):
		# Too narrow a window to hold the grid. Say so where the pane is rather than
		# painting a 600px board over the Tetris field.
		var r := layout.right_pane
		draw_string(_font, Vector2(r.position.x - position.x, r.position.y - position.y + 40.0),
			"window too narrow
for an %dx%d board" % [sim.rules.cols, sim.rules.rows],
			HORIZONTAL_ALIGNMENT_CENTER, maxf(80.0, r.size.x), 13, TEXT_DIM)
		return
	draw_rect(Rect2(Vector2.ZERO, size), BG)
	_grid(size)
	_ghost()
	# The frame last, so the ghost cannot paint over the boundary a shot bounces off.
	draw_rect(Rect2(Vector2.ZERO, size), FRAME if focused else Color(FRAME, 0.35), false, 2.0)
	_hint(size)

## Whatever the board most needs to say. The input warning wins: nothing else matters if
## the keys are not wired.
func _hint(size: Vector2) -> void:
	if input_warning != "":
		var box := Rect2(Vector2(10.0, size.y * 0.5 - 30.0), Vector2(size.x - 20.0, 60.0))
		draw_rect(box, Color(0.10, 0.06, 0.06, 0.92))
		draw_rect(box, Color(1.0, 0.45, 0.45, 0.7), false, 1.0)
		draw_string(_font, Vector2(0.0, size.y * 0.5 - 8.0), "INPUT MAP INCOMPLETE",
			HORIZONTAL_ALIGNMENT_CENTER, size.x, 15, Color(1.0, 0.55, 0.55))
		draw_string(_font, Vector2(0.0, size.y * 0.5 + 12.0), input_warning,
			HORIZONTAL_ALIGNMENT_CENTER, size.x, 12, Color(0.85, 0.70, 0.70))
		return
	if sim.placed.is_empty() and not Engine.is_editor_hint():
		draw_string(_font, Vector2(0.0, size.y * 0.5), "TAB to build",
			HORIZONTAL_ALIGNMENT_CENTER, size.x, 15, TEXT_DIM)

func _grid(size: Vector2) -> void:
	var cs := float(layout.cell)
	for x in sim.rules.cols + 1:
		var px := float(x) * cs
		var col := GRID_STRONG if x % GRID_EMPHASIS == 0 else GRID
		draw_line(Vector2(px, 0.0), Vector2(px, size.y), col, 1.0)
	for y in sim.rules.rows + 1:
		var py := float(y) * cs
		var col2 := GRID_STRONG if y % GRID_EMPHASIS == 0 else GRID
		draw_line(Vector2(0.0, py), Vector2(size.x, py), col2, 1.0)

## The build ghost: the footprint it would occupy, and a line showing where it would fire.
##
## Translucent fill plus an outline is the same visual language as the Tetris ghost piece,
## which is the cheapest way to make the two halves feel like one game.
func _ghost() -> void:
	if cursor == null or not focused:
		return
	var d := cursor.def()
	if d == null:
		return
	var cs := float(layout.cell)
	var ok := cursor.can_build(sim) if sim != null else false
	var fill := GHOST_OK if ok else GHOST_BAD
	var cannon_cell := cursor.origin + d.cannon(cursor.rot)
	for c in cursor.cells():
		var at := Vector2(c) * cs
		draw_rect(Rect2(at, Vector2(cs, cs)), GHOST_CANNON if c == cannon_cell else fill)
		draw_rect(Rect2(at, Vector2(cs, cs)), GHOST_EDGE, false, 1.0)
	# The cannon cell gets a ring as well as a brighter fill, so it reads at a glance and
	# at any cell size -- a tint alone is easy to miss against the grid.
	var hub := (Vector2(cannon_cell) + Vector2(0.5, 0.5)) * cs
	draw_arc(hub, cs * 0.3, 0.0, TAU, 20, AIM_LINE, 2.0)
	draw_circle(hub, cs * 0.1, AIM_LINE)

	# Where the shot would leave, and which way. Drawn even when the spot is illegal, so
	# the player can aim first and then find a home for it.
	var m: Dictionary = d.muzzle(cursor.rot, cursor.aim)
	if not bool(m["ok"]):
		return
	var from := (Vector2(cursor.origin) + Vector2(m["offset"])) * cs
	var away := Aim.vector(cursor.aim)
	# From the cannon hub to the hull, then onward: the kink shows the shot crossing its
	# own body, which is the thing that is otherwise invisible about an interior cannon.
	draw_line(hub, from, Color(AIM_LINE, 0.35), 1.0)
	var tip := from + away * cs * 3.0
	draw_line(from, tip, AIM_LINE, 2.0)
	# A head, so the direction is unambiguous at 22.5-degree steps.
	var wing := away.rotated(PI * 0.82) * cs * 0.4
	draw_line(tip, tip + wing, AIM_LINE, 2.0)
	draw_line(tip, tip + wing.reflect(away), AIM_LINE, 2.0)
	draw_circle(from, 3.5, AIM_LINE)
	draw_string(_font, tip + Vector2(6.0, -4.0), "%s deg" % Aim.label(cursor.aim),
		HORIZONTAL_ALIGNMENT_LEFT, -1, 11, Color(AIM_LINE, 0.9))
