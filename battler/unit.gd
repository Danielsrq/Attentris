class_name BattlerUnit
extends Node2D

## One fighter: its own target, its own step, its own damage.
##
## Under a standing ATTACK-MOVE order, which is the whole behaviour. Walk along the lane; if
## an enemy comes within aggro, walk at it; if the path is blocked, rotate and go round it;
## swing when within reach. Flanking is not a feature in here -- it is what "go round it"
## looks like from the outside.
##
## The movement and targeting GEOMETRY lives in Steering and BattleField, as pure static
## functions over positions and radii. This file is the state machine and the animation and
## holds no geometry of its own, which is why the awkward parts of crowd movement are
## testable with no scene tree.
##
## The three Area2D children are NO LONGER READ. Radial targeting with ranked candidates, a
## swept step test and neighbour gathering are all computations over a list, which an overlap
## query answers badly, so detection moved to BattleField. The nodes and their wiring check
## remain only so that deleting them can be a separate, reviewable step.

enum State { WALK, ATTACK, DYING }

## Frame centre to feet, in pixels.
##
## `position` is the unit's GROUND point -- what collides, what sorts by depth, what the lane
## confines. The sprite hangs above it by this much, so a 64px walk frame and a 192px swing
## stand on the same spot and the body does not drift when the clip changes.
const FEET_BELOW_CENTRE := 30.0

var def: UnitDef
var rules: BattlerRules
var hp: int = 0
## Authoritative liveness, set SYNCHRONOUSLY the moment hp hits zero.
##
## Everything checks this and not is_instance_valid(): a corpse stays a perfectly valid
## instance for the whole death animation -- 6 frames at 0.2s is 1.2 seconds of swinging at a
## body while the next enemy walks past.
var alive: bool = true
## Mirrored from the def.
##
## Duplicated deliberately. BattleField's entire contract is `alive`, `position` and `team`,
## which is what lets its queries be tested against a three-line stand-in; reaching through
## `def.team` would drag UnitDef into that contract.
var team: int = UnitDef.TEAL

## The enemy this unit is committed to, or null. Held across frames by hysteresis.
var _target: BattlerUnit = null
## Which way this unit turns when blocked: -1 or +1, assigned at spawn and never changed.
##
## An IMMUTABLE bias is what makes the steering work, and it is stronger than choosing a side
## at each block. On a head-on obstruction both sides score alike, so a unit that re-picks
## every frame cancels its own deflections and vibrates in place. Worse, re-picking leaves the
## search as greedy hill-climbing, which cannot get off a wall at all, because getting off
## means moving temporarily FURTHER from the goal. A constant bias makes it wall-following.
##
## Alternated between spawns, so two units blocked by the same body go round opposite sides
## and a crowd splits instead of funnelling into one queue.
var _bias: int = 1
## Last frame's actual displacement. What the facing is drawn from.
var _velocity: Vector2 = Vector2.ZERO
var _facing: int = SpriteSheet.RIGHT
var _clip: StringName = &"walk"

var _state: int = State.WALK
var _ledger: HitLedger
var _contact_pending: bool = false
var _sprite: AnimatedSprite2D
var _hurt: Area2D
var _hit: Area2D
var _sight: Area2D
var _hurt_shape: CollisionShape2D
var _hit_shape: CollisionShape2D

func setup(d: UnitDef, r: BattlerRules, bias: int = 1) -> void:
	def = d
	rules = r
	team = d.team
	hp = d.hp
	_bias = 1 if bias >= 0 else -1
	_facing = d.facing
	_ledger = HitLedger.new(d.iframe_sec)
	_resolve()
	_build_shapes()
	_wire_layers()
	_sprite.sprite_frames = SpriteSheet.build(d, r.frame_seconds)
	_sprite.centered = true
	_sprite.texture_filter = CanvasItem.TEXTURE_FILTER_NEAREST
	# Hung above the ground point, so `position` means the feet.
	_sprite.position = Vector2(0.0, -FEET_BELOW_CENTRE)
	if not _sprite.frame_changed.is_connected(_on_frame_changed):
		_sprite.frame_changed.connect(_on_frame_changed)
	for problem in wiring_errors(_hurt, _hit, _sight, d):
		push_error("BattlerUnit %s: %s" % [d.id, problem])
	_play(&"walk")

## Looked up on demand rather than with @onready, because a unit is set up immediately after
## add_child and its parent may not be in the tree yet -- in which case every @onready is
## still null, silently.
func _resolve() -> void:
	_sprite = get_node_or_null("Sprite")
	_hurt = get_node_or_null("Hurtbox")
	_hit = get_node_or_null("Hitbox")
	_sight = get_node_or_null("Sight")
	_hurt_shape = _hurt.get_node_or_null("Shape") if _hurt != null else null
	_hit_shape = _hit.get_node_or_null("Shape") if _hit != null else null

# ------------------------------------------------------------------------------
# Tick
# ------------------------------------------------------------------------------

## One step of simulation, called BY THE ARENA in spawn order rather than from
## _physics_process.
##
## Collision-aware movement needs a defined order -- whoever steps first takes the contested
## space -- and a node cannot choose when it runs relative to its siblings. Being driven also
## means the whole crowd can be stepped by hand in a test, with no physics frames at all.
func advance(delta: float, field: BattleField) -> void:
	if def == null or not alive:
		return
	_ledger.tick(delta)
	_target = _pick_target(field)
	if _target != null and position.distance_to(_target.position) <= def.reach_px:
		_attack_step()
	else:
		_move_step(delta, field)

## The four target rules, in priority order. There is no reachability test and no occupancy
## count: an unreachable target is handled by the steering walking around it.
func _pick_target(field: BattleField) -> BattlerUnit:
	# 1. Fight what you are touching. Also what stops a unit orbiting a packed target while
	#    something else stands right next to it.
	var touching := field.nearest_enemy(position, team, def.reach_px, self) as BattlerUnit
	if touching != null:
		return touching
	# 2. Hysteresis. Without it two near-equidistant enemies swap a unit back and forth every
	#    frame and it closes on neither.
	if _target != null and is_instance_valid(_target) and _target.alive:
		if position.distance_to(_target.position) <= rules.aggro_px * rules.target_keep_factor:
			return _target
	# 3. Otherwise the nearest enemy in aggro. 4. Otherwise none, and the lane flow decides.
	return field.nearest_enemy(position, team, rules.aggro_px, self) as BattlerUnit

func _move_step(delta: float, field: BattleField) -> void:
	if _state != State.WALK:
		_state = State.WALK
		_play(&"walk")
	var desired := field.flow_at(position, team)
	if _target != null:
		desired = _target.position - position
	_velocity = field.step(position, desired, rules.walk_speed * delta, _bias, self,
		rules.turn_step_deg, rules.turn_max_deg, rules.min_step_scale)
	position += _velocity
	_face(_velocity)

func _attack_step() -> void:
	if _state != State.ATTACK:
		_enter_attack()
	_velocity = Vector2.ZERO
	if not _contact_pending:
		return
	_contact_pending = false
	if _target != null and is_instance_valid(_target) and _target.alive:
		_target.take_damage(def.damage, get_instance_id())

func take_damage(amount: int, from: int) -> void:
	if not alive or not _ledger.can_hit(from):
		return
	_ledger.register(from)
	hp -= amount
	if hp <= 0:
		_die()

## Order matters, and the first step is the one that has to be synchronous.
func _die() -> void:
	alive = false                                   # before anything deferred
	_hit_shape.set_deferred("disabled", true)
	_hurt_shape.set_deferred("disabled", true)
	# A corpse sinks behind living siblings, but not below 1: z_index is relative to the
	# parent, and the arena paints an opaque pane fill in its own _draw, so 0 and below is
	# where a sprite goes to become invisible rather than to go behind something.
	z_index = maxi(1, z_index - 1)
	_state = State.DYING
	_play(&"death")
	_sprite.animation_finished.connect(queue_free, CONNECT_ONE_SHOT)

## Facing is set ONCE here, from the target, and then left alone for the whole swing.
##
## Re-facing mid-attack would be worse than merely wrong. Changing a clip restarts its frame
## counter, so a unit whose target drifts across the diagonal would restart its swing every
## few frames and never reach the contact frame -- landing no hits at all while looking
## perfectly busy. That is the same shape as the contact-frame-0 trap.
func _enter_attack() -> void:
	_state = State.ATTACK
	_contact_pending = false
	if _target != null and is_instance_valid(_target):
		_facing = Steering.facing_from(_target.position - position, _facing)
	_play(&"attack")

## Facing follows motion, but only while walking -- see _enter_attack.
func _face(v: Vector2) -> void:
	if _state != State.WALK:
		return
	var f := Steering.facing_from(v, _facing)
	if f == _facing:
		return
	_facing = f
	_play(_clip)

func _play(clip: StringName) -> void:
	_clip = clip
	var anim := StringName("%s_%d" % [clip, _facing])
	if _sprite.animation != anim:
		_sprite.play(anim)

## Flags the contact; it is CONSUMED in advance() rather than acted on here.
##
## frame_changed fires from the sprite's own _process, outside the simulation step. Dealing
## damage from here would interleave it with rendering non-deterministically and make it
## untestable. The flag makes damage land on a simulation step, always.
##
## A looping attack clip emits one contact per loop, which gives the attack cadence for free
## -- but a contact frame of 0 would never fire, because play() on a clip already sitting at
## frame 0 emits no signal at all.
func _on_frame_changed() -> void:
	if _state != State.ATTACK or def == null:
		return
	if _sprite.frame == def.contact_frame(&"attack"):
		_contact_pending = true

# ------------------------------------------------------------------------------
# The Area2D children, kept only until their deletion can be reviewed on its own
# ------------------------------------------------------------------------------

func _build_shapes() -> void:
	_rect(_hurt_shape, Vector2(22.0, 46.0), Vector2.ZERO)
	var ahead := float(def.heading)
	_rect(_hit_shape, Vector2(def.reach_px, 40.0), Vector2(ahead * def.reach_px * 0.5, 0.0))
	var s: CollisionShape2D = _sight.get_node_or_null("Shape") if _sight != null else null
	_rect(s, Vector2(def.sight_px, 30.0), Vector2(ahead * def.sight_px * 0.5, 0.0))

func _rect(shape: CollisionShape2D, size: Vector2, at: Vector2) -> void:
	if shape == null:
		return
	var r := RectangleShape2D.new()
	r.size = size
	shape.shape = r
	shape.position = at

func _wire_layers() -> void:
	_hurt.collision_layer = def.own_layer()
	_hurt.collision_mask = 0
	_hurt.monitoring = false
	_hurt.monitorable = true
	for a in [_hit, _sight]:
		a.collision_layer = 0
		a.collision_mask = def.enemy_layer()
		a.monitoring = true
		a.monitorable = false

## Everything about the areas that would otherwise fail silently. Returns one string per
## fault. Static and pure enough to test without a tree: an Area2D owns its PhysicsServer RID
## from construction, so these properties are readable off a bare Area2D.new().
static func wiring_errors(hurt: Area2D, hit: Area2D, sight: Area2D, d: UnitDef) -> Array:
	var out: Array = []
	if hurt == null or hit == null or sight == null:
		return ["a required area is missing"]
	if hurt.collision_layer != d.own_layer():
		out.append("hurtbox must sit on its own team's layer")
	if hurt.collision_mask != 0:
		out.append("hurtbox must mask nothing -- it is found, it does not look")
	if not hurt.monitorable:
		out.append("hurtbox must be monitorable or nothing can find it")
	if hurt.monitoring:
		out.append("hurtbox must not monitor -- that inverts the damage direction")
	for pair in [["hitbox", hit], ["sight", sight]]:
		var a: Area2D = pair[1]
		if a.collision_mask != d.enemy_layer():
			out.append("%s must mask the enemy layer" % pair[0])
		if a.collision_layer != 0:
			out.append("%s must sit on no layer -- it is a query, not a target" % pair[0])
		if not a.monitoring:
			out.append("%s must monitor or it never detects anything" % pair[0])
		if a.monitorable:
			out.append("%s must not be monitorable or swings discover each other" % pair[0])
	for pair2 in [["hurtbox", hurt], ["hitbox", hit], ["sight", sight]]:
		var sh: CollisionShape2D = (pair2[1] as Area2D).get_node_or_null("Shape")
		if sh == null or sh.shape == null:
			out.append("%s has no shape -- the other silent never-overlaps" % pair2[0])
	return out
