extends TestHarness

## Auto-battler: the spawn rule, the sheet maths, the hit ledger, and four tree tests.
##
## Almost all the value is in the pure groups. The old tower defence was a headless sim with
## ~250 assertions; this design has no central sim, so combat correctness is split between
## arithmetic that can be unit-tested and behaviour that needs the physics server. The four
## tree tests are deliberately few: they are the only place a flake can enter the suite, and
## the arithmetic above already covers everything they would otherwise re-prove.

func run() -> void:
	test_step_free()
	test_escalation_ladder()
	test_choose_step()
	test_wall_following()
	test_concave_trap()
	test_facing_from()
	test_battle_field_bounds()
	test_battle_field_queries()
	test_spawn_rule()
	test_sheet_geometry()
	test_frame_table_matches_the_art()
	await test_hit_flash()
	await test_clip_startup()
	await test_units_move()
	await test_units_fight()
	await test_arena()

# ------------------------------------------------------------------------------
# The spawn rule
# ------------------------------------------------------------------------------

func test_spawn_rule() -> void:
	group("BattlerRules: which clears send a fighter")

	check(BattlerRules.spawns_unit(make_result(4, SpinDetect.Kind.NONE, &"I", -1, 0)),
		"a quad sends one")
	check(BattlerRules.spawns_unit(make_result(1, SpinDetect.Kind.FULL, &"T", -1, 0)),
		"a T-spin single sends one -- any spin counts")
	check(BattlerRules.spawns_unit(make_result(2, SpinDetect.Kind.MINI, &"S", -1, 0)),
		"so does a mini")
	check(not BattlerRules.spawns_unit(make_result(2, SpinDetect.Kind.NONE, &"O", -1, 0)),
		"a plain double sends nothing")
	check(not BattlerRules.spawns_unit(make_result(3, SpinDetect.Kind.NONE, &"L", -1, 0)),
		"nor a plain triple")
	check(not BattlerRules.spawns_unit(null), "and a null result is not an error")

	# Through the real clear path, so this is not only a test of hand-built fixtures: an
	# I piece dropped into the open column completes four rows.
	var b := make_board(["XXXXXXXXX.", "XXXXXXXXX.", "XXXXXXXXX.", "XXXXXXXXX."])
	var g := make_game(b)
	g.start()
	place(g, &"I", 1, Vector2i(7, g.board.total_height - 4))
	var events: Array = []
	g.lock_piece(events)
	var quad := first_clear(events)
	check(quad != null, "the fixture cleared rows at all")
	if quad != null:
		eq(quad.lines, 4, "   four of them")
		check(BattlerRules.spawns_unit(quad), "   so a real quad sends a fighter")

# ------------------------------------------------------------------------------
# Sheet geometry
# ------------------------------------------------------------------------------

func test_sheet_geometry() -> void:
	group("SpriteSheet: the grid is derived, never declared")

	# The four real sizes in this art set. These lines are why nothing declares a column
	# count: the two oversize sheets DISAGREE, and a shared constant would be silently wrong
	# for one of the two characters.
	eq(SpriteSheet.grid(Vector2i(832, 256), 64), Vector2i(13, 4), "a standard sheet is 13x4")
	eq(SpriteSheet.grid(Vector2i(832, 64), 64), Vector2i(13, 1),
		"hurt.png is one row, so it has no facings")
	eq(SpriteSheet.grid(Vector2i(1536, 768), 192), Vector2i(8, 4),
		"teal's thrust_oversize is 8 columns")
	eq(SpriteSheet.grid(Vector2i(1152, 768), 192), Vector2i(6, 4),
		"   while red's slash_oversize is 6 -- they are not the same")
	eq(SpriteSheet.grid(Vector2i(832, 256), 0), Vector2i.ZERO, "a zero frame size is not a crash")

	eq(SpriteSheet.region(64, 0, 3), Rect2i(0, 192, 64, 64), "frame 0 facing right")
	eq(SpriteSheet.region(64, 8, 3), Rect2i(512, 192, 64, 64), "the last walk frame")
	eq(SpriteSheet.region(192, 7, 1), Rect2i(1344, 192, 192, 192), "and an oversize frame")

	# LPC row order, with the single-row sheet as the general case rather than a special one.
	eq(SpriteSheet.facing_row(SpriteSheet.RIGHT, 4), 3, "right is row 3")
	eq(SpriteSheet.facing_row(SpriteSheet.LEFT, 4), 1, "left is row 1")
	eq(SpriteSheet.facing_row(SpriteSheet.RIGHT, 1), 0, "a one-row sheet collapses every facing")

	# A built SpriteFrames has every clip, under every facing, at the declared length.
	SpriteSheet.clear_cache()
	var d := UnitDef.teal()
	var sf := SpriteSheet.build(d, 0.2)
	for clip in d.clips:
		for facing in [SpriteSheet.LEFT, SpriteSheet.RIGHT]:
			var name := StringName("%s_%d" % [clip, facing])
			check(sf.has_animation(name), "built %s" % name)
			eq(sf.get_frame_count(name), d.frame_count(clip),
				"   with %d frames" % d.frame_count(clip))
	eq(sf.get_animation_loop(&"death_3"), false,
		"death does not loop, or the corpse never finishes and never frees")
	check(sf.get_animation_loop(&"walk_3"), "walk does")
	# Retiming must not need a rebuild, so frame_seconds can be a live knob.
	SpriteSheet.apply_speed(sf, 0.05)
	check(absf(sf.get_animation_speed(&"walk_3") - 20.0) < 0.01, "apply_speed retimes in place")
	eq(SpriteSheet.build(d, 0.2), sf, "and a second build is the shared cached one")

	# Attack speed is a STAT, divided across whatever frames the clip has -- not a consequence
	# of how many the artist drew. Teal thrusts in 8 and red slashes in 6, and both must take
	# the same second, or red kills a third faster for no reason anyone chose.
	eq(SpriteSheet.clip_fps(&"attack", 8, 0.15, 1.0), 8.0,
		"an 8-frame swing runs at 8fps to finish in a second")
	eq(SpriteSheet.clip_fps(&"attack", 6, 0.15, 1.0), 6.0,
		"   and a 6-frame swing at 6fps, so both take exactly as long")
	eq(SpriteSheet.clip_fps(&"walk", 9, 0.15, 1.0), 1.0 / 0.15,
		"walking is still paced per frame, not per cycle")
	eq(SpriteSheet.clip_fps(&"attack", 8, 0.15, 0.0), 1.0 / 0.15,
		"and attack_sec of 0 falls back to the per-frame rate")

	SpriteSheet.clear_cache()
	var paced := SpriteSheet.build(UnitDef.teal(), 0.15, 2.0)
	eq(paced.get_animation_speed(&"attack_3"), 4.0,
		"build paces the attack clip: 8 frames over 2s is 4fps")
	check(absf(paced.get_animation_speed(&"walk_3") - 1.0 / 0.15) < 0.01,
		"   while leaving walk on the per-frame rate")
	SpriteSheet.apply_speed(paced, 0.15, 1.0)
	eq(paced.get_animation_speed(&"attack_3"), 8.0,
		"and apply_speed re-paces it in place, by clip prefix")

## The declared frame counts against the actual art.
##
## Frame counts are declared rather than detected, which is cheap and self-documenting but
## would let a re-export from the LPC generator silently start playing blank frames. This
## closes that: for each clip, the last declared frame must have pixels and the one after it
## must not. Occupancy is a contiguous prefix in every row of every sheet, so "first empty
## frame" is a well-defined boundary.
func test_frame_table_matches_the_art() -> void:
	group("UnitDef: the declared frame counts match the sheets")

	for d in [UnitDef.teal(), UnitDef.red()]:
		for clip in d.clips:
			var spec: Dictionary = d.clips[clip]
			var tex: Texture2D = spec["sheet"]
			var px: int = spec["frame_px"]
			var declared: int = spec["frames"]
			var img := tex.get_image()
			var g := SpriteSheet.grid(Vector2i(tex.get_size()), px)
			var last := img.get_region(SpriteSheet.region(px, declared - 1, 0)).get_used_rect()
			check(last.size != Vector2i.ZERO,
				"%s/%s frame %d has art" % [d.id, clip, declared - 1])
			if declared < g.x:
				var past := img.get_region(SpriteSheet.region(px, declared, 0)).get_used_rect()
				eq(past.size, Vector2i.ZERO,
					"   and frame %d is blank, so %d is the real count" % [declared, declared])

# ------------------------------------------------------------------------------
# The hit ledger
# ------------------------------------------------------------------------------

func test_hit_flash() -> void:
	group("BattlerUnit: the hit flash, which is the only feedback a hit now gives")

	SpriteSheet.clear_cache()
	var r := BattlerRules.testing()
	var f := BattleField.new(Rect2(0.0, 0.0, 600.0, 256.0), r.body_radius)
	var u := _spawn(UnitDef.teal(), r, f, Vector2(100.0, 128.0))
	var sprite: AnimatedSprite2D = u.get_node("Sprite")

	eq(sprite.modulate.a, 1.0, "a fresh unit is fully opaque")
	check(not u.flashing(), "   and not flashing")

	u.take_damage(1)
	check(u.flashing(), "taking damage starts a flash")
	eq(u.hp, UnitDef.teal().hp - 1, "   and the damage still lands")

	# Quantised, and never so faint that the unit disappears -- several may flash at once.
	var seen := {}
	var in_range := true
	for _i in 10:
		u.advance(0.005, f)
		var a := sprite.modulate.a
		if a < r.flash_alpha_min - 0.001 or a > 1.001:
			in_range = false
		seen[snappedf(a, 0.01)] = true
	check(in_range, "   the alpha stays between flash_alpha_min and full")
	check(seen.size() > 1, "   and actually varies, rather than sitting on one value")

	# A hit landing mid-flash must not extend it, or a unit under attack from several sides
	# flickers forever and reads as a rendering fault.
	var left_before := u.flashing()
	u.take_damage(1)
	check(left_before and u.flashing(), "a second hit mid-flash does not restart it")

	# It ends, and it ends at full opacity rather than wherever the jitter happened to stop.
	for _i in 80:
		u.advance(0.01, f)
	check(not u.flashing(), "the flash finishes")
	eq(sprite.modulate.a, 1.0, "   and leaves the sprite fully opaque")

	# No invulnerability: damage is limited by the swing animation, not by a window, so two
	# calls in a row both land. This is the assertion that would catch i-frames coming back
	# by accident.
	var before := u.hp
	u.take_damage(3)
	u.take_damage(3)
	eq(u.hp, before - 6, "there is no immunity window -- both hits count")

	u.queue_free()
	await physics_frames(2)

const UNIT_SCENE := preload("res://battler/unit.tscn")
const ARENA_SCRIPT := preload("res://battler/arena.gd")

## A live arena in the suite's own tree. Its lane is the real one from ViewLayout, so the
## depth the tests spawn into is the depth the game uses.
func _arena(r: BattlerRules) -> Node2D:
	var a := Node2D.new()
	a.set_script(ARENA_SCRIPT)
	add_child(a)
	var l := ViewLayout.new(make_board([]), Vector2(1280.0, 720.0), 30)
	a.setup(r, l, 99)
	return a

## Fast timings, so a fight resolves in a handful of steps rather than seconds of real time.
## attack_sec has to come down too: it OVERRIDES frame_seconds for the swing, so leaving it
## at the default second would pace the fight regardless of how fast the other clips run.
func _fast_rules() -> BattlerRules:
	var r := BattlerRules.testing()
	r.frame_seconds = 0.02
	r.attack_sec = 0.12
	r.walk_speed = 200.0
	return r

func test_clip_startup() -> void:
	group("BattlerUnit: both teams actually START animating")

	# Red faces LEFT, and "walk_1" is the first animation SpriteSheet builds, so assigning the
	# SpriteFrames already left `animation` sitting on that exact name with nothing playing.
	# A name-only guard then skipped the opening play() and red slid across the lane on a
	# frozen frame until its first fight changed the clip and unstuck it. Teal faces RIGHT,
	# wanted "walk_3", and never hit the case -- which is why it presented as one team only.
	SpriteSheet.clear_cache()
	var r := _fast_rules()
	var f := BattleField.new(Rect2(0.0, 0.0, 600.0, 256.0), r.body_radius)
	var depth := 100.0
	for d in [UnitDef.teal(), UnitDef.red()]:
		var u := _spawn(d, r, f, Vector2(300.0, depth))
		depth += 60.0
		var sp: AnimatedSprite2D = u.get_node("Sprite")
		eq(sp.animation, StringName("walk_%d" % d.facing),
			"%s starts on the walk clip for its own facing" % d.id)
		check(sp.is_playing(), "   and is PLAYING it, rather than frozen on frame 0")
	for u in f.units:
		if is_instance_valid(u):
			u.queue_free()
	await physics_frames(2)

func _spawn(d: UnitDef, r: BattlerRules, f: BattleField, at: Vector2) -> BattlerUnit:
	var u: BattlerUnit = UNIT_SCENE.instantiate()
	# Parented to the SUITE, so the runner's queue_free takes the units with it and a later
	# suite never sees them.
	add_child(u)
	u.setup(d, r)
	u.position = at
	f.track(u)
	return u

## One simulation step for everyone, in roster order -- what the arena does per frame.
func _step(f: BattleField, delta: float = 0.0166) -> void:
	for u in f.units.duplicate():
		if u != null and is_instance_valid(u):
			u.advance(delta, f)

func test_units_move() -> void:
	group("BattlerUnit: attack-move, with no scene tree driving it")

	# Movement needs no physics frames at all now: the arena sequences advance(), so a test
	# can step the crowd by hand. Only the animation-timed DAMAGE still needs the tree.
	SpriteSheet.clear_cache()
	var r := _fast_rules()
	var f := BattleField.new(Rect2(0.0, 0.0, 600.0, 256.0), r.body_radius)

	var lone := _spawn(UnitDef.teal(), r, f, Vector2(50.0, 128.0))
	var x0 := lone.position.x
	for _i in 10:
		_step(f)
	check(lone.position.x > x0, "a teal unit with nothing in aggro attack-moves right")
	eq(lone.position.y, 128.0, "   straight along the lane, since nothing deflects it")

	var red := _spawn(UnitDef.red(), r, f, Vector2(550.0, 60.0))
	var rx0 := red.position.x
	for _i in 10:
		_step(f)
	check(red.position.x < rx0, "and a red unit attack-moves left")

	# Aggro pulls a unit OFF the flow line and onto an enemy at a different depth, which is
	# the whole difference between this and the old single file.
	var chaser := _spawn(UnitDef.teal(), r, f, Vector2(300.0, 200.0))
	var bait := _spawn(UnitDef.red(), r, f, Vector2(340.0, 140.0))
	var gap0 := chaser.position.distance_to(bait.position)
	var dy0 := absf(chaser.position.y - bait.position.y)
	for _i in 6:
		_step(f)
	check(chaser.position.distance_to(bait.position) < gap0,
		"a unit closes on an enemy within aggro")
	check(absf(chaser.position.y - bait.position.y) < dy0,
		"   including across the lane's depth, not just along it")

	# Bodies cannot occupy the same ground. The guarantee is the step test, so it holds for
	# every unit on the field at every step rather than being a tendency.
	var crowd: Array = []
	for i in 6:
		crowd.append(_spawn(UnitDef.teal(), r, f, Vector2(40.0, 40.0 + 34.0 * float(i))))
	var overlapped := 0
	for _i in 40:
		_step(f)
		for u in f.living():
			for v in f.living():
				if u != v and u.position.distance_to(v.position) < r.body_radius * 2.0 - 0.01:
					overlapped += 1
	eq(overlapped, 0, "no two bodies ever overlap, at any step")

	# And nobody leaves the lane across its depth.
	var escaped := 0
	for u in f.living():
		if u.position.y < f.centre_bounds.position.y - 0.01 				or u.position.y > f.centre_bounds.end.y + 0.01:
			escaped += 1
	eq(escaped, 0, "   and nobody is pushed out through the lane's edge")

	for u in f.units:
		if is_instance_valid(u):
			u.queue_free()
	await physics_frames(2)

func test_units_fight() -> void:
	group("BattlerUnit: meeting, hitting, and dying")

	SpriteSheet.clear_cache()
	var r := _fast_rules()
	var f := BattleField.new(Rect2(0.0, 0.0, 600.0, 256.0), r.body_radius)

	# Placed just outside reach of each other, so they close and engage.
	var teal := _spawn(UnitDef.teal(), r, f, Vector2(280.0, 128.0))
	var red := _spawn(UnitDef.red(), r, f, Vector2(320.0, 128.0))

	# The tree is still needed here, and only here: the contact frame comes from the sprite's
	# own animation, so damage lands on real frames even though movement does not.
	var died := false
	for _i in 400:
		await physics_frames(1)
		_step(f)
		if not red.alive or not teal.alive:
			died = true
			break
	check(died, "a fight resolves rather than stalling forever")
	check(teal.hp < UnitDef.teal().hp or red.hp < UnitDef.red().hp,
		"   with real damage dealt")

	# A corpse must be untargetable AND must stop blocking. `alive` is the synchronous half;
	# the shape disable is deferred, which is exactly why the flag has to carry the truth.
	var corpse := red if not red.alive else teal
	check(not corpse.alive, "the loser is marked dead synchronously")
	eq(corpse.hp <= 0, true, "   with its HP spent")
	corpse.take_damage(99)
	eq(corpse.hp <= 0, true, "and further damage to a corpse does nothing")
	# Asked from the SURVIVOR's side: the corpse is the only enemy it has, and must not be
	# offered as one. Asking from the corpse's own team would return the survivor and prove
	# nothing.
	var survivor := teal if teal.alive else red
	eq(f.nearest_enemy(survivor.position, survivor.team, 400.0, survivor), null,
		"   and it is no longer anybody's nearest enemy")
	check(f.has_room(corpse.position, r.body_radius, null),
		"   nor does it hold ground, so the line can erode")

	for u in [teal, red]:
		if is_instance_valid(u):
			u.queue_free()
	await physics_frames(2)

func test_arena() -> void:
	group("BattlerArena: spawning into a crowd, sweeping, and banking breaches")

	SpriteSheet.clear_cache()
	var r := _fast_rules()
	# Long enough that the timer never fires during the test, so every unit below is one this
	# test put there.
	r.red_interval = 1000.0
	r.red_jitter = 0.0
	var a: Node2D = _arena(r)
	var lane: Rect2 = a.layout.lane_rect

	# The transform must stay untouched: this is a @tool script, and writing `position` at
	# editor load marks main.tscn modified, which gives F5 a save the user never asked for.
	eq(a.position, Vector2.ZERO, "the arena never writes its own transform")
	check(lane.position.y > a.layout.right_pane.position.y,
		"   and the lane is a band inside the pane, not the whole of it")
	eq(a.pending(), 0, "nothing is queued to start with")

	# A spawn is QUEUED and placed on the next tick, not placed inline.
	a.spawn_teal()
	a.spawn_red()
	eq(a.living(), 0, "a spawn does not place a unit inline")
	eq(a.pending(), 2, "   it queues")
	await physics_frames(2)
	eq(a.living(), 2, "and the next tick places both")
	eq(a.pending(), 0, "   clearing the queue")

	var teal: BattlerUnit = null
	var red: BattlerUnit = null
	for u in a.field.living():
		if u.team == UnitDef.TEAL:
			teal = u
		else:
			red = u
	check(teal != null and red != null, "one of each team")
	check(teal.position.x < red.position.x, "teal enters left of red")
	check(lane.has_point(teal.position), "both enter inside the lane")
	check(lane.has_point(red.position), "   in SCREEN coordinates, not pane-local")

	# Depth sorting, recomputed every tick now that depth is something units walk around in.
	check(teal.z_index > 0,
		"a unit's z keeps it in front of the pane fill rather than hidden behind it")

	# A teal reaching the far edge just leaves; a red reaching the near edge is the breach.
	teal.position.x = lane.end.x + 1.0
	await physics_frames(2)
	eq(a.take_breaches(), 0, "a teal unit reaching its far end banks no breach")
	red.position.x = lane.position.x - 1.0
	await physics_frames(2)
	eq(a.take_breaches(), 1, "a red unit reaching the near edge banks one")
	eq(a.take_breaches(), 0, "   and taking it clears the counter")
	eq(a.living(), 0, "both units are gone")
	eq(a.field.units.size(), 0, "   and the roster is pruned rather than growing")

	# A crowded entry edge must QUEUE, never drop. The lane holds 7 bodies across its usable
	# depth, so twelve at once cannot all fit, and a fighter lost here would be invisible and
	# indistinguishable from a broken spawn rule.
	for _i in 12:
		a.spawn_teal()
	await physics_frames(2)
	check(a.living() < 12, "twelve at once do not all fit at the entry edge")
	check(a.pending() > 0, "   so the remainder stays queued")
	var placed: int = a.living() + a.pending()
	eq(placed, 12, "   and nothing is dropped: placed plus queued is still twelve")
	for _i in 90:
		await physics_frames(1)
	eq(a.pending(), 0, "the queue drains as the first arrivals walk clear")
	eq(a.living(), 12, "   and every one of the twelve made it onto the field")

	# The holes a breach turns into. One column per row, in range, and as many as asked for.
	var holes: Array = a.take_holes(4)
	eq(holes.size(), 4, "take_holes returns one column per row")
	var in_range := true
	for h in holes:
		if int(h) < 0 or int(h) > 9:
			in_range = false
	check(in_range, "   every column is on the board")
	eq(holes[0], holes[3], "   and a single flush is one clean column, not scattered")

	# The breach has to actually reach the Tetris board, which is the only part of the
	# minigame that can end a run.
	var g := make_game(make_board([]))
	var before := 0
	for y in g.board.total_height:
		if not g.board.row_is_empty(y):
			before += 1
	var ev: Array = []
	g.receive_garbage(a.take_holes(2), ev)
	var after := 0
	for y in g.board.total_height:
		if not g.board.row_is_empty(y):
			after += 1
	eq(after - before, 2, "a breach's rows land on the Tetris board")

	a.queue_free()
	await physics_frames(2)

# ------------------------------------------------------------------------------
# Steering: pure geometry, no nodes, no physics server, no scene tree
# ------------------------------------------------------------------------------

## Room enough that only the obstacles a test places can reject a step.
const OPEN := Rect2(-1000.0, -1000.0, 2000.0, 2000.0)
const R := 16.0

func test_step_free() -> void:
	group("Steering: the swept step test refuses what a raycast would allow")

	var none := PackedVector2Array()
	check(Steering.step_free(Vector2.ZERO, Vector2(10.0, 0.0), R, none, R, OPEN),
		"an empty field is free")
	var ahead := PackedVector2Array([Vector2(20.0, 0.0)])
	check(not Steering.step_free(Vector2.ZERO, Vector2(10.0, 0.0), R, ahead, R, OPEN),
		"a body straight ahead blocks")
	var aside := PackedVector2Array([Vector2(0.0, 80.0)])
	check(Steering.step_free(Vector2.ZERO, Vector2(10.0, 0.0), R, aside, R, OPEN),
		"   and one well off the path does not")

	# THE trap this test exists for. Two bodies 40px apart leave a 40px centre gap, which a
	# RAY down the middle passes cleanly -- it stays 20px from each, and each body is only
	# 16px in radius. A 32px body cannot fit, and committing to that squeeze is how a unit
	# wedges itself at an opening.
	var pinch := PackedVector2Array([Vector2(30.0, -20.0), Vector2(30.0, 20.0)])
	check(not Steering.step_free(Vector2.ZERO, Vector2(60.0, 0.0), R, pinch, R, OPEN),
		"a gap narrower than the body is refused, though a ray would pass it")
	var wide := PackedVector2Array([Vector2(30.0, -40.0), Vector2(30.0, 40.0)])
	check(Steering.step_free(Vector2.ZERO, Vector2(60.0, 0.0), R, wide, R, OPEN),
		"   and a gap the body does fit through is allowed")

	# Swept, not endpoint-tested: a step that starts and ends clear but passes through a body
	# must still be refused.
	var midway := PackedVector2Array([Vector2(50.0, 0.0)])
	check(not Steering.step_free(Vector2.ZERO, Vector2(100.0, 0.0), R, midway, R, OPEN),
		"a body in the MIDDLE of a long step blocks it, not just at the ends")

	# Bounds behave exactly like a body, which is what lets one rotation search handle walls.
	var box := Rect2(0.0, 0.0, 100.0, 100.0)
	check(Steering.step_free(Vector2(50.0, 50.0), Vector2(60.0, 50.0), R, none, R, box),
		"a step inside the bounds is free")
	check(not Steering.step_free(Vector2(50.0, 50.0), Vector2(50.0, 120.0), R, none, R, box),
		"   and one leaving them is refused like any other obstacle")

func test_escalation_ladder() -> void:
	group("Steering: the escalation ladder")

	var ladder := Steering.angles(25.0, 150.0, 1)
	eq(ladder[0], 0.0, "the ladder tries straight ahead first")
	eq(ladder[1], 25.0, "   then the committed side, shallowest first")
	eq(ladder[2], 50.0, "   escalating FURTHER on that side rather than crossing over")
	# Measured, not assumed: interleaving the two sides was written first and left a unit
	# oscillating in one scallop of a wall for 400 frames. Exhausting one side is what turns
	# the search into wall-following, so the ordering is the behaviour and is worth pinning.
	var crossover := -1
	for i in ladder.size():
		if ladder[i] < 0.0:
			crossover = i
			break
	eq(crossover, 7, "the far side is only reached after all 6 rungs of the near side")
	var deepest := 0.0
	for a in ladder:
		deepest = maxf(deepest, absf(a))
	eq(deepest, 150.0, "and it escalates to the full deflection")
	check(deepest > 90.0,
		"   past 90 degrees, which is what moves AWAY and escapes a pocket")
	var mirrored := Steering.angles(25.0, 150.0, -1)
	eq(mirrored[1], -25.0, "a negative bias tries the other side first")

	eq(Steering.step_scale(0.0, 0.35), 1.0, "a straight step moves the full distance")
	check(Steering.step_scale(90.0, 0.35) >= 0.35, "a sideways step still moves")
	check(Steering.step_scale(150.0, 0.35) >= 0.35,
		"   and so does a backward one -- a deflected unit creeps, never freezes")

func test_choose_step() -> void:
	group("Steering: choosing a step")

	var none := PackedVector2Array()
	var straight := Steering.choose_step(Vector2.ZERO, Vector2.RIGHT, 10.0, R,
		none, R, OPEN, 1, 25.0, 150.0, 0.35)
	eq(straight, Vector2(10.0, 0.0), "with nothing in the way, it walks straight at the goal")
	check(not Steering.deflected(straight, Vector2.RIGHT), "   and reports no deflection")

	# 50px out, stepping 30: the start is clear (50 > the 32px clearance) but a straight step
	# would close to 20, so the ladder has to find a way round.
	var wall_one := PackedVector2Array([Vector2(50.0, 0.0)])
	var around := Steering.choose_step(Vector2.ZERO, Vector2.RIGHT, 30.0, R,
		wall_one, R, OPEN, 1, 25.0, 150.0, 0.35)
	check(not around.is_zero_approx(), "a body in the way does not stop it")
	check(Steering.deflected(around, Vector2.RIGHT), "   it goes around, and says so")
	check(Steering.step_free(Vector2.ZERO, around, R, wall_one, R, OPEN),
		"   and the step it picked is one it can actually take")

	# Fully enclosed: a ring tight enough that no rung of the ladder fits.
	var ring := PackedVector2Array()
	for i in 16:
		var a := TAU * float(i) / 16.0
		ring.append(Vector2(cos(a), sin(a)) * 33.0)
	var boxed := Steering.choose_step(Vector2.ZERO, Vector2.RIGHT, 10.0, R,
		ring, R, OPEN, 1, 25.0, 150.0, 0.35)
	eq(boxed, Vector2.ZERO, "a fully enclosed unit holds position rather than jittering")
	check(Steering.deflected(boxed, Vector2.RIGHT), "   and a held step counts as deflected")

	# A flat wall should produce a step that runs ALONG it, which is the wall-following case.
	var wall := PackedVector2Array()
	for i in 9:
		wall.append(Vector2(50.0, -112.0 + 28.0 * float(i)))
	var slide := Steering.choose_step(Vector2.ZERO, Vector2.RIGHT, 30.0, R,
		wall, R, OPEN, 1, 25.0, 150.0, 0.35)
	check(not slide.is_zero_approx(), "a flat wall still yields a step")
	check(Steering.deflected(slide, Vector2.RIGHT), "   deflected rather than straight")
	check(absf(slide.y) > 5.0, "   with real movement ALONG the wall, not into it")

func test_wall_following() -> void:
	group("Steering: wall-following, which is what stops a unit pressing into its own line")

	# A line of bodies between the unit and its goal, sealed (28px spacing needs 64px to pass)
	# but FINITE -- exactly the shape an allied front line makes.
	var wall := PackedVector2Array()
	for i in 8:
		wall.append(Vector2(50.0 + 28.0 * float(i), 0.0))
	var goal := Vector2(150.0, 60.0)
	var at := Vector2(150.0, -60.0)
	var start_away := at.distance_to(goal)

	# Committed bias, held until a straight step succeeds -- the whole point of the test.
	var bias := 1
	var stuck := 0
	for _i in 400:
		var desired := (goal - at).normalized()
		var s := Steering.choose_step(at, desired, 6.0, R, wall, R, OPEN,
			bias, 25.0, 150.0, 0.35)
		if s.is_zero_approx():
			stuck += 1
			break
		if not Steering.deflected(s, desired):
			bias = 1          # unobstructed again: commitment released
		at += s
		if at.distance_to(goal) < 8.0:
			break

	eq(stuck, 0, "it never has to hold position -- the wall has an end")
	check(at.distance_to(goal) < 8.0, "and it rounds the end of the wall and reaches the goal")
	check(at.distance_to(goal) < start_away, "   having closed the distance, not circled")

	# Without commitment the same unit vibrates: flipping the side every frame means the two
	# deflections cancel and it makes no progress. This is the regression the commitment
	# exists to prevent, so it is worth asserting that the naive version really does fail.
	var flip_at := Vector2(150.0, -60.0)
	var flip := 1
	for _i in 400:
		var desired := (goal - flip_at).normalized()
		var s := Steering.choose_step(flip_at, desired, 6.0, R, wall, R, OPEN,
			flip, 25.0, 150.0, 0.35)
		if s.is_zero_approx():
			break
		flip_at += s
		flip = -flip          # the bug: a new side every frame
	check(flip_at.distance_to(goal) > at.distance_to(goal),
		"while a unit that re-picks its side every frame does strictly worse")

func test_concave_trap() -> void:
	group("Steering: a closed pocket is a known limitation, and a safe one")

	# A U opening upward with the goal below its closed floor. Reaching the goal means
	# travelling AWAY from it, out of the mouth and around the outside.
	#
	# THIS ALGORITHM DOES NOT ESCAPE IT, and that was measured rather than assumed. Letting
	# the ladder cross to the far side makes the unit oscillate between the two inner
	# corners; forbidding the crossover wedges it in one corner instead, held for 2964 of
	# 3000 frames -- and that same strict version breaks the FLAT wall, which is the case
	# that actually occurs, taking it from 109 frames to never. So the crossover stays and
	# the pocket stays unsolved.
	#
	# It is tolerable because marching allies form lines, not pockets: a unit in the middle
	# of a crowd is ENCLOSED, which is a different state and is handled by holding still. A
	# genuine U would have to be built deliberately.
	#
	# What is asserted here is the invariant that makes the limitation safe rather than
	# dangerous: a trapped unit never tunnels into a body, never produces a NaN, and never
	# thrashes its way out of the lane. If a future steering change breaks any of those, this
	# fails even though the escape never worked.
	var u := PackedVector2Array()
	for i in 8:
		u.append(Vector2(50.0 + 28.0 * float(i), 0.0))
	for i in 4:
		u.append(Vector2(50.0, -28.0 * float(i + 1)))
		u.append(Vector2(246.0, -28.0 * float(i + 1)))
	var goal := Vector2(148.0, 80.0)
	var at := Vector2(148.0, -50.0)
	var clearance := R + R

	var min_seen := 9999.0
	var finite := true
	for _i in 600:
		var desired := (goal - at).normalized()
		var s2 := Steering.choose_step(at, desired, 6.0, R, u, R, OPEN,
			1, 25.0, 150.0, 0.35)
		at += s2
		if not (is_finite(at.x) and is_finite(at.y)):
			finite = false
			break
		for o in u:
			min_seen = minf(min_seen, at.distance_to(o))

	check(finite, "a trapped unit's position stays finite")
	check(min_seen >= clearance - 0.01,
		"   and it never tunnels inside a body, however long it is stuck")
	check(OPEN.has_point(at), "   nor escapes through the bounds")

func test_facing_from() -> void:
	group("Steering: which way a unit is drawn")

	eq(Steering.facing_from(Vector2(5.0, 0.0), SpriteSheet.UP), SpriteSheet.RIGHT,
		"moving right faces right")
	eq(Steering.facing_from(Vector2(-5.0, 0.0), SpriteSheet.UP), SpriteSheet.LEFT,
		"   and left, left")
	eq(Steering.facing_from(Vector2(0.0, 5.0), SpriteSheet.UP), SpriteSheet.DOWN,
		"screen-down faces the camera")
	eq(Steering.facing_from(Vector2(0.0, -5.0), SpriteSheet.DOWN), SpriteSheet.UP,
		"   and screen-up faces away")
	eq(Steering.facing_from(Vector2(5.0, 5.0), SpriteSheet.UP), SpriteSheet.RIGHT,
		"a diagonal breaks to the horizontal, which reads better along the lane")
	eq(Steering.facing_from(Vector2.ZERO, SpriteSheet.LEFT), SpriteSheet.LEFT,
		"and standing still keeps the facing it had")

# ------------------------------------------------------------------------------
# BattleField
# ------------------------------------------------------------------------------

func _field(depth: float = 256.0) -> BattleField:
	return BattleField.new(Rect2(100.0, 200.0, 600.0, depth), R)

func test_battle_field_bounds() -> void:
	group("BattleField: the lane's shape")

	var f := _field()
	# Inset across the flow, so a body never half-leaves the lane.
	eq(f.centre_bounds.position.y, 216.0, "the centre bounds inset by a body radius on top")
	eq(f.centre_bounds.end.y, 440.0, "   and on the bottom")
	eq(f.centre_bounds.size.y, 224.0, "   leaving 7 bodies of usable depth out of 8")
	# Open along it, or nothing could ever leave the lane and a breach would be impossible.
	check(f.centre_bounds.position.x < f.lane.position.x,
		"and it runs well past the lane's near end")
	check(f.centre_bounds.end.x > f.lane.end.x,
		"   and its far end, so a unit can walk out to breach or retire")

	eq(f.flow_at(Vector2.ZERO, UnitDef.TEAL), Vector2.RIGHT, "teal marches right")
	eq(f.flow_at(Vector2.ZERO, UnitDef.RED), Vector2.LEFT, "   and red marches left")
	eq(f.spawn_x(UnitDef.TEAL), 100.0, "teal enters at the near edge")
	eq(f.spawn_x(UnitDef.RED), 700.0, "   and red at the far one")
	check(f.past_far_end(Vector2(701.0, 300.0), UnitDef.TEAL), "a teal past the right is done")
	check(not f.past_far_end(Vector2(99.0, 300.0), UnitDef.TEAL), "   but not one at the left")
	check(f.past_far_end(Vector2(99.0, 300.0), UnitDef.RED), "a red past the left has breached")

func test_battle_field_queries() -> void:
	group("BattleField: who is there, and does a body fit")

	var f := _field()
	var a := Body.new(Vector2(200.0, 300.0), UnitDef.TEAL)
	var b := Body.new(Vector2(260.0, 300.0), UnitDef.RED)
	var corpse := Body.new(Vector2(210.0, 300.0), UnitDef.RED, false)
	for u in [a, b, corpse]:
		f.track(u)
	eq(f.units.size(), 3, "three units tracked")
	f.track(a)
	eq(f.units.size(), 3, "   and tracking one twice does not duplicate it")

	eq(f.living().size(), 2, "a corpse is not living")
	eq(f.living_count(UnitDef.TEAL), 1, "one teal alive")
	eq(f.living_count(UnitDef.RED), 1, "   and one red, the corpse excluded")

	var obs := f.obstacles(a)
	eq(obs.size(), 1, "obstacles exclude the mover itself AND the corpse")
	eq(obs[0], Vector2(260.0, 300.0), "   leaving only the living enemy")
	check(not f.has_room(Vector2(265.0, 300.0), R, a),
		"there is no room on top of a living body")
	check(f.has_room(Vector2(210.0, 300.0), R, a),
		"   but a corpse does not hold ground, so a line can erode")
	check(not f.has_room(Vector2(200.0, 205.0), R, null),
		"and no room half-outside the lane")

	# Nearest, ignoring allies and corpses, respecting the radius.
	var far := Body.new(Vector2(500.0, 300.0), UnitDef.RED)
	var ally := Body.new(Vector2(205.0, 300.0), UnitDef.TEAL)
	f.track(far)
	f.track(ally)
	eq(f.nearest_enemy(a.position, UnitDef.TEAL, 400.0, a), b,
		"the nearest enemy wins, and an adjacent ally is not one")
	eq(f.nearest_enemy(a.position, UnitDef.TEAL, 40.0, a), null,
		"   and nothing outside the radius is a target at all")
	b.alive = false
	eq(f.nearest_enemy(a.position, UnitDef.TEAL, 400.0, a), far,
		"killing the nearest promotes the next, rather than holding a corpse")

## A unit stand-in: the three properties BattleField reads, and nothing else.
##
## Its existence is the assertion -- if BattleField ever reaches for a node, a physics body
## or a UnitDef, these groups stop compiling and the contract is back to being implicit.
class Body:
	var alive := true
	var position := Vector2.ZERO
	var team := UnitDef.TEAL

	func _init(p: Vector2, t: int = UnitDef.TEAL, is_alive: bool = true) -> void:
		position = p
		team = t
		alive = is_alive
