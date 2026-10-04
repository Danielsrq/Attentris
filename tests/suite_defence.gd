extends TestHarness

## Defence: the whole board
##
## Run by tests/run_tests.gd, which owns the totals. See tests/test_harness.gd for
## group/check/eq and the shared board fixtures.

func run() -> void:
	test_aim()
	test_tower_compile()
	test_muzzle()
	test_garbage_buffer()
	test_defence_field()
	test_defence_sim()
	test_build_cursor()
	test_defence_coupling()
	test_defence_view_pool()

# ------------------------------------------------------------------------------
# Defence: aim lattice
# ------------------------------------------------------------------------------

func test_aim() -> void:
	group("Aim: the 16-direction lattice")

	eq(Aim.COUNT, 16, "sixteen directions")
	# 0 is +x; 4 is up, because board space is y-down and the table negates y.
	check(Aim.vector(0).is_equal_approx(Vector2.RIGHT), "direction 0 points +x")
	check(Aim.vector(4).is_equal_approx(Vector2.UP), "direction 4 points up on screen")
	check(Aim.vector(8).is_equal_approx(Vector2.LEFT), "direction 8 points -x")
	check(Aim.vector(12).is_equal_approx(Vector2.DOWN), "direction 12 points down")

	var off: Array = []
	for d in Aim.COUNT:
		if absf(Aim.vector(d).length() - 1.0) > 1e-6:
			off.append(d)
	check(off.is_empty(), "every direction is a unit vector%s" % [
		"" if off.is_empty() else " (%s)" % str(off)])

	# Reflection is an involution on the index, and the two together are a reversal. These
	# identities are what guarantee a shot is still on the lattice after any number of
	# bounces, which is what makes a ricochet plannable.
	var bad_x: Array = []
	var bad_y: Array = []
	var bad_corner: Array = []
	for d in Aim.COUNT:
		if Aim.reflect_x(Aim.reflect_x(d)) != d:
			bad_x.append(d)
		if Aim.reflect_y(Aim.reflect_y(d)) != d:
			bad_y.append(d)
		if Aim.reflect_x(Aim.reflect_y(d)) != Aim.reverse(d):
			bad_corner.append(d)
	check(bad_x.is_empty(), "reflecting twice off a vertical surface is the identity")
	check(bad_y.is_empty(), "   and off a horizontal one")
	check(bad_corner.is_empty(), "both at once is exactly a reversal, which is a corner hit")

	# And the index arithmetic must agree with the vectors it stands for.
	var mismatch: Array = []
	for d in Aim.COUNT:
		var v := Aim.vector(d)
		var rx := Aim.vector(Aim.reflect_x(d))
		var ry := Aim.vector(Aim.reflect_y(d))
		if absf(rx.x + v.x) > 1e-6 or absf(rx.y - v.y) > 1e-6:
			mismatch.append("x:%d" % d)
		if absf(ry.y + v.y) > 1e-6 or absf(ry.x - v.x) > 1e-6:
			mismatch.append("y:%d" % d)
	check(mismatch.is_empty(), "reflect_x flips only x and reflect_y only y%s" % [
		"" if mismatch.is_empty() else " (%s)" % str(mismatch)])

	# Speed survives a bounce because only the index changes -- the magnitude is never
	# recomputed, so it cannot creep. Checked relatively: at length 240 the float
	# representation of a unit vector is already worth ~2e-5 in absolute terms, so an
	# absolute epsilon here would be measuring Vector2.rotated, not the bounce.
	var v0 := Aim.velocity(3, 240.0)
	var v1 := Aim.velocity(Aim.reflect_x(3), 240.0)
	check(absf(v0.length() - v1.length()) / 240.0 < 1e-6, "a bounce does not change speed")
	# Worst case over every direction and every reflection, so one bad entry cannot hide.
	var worst := 0.0
	for d in Aim.COUNT:
		for n in [Aim.reflect_x(d), Aim.reflect_y(d), Aim.reverse(d)]:
			worst = maxf(worst, absf(Aim.velocity(n, 240.0).length() - 240.0) / 240.0)
	check(worst < 1e-6, "   for every direction and every reflection (worst %.9f)" % worst)

	eq(Aim.turn(15, 1), 0, "turning past the last direction wraps")
	eq(Aim.turn(0, -1), 15, "   and so does turning back past the first")
	eq(Aim.vector(16), Aim.vector(0), "an out-of-range index is wrapped, not an error")
	eq(Aim.label(0), "0", "cardinals read as whole degrees")
	eq(Aim.label(3), "67.5", "and the half steps keep their half")

# ------------------------------------------------------------------------------
# Defence: tower compile
# ------------------------------------------------------------------------------

func test_tower_compile() -> void:
	group("TowerSet: masks compile to four rotations with their cannon")

	var ts := TowerSet.standard()
	check(ts.size() >= 2, "the standard set has towers in it")

	var basic := ts.get_def(&"basic")
	check(basic != null, "basic is registered by id")
	eq(basic.box, 2, "a 2x2 mask gets box 2")
	eq(basic.cell_count(), 4, "   and four cells")
	eq(basic.extent(0), Vector2i(2, 2), "   spanning 2x2")

	var angled := ts.get_def(&"angled")
	eq(angled.box, 3, "XXC/XX. gets box 3, derived from max(width, height)")
	eq(angled.cell_count(), 5, "   and five cells -- C is a body cell too")
	eq(angled.extent(0), Vector2i(3, 2), "   spanning 3x2 at rotation 0")
	eq(angled.extent(1), Vector2i(2, 3), "   and 2x3 turned once")
	eq(angled.extent(2), Vector2i(3, 2), "   back to 3x2 at 180")

	for d in ts.defs:
		var def: TowerDef = d
		# The cannon must land inside its own footprint at every rotation. This is the
		# invariant that a remembered cell INDEX would break, because sorting reorders.
		var outside: Array = []
		var wrong_size: Array = []
		var unnormalised: Array = []
		for rot in 4:
			if not def.covers(rot, def.cannon(rot)):
				outside.append(rot)
			if def.cells(rot).size() != def.cell_count():
				wrong_size.append(rot)
			var lo := TowerSet._min_corner(def.cells(rot))
			if lo != Vector2i.ZERO:
				unnormalised.append(rot)
		check(outside.is_empty(), "%s: the cannon is inside the footprint at every rotation%s" % [
			def.id, "" if outside.is_empty() else " (%s)" % str(outside)])
		check(wrong_size.is_empty(), "%s: rotating never loses or gains a cell" % def.id)
		check(unnormalised.is_empty(), "%s: every rotation is normalised to its own corner" % def.id)

		# Four turns is the identity, so a player cycling the footprint always comes back.
		eq(def.cells(4), def.cells(0), "%s: four turns returns to the start" % def.id)
		eq(def.cannon(4), def.cannon(0), "%s:    with the cannon back where it began" % def.id)
		# Extents swap on an odd turn and restore on an even one.
		eq(def.extent(1), Vector2i(def.extent(0).y, def.extent(0).x),
			"%s: one turn transposes the extent" % def.id)

	# Spelled out, so a change to rotate_cw shows up as a diff and not as a vibe.
	# XC / XX  turned clockwise once puts the cannon at the bottom-right.
	eq(basic.cannon(0), Vector2i(1, 0), "basic: the cannon starts top-right")
	eq(basic.cannon(1), Vector2i(1, 1), "   one turn clockwise takes it to bottom-right")
	eq(basic.cannon(2), Vector2i(0, 1), "   then bottom-left")
	eq(basic.cannon(3), Vector2i(0, 0), "   then top-left")

	# A mask is not mutated by compiling it, and two sets are independent.
	var again := TowerSet.standard()
	eq(again.get_def(&"angled").cells(0), angled.cells(0),
		"compiling twice gives the same footprint")

# ------------------------------------------------------------------------------
# Defence: muzzle exit
# ------------------------------------------------------------------------------

func test_muzzle() -> void:
	group("Muzzle: a shot leaves the hull, not the cannon cell")

	var ts := TowerSet.standard()
	var basic := ts.get_def(&"basic")
	var angled := ts.get_def(&"angled")

	# Every rotation and every direction must resolve. "The cannon is walled in" is
	# geometrically impossible for a solid finite footprint -- a ray always leaves a
	# bounded set -- so a false here means the mask is not solid or the walk is broken.
	var failed: Array = []
	for d in ts.defs:
		var def: TowerDef = d
		for rot in 4:
			for dir in Aim.COUNT:
				if not bool(def.muzzle(rot, dir)["ok"]):
					failed.append("%s r%d d%d" % [def.id, rot, dir])
	check(failed.is_empty(), "all 64 entries resolve for every tower%s" % [
		"" if failed.is_empty() else " (%s)" % str(failed)])
	eq(basic.muzzles.size(), 4 * Aim.COUNT, "the table is 4 rotations x 16 directions")

	# The exit must be ON the hull boundary: outside every footprint cell, but touching
	# one. If it were inside, the shot would spawn in solid geometry; if it were far
	# outside, the walk overshot.
	var inside: Array = []
	var adrift: Array = []
	for d2 in ts.defs:
		var def2: TowerDef = d2
		for rot in 4:
			var span := def2.extent(rot)
			for dir in Aim.COUNT:
				var off: Vector2 = def2.muzzle(rot, dir)["offset"]
				var c := Vector2i(floori(off.x), floori(off.y))
				# A boundary point floors into the cell it is entering, which is outside
				# the footprint by construction -- unless it landed back inside.
				if def2.covers(rot, c) and not is_equal_approx(off.x, roundf(off.x)) 						and not is_equal_approx(off.y, roundf(off.y)):
					inside.append("%s r%d d%d" % [def2.id, rot, dir])
				if off.x < -1.0 or off.y < -1.0 or off.x > float(span.x) + 1.0 						or off.y > float(span.y) + 1.0:
					adrift.append("%s r%d d%d %s" % [def2.id, rot, dir, str(off)])
	check(inside.is_empty(), "no exit point sits inside solid cells%s" % [
		"" if inside.is_empty() else " (%s)" % str(inside)])
	check(adrift.is_empty(), "and none overshoots the footprint%s" % [
		"" if adrift.is_empty() else " (%s)" % str(adrift)])

	# Spelled out for the simple case. basic is XC/XX with the cannon top-right at (1,0),
	# so firing +x leaves the right edge of a 2-wide footprint at the middle of row 0.
	var m0: Dictionary = basic.muzzle(0, 0)
	check(is_equal_approx(float(m0["offset"].x), 2.0), "basic firing +x exits at x = 2 cells")
	check(is_equal_approx(float(m0["offset"].y), 0.5), "   halfway down its own row")
	# Firing up from the same cannon leaves the top edge immediately.
	var m4: Dictionary = basic.muzzle(0, 4)
	check(is_equal_approx(float(m4["offset"].y), 0.0), "firing up exits at y = 0")
	check(is_equal_approx(float(m4["offset"].x), 1.5), "   from the middle of its column")
	# Firing -x from the top-right cannon must cross the body cell at (0,0) first, which is
	# the whole reason the exit is not the cannon centre.
	var m8: Dictionary = basic.muzzle(0, 8)
	check(is_equal_approx(float(m8["offset"].x), 0.0), "firing -x crosses the body and exits at x = 0")

	# The interior-cannon case the design asked for: angled is XXC/XX., and turned twice
	# the cannon sits at the far corner with body between it and most directions.
	var any_crossing := false
	for dir2 in Aim.COUNT:
		var off2: Vector2 = angled.muzzle(2, dir2)["offset"]
		if (Vector2(angled.cannon(2)) + Vector2(0.5, 0.5)).distance_to(off2) > 1.01:
			any_crossing = true
	check(any_crossing, "an interior cannon fires through its own body to reach the hull")

# ------------------------------------------------------------------------------
# Defence: the garbage window
# ------------------------------------------------------------------------------

func test_garbage_buffer() -> void:
	group("GarbageBuffer: the 2s window between a breach and the garbage")

	var r := DefenceRules.testing()
	var gb := GarbageBuffer.new(r, 1234)

	check(not gb.is_open(), "the window starts closed")
	eq(gb.tick(1.0), 0, "   and ticking a closed window sends nothing")

	# The first breach opens the window and counts. A lone breach is one row, not free --
	# the window is about batching, not about forgiving the first one.
	gb.touch()
	check(gb.is_open(), "the first breach opens the window")
	eq(gb.incoming(), 1, "   and counts itself")
	eq(gb.tick(1.0), 0, "half way through, nothing has been sent")
	check(gb.is_open(), "   the window is still open")
	gb.touch()
	gb.touch()
	eq(gb.incoming(), 3, "breaches during the window pile up")
	eq(gb.tick(1.1), 3, "when it closes, the whole buffer goes at once")
	check(not gb.is_open(), "   and the window closes")
	eq(gb.incoming(), 0, "   leaving nothing pending")

	# A later breach opens a fresh window rather than extending a dead one.
	gb.touch()
	check(gb.is_open(), "a breach after a flush opens a new window")
	eq(gb.tick(r.window_sec + 0.1), 1, "   which sends on its own")

	group("GarbageBuffer: the cap, and the carry that drains")
	var gb2 := GarbageBuffer.new(r, 99)
	for _i in 11:
		gb2.touch()
	eq(gb2.tick(r.window_sec + 0.1), r.flush_cap, "eleven breaches send only the cap of 8")
	eq(gb2.pending, 3, "   and carry the other 3 rather than dropping them")
	# drain_carry re-opens the window, so the carry arrives without another breach. Without
	# it those rows would sit forever if nothing ever breached again.
	check(gb2.is_open(), "the carry re-opens the window on its own")
	eq(gb2.tick(r.window_sec + 0.1), 3, "   and the carried rows follow one window later")
	eq(gb2.pending, 0, "   with nothing left over")
	check(not gb2.is_open(), "   and the window finally closes")

	var r_nodrain := DefenceRules.testing()
	r_nodrain.drain_carry = false
	var gb3 := GarbageBuffer.new(r_nodrain, 99)
	for _i in 11:
		gb3.touch()
	eq(gb3.tick(r.window_sec + 0.1), r.flush_cap, "with drain_carry off the cap still applies")
	check(not gb3.is_open(), "   but the window does not re-open")
	eq(gb3.tick(100.0), 0, "   and the carry waits for a real breach")
	eq(gb3.pending, 3, "      still held, not lost")

	group("GarbageBuffer: hole columns")
	var gb4 := GarbageBuffer.new(r, 7)
	check(gb4.hole >= 0 and gb4.hole <= 9, "the starting hole is a real column")
	# Clean garbage shares one column across every row, which is what makes it a free well.
	var holes := gb4.holes_for(4)
	eq(holes.size(), 4, "holes_for returns one entry per row")
	eq(holes, [gb4.hole, gb4.hole, gb4.hole, gb4.hole], "clean garbage repeats one column")
	# Cheese varies per row, and must stay in bounds however it walks.
	var r_cheese := DefenceRules.testing()
	r_cheese.style = DefenceRules.GarbageStyle.CHEESE
	var gb5 := GarbageBuffer.new(r_cheese, 7)
	var out_of_range: Array = []
	var varied := false
	for _t in 40:
		var hs := gb5.holes_for(8)
		for h in hs:
			if int(h) < 0 or int(h) > 9:
				out_of_range.append(h)
		if hs.min() != hs.max():
			varied = true
	check(out_of_range.is_empty(), "cheese holes stay inside the field%s" % [
		"" if out_of_range.is_empty() else " (%s)" % str(out_of_range)])
	check(varied, "   and actually vary between rows")

	# The walk is at most one column per flush, which is what keeps a well plannable.
	var gb6 := GarbageBuffer.new(r, 2024)
	var jumps: Array = []
	for _t in 200:
		var was: int = gb6.hole
		gb6.touch()
		gb6.tick(r.window_sec + 0.1)
		if absi(gb6.hole - was) > 1:
			jumps.append("%d->%d" % [was, gb6.hole])
	check(jumps.is_empty(), "the hole never moves more than one column per flush%s" % [
		"" if jumps.is_empty() else " (%s)" % str(jumps)])

	# Seeded, so the walk is reproducible even though the rest of the defence side is not.
	var a := GarbageBuffer.new(r, 555)
	var b := GarbageBuffer.new(r, 555)
	var same := true
	for _t in 50:
		a.touch(); b.touch()
		a.tick(r.window_sec + 0.1); b.tick(r.window_sec + 0.1)
		if a.hole != b.hole:
			same = false
	check(same, "two buffers on the same seed walk the hole identically")

# ------------------------------------------------------------------------------
# Defence: the field
# ------------------------------------------------------------------------------

func test_defence_field() -> void:
	group("DefenceField: placement occupancy")

	var ts := TowerSet.standard()
	var basic := ts.get_def(&"basic")
	var f := DefenceField.new(20, 18)

	eq(f.occupied_count(), 0, "a new field is empty")
	check(f.is_free(0, 0), "   and its corner is free")
	check(not f.in_bounds(20, 0), "column 20 is off a 20-wide grid")
	check(not f.in_bounds(0, 18), "and row 18 is off an 18-tall one")

	check(f.can_place(basic, 0, Vector2i(0, 0)), "a 2x2 fits in the corner")
	check(not f.can_place(basic, 0, Vector2i(19, 0)), "   but not hanging off the right edge")
	check(not f.can_place(basic, 0, Vector2i(0, 17)), "   nor off the bottom")

	var t := DefenceTower.new(7, basic, Vector2i(3, 4))
	f.place(t)
	eq(f.occupied_count(), 4, "placing a 2x2 occupies four cells")
	eq(f.at(3, 4), 7, "   stamped with the tower's id")
	eq(f.at(4, 5), 7, "   across its whole footprint")
	check(not f.can_place(basic, 0, Vector2i(4, 5)), "an overlapping tower does not fit")
	check(f.can_place(basic, 0, Vector2i(5, 4)), "   but one beside it does")

	f.remove(t)
	eq(f.occupied_count(), 0, "removing it frees every cell")

	# The cursor is clamped by EXTENT, which transposes with rotation -- so a 3x2 footprint
	# turned once can be walked one column further right and one row less far down.
	var angled := ts.get_def(&"angled")
	eq(f.clamp_origin(Vector2i(99, 99), angled.extent(0)), Vector2i(17, 16),
		"a 3x2 footprint clamps to leave room for itself")
	eq(f.clamp_origin(Vector2i(99, 99), angled.extent(1)), Vector2i(18, 15),
		"   and turned once it clamps differently, because the extent transposed")
	eq(f.clamp_origin(Vector2i(-5, -5), angled.extent(0)), Vector2i(0, 0),
		"and it cannot be walked off the top-left either")

	eq(f.cell_at(Vector2(0.0, 0.0), 30), Vector2i(0, 0), "pixel 0,0 is cell 0,0")
	eq(f.cell_at(Vector2(89.0, 31.0), 30), Vector2i(2, 1), "and pixels map into their cell")
	eq(f.pixel_size(30), Vector2(600.0, 540.0), "a 20x18 grid of 30px cells is 600x540")

func test_defence_sim() -> void:
	group("DefenceSim: building")

	var sim := _def_sim()
	var basic: TowerDef = sim.towers.get_def(&"basic")
	var ev: Array = []

	var t := sim.build(basic, 0, Vector2i(2, 2), 0, ev)
	check(t != null, "a tower can be built on empty cells")
	eq(_of_type(ev, DefenceEvent.Type.TOWER_BUILT).size(), 1, "   and reports itself once")
	eq(sim.field.at(2, 2), t.id, "   occupying the grid")
	ev.clear()
	check(sim.build(basic, 0, Vector2i(2, 2), 0, ev) == null, "a second tower cannot overlap it")
	eq(ev.size(), 0, "   and reports nothing")
	eq(sim.tower_at(Vector2i(3, 3)), t, "a cell lookup finds the tower that owns it")
	check(sim.tower_at(Vector2i(9, 9)) == null, "   and nothing where there is none")

	# max_towers stands in for cost while towers are free.
	sim.rules.max_towers = 2
	ev.clear()
	check(sim.build(basic, 0, Vector2i(6, 2), 0, ev) != null, "the second of two is allowed")
	check(sim.build(basic, 0, Vector2i(9, 2), 0, ev) == null, "   the third is not")
	sim.remove_tower(t, ev)
	check(sim.build(basic, 0, Vector2i(9, 2), 0, ev) != null, "   removing one makes room again")

	group("DefenceSim: speed is the body's own")
	# Speed lives on DefenceEnemy, not read from the rules each tick, so types can differ
	# without touching the move loop -- and changing the rule does not retroactively speed
	# up bodies already on the field.
	var sp := _def_sim()
	var creeper := DefenceEnemy.new(sp._id(), Vector2(500.0, 100.0), 30, 3,
		sp.rules.enemy_half_extents, 10.0)
	var sprinter := DefenceEnemy.new(sp._id(), Vector2(500.0, 200.0), 30, 6,
		sp.rules.enemy_half_extents, 80.0)
	sp.enemies.append(creeper)
	sp.enemies.append(sprinter)
	for _i in 60:
		sp.tick(1.0 / 60.0)
	# One second: 10px and 80px.
	check(absf((500.0 - creeper.pos.x) - 10.0) < 0.5, "a slow body moves at its own speed")
	check(absf((500.0 - sprinter.pos.x) - 80.0) < 0.5, "   and a fast one at its own, eight times further")
	sp.rules.enemy_speed = 999.0
	var was := sprinter.pos.x
	sp.tick(1.0 / 60.0)
	check(absf((was - sprinter.pos.x) - 80.0 / 60.0) < 0.1,
		"changing the rule does not retroactively speed up a body already on the field")

	group("DefenceSim: enemies drift and breach")
	# Built WITH spawning on, so the first interval has to elapse. Setting the interval
	# after construction would leave the timer at zero and spawn on tick one as well.
	var r2 := DefenceRules.testing()
	r2.spawn_interval = 1.0
	var s2 := DefenceSim.new(r2, TowerSet.standard(), 30, 1)
	var ev2: Array = []
	for _i in 60:
		ev2.append_array(s2.tick(1.0 / 60.0))
	eq(_of_type(ev2, DefenceEvent.Type.ENEMY_SPAWNED).size(), 1, "one spawn per interval")
	eq(s2.enemies.size(), 1, "   and it is on the field")
	var e: DefenceEnemy = s2.enemies[0]
	check(e.pos.x < 600.0, "it has moved left")
	check(e.lane >= 0 and e.lane < s2.rules.rows, "   in a real lane")

	s2.rules.spawn_interval = 0.0
	ev2.clear()
	# Budget derived from the rules, not hardcoded: a 1200-tick loop was tuned to 40px/s and
	# silently stopped reaching the wall the moment the board was slowed to 12.
	var crossing := int((600.0 - r2.enemy_half_extents.x) / r2.enemy_speed * 60.0) + 120
	for _i in crossing:
		ev2.append_array(s2.tick(1.0 / 60.0))
		if s2.enemies.is_empty():
			break
	eq(_of_type(ev2, DefenceEvent.Type.ENEMY_BREACHED).size(), 1, "reaching the wall is a breach")
	eq(s2.breaches, 1, "   counted")
	check(s2.enemies.is_empty(), "   and the body is despawned")
	check(s2.garbage.is_open(), "   opening the garbage window")

	group("DefenceSim: firing, and the line-clear shot")
	var s3 := _def_sim()
	var ev3: Array = []
	var t3 := s3.build(s3.towers.get_def(&"basic"), 0, Vector2i(5, 5), 0, ev3)
	ev3.clear()
	s3.tick(1.0 / 60.0)
	eq(s3.shots_fired, 1, "a fresh tower fires immediately")
	var fired := s3.shots_fired
	for _i in 30:
		s3.tick(1.0 / 60.0)
	eq(s3.shots_fired, fired, "   then waits out its cooldown")
	# A line clear grants a shot that does NOT reset the cooldown, so clearing can only add
	# damage. If it reset the timer, clearing lines would feel like a penalty.
	var before := t3.cooldown
	s3.grant_free_shots(1)
	ev3.clear()
	s3.tick(1.0 / 60.0)
	eq(s3.shots_fired, fired + 1, "a line clear fires an extra shot during the cooldown")
	var causes := _of_type(ev3, DefenceEvent.Type.SHOT_FIRED)
	if causes.size() == 1:
		eq(String(causes[0].data["cause"]), "line_clear", "   reported as a line_clear shot")
	check(t3.cooldown < before, "   and the cooldown kept running rather than resetting")

	group("DefenceSim: bouncing stays on the lattice")
	var s4 := _def_sim()
	var ev4: Array = []
	s4.build(s4.towers.get_def(&"basic"), 0, Vector2i(1, 8), 0, ev4)
	ev4.clear()
	for _i in 600:
		ev4.append_array(s4.tick(1.0 / 60.0))
	var expired := _of_type(ev4, DefenceEvent.Type.SHOT_EXPIRED)
	check(expired.size() > 0, "shots expire rather than accumulating forever")
	var bounced := _of_type(ev4, DefenceEvent.Type.SHOT_BOUNCED)
	check(bounced.size() > 0, "   having bounced off the walls on the way")
	var over: Array = []
	for sh in s4.shots:
		if sh.bounces > s4.rules.max_bounces:
			over.append(sh.bounces)
	check(over.is_empty(), "no shot outlives its bounce budget")

	var s5 := _def_sim()
	var ev5: Array = []
	s5.build(s5.towers.get_def(&"basic"), 0, Vector2i(1, 8), 2, ev5)
	var off_lattice := false
	var sped := false
	for _i in 600:
		s5.tick(1.0 / 60.0)
		for sh in s5.shots:
			if sh.dir < 0 or sh.dir > 15:
				off_lattice = true
			if absf(sh.vel.length() - s5.rules.projectile_speed) / s5.rules.projectile_speed > 1e-6:
				sped = true
	check(not off_lattice, "bouncing never leaves the 16 directions")
	check(not sped, "   and never changes speed")

	# Shots must stay inside the field. De-penetration is what guarantees this: without it
	# a shot sinks into a wall and the overlap test fires every tick.
	var s5b := _def_sim()
	var ev5b: Array = []
	s5b.build(s5b.towers.get_def(&"basic"), 0, Vector2i(1, 8), 3, ev5b)
	var escaped := false
	for _i in 600:
		s5b.tick(1.0 / 60.0)
		for sh in s5b.shots:
			var r := s5b.rules.projectile_radius
			if sh.pos.x < -r or sh.pos.y < -r or sh.pos.x > 600.0 + r or sh.pos.y > 540.0 + r:
				escaped = true
	check(not escaped, "no shot ever leaves the field")

	group("DefenceSim: a bounce is a hit")
	# One fresh shot per hit, each approaching from the side so the contact is a clean
	# single-axis one. Deliberately NOT one shot ping-ponging: this is a test of the damage
	# model, and tying it to a trajectory would make it a test of the geometry instead.
	# (Starting a shot at the enemy's exact centre overlaps both axes equally, which is a
	# true corner and correctly reverses -- a real behaviour, just the wrong thing to lean
	# a damage test on.)
	var s6 := _def_sim()
	var ev6: Array = []
	var e6 := DefenceEnemy.new(999, Vector2(300.0, 255.0), s6.rules.enemy_hp, 8,
		s6.rules.enemy_half_extents)
	s6.enemies.append(e6)
	var hp_after: Array = []
	var hits := 0
	for n in 3:
		var sh := DefenceShot.new(900 + n, Vector2(e6.pos.x - 24.0, e6.pos.y), 0,
			s6.rules.projectile_speed, 0)
		sh.ignore_towers = false
		s6.shots.append(sh)
		for _i in 40:
			var evs := s6.tick(1.0 / 60.0)
			var got := _of_type(evs, DefenceEvent.Type.ENEMY_HIT)
			ev6.append_array(evs)
			if got.size() > 0:
				hits += got.size()
				hp_after.append(int(got[0].data["hp_left"]))
				break
		s6.shots.clear()          # retire it; the next hit gets its own shot
		if s6.enemies.is_empty():
			break
	eq(hits, 3, "three bounces land three hits")
	eq(hp_after, [20, 10, 0], "   taking 10 HP each, from 30 to nothing")
	eq(_of_type(ev6, DefenceEvent.Type.ENEMY_KILLED).size(), 1, "   and the third kills")
	eq(s6.kills, 1, "      counted")
	check(s6.enemies.is_empty(), "      and the body is gone")

	group("DefenceSim: a shot clears its own tower first")
	var s7 := _def_sim()
	var ev7: Array = []
	s7.build(s7.towers.get_def(&"basic"), 0, Vector2i(5, 5), 0, ev7)
	s7.tick(1.0 / 60.0)
	eq(s7.shots.size(), 1, "the tower fired")
	if s7.shots.size() == 1:
		var sh7: DefenceShot = s7.shots[0]
		# It spawns ON its own hull, so without the flag it would bounce on frame one.
		eq(sh7.bounces, 0, "   and the new shot has not bounced off its own tower")
		for _i in 20:
			s7.tick(1.0 / 60.0)
		check(not sh7.ignore_towers or s7.shots.is_empty(),
			"the flag clears once the shot is clear of every footprint")

	# The discrete overlap test is only sound while a shot cannot cross an obstacle in one
	# step. This is the guard that says so out loud.
	check(s7.rules.step_is_safe(30), "the default speed cannot tunnel through a 30px cell")
	var fast := DefenceRules.testing()
	fast.projectile_speed = 2000.0
	check(not fast.step_is_safe(30), "   and a tenfold speed buff is correctly flagged unsafe")
	fast.substeps = 8
	check(fast.step_is_safe(30), "   which substeps buys back")

	group("DefenceSim: the garbage window reaches the event stream")
	var s8 := _def_sim()
	var ev8: Array = []
	s8.garbage.touch()
	for _i in 200:
		ev8.append_array(s8.tick(1.0 / 60.0))
	var flushed := _of_type(ev8, DefenceEvent.Type.GARBAGE_FLUSHED)
	eq(flushed.size(), 1, "the window closing emits one flush")
	if flushed.size() == 1:
		eq(int(flushed[0].data["rows"]), 1, "   of one row")
		eq((flushed[0].data["holes"] as Array).size(), 1, "   with one hole column per row")

# ------------------------------------------------------------------------------
# Defence: the build cursor
# ------------------------------------------------------------------------------

func test_build_cursor() -> void:
	group("BuildCursor: intent that is always legal")

	var ts := TowerSet.standard()
	var f := DefenceField.new(20, 18)
	var cur := BuildCursor.new(ts, f)

	eq(cur.origin, Vector2i(0, 0), "the cursor starts in the corner")
	check(cur.def() != null, "   with a tower selected")

	cur.move(3, 4)
	eq(cur.origin, Vector2i(3, 4), "it moves by whole cells")
	cur.move(-99, -99)
	eq(cur.origin, Vector2i(0, 0), "   and cannot be walked off the top-left")
	cur.move(99, 99)
	# basic is 2x2 on a 20x18 grid, so the furthest legal origin is (18, 16).
	eq(cur.origin, Vector2i(18, 16), "   nor off the bottom-right: it clamps for its own size")

	# Turning can transpose the extent, so turning has to clamp too. A 3x2 at the right
	# edge becomes 2x3 and would hang off the bottom if nothing re-checked.
	cur.select(1)                                  # angled: 3x2
	eq(cur.def().id, &"angled", "a different tower can be selected")
	cur.origin = Vector2i(17, 16)
	cur.turn_footprint(1)                          # now 2x3
	check(f.clamp_origin(cur.origin, cur.def().extent(cur.rot)) == cur.origin,
		"turning the footprint re-clamps, because the extent transposed")
	var span := cur.def().extent(cur.rot)
	check(cur.origin.x + span.x <= 20 and cur.origin.y + span.y <= 18,
		"   so the footprint is still fully on the grid")

	# Aim is independent of the footprint and wraps.
	cur.aim = 0
	cur.turn_aim(1)
	eq(cur.aim, 1, "aim turns one step at a time")
	cur.turn_aim(-2)
	eq(cur.aim, 15, "   and wraps past zero")
	var was_rot := cur.rot
	cur.turn_aim(4)
	eq(cur.rot, was_rot, "turning the aim does not move the footprint")
	var was_aim := cur.aim
	cur.turn_footprint(1)
	eq(cur.aim, was_aim, "   and turning the footprint does not move the aim")

	# Selecting cycles and stays in range.
	cur.selected = 0
	cur.select_next(-1)
	eq(cur.selected, ts.defs.size() - 1, "selecting backwards from the first wraps to the last")
	cur.select(99)
	eq(cur.selected, ts.defs.size() - 1, "   and an out-of-range selection clamps")

	# can_build asks the sim, so the cursor reports permission rather than assuming it.
	var r := DefenceRules.testing()
	r.spawn_interval = 0.0
	var sim := DefenceSim.new(r, ts, 30, 1)
	var cur2 := BuildCursor.new(ts, sim.field)
	cur2.select(0)
	cur2.origin = Vector2i(5, 5)
	check(cur2.can_build(sim), "an empty spot can be built on")
	var ev: Array = []
	sim.build(cur2.def(), cur2.rot, cur2.origin, cur2.aim, ev)
	check(not cur2.can_build(sim), "   and not twice")
	eq(cur2.cells().size(), cur2.def().cell_count(),
		"the ghost covers exactly the footprint's cells")

# ------------------------------------------------------------------------------
# Defence: the coupling to the Tetris board
# ------------------------------------------------------------------------------

func test_defence_coupling() -> void:
	group("Coupling: a breach lands garbage in the Tetris board")

	var r := DefenceRules.testing()
	r.spawn_interval = 0.0
	var sim := DefenceSim.new(r, TowerSet.standard(), 30, 5)
	var g := make_game(Board.new(10, 20, 20))
	g.start()

	# Breach, wait out the window, and hand the flush to Game the way main.gd does.
	sim.garbage.touch()
	var flushed: Array = []
	for _i in 200:
		for e in sim.tick(1.0 / 60.0):
			if e.type == DefenceEvent.Type.GARBAGE_FLUSHED:
				flushed.append(e)
	eq(flushed.size(), 1, "the window closing produces one flush")
	var ev: Array = []
	g.receive_garbage(flushed[0].data["holes"], ev)
	var floor_y: int = g.board.total_height - 1
	var row := _row(g.board, floor_y)
	eq(row.length(), 10, "a garbage row reaches the board")
	eq(row.count("G"), 9, "   solid but for its hole")
	eq(ev.filter(func(e): return e.type == GameEvent.Type.GARBAGE_RECEIVED).size(), 1,
		"   reported through the Tetris event stream")

	group("Coupling: a line clear is an extra shot, not a reset cooldown")
	var r2 := DefenceRules.testing()
	r2.spawn_interval = 0.0
	var sim2 := DefenceSim.new(r2, TowerSet.standard(), 30, 5)
	var ev2: Array = []
	var t := sim2.build(sim2.towers.get_def(&"basic"), 0, Vector2i(4, 4), 0, ev2)
	sim2.tick(1.0 / 60.0)                      # spends the opening shot
	var after_first := sim2.shots_fired
	check(t.cooldown > 0.0, "the tower is on cooldown after firing")
	sim2.grant_free_shots(1)
	eq(t.free_shots, 1, "a clear grants one free shot")
	var cd := t.cooldown
	sim2.tick(1.0 / 60.0)
	eq(sim2.shots_fired, after_first + 1, "   which fires during the cooldown")
	eq(t.free_shots, 0, "   and is spent")
	check(t.cooldown < cd, "   while the cooldown kept counting down rather than resetting")

	group("Coupling: switching boards freezes Tetris, not just its gravity")
	# The trap: scaling gravity to zero stops the FALL but not the LOCK. A piece already
	# resting on the stack is governed by lock_delay_sec, so with gravity alone it locks
	# itself half a second after the player looks away -- which is exactly the thing that
	# reads as "Tetris carried on without me".
	var gf := make_game(Board.new(10, 20, 20))
	gf.start()
	var inp0 := InputState.new()
	# Drop it to the floor so the lock timer is the only thing that can advance.
	while gf.board.can_place(gf.active.cells(), gf.active.pos + Vector2i(0, 1)):
		gf.active.pos.y += 1
	var grounded: int = gf.active.pos.y
	var piece_was: int = gf.pieces_placed

	# Gravity off alone is NOT enough: the piece still locks.
	gf.gravity_scale = 0.0
	for _i in 120:
		gf.tick(inp0)
	check(gf.pieces_placed > piece_was, "with gravity merely zeroed a grounded piece still locks")

	# advancing = false holds it.
	var gh := make_game(Board.new(10, 20, 20))
	gh.start()
	while gh.board.can_place(gh.active.cells(), gh.active.pos + Vector2i(0, 1)):
		gh.active.pos.y += 1
	var held_y: int = gh.active.pos.y
	var held_pieces: int = gh.pieces_placed
	gh.advancing = false
	for _i in 300:
		gh.tick(inp0)
	eq(gh.pieces_placed, held_pieces, "frozen, five seconds grounded does not lock it")
	eq(gh.active.pos.y, held_y, "   and it has not moved")
	check(gh.active != null, "   the piece is still in play")

	# And the timer RESUMES rather than resetting, so the freeze is not free lock delay.
	var partial := gh.active.lock_timer
	gh.advancing = true
	for _i in 10:
		gh.tick(inp0)
	check(gh.active == null or gh.active.lock_timer > partial,
		"unfreezing resumes the lock timer where it stopped")

	# A piece in the AIR is held too, so the freeze covers the fall as well as the lock.
	var ga := make_game(Board.new(10, 20, 20))
	ga.start()
	var air_y: int = ga.active.pos.y
	ga.advancing = false
	for _i in 600:
		ga.tick(inp0)
	eq(ga.active.pos.y, air_y, "an airborne piece does not fall while frozen")

	group("Coupling: gravity scales while the defence board has focus")
	var g2 := make_game(Board.new(10, 20, 20))
	g2.start()
	eq(g2.gravity_scale, 1.0, "gravity is unscaled by default")
	var full := g2.current_gravity()
	g2.gravity_scale = 0.35
	# current_gravity reports the UNSCALED rate; the scale is applied in _handle_gravity,
	# so the readout keeps meaning "this level's gravity" rather than "this frame's".
	eq(g2.current_gravity(), full, "current_gravity still reports the level's rate")
	# Measured through the sim: a scaled piece falls strictly less far in the same ticks.
	var inp := InputState.new()
	var slow_start: int = g2.active.pos.y
	for _i in 120:
		g2.tick(inp)
	var slow_fall: int = g2.active.pos.y - slow_start if g2.active != null else 999
	var g3 := make_game(Board.new(10, 20, 20))
	g3.start()
	var fast_start: int = g3.active.pos.y
	for _i in 120:
		g3.tick(inp)
	var fast_fall: int = g3.active.pos.y - fast_start if g3.active != null else 999
	check(slow_fall < fast_fall, "a scaled piece falls less far in the same time")
	# Zero scale must behave exactly like gravity off, or F3 and focus would disagree.
	var g4 := make_game(Board.new(10, 20, 20))
	g4.start()
	g4.gravity_scale = 0.0
	var held: int = g4.active.pos.y
	for _i in 300:
		g4.tick(inp)
	eq(g4.active.pos.y, held, "a zero scale holds the piece exactly like gravity off")

func test_defence_view_pool() -> void:
	group("DefenceView: entity nodes are pooled, not churned")

	var r := DefenceRules.testing()
	r.spawn_interval = 0.0
	var ts := TowerSet.standard()
	var sim := DefenceSim.new(r, ts, 30, 3)
	var cur := BuildCursor.new(ts, sim.field)
	var l := ViewLayout.new(Board.new(10, 20, 20), Vector2(1280, 720), 30)
	var v := _defence_view(sim, cur, l)

	eq(v.get_child_count(), 0, "a fresh view has no entity nodes")

	# Building emits TOWER_BUILT, which is what creates the node -- lifecycle travels as
	# events, positions do not.
	var ev: Array = []
	var t := sim.build(ts.get_def(&"basic"), 0, Vector2i(4, 4), 0, ev)
	v.handle_events(ev)
	eq(v.get_child_count(), 1, "building a tower instances one node")
	var node: Node2D = v.get_child(0)
	check(node.visible, "   and it is visible")

	v.sync()
	eq(node.position, Vector2(4.0 * 30.0, 4.0 * 30.0),
		"sync puts it at its grid position in FIELD-LOCAL pixels")

	# The board's screen offset lives on the view's own transform, so children inherit it.
	# Getting this wrong is not subtle but it is silent: the grid draws in the pane while
	# every tower, enemy and shot renders over the Tetris board, because the sim works in
	# field-local pixels that start at 0,0.
	eq(v.position, l.defence_origin(r.cols, r.rows),
		"the view carries the board's screen offset")
	var on_screen := v.position + node.position
	check(l.defence_rect.has_point(on_screen),
		"so a tower ends up inside the defence rect, not at the window origin")
	var field_rect := Rect2(l.field_origin, l.field_size)
	check(not field_rect.has_point(on_screen), "   and nowhere near the Tetris playfield")

	# The worst case is an enemy at the spawn edge, which is the furthest-right thing the
	# sim ever produces -- that is what showed the bug.
	var spawn_x := r.cols * 30 - int(r.enemy_half_extents.x)
	var at_edge := v.position + Vector2(float(spawn_x), 255.0)
	check(l.defence_rect.has_point(at_edge), "an enemy at the spawn edge is still in the pane")
	check(not field_rect.has_point(at_edge), "   and not over the Tetris board")

	# Releasing hides and pools rather than freeing. That is the point: a freed node with
	# a live id in an event loop is the bug class this avoids.
	ev.clear()
	sim.remove_tower(t, ev)
	v.handle_events(ev)
	eq(v.get_child_count(), 1, "removing it does not free the node")
	check(not node.visible, "   it is hidden instead")

	ev.clear()
	sim.build(ts.get_def(&"basic"), 0, Vector2i(8, 4), 0, ev)
	v.handle_events(ev)
	eq(v.get_child_count(), 1, "building again reuses the pooled node rather than adding one")
	check(node.visible, "   which is visible once more")

	# The textures live on the Sprite2D nodes in the scenes, not behind an export on the
	# root. Assert they actually arrive: an empty-looking sprite in the editor was the
	# symptom that the export indirection was wrong, and a silently untextured tile would
	# look identical at runtime to art that failed to import.
	var tiles := node.get_node_or_null("Body")
	check(tiles != null, "a tower node has a Body")
	if tiles != null:
		var untextured: Array = []
		var shown := 0
		for c in tiles.get_children():
			var sp := c as Sprite2D
			if sp == null or not sp.visible:
				continue
			shown += 1
			if sp.texture == null:
				untextured.append(String(sp.name))
		eq(shown, 4, "   with one visible tile per cell of a 2x2 footprint")
		check(untextured.is_empty(), "   and every tile textured from the scene%s" % [
			"" if untextured.is_empty() else " (%s)" % str(untextured)])
	for n2 in ["Barrel", "Mount"]:
		var sp2 := node.get_node_or_null(n2) as Sprite2D
		check(sp2 != null and sp2.texture != null, "the %s sprite carries its own texture" % n2)

	# A 5-cell footprint needs more tiles than the 2x2 did, which exercises duplicating the
	# template tile -- and the duplicates must be textured too.
	ev.clear()
	var t5 := sim.build(ts.get_def(&"angled"), 0, Vector2i(12, 10), 0, ev)
	v.handle_events(ev)
	v.sync()
	var n5: Node2D = v.get_child(v.get_child_count() - 1)
	var body5 := n5.get_node_or_null("Body")
	if body5 != null:
		var vis5 := 0
		var bare5 := 0
		for c in body5.get_children():
			var sp3 := c as Sprite2D
			if sp3 != null and sp3.visible:
				vis5 += 1
				if sp3.texture == null:
					bare5 += 1
		eq(vis5, 5, "a 5-cell tower grows to five visible tiles")
		eq(bare5, 0, "   all of them textured, including the duplicated ones")
	# Release it through the EVENT, not just out of the sim: dropping the tower without
	# telling the view leaves its node visible forever, which is the bug the pool exists to
	# make impossible to hit by accident.
	ev.clear()
	sim.remove_tower(t5, ev)
	v.handle_events(ev)

	# Shots and enemies pool independently, so a shot cannot be handed an enemy's node.
	ev.clear()
	var e := DefenceEnemy.new(sim._id(), Vector2(500.0, 100.0), r.enemy_hp, 3, r.enemy_half_extents)
	sim.enemies.append(e)
	ev.append(DefenceEvent.new(DefenceEvent.Type.ENEMY_SPAWNED, {"id": e.id, "enemy": e}))
	var sh := DefenceShot.new(sim._id(), Vector2(300.0, 200.0), 0, r.projectile_speed, 0)
	sim.shots.append(sh)
	ev.append(DefenceEvent.new(DefenceEvent.Type.SHOT_FIRED,
		{"id": sh.id, "shot": sh, "tower_id": 0, "cause": "cooldown"}))
	v.handle_events(ev)
	eq(_visible_children(v), 3, "an enemy and a shot each get their own node")
	v.sync()

	# Ids are never reused, so a release cannot hide the wrong node.
	ev.clear()
	ev.append(DefenceEvent.new(DefenceEvent.Type.SHOT_EXPIRED, {"id": sh.id, "reason": "life"}))
	v.handle_events(ev)
	eq(_visible_children(v), 2, "expiring the shot hides exactly one node")

	# A breach releases the body too -- the sim has already despawned it, so a node left
	# visible would be a ghost standing at the wall forever.
	ev.clear()
	ev.append(DefenceEvent.new(DefenceEvent.Type.ENEMY_BREACHED, {"id": e.id, "pos": e.pos}))
	v.handle_events(ev)
	eq(_visible_children(v), 1, "a breach releases the enemy's node")

	# setup() again is a restart: everything hides and returns to its pool, and nothing is
	# instanced a second time.
	var before := v.get_child_count()
	v.setup(sim, cur, l)
	eq(v.get_child_count(), before, "a restart reuses every node it already made")
	eq(_visible_children(v), 0, "   and leaves none of them showing")

	v.free()

# ------------------------------------------------------------------------------
# Defence: the sim
# ------------------------------------------------------------------------------

## A sim with spawning off, so a test only sees the traffic it asks for.
func _def_sim(seed_v: int = 1) -> DefenceSim:
	var r := DefenceRules.testing()
	r.spawn_interval = 0.0
	return DefenceSim.new(r, TowerSet.standard(), 30, seed_v)

func _of_type(events: Array, t: int) -> Array:
	return events.filter(func(e): return e.type == t)

# ------------------------------------------------------------------------------
# Defence: the view's entity pool
# ------------------------------------------------------------------------------

## DefenceView is a Node2D with no class_name, so it is loaded by path. It needs no tree:
## _draw never runs here, and add_child works on a detached node.
## Live entity nodes. Counted instead of get_child_count() because the pool only ever
## grows: a test that asserts on children is really asserting how many entities every
## earlier line happened to create.
func _visible_children(v: Node2D) -> int:
	var n := 0
	for c in v.get_children():
		if (c as Node2D) != null and (c as Node2D).visible:
			n += 1
	return n

func _defence_view(sim: DefenceSim, cur: BuildCursor, l: ViewLayout) -> Node2D:
	var v: Node2D = load("res://view/defence_view.gd").new()
	v.setup(sim, cur, l)
	return v
