extends TestHarness

## Pieces: rotation, kicks and spawning
##
## Run by tests/run_tests.gd, which owns the totals. See tests/test_harness.gd for
## group/check/eq and the shared board fixtures.

func run() -> void:
	test_rotation_primitive()
	test_kick_table_values()
	test_kick_table_structure()
	test_spawn_positions()
	test_tsd()
	test_tsd_via_hard_drop()
	test_tst()
	test_tss_mini()
	test_i_piece_kick()
	test_left_wall_kick()
	test_o_never_kicks()

# ------------------------------------------------------------------------------
# Rotation primitive and kick data
# ------------------------------------------------------------------------------

func test_rotation_primitive() -> void:
	group("rotation primitive")
	var ps := PieceSet.tetrominoes()
	for d in ps.defs:
		eq(d.rotations.size(), 4, "%s has 4 rotation states" % d.id)
		var counts_match := true
		for r in 4:
			if d.rotations[r].size() != d.rotations[0].size():
				counts_match = false
		check(counts_match, "%s keeps its cell count through every rotation" % d.id)
		# Four clockwise turns must be the identity, for any box size.
		var cur: Array = d.rotations[0]
		for _i in 4:
			cur = PieceSet.rotate_cw(cur, d.box)
		check(same_cells(cur, d.rotations[0]), "%s: 4x cw == identity (box %d)" % [d.id, d.box])
	eq(ps.get_def(&"I").box, 4, "I uses a 4x4 box")
	eq(ps.get_def(&"O").box, 2, "O uses a 2x2 box (rotation-invariant by construction)")
	eq(ps.get_def(&"T").box, 3, "T uses a 3x3 box")
	eq(ps.get_def(&"I").first_filled_row, 1, "I spawn row sits on box row 1")
	eq(ps.get_def(&"T").first_filled_row, 0, "T spawn row sits on box row 0")

func test_kick_table_values() -> void:
	group("kick table values (y-down, after the single y-flip)")
	# jlstz 0 -> R. y-up (0,0)(-1,0)(-1,+1)(0,-2)(-1,-2) becomes:
	eq(KickTables.get_kicks(&"jlstz", 0, 1),
		[Vector2i(0, 0), Vector2i(-1, 0), Vector2i(-1, -1), Vector2i(0, 2), Vector2i(-1, 2)],
		"jlstz 0->R")
	eq(KickTables.get_kicks(&"jlstz", 1, 0),
		[Vector2i(0, 0), Vector2i(1, 0), Vector2i(1, 1), Vector2i(0, -2), Vector2i(1, -2)],
		"jlstz R->0")
	eq(KickTables.get_kicks(&"jlstz", 3, 2),
		[Vector2i(0, 0), Vector2i(-1, 0), Vector2i(-1, 1), Vector2i(0, -2), Vector2i(-1, -2)],
		"jlstz L->2")
	eq(KickTables.get_kicks(&"jlstz", 0, 3),
		[Vector2i(0, 0), Vector2i(1, 0), Vector2i(1, -1), Vector2i(0, 2), Vector2i(1, 2)],
		"jlstz 0->L")
	# I has its own table; the magnitude-2 offsets are what make I-spins possible.
	eq(KickTables.get_kicks(&"i", 0, 1),
		[Vector2i(0, 0), Vector2i(-2, 0), Vector2i(1, 0), Vector2i(-2, 1), Vector2i(1, -2)],
		"i 0->R")
	eq(KickTables.get_kicks(&"i", 1, 0),
		[Vector2i(0, 0), Vector2i(2, 0), Vector2i(-1, 0), Vector2i(2, -1), Vector2i(-1, 2)],
		"i R->0")
	check(KickTables.get_kicks(&"jlstz", 0, 1) != KickTables.get_kicks(&"i", 0, 1),
		"I and JLSTZ tables are distinct")

func test_kick_table_structure() -> void:
	group("kick table structure")
	var quarters := [[0, 1], [1, 0], [1, 2], [2, 1], [2, 3], [3, 2], [3, 0], [0, 3]]
	for cls in [&"jlstz", &"i"]:
		for t in quarters:
			eq(KickTables.get_kicks(cls, t[0], t[1]).size(), 5,
				"%s %d->%d has 5 offsets" % [cls, t[0], t[1]])
		for t in [[0, 2], [2, 0], [1, 3], [3, 1]]:
			check(KickTables.get_kicks(cls, t[0], t[1]).size() >= 1,
				"%s %d->%d (180) has offsets" % [cls, t[0], t[1]])
	var o_all_identity := true
	for a in 4:
		for b in 4:
			if a == b:
				continue
			var k := KickTables.get_kicks(&"o", a, b)
			if k.size() != 1 or k[0] != Vector2i.ZERO:
				o_all_identity = false
	check(o_all_identity, "o table is identity-only in every direction")

func test_spawn_positions() -> void:
	group("spawn positions")
	var g := make_game(Board.new(10, 20, 20))
	for d in g.piece_set.defs:
		var pos := g.spawn_pos_for(d)
		var top := 0x7FFFFFFF
		var lo := 0x7FFFFFFF
		var hi := -0x7FFFFFFF
		for c in d.rotations[0]:
			top = mini(top, pos.y + c.y)
			lo = mini(lo, pos.x + c.x)
			hi = maxi(hi, pos.x + c.x)
		eq(top, g.board.buffer_height, "%s spawns with its top row on the first visible row" % d.id)
		check(lo >= 0 and hi < g.board.width, "%s spawns inside the field (cols %d..%d)" % [d.id, lo, hi])
	# Guideline columns, 0-indexed.
	eq(g.spawn_pos_for(g.piece_set.get_def(&"T")).x, 3, "T spawns at box column 3")
	eq(g.spawn_pos_for(g.piece_set.get_def(&"I")).x, 3, "I spawns at box column 3 (cells 3..6)")
	eq(g.spawn_pos_for(g.piece_set.get_def(&"O")).x, 4, "O spawns at box column 4 (cells 4..5)")

# ------------------------------------------------------------------------------
# Hand-derived kick cases
# ------------------------------------------------------------------------------

func test_tsd() -> void:
	group("T-spin Double")
	# Slot at columns 3..5. The T enters vertically through the 2-wide gap at
	# columns 4..5 and rotates L->2; offsets 0 and 1 are blocked, offset 2 fits.
	var b := make_board([
		"..........",
		"XXXX..XXXX",
		"XXX...XXXX",
		"XXXX.XXXXX",
	])
	var g := make_game(b)
	var p := place(g, &"T", 3, Vector2i(4, 36))
	check(b.can_place(p.cells(), p.pos), "start position is legal")
	check(g.try_rotate(2), "L->2 rotation succeeds")
	eq(p.last_kick_index, 2, "used kick offset index 2")
	eq(p.pos, Vector2i(3, 37), "kicked to (3, 37)")
	eq(SpinDetect.detect(b, p, g.rules), SpinDetect.Kind.FULL, "full T-spin (both front corners filled)")
	var events: Array = []
	g.lock_piece(events)
	var r := first_clear(events)
	check(r != null, "a clear was reported")
	if r != null:
		eq(r.lines, 2, "cleared 2 lines")
		eq(r.spin, SpinDetect.Kind.FULL, "recorded as a full spin")
		eq(r.piece_id, &"T", "credited to the T")
		check(r.is_difficult(), "counts as a difficult clear for B2B")

func test_tsd_via_hard_drop() -> void:
	group("T-spin Double through the full input path")
	# The way it actually happens in play: rotate, then hard drop. A seated piece
	# cannot fall, so the spin must survive the drop.
	var b := make_board([
		"..........",
		"XXXX..XXXX",
		"XXX...XXXX",
		"XXXX.XXXXX",
	])
	var g := make_game(b)
	place(g, &"T", 3, Vector2i(4, 36))
	var rotate := InputState.new()
	rotate.rotate_ccw = true
	g.tick(rotate)
	eq(g.active.rot, 2, "tick() applied the ccw rotation")
	eq(g.active.last_kick_index, 2, "same kick as the direct call")

	var drop := InputState.new()
	drop.hard_drop = true
	var r := first_clear(g.tick(drop))
	check(r != null, "hard drop produced a clear")
	if r != null:
		eq(r.lines, 2, "cleared 2 lines")
		eq(r.spin, SpinDetect.Kind.FULL, "the T-spin survived the hard drop")

	# The converse: a rotation followed by a drop that actually moves is not a spin.
	var b2 := make_board(["..........", "..........", "..........", ".........."])
	var g2 := make_game(b2)
	place(g2, &"T", 0, Vector2i(3, 30))
	g2.tick(rotate)
	var events2 := g2.tick(drop)
	eq(g2.last_spin, SpinDetect.Kind.NONE, "a rotation then a real fall is not a spin")
	check(events2.size() > 0, "the drop still locked the piece")

func test_tst() -> void:
	group("T-spin Triple")
	# The 1x2 tuck. Column 3 is a 3-deep well; the overhang at (3, 35) blocks the
	# early offsets, so only offset index 4 -- the mini-to-full upgrade kick -- fits.
	var b := make_board([
		"...X......",
		"..........",
		"XXX.XXXXXX",
		"XX..XXXXXX",
		"XXX.XXXXXX",
	])
	var g := make_game(b)
	var p := place(g, &"T", 0, Vector2i(1, 35))
	check(b.can_place(p.cells(), p.pos), "start position is legal")
	check(g.try_rotate(3), "0->L rotation succeeds")
	eq(p.last_kick_index, 4, "used kick offset index 4 (the 1x2 tuck)")
	eq(p.pos, Vector2i(2, 37), "kicked to (2, 37)")
	eq(SpinDetect.detect(b, p, g.rules), SpinDetect.Kind.FULL, "full T-spin")
	var events: Array = []
	g.lock_piece(events)
	var r := first_clear(events)
	check(r != null, "a clear was reported")
	if r != null:
		eq(r.lines, 3, "cleared 3 lines")
		eq(r.spin, SpinDetect.Kind.FULL, "recorded as a full spin")

func test_tss_mini() -> void:
	group("T-spin Mini")
	# Three corners filled but only one of the two front corners, and no tuck kick,
	# so this must come out Mini rather than full.
	var b := make_board([
		"...X......",
		"..........",
		"...X.X....",
	])
	var g := make_game(b)
	var p := place(g, &"T", 1, Vector2i(3, 37))
	check(b.can_place(p.cells(), p.pos), "start position is legal")
	check(g.try_rotate(0), "R->0 rotation succeeds")
	eq(p.last_kick_index, 0, "no kick needed (offset index 0)")
	eq(p.pos, Vector2i(3, 37), "position unchanged")
	eq(SpinDetect.detect(b, p, g.rules), SpinDetect.Kind.MINI, "detected as Mini")
	# With the upgrade rule on but kick index 0, it must stay Mini.
	check(g.rules.mini_upgrade_on_kick5, "upgrade-on-tuck rule is enabled")

func test_i_piece_kick() -> void:
	group("I-piece kick (uses the I table, not JLSTZ)")
	# Vertical I in column 8. Offsets 0-3 are blocked by (6,37) or by the right wall,
	# so offset 4 lands it two rows lower -- an offset the JLSTZ table does not have.
	var b := make_board([
		"..........",
		"......X...",
		"..........",
		"..........",
	])
	var g := make_game(b)
	var p := place(g, &"I", 1, Vector2i(6, 36))
	check(b.can_place(p.cells(), p.pos), "start position is legal")
	check(g.try_rotate(0), "R->0 rotation succeeds")
	eq(p.last_kick_index, 4, "used kick offset index 4")
	eq(p.pos, Vector2i(5, 38), "kicked to (5, 38)")
	check(not KickTables.get_kicks(&"jlstz", 1, 0).has(Vector2i(-1, 2)),
		"that offset does not exist in the JLSTZ table, so the I table was used")

func test_left_wall_kick() -> void:
	group("left wall kick")
	var b := make_board(["..........", "..........", ".........."])
	var g := make_game(b)
	var p := place(g, &"T", 1, Vector2i(-1, 37))
	check(b.can_place(p.cells(), p.pos), "vertical T flush against the left wall is legal")
	check(g.try_rotate(2), "R->2 rotation succeeds")
	eq(p.last_kick_index, 1, "used kick offset index 1 (pushed right off the wall)")
	eq(p.pos, Vector2i(0, 37), "kicked to (0, 37)")

func test_o_never_kicks() -> void:
	group("O never kicks")
	var b := make_board(["..........", ".........."])
	var g := make_game(b)
	var p := place(g, &"O", 0, Vector2i(4, 38))
	var before := p.cells().duplicate()
	for target in [1, 2, 3, 0]:
		check(g.try_rotate(target), "O rotates to state %d" % target)
		eq(p.pos, Vector2i(4, 38), "O position unchanged after rotating to %d" % target)
		check(same_cells(p.cells(), before), "O occupies the same cells in state %d" % target)

func same_cells(a: Array, b: Array) -> bool:
	if a.size() != b.size():
		return false
	var set := {}
	for c in a:
		set[c] = true
	for c in b:
		if not set.has(c):
			return false
	return true
