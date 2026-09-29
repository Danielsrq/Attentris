extends Node

## Headless assertions over the pure sim. Run with:
##   godot --headless --path . res://tests/run_tests.tscn
## Exits 0 when everything passes, 1 otherwise.
##
## The kick cases below are hand-derived: each one states the expected kick offset
## index, so a sign error in the y-up -> y-down flip in KickTables fails loudly here
## instead of feeling subtly wrong in play.

var _pass := 0
var _fail := 0
var _group := ""

func _ready() -> void:
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
	test_board_clears()
	test_board_perfect_clear_and_remove_row()
	test_randomizer_bag()
	test_randomizer_seed()
	test_all_spin_immobility()
	test_scorer_values()
	test_b2b_and_combo_chain()
	test_hold_swap()
	test_width_12()
	test_control_profiles()
	test_outcome_names()
	test_zero_line_spin_scoring()
	test_audio_events()
	test_audio_cue_mapping()
	test_placement_enumeration()
	test_quad_evaluator()
	test_well_detection_with_holes()
	test_parity_property()
	test_attack_table()
	test_dependency_attribution()
	test_hole_topology()
	test_tspin_slots()
	test_garbage_push()
	test_ghost_pocket()
	test_shape_resources()
	test_structures()
	test_resources()
	test_pane_tab()

	print("")
	print("%d passed, %d failed" % [_pass, _fail])
	get_tree().quit(1 if _fail > 0 else 0)

# ------------------------------------------------------------------------------
# Harness
# ------------------------------------------------------------------------------

func group(name: String) -> void:
	_group = name
	print("")
	print("-- %s" % name)

func check(cond: bool, what: String) -> void:
	if cond:
		_pass += 1
		print("   ok    %s" % what)
	else:
		_fail += 1
		printerr("   FAIL  [%s] %s" % [_group, what])
		print("   FAIL  %s" % what)

func eq(got, want, what: String) -> void:
	if typeof(got) == typeof(want) and got == want:
		_pass += 1
		print("   ok    %s" % what)
	else:
		_fail += 1
		printerr("   FAIL  [%s] %s -- got %s, want %s" % [_group, what, got, want])
		print("   FAIL  %s -- got %s, want %s" % [what, got, want])

## Build a board from ASCII rows. Rows are given top-to-bottom and anchored to the
## BOTTOM of the field, so the last string is the floor row.
func make_board(rows: Array, w: int = 10) -> Board:
	var b := Board.new(w, 20, 20)
	var base := b.total_height - rows.size()
	for i in rows.size():
		var line: String = rows[i]
		for x in line.length():
			if line[x] == "X":
				b.set_cell(x, base + i, 1)
	return b

func make_game(b: Board, rs: RuleSet = null) -> Game:
	var rules := rs if rs != null else RuleSet.competitive()
	return Game.new(rules, PieceSet.tetrominoes(), b, Scorer.new(ScoreTable.new()), 12345)

## Place a piece by hand without spawning, for kick tests.
func place(g: Game, id: StringName, rot: int, pos: Vector2i) -> ActivePiece:
	var p := ActivePiece.new()
	p.def = g.piece_set.get_def(id)
	p.rot = rot
	p.pos = pos
	p.lowest_row = pos.y
	g.active = p
	return p

func first_clear(events: Array) -> ClearResult:
	for e in events:
		if e.type == GameEvent.Type.LINES_CLEARED:
			return e.data["result"]
	return null

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

# ------------------------------------------------------------------------------
# Board
# ------------------------------------------------------------------------------

func test_board_clears() -> void:
	group("board clears")
	var b := make_board([
		"X.........",
		"XXXXXXXXXX",
		"..X.......",
		"XXXXXXXXXX",
	])
	eq(b.find_full_rows(), [37, 39], "found the two full rows")
	eq(b.clear_rows([37, 39]), 2, "cleared 2 rows")
	check(b.find_full_rows().is_empty(), "no full rows remain")
	# Non-contiguous clear: the survivors keep their order and fall to the bottom.
	eq(b.get_cell(2, 39), 1, "row that sat between the cleared rows is now the floor")
	eq(b.get_cell(0, 38), 1, "row above the cleared rows dropped by two")
	check(b.row_is_empty(37), "the rows vacated at the top are empty")

	var b2 := make_board(["XXXXXXXXXX", "XXXXXXXXXX", "XXXXXXXXXX", "XXXXXXXXXX"])
	eq(b2.find_full_rows().size(), 4, "tetris: four full rows")
	b2.clear_rows(b2.find_full_rows())
	check(b2.is_empty(), "board empty after clearing all four")

func test_board_perfect_clear_and_remove_row() -> void:
	group("perfect clear and remove_row")
	var b := make_board(["XXXXXXXXXX"])
	b.clear_rows(b.find_full_rows())
	check(b.is_empty(), "perfect clear detected as an empty board")

	var b2 := make_board(["X.........", ".X........", "..X......."])
	b2.remove_row(38)   # the middle of the three
	eq(b2.get_cell(2, 39), 1, "bottom row untouched")
	eq(b2.get_cell(0, 38), 1, "row above dropped into the gap")
	check(b2.row_is_empty(37), "top vacated")

# ------------------------------------------------------------------------------
# Randomizer
# ------------------------------------------------------------------------------

func test_randomizer_bag() -> void:
	group("7-bag randomizer")
	var ps := PieceSet.tetrominoes()
	var r := Randomizer.new(ps, 99, ["S", "Z", "O"])
	var first_bag := {}
	for _i in 7:
		first_bag[r.next().id] = true
	eq(first_bag.size(), 7, "first 7 draws are a permutation of all 7 pieces")

	var counts := {}
	var r2 := Randomizer.new(ps, 7, [])
	for _i in 70:
		var id = r2.next().id
		counts[id] = counts.get(id, 0) + 1
	var all_ten := true
	for id in counts:
		if counts[id] != 10:
			all_ten = false
	eq(counts.size(), 7, "70 draws cover all 7 pieces")
	check(all_ten, "70 draws yield exactly 10 of each")

	var no_bad_opener := true
	for s in range(200):
		var r3 := Randomizer.new(ps, s, ["S", "Z", "O"])
		var id := String(r3.next().id)
		if id == "S" or id == "Z" or id == "O":
			no_bad_opener = false
			break
	check(no_bad_opener, "opening piece is never S, Z or O across 200 seeds")

func test_randomizer_seed() -> void:
	group("randomizer determinism")
	var ps := PieceSet.tetrominoes()
	var a: Array = []
	var b: Array = []
	var ra := Randomizer.new(ps, 424242, [])
	var rb := Randomizer.new(ps, 424242, [])
	for _i in 35:
		a.append(ra.next().id)
		b.append(rb.next().id)
	eq(a, b, "same seed produces the same 35-piece sequence")
	var rc := Randomizer.new(ps, 424243, [])
	var c: Array = []
	for _i in 35:
		c.append(rc.next().id)
	check(a != c, "a different seed produces a different sequence")

# ------------------------------------------------------------------------------
# Spin detection
# ------------------------------------------------------------------------------

func test_all_spin_immobility() -> void:
	group("all-spin (immobility)")
	# S wedged so it cannot move left, right, up or down.
	var b := make_board([
		".X........",
		"...X......",
		"..........",
	])
	var g := make_game(b)
	var p := place(g, &"S", 0, Vector2i(0, 38))
	check(b.can_place(p.cells(), p.pos), "S position is legal")
	check(SpinDetect.is_immobile(b, p), "S is immobile in all four directions")
	p.last_action_was_rotation = true
	eq(SpinDetect.detect(b, p, g.rules), SpinDetect.Kind.FULL, "counts as a full S-spin")

	# The same S, but the rotation flag clear: nothing counts.
	p.last_action_was_rotation = false
	eq(SpinDetect.detect(b, p, g.rules), SpinDetect.Kind.NONE,
		"no spin unless the last action was a rotation")

	# And with all-spin off, a non-T piece never scores a spin.
	var strict := RuleSet.guideline()
	p.last_action_was_rotation = true
	eq(SpinDetect.detect(b, p, strict), SpinDetect.Kind.NONE, "all_spin off: S-spin ignored")

	# Free space is never a spin.
	var empty := Board.new(10, 20, 20)
	var g2 := make_game(empty)
	var p2 := place(g2, &"S", 0, Vector2i(4, 30))
	p2.last_action_was_rotation = true
	check(not SpinDetect.is_immobile(empty, p2), "S in open space is mobile")
	eq(SpinDetect.detect(empty, p2, g2.rules), SpinDetect.Kind.NONE, "open-space rotation is not a spin")

# ------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------

func make_result(lines: int, spin: int, piece: StringName, b2b: int, combo: int) -> ClearResult:
	var r := ClearResult.new()
	r.lines = lines
	r.spin = spin
	r.piece_id = piece
	r.b2b = b2b
	r.combo = combo
	r.level = 1
	return r

func test_scorer_values() -> void:
	group("scorer")
	var s := Scorer.new(ScoreTable.new())
	eq(s.score_clear(make_result(1, SpinDetect.Kind.NONE, &"L", -1, 0)), 100, "single = 100")
	eq(s.score_clear(make_result(2, SpinDetect.Kind.NONE, &"L", -1, 0)), 300, "double = 300")
	eq(s.score_clear(make_result(3, SpinDetect.Kind.NONE, &"L", -1, 0)), 500, "triple = 500")
	eq(s.score_clear(make_result(4, SpinDetect.Kind.NONE, &"I", -1, 0)), 800, "tetris = 800")
	eq(s.score_clear(make_result(4, SpinDetect.Kind.NONE, &"I", 1, 0)), 1200, "b2b tetris = 800 x 1.5")
	eq(s.score_clear(make_result(0, SpinDetect.Kind.FULL, &"T", -1, -1)), 400, "T-spin no lines = 400")
	eq(s.score_clear(make_result(1, SpinDetect.Kind.FULL, &"T", -1, 0)), 800, "TSS = 800")
	eq(s.score_clear(make_result(2, SpinDetect.Kind.FULL, &"T", -1, 0)), 1200, "TSD = 1200")
	eq(s.score_clear(make_result(3, SpinDetect.Kind.FULL, &"T", -1, 0)), 1600, "TST = 1600")
	eq(s.score_clear(make_result(1, SpinDetect.Kind.MINI, &"T", -1, 0)), 200, "T-spin Mini single = 200")
	eq(s.score_clear(make_result(1, SpinDetect.Kind.NONE, &"L", -1, 3)), 250, "single with 3 combo = 100 + 150")
	# A non-T spin uses the separate all-spin table, so it is tunable independently.
	eq(s.score_clear(make_result(1, SpinDetect.Kind.FULL, &"S", -1, 0)), 200, "S-spin single uses other_spin_scores")

func test_b2b_and_combo_chain() -> void:
	group("B2B and combo bookkeeping")
	var g := make_game(Board.new(10, 20, 20))
	eq(g.b2b, -1, "b2b starts inactive")
	eq(g.combo, -1, "combo starts inactive")
	# Two tetrises in a row: b2b -1 -> 0 -> 1, combo -1 -> 0 -> 1.
	for attempt in 2:
		for y in range(36, 40):
			for x in range(1, 10):
				g.board.set_cell(x, y, 1)
		# Leave a stray block outside the cleared rows so this is not a perfect clear.
		g.board.set_cell(5, 34, 1)
		place(g, &"I", 1, Vector2i(-2, 36))
		var events: Array = []
		g.lock_piece(events)
		var r := first_clear(events)
		check(r != null, "tetris %d reported a clear" % (attempt + 1))
		if r != null:
			eq(r.lines, 4, "tetris %d cleared 4 lines" % (attempt + 1))
			eq(r.b2b, attempt, "tetris %d has b2b == %d" % [attempt + 1, attempt])
			eq(r.combo, attempt, "tetris %d has combo == %d" % [attempt + 1, attempt])
			eq(r.b2b_active(), attempt == 1, "tetris %d b2b bonus active == %s" % [attempt + 1, attempt == 1])
		g.board.clear()
	# A non-difficult clear breaks the chain.
	g.b2b = 3
	for x in g.board.width:
		g.board.set_cell(x, 39, 1)
	g.board.set_cell(4, 39, 0)
	place(g, &"T", 2, Vector2i(3, 37))
	var ev2: Array = []
	g.lock_piece(ev2)
	eq(g.b2b, -1, "a plain single breaks the back-to-back chain")

func test_hold_swap() -> void:
	group("hold / swap")
	var g := make_game(Board.new(10, 20, 20))
	g.start()
	var first: StringName = g.active.def.id
	var queued: StringName = g.next_queue[0].id
	var input := InputState.new()
	input.hold = true
	g.tick(input)
	eq(g.hold_def.id, first, "first piece went into hold")
	eq(g.active.def.id, queued, "the queued piece came out")
	check(g.hold_used, "hold is now spent for this piece")

	var swapped: StringName = g.active.def.id
	g.tick(input)
	eq(g.active.def.id, swapped, "a second hold in the same piece is refused")

	# After a lock the hold is available again, and the next swap returns the piece
	# that has been sitting in hold since the first swap. Drop the piece to the floor
	# first -- locking it at the spawn position would block out the next spawn.
	while g.board.can_place(g.active.cells(), g.active.pos + Vector2i(0, 1)):
		g.active.pos.y += 1
	var events: Array = []
	g.lock_piece(events)
	check(not g.topped_out, "locking at the floor does not top out")
	check(not g.hold_used, "hold is released on lock")
	var third: StringName = g.active.def.id
	g.tick(input)
	eq(g.hold_def.id, third, "the new piece went into hold")
	eq(g.active.def.id, first, "the originally held piece came back out")

func test_width_12() -> void:
	group("12-wide field (parameterisation smoke test)")
	var b := Board.new(12, 20, 20)
	# Fast fixed gravity so a short run actually places pieces.
	var rs := RuleSet.competitive()
	rs.gravity_mode = RuleSet.GravityMode.FIXED
	rs.fixed_gravity_cells_per_sec = 40.0
	var g := make_game(b, rs)
	eq(g.spawn_pos_for(g.piece_set.get_def(&"T")).x, 4, "T re-centres on a 12-wide field")
	eq(g.spawn_pos_for(g.piece_set.get_def(&"I")).x, 4, "I re-centres on a 12-wide field")
	g.start()
	var input := InputState.new()
	var ticks := 0
	while ticks < 3000 and not g.topped_out:
		g.tick(input)
		ticks += 1
	check(g.pieces_placed >= 10, "unattended run placed %d pieces before topping out" % g.pieces_placed)

	# The extra two columns must actually be usable: slide a piece to the right wall
	# and confirm it reaches column 11.
	var g2 := make_game(Board.new(12, 20, 20))
	g2.start()
	var slide := InputState.new()
	slide.right = true
	for _i in 30:
		g2.tick(slide)
	var rightmost := -1
	for c in g2.active.board_cells():
		rightmost = maxi(rightmost, c.x)
	eq(rightmost, 11, "a piece slides all the way to column 11 on a 12-wide field")

func test_control_profiles() -> void:
	group("control profiles")
	var meta_keys := {}
	for action in InputReader.META:
		for k in InputReader.META[action]:
			meta_keys[k] = action

	for profile_name in ControlProfiles.names():
		var p := ControlProfiles.get_profile(profile_name)
		var missing: Array = []
		for action in ControlProfiles.ACTIONS:
			if not p.has(action):
				missing.append(action)
		check(missing.is_empty(), "%s defines every action%s" % [
			profile_name, "" if missing.is_empty() else " (missing %s)" % str(missing)])

		# No key may drive two actions inside one profile, and no key may collide
		# with a meta key, or restart/debug would fire mid-game.
		var seen := {}
		var dupes: Array = []
		var clashes: Array = []
		for action in ControlProfiles.ACTIONS:
			for k in p.get(action, []):
				if seen.has(k):
					dupes.append("%s uses the same key as %s" % [action, seen[k]])
				seen[k] = action
				if meta_keys.has(k):
					clashes.append("%s clashes with meta %s" % [action, meta_keys[k]])
		check(dupes.is_empty(), "%s has no duplicate bindings%s" % [
			profile_name, "" if dupes.is_empty() else " (%s)" % str(dupes)])
		check(clashes.is_empty(), "%s does not collide with the meta keys%s" % [
			profile_name, "" if clashes.is_empty() else " (%s)" % str(clashes)])

		# Every action except 180 must actually be bound to something.
		var unbound: Array = []
		for action in ControlProfiles.ACTIONS:
			if action != "rotate_180" and p.get(action, []).is_empty():
				unbound.append(action)
		check(unbound.is_empty(), "%s binds every essential action%s" % [
			profile_name, "" if unbound.is_empty() else " (unbound %s)" % str(unbound)])

	# The personal profile, spelled out.
	var w := ControlProfiles.wasd_numpad()
	eq(w["left"], [KEY_A], "wasd_numpad: left is A")
	eq(w["right"], [KEY_D], "wasd_numpad: right is D")
	eq(w["soft_drop"], [KEY_W], "wasd_numpad: soft drop is W")
	eq(w["hard_drop"], [KEY_S], "wasd_numpad: hard drop is S")
	eq(w["rotate_ccw"], [KEY_KP_7], "wasd_numpad: ccw is numpad 7")
	eq(w["rotate_cw"], [KEY_KP_9], "wasd_numpad: cw is numpad 9")

	# Cycling wraps and always lands on a real profile.
	var reader := InputReader.new("wasd_numpad")
	eq(reader.profile_name, "wasd_numpad", "reader starts on the requested profile")
	var visited: Array = []
	for _i in ControlProfiles.names().size():
		visited.append(reader.profile_name)
		reader.next_profile()
	eq(visited.size(), ControlProfiles.names().size(), "cycling visits every profile")
	eq(reader.profile_name, "wasd_numpad", "cycling wraps back to the start")
	check(reader.binding_text().size() > 0, "binding text renders for the overlay")
	eq(ControlProfiles.keys_text(ControlProfiles.guideline(), "rotate_180"), "-",
		"an unbound action reads as a dash")

func test_outcome_names() -> void:
	group("outcome naming")
	eq(make_result(1, SpinDetect.Kind.NONE, &"L", -1, 0).describe(), "Single", "1 line")
	eq(make_result(2, SpinDetect.Kind.NONE, &"L", -1, 0).describe(), "Double", "2 lines")
	eq(make_result(3, SpinDetect.Kind.NONE, &"L", -1, 0).describe(), "Triple", "3 lines")
	eq(make_result(4, SpinDetect.Kind.NONE, &"I", -1, 0).describe(), "Quad", "4 lines")
	eq(make_result(0, SpinDetect.Kind.NONE, &"L", -1, -1).describe(), "",
		"a placement that neither spins nor clears has no name")

	eq(make_result(1, SpinDetect.Kind.FULL, &"T", -1, 0).describe(), "T-Spin Single", "TSS")
	eq(make_result(2, SpinDetect.Kind.FULL, &"T", -1, 0).describe(), "T-Spin Double", "TSD")
	eq(make_result(3, SpinDetect.Kind.FULL, &"T", -1, 0).describe(), "T-Spin Triple", "TST")
	eq(make_result(0, SpinDetect.Kind.FULL, &"T", -1, -1).describe(), "T-Spin Zero", "spin, no lines")
	eq(make_result(1, SpinDetect.Kind.MINI, &"T", -1, 0).describe(), "T-Spin Mini Single", "mini")
	eq(make_result(0, SpinDetect.Kind.MINI, &"T", -1, -1).describe(), "T-Spin Mini Zero", "mini, no lines")

	for id in [&"S", &"Z", &"L", &"J", &"I"]:
		eq(make_result(0, SpinDetect.Kind.FULL, id, -1, -1).describe(), "%s-Spin Zero" % id,
			"%s-spin with no clear is named" % id)
	eq(make_result(2, SpinDetect.Kind.FULL, &"S", -1, 0).describe(), "S-Spin Double", "S-spin double")

	# Beyond four lines, which pentominoes will reach.
	eq(make_result(5, SpinDetect.Kind.NONE, &"L", -1, 0).describe(), "5 Lines", "5-line clear falls back")

	var r := make_result(4, SpinDetect.Kind.NONE, &"I", 2, 3)
	r.perfect_clear = true
	eq(r.tags(), ["B2B x2", "Combo 3", "Perfect Clear"], "tags list every modifier")
	eq(r.describe(), "Quad", "modifiers stay out of the name")
	eq(make_result(1, SpinDetect.Kind.NONE, &"L", -1, 0).tags(), [], "a plain single has no tags")

	# The HUD draws the two halves on separate lines, so they must stand alone.
	var tsd := make_result(2, SpinDetect.Kind.FULL, &"T", -1, 0)
	eq(tsd.spin_name(), "T-Spin", "spin half reads alone")
	eq(tsd.count_name(), "Double", "count half reads alone")
	eq("%s %s" % [tsd.spin_name(), tsd.count_name()], tsd.describe(), "the halves rejoin into describe()")
	var quad := make_result(4, SpinDetect.Kind.NONE, &"I", -1, 0)
	eq(quad.spin_name(), "", "a plain clear has no spin half")
	eq(quad.count_name(), "Quad", "and carries the whole name in the count half")

	# is_named() gates whether the HUD readout updates at all.
	check(make_result(1, SpinDetect.Kind.NONE, &"L", -1, 0).is_named(), "a clear is named")
	check(make_result(0, SpinDetect.Kind.FULL, &"S", -1, -1).is_named(), "a zero-line spin is named")
	check(not make_result(0, SpinDetect.Kind.NONE, &"L", -1, -1).is_named(),
		"an ordinary placement is not named, so it will not wipe the readout")

func test_zero_line_spin_scoring() -> void:
	group("zero-line spin scoring")
	var s := Scorer.new(ScoreTable.new())
	eq(s.score_clear(make_result(0, SpinDetect.Kind.FULL, &"T", -1, -1)), 400, "T-Spin Zero scores 400")
	eq(s.score_clear(make_result(0, SpinDetect.Kind.MINI, &"T", -1, -1)), 100, "T-Spin Mini Zero scores 100")
	eq(s.score_clear(make_result(0, SpinDetect.Kind.FULL, &"L", -1, -1)), 0,
		"an incidental L-Spin Zero scores nothing by default")
	var loose := ScoreTable.new()
	loose.score_non_t_zero_line_spins = true
	eq(Scorer.new(loose).score_clear(make_result(0, SpinDetect.Kind.FULL, &"L", -1, -1)), 100,
		"...but does once the flag is on")

	# End to end: a T-Spin that clears nothing must still award points and must not
	# touch the back-to-back chain.
	var b := make_board([
		"...X......",
		"..........",
		"...X.X....",
	])
	var g := make_game(b)
	g.b2b = 2
	var before := g.score
	place(g, &"T", 1, Vector2i(3, 37))
	check(g.try_rotate(0), "rotated into the mini slot")
	var events: Array = []
	g.lock_piece(events)
	eq(g.last_result.describe(), "T-Spin Mini Zero", "named as a mini with no clear")
	eq(g.score - before, 100, "awarded the T-Spin Mini value")
	eq(g.last_points, 100, "points recorded on the result")
	eq(g.b2b, 2, "a spin with no line clear leaves the b2b chain alone")
	eq(g.combo, -1, "but it does break the combo")

	# The locked event carries the result, so a consumer needs nothing else.
	var carried: ClearResult = null
	for e in events:
		if e.type == GameEvent.Type.PIECE_LOCKED:
			carried = e.data["result"]
	check(carried != null, "PIECE_LOCKED carries a ClearResult")
	if carried != null:
		eq(carried.describe(), "T-Spin Mini Zero", "and it is the same result")

## Seat the active piece on the stack. Locking at the spawn position would block out
## the next spawn, which is correct behaviour but wrong for these setups.
func drop_to_floor(g: Game) -> void:
	while g.board.can_place(g.active.cells(), g.active.pos + Vector2i(0, 1)):
		g.active.pos.y += 1

func events_of(events: Array, type: int) -> Array:
	var out: Array = []
	for e in events:
		if e.type == type:
			out.append(e)
	return out

func test_audio_events() -> void:
	group("events for audio cues")

	# --- movement is coalesced: one event per tick regardless of cells travelled ---
	var g := make_game(Board.new(10, 20, 20))
	g.start()
	var right := InputState.new()
	right.right = true
	var tap := g.tick(right)
	eq(events_of(tap, GameEvent.Type.PIECE_MOVED).size(), 1, "a tap emits one move event")
	eq(events_of(tap, GameEvent.Type.PIECE_MOVED)[0].data["cells"], 1, "tap moved one cell")
	eq(events_of(tap, GameEvent.Type.PIECE_MOVED)[0].data["dir"], 1, "direction reported")

	# Charge DAS, then the ARR 0 slide must still be a single event.
	var slide_events: Array = []
	for _i in 20:
		slide_events = g.tick(right)
		var moved := events_of(slide_events, GameEvent.Type.PIECE_MOVED)
		if not moved.is_empty() and moved[0].data["cells"] > 1:
			break
	var multi := events_of(slide_events, GameEvent.Type.PIECE_MOVED)
	eq(multi.size(), 1, "an ARR 0 wall slide is still ONE event, not one per cell")
	check(multi[0].data["cells"] > 1, "and it reports the whole distance (%d cells)" % multi[0].data["cells"])

	# Held against the wall, nothing moves, so nothing fires.
	var at_wall: Array = []
	for _i in 10:
		at_wall = g.tick(right)
	check(events_of(at_wall, GameEvent.Type.PIECE_MOVED).is_empty(),
		"a blocked piece emits no move event, so the cue does not machine-gun")

	# --- rotation reports the spin state at that moment ---
	var b := make_board([
		"..........",
		"XXXX..XXXX",
		"XXX...XXXX",
		"XXXX.XXXXX",
	])
	var g2 := make_game(b)
	place(g2, &"T", 3, Vector2i(4, 36))
	var rot: Array = []
	g2.try_rotate(2, rot)
	var rotated := events_of(rot, GameEvent.Type.PIECE_ROTATED)
	eq(rotated.size(), 1, "a successful rotation emits one event")
	eq(rotated[0].data["spin"], SpinDetect.Kind.FULL, "and reports the spin BEFORE the clear happens")
	eq(rotated[0].data["kick"], 2, "carrying the kick index")

	# A rotation in open space reports no spin, so the click must not fire.
	var g3 := make_game(Board.new(10, 20, 20))
	place(g3, &"T", 0, Vector2i(3, 30))
	var rot3: Array = []
	g3.try_rotate(1, rot3)
	eq(events_of(rot3, GameEvent.Type.PIECE_ROTATED)[0].data["spin"], SpinDetect.Kind.NONE,
		"an open-space rotation reports no spin")

	# A rotation that fails every kick emits nothing at all.
	var packed := make_board(["XXXXXXXXXX", "XXXX..XXXX", "XXXXXXXXXX"])
	var g4 := make_game(packed)
	place(g4, &"T", 0, Vector2i(3, 30))
	g4.active.pos = Vector2i(4, 38)
	var rot4: Array = []
	var ok := g4.try_rotate(1, rot4)
	check(not ok or events_of(rot4, GameEvent.Type.PIECE_ROTATED).size() == 1,
		"a rejected rotation emits nothing")

	# --- lock cause distinguishes a slam from a settle ---
	var g5 := make_game(Board.new(10, 20, 20))
	g5.start()
	var drop := InputState.new()
	drop.hard_drop = true
	var locked := events_of(g5.tick(drop), GameEvent.Type.PIECE_LOCKED)
	eq(locked.size(), 1, "hard drop locks the piece")
	eq(locked[0].data["cause"], "hard_drop", "reported as a hard drop")

	var g6 := make_game(Board.new(10, 20, 20))
	g6.start()
	var manual: Array = []
	g6.lock_piece(manual)
	eq(events_of(manual, GameEvent.Type.PIECE_LOCKED)[0].data["cause"], "lock_delay",
		"a natural lock reports lock_delay")

	# --- combo_broken fires exactly once, on the placement that ends the chain ---
	var g7 := make_game(Board.new(10, 20, 20))
	g7.start()
	g7.combo = 3
	var brk: Array = []
	drop_to_floor(g7)
	g7.lock_piece(brk)
	eq(events_of(brk, GameEvent.Type.PIECE_LOCKED)[0].data["combo_broken"], true,
		"ending a combo chain is reported")
	eq(g7.combo, -1, "and the chain is reset")
	var brk2: Array = []
	drop_to_floor(g7)
	g7.lock_piece(brk2)
	eq(events_of(brk2, GameEvent.Type.PIECE_LOCKED)[0].data["combo_broken"], false,
		"the next placement does not report it again")

	# --- the cue routing itself, without needing an audio device ---
	group("cue routing")
	eq(_cue_for(4, SpinDetect.Kind.NONE), "line_clear_plus", "quad -> line_clear_plus")
	eq(_cue_for(3, SpinDetect.Kind.NONE), "line_clear", "triple -> line_clear")
	eq(_cue_for(2, SpinDetect.Kind.NONE), "line_clear", "double -> line_clear")
	eq(_cue_for(1, SpinDetect.Kind.NONE), "line_clear", "single -> line_clear")
	eq(_cue_for(1, SpinDetect.Kind.FULL), "line_clear_plus", "spin single -> line_clear_plus")
	eq(_cue_for(2, SpinDetect.Kind.MINI), "line_clear_plus", "mini spin double -> line_clear_plus")
	eq(_cue_for(0, SpinDetect.Kind.FULL), "", "a spin with no clear plays no clear cue")
	eq(_cue_for(0, SpinDetect.Kind.NONE), "", "a plain placement plays no clear cue")

## Mirrors AudioBus._on_locked so the routing is checked without an audio device.
func _cue_for(lines: int, spin: int) -> String:
	if lines <= 0:
		return ""
	return "line_clear_plus" if (lines >= 4 or spin != SpinDetect.Kind.NONE) else "line_clear"

func test_audio_cue_mapping() -> void:
	group("audio cue mapping")
	var bus := AudioBus.new()
	var cues := [
		AudioBus.CUE_CLICK, AudioBus.CUE_MOVE, AudioBus.CUE_DROP,
		AudioBus.CUE_CLEAR, AudioBus.CUE_CLEAR_PLUS, AudioBus.CUE_COMBO_BREAK,
	]
	for cue in cues:
		check(bus.stream_for(cue) != null, "cue '%s' resolves to a stream" % cue)
		bus.set_cue_gain(cue, -7.5)
		eq(bus.gain_for(cue), -7.5, "cue '%s' gain round-trips" % cue)
	eq(bus.stream_for("nonexistent"), null, "an unknown cue resolves to nothing")
	eq(bus.gain_for("nonexistent"), 0.0, "an unknown cue has no trim")

	# Shipped balance.
	var fresh := AudioBus.new()
	eq(fresh.gain_click, 0.0, "click ships flat (the louder sample sits there now)")
	eq(fresh.gain_combo_break, -3.0, "combo break ships at -3 dB")
	eq(fresh.stream_click.resource_path, "res://audio/spin_click.ogg", "click plays spin_click")
	eq(fresh.stream_move.resource_path, "res://audio/piece_move.ogg", "move plays piece_move")
	fresh.free()

	# Clearing a slot must silence that cue rather than erroring.
	bus.stream_move = null
	eq(bus.stream_for(AudioBus.CUE_MOVE), null, "an emptied slot silences its cue")
	bus.play(AudioBus.CUE_MOVE)
	check(true, "playing a silenced cue is a no-op, not a crash")
	bus.free()

func test_placement_enumeration() -> void:
	group("solver placement enumeration")
	var empty := Board.new(10, 20, 20)
	var ps := PieceSet.tetrominoes()
	var t := ps.get_def(&"T")
	var spawn := Vector2i((empty.width - t.box) / 2, empty.buffer_height - t.first_filled_row)
	var open_field := PlacementEnumerator.placements_for(empty, t, spawn)
	# T has 4 distinct rotations; 8 columns for the 3-wide ones, 9 for the vertical.
	check(open_field.size() >= 30, "open field yields %d T placements" % open_field.size())
	var rots := {}
	for p in open_field:
		rots[p.rot] = true
	eq(rots.size(), 4, "all four rotations are enumerated (straight-drop is not rotation-blind)")

	# O is rotation-invariant, so every rotation collapses to the same cells.
	var o := ps.get_def(&"O")
	var o_spawn := Vector2i((empty.width - o.box) / 2, empty.buffer_height)
	eq(PlacementEnumerator.placements_for(empty, o, o_spawn).size(), 9,
		"O yields exactly one placement per column, duplicates collapsed")

	# The reachability claim: a tower between spawn and the target must block it.
	# Column 0 is walled off from the spawn column by a full-height stack at column 2.
	var walled := Board.new(10, 20, 20)
	for y in range(walled.buffer_height - 2, walled.total_height):
		walled.set_cell(2, y, 1)
	var reachable := PlacementEnumerator.placements_for(walled, t, spawn)
	var min_x := 99
	for p in reachable:
		for c in p.cells:
			min_x = mini(min_x, c.x)
	check(min_x >= 3, "nothing left of the tower is proposed (leftmost cell %d)" % min_x)
	check(not reachable.is_empty(), "placements right of the tower are still found")

	# Every placement must be legal on the board it was enumerated against.
	var all_legal := true
	for p in reachable:
		if not walled.can_place(p.def.rotations[p.rot], p.pos):
			all_legal = false
	check(all_legal, "every enumerated placement is collision-free")

	# And every one must be resting, not floating.
	var all_seated := true
	for p in reachable:
		if walled.can_place(p.def.rotations[p.rot], p.pos + Vector2i(0, 1)):
			all_seated = false
	check(all_seated, "every enumerated placement is resting on something")

func test_quad_evaluator() -> void:
	group("quad evaluator")
	var w := EvalWeights.quad()
	var ev := BoardEvaluator.new(w)
	var well: int = w.well_column
	check(well >= 0, "the quad preset pins the well to a fixed column, not a moving one")

	# Same stack twice; the only difference is one cell sitting in the reserved well.
	var clean := Board.new(10, 20, 20)
	var dirty := Board.new(10, 20, 20)
	for b in [clean, dirty]:
		for x in b.width:
			if x == well:
				continue
			for y in range(b.total_height - 4, b.total_height):
				b.set_cell(x, y, 1)
	dirty.set_cell(well, dirty.total_height - 1, 1)

	var s_clean := ev.score(clean, well, 0, clean.total_height - 4)
	var s_dirty := ev.score(dirty, well, 0, dirty.total_height - 1)
	check(s_clean > s_dirty, "keeping the well empty beats filling it (%.1f vs %.1f)" % [s_clean, s_dirty])

	# A well 4 deep must be worth more than one 2 deep: that gradient is what gives
	# the search something multi-piece to plan toward.
	var shallow := Board.new(10, 20, 20)
	for x in shallow.width:
		if x == well:
			continue
		for y in range(shallow.total_height - 2, shallow.total_height):
			shallow.set_cell(x, y, 1)
	check(ev.score(clean, well, 0, clean.total_height - 4) - ev.score(shallow, well, 0, shallow.total_height - 2) != 0.0,
		"well depth changes the score")

	# A quad must beat a single on an otherwise identical board.
	var quad_score := ev.score(clean, well, 4, clean.total_height - 4)
	var single_score := ev.score(clean, well, 1, clean.total_height - 4)
	check(quad_score > single_score, "a quad outscores a single (%.1f vs %.1f)" % [quad_score, single_score])

	# Danger release: once the stack is tall the well ban must mostly lift, or an
	# I drought becomes a death sentence.
	var tall_clean := Board.new(10, 20, 20)
	var tall_dirty := Board.new(10, 20, 20)
	var rows: int = w.danger_height + 1
	for b in [tall_clean, tall_dirty]:
		for x in b.width:
			if x == well:
				continue
			for y in range(b.total_height - rows, b.total_height):
				b.set_cell(x, y, 1)
	tall_dirty.set_cell(well, tall_dirty.total_height - 1, 1)
	var gap_safe := ev.score(clean, well, 0, 0) - ev.score(dirty, well, 0, 0)
	var gap_danger := ev.score(tall_clean, well, 0, 0) - ev.score(tall_dirty, well, 0, 0)
	check(gap_danger < gap_safe,
		"the well ban relaxes in danger (penalty gap %.1f -> %.1f)" % [gap_safe, gap_danger])

func test_well_detection_with_holes() -> void:
	group("well detection under holes")
	var w := EvalWeights.quad()
	var ev := BoardEvaluator.new(w)
	var well: int = w.well_column

	# Bottom four rows complete except the well: a genuine quad.
	var real := Board.new(10, 20, 20)
	for x in real.width:
		if x == well:
			continue
		for y in range(real.total_height - 4, real.total_height):
			real.set_cell(x, y, 1)
	ev.score(real, well, 0, 0)   # populates the height cache
	eq(ev.clean_well_rows(real, well), 4, "four complete rows read as a quad")

	# Same surface heights, but every neighbour column is hollow underneath. The old
	# surface-height measure reported a 4-deep well here; no row is complete.
	var hollow := Board.new(10, 20, 20)
	for x in hollow.width:
		if x == well:
			continue
		hollow.set_cell(x, hollow.total_height - 4, 1)   # a lid, nothing below it
	ev.score(hollow, well, 0, 0)
	eq(ev.clean_well_rows(hollow, well), 0,
		"tall-but-hollow neighbours are NOT a quad (this is the bug that holes caused)")

	# One short column far from the well also kills it, even though the well's
	# immediate neighbours look perfect.
	var far_gap := Board.new(10, 20, 20)
	var far: int = 0 if well > 4 else far_gap.width - 1
	for x in far_gap.width:
		if x == well or x == far:
			continue
		for y in range(far_gap.total_height - 4, far_gap.total_height):
			far_gap.set_cell(x, y, 1)
	ev.score(far_gap, well, 0, 0)
	eq(ev.clean_well_rows(far_gap, well), 0,
		"a gap in a distant column is seen, not just the two neighbours")

	# Partial progress: three complete rows plus an incomplete fourth counts three.
	var partial := Board.new(10, 20, 20)
	for x in partial.width:
		if x == well:
			continue
		for y in range(partial.total_height - 3, partial.total_height):
			partial.set_cell(x, y, 1)
	partial.set_cell(0, partial.total_height - 4, 1)   # lone cell on the fourth row
	ev.score(partial, well, 0, 0)
	eq(ev.clean_well_rows(partial, well), 3, "an incomplete row stops the count")

	# A blocked well reads as zero regardless of what is above it.
	var blocked := Board.new(10, 20, 20)
	for x in blocked.width:
		for y in range(blocked.total_height - 4, blocked.total_height):
			blocked.set_cell(x, y, 1)
	ev.score(blocked, well, 0, 0)
	eq(ev.clean_well_rows(blocked, well), 0, "a filled well offers no quad")

func test_parity_property() -> void:
	group("checkerboard parity of the piece set")
	# Colour every cell by (x + y) % 2. The claim: every tetromino except T covers
	# exactly two of each colour, in every rotation AND at every translation, so only
	# T can change a board's parity imbalance -- by exactly 2.
	var ps := PieceSet.tetrominoes()
	for d in ps.defs:
		var balances := {}
		for rot in 4:
			# Both translation parities, since shifting a piece one cell flips every
			# colour it covers.
			for shift in 2:
				var a := 0
				var b := 0
				for c in d.rotations[rot]:
					if (c.x + c.y + shift) % 2 == 0:
						a += 1
					else:
						b += 1
				balances[absi(a - b)] = true
		var expected := 2 if d.id == &"T" else 0
		eq(balances.keys(), [expected],
			"%s always covers a %s colour split" % [d.id, "3-1" if expected == 2 else "2-2"])

	# A cleared row must be parity-neutral, or the imbalance would drift on its own.
	# True for any EVEN board width; an odd width would break the invariant.
	var b := Board.new(10, 20, 20)
	var row_a := 0
	var row_b := 0
	for x in b.width:
		if (x + 39) % 2 == 0:
			row_a += 1
		else:
			row_b += 1
	eq(row_a, row_b, "a full row on an even-width board is parity-neutral")
	check(b.width % 2 == 0, "board width is even, so clears cannot shift parity")

func attack_result(lines: int, spin: int, piece: StringName, b2b: int) -> ClearResult:
	var r := ClearResult.new()
	r.lines = lines
	r.spin = spin
	r.piece_id = piece
	r.b2b = b2b
	r.combo = -1
	r.level = 1
	return r

func test_attack_table() -> void:
	group("attack table")
	var t := AttackTable.new()
	var none := SpinDetect.Kind.NONE
	eq(t.attack_for(attack_result(1, none, &"L", -1)), 0, "single sends nothing")
	eq(t.attack_for(attack_result(2, none, &"L", -1)), 1, "double sends 1")
	eq(t.attack_for(attack_result(3, none, &"L", -1)), 2, "triple sends 2")
	eq(t.attack_for(attack_result(4, none, &"I", -1)), 4, "quad sends 4")
	eq(t.attack_for(attack_result(0, none, &"L", -1)), 0, "a placement with no clear sends nothing")

	var full := SpinDetect.Kind.FULL
	eq(t.attack_for(attack_result(1, full, &"T", -1)), 2, "T-spin single sends 2")
	eq(t.attack_for(attack_result(2, full, &"T", -1)), 4, "T-spin double matches a quad at 4")
	eq(t.attack_for(attack_result(3, full, &"T", -1)), 6, "T-spin triple sends 6")
	eq(t.attack_for(attack_result(0, full, &"T", -1)), 0, "a spin that clears nothing sends nothing")
	eq(t.attack_for(attack_result(1, SpinDetect.Kind.MINI, &"T", -1)), 1, "T-spin mini single sends 1")

	# B2B adds to difficult clears only.
	eq(t.attack_for(attack_result(4, none, &"I", 1)), 5, "b2b quad sends 4 + 1")
	eq(t.attack_for(attack_result(2, full, &"T", 1)), 5, "b2b TSD sends 4 + 1")
	eq(t.attack_for(attack_result(2, none, &"L", 1)), 1, "a double is not difficult, so no b2b bonus")

	# Combos are off by default, as specified.
	var combo_r := attack_result(2, none, &"L", -1)
	combo_r.combo = 5
	eq(t.attack_for(combo_r), 1, "combos contribute nothing while use_combo is off")
	t.use_combo = true
	check(t.attack_for(combo_r) > 1, "...and do once it is switched on")

	# End to end through the real lock path.
	group("attack through Game")
	var g := make_game(Board.new(10, 20, 20))
	eq(g.attack_sent, 0, "starts at zero")
	for attempt in 2:
		for y in range(36, 40):
			for x in range(1, 10):
				g.board.set_cell(x, y, 1)
		g.board.set_cell(5, 34, 1)
		place(g, &"I", 1, Vector2i(-2, 36))
		var events: Array = []
		g.lock_piece(events)
		g.board.clear()
	# First quad 4, second quad 4 + 1 for the live chain.
	eq(g.attack_sent, 9, "two quads send 4 then 5 with the b2b bonus")
	eq(g.last_result.attack, 5, "the result carries its own attack value")

## Board with one 1-wide notch at `col`, `depth` deep, and the neighbours set so the
## left and right overhangs are exactly as asked.
func notch_board(col: int, left_ov: int, right_ov: int) -> Board:
	var b := Board.new(10, 20, 20)
	var floor_h := 4
	for x in b.width:
		var h := floor_h
		if x == col - 1:
			h = floor_h + left_ov
		elif x == col + 1:
			h = floor_h + right_ov
		elif x == col:
			h = floor_h
		for i in h:
			b.set_cell(x, b.total_height - 1 - i, 1)
	return b

func test_dependency_attribution() -> void:
	group("dependency attribution (I / L / J)")
	var w := EvalWeights.lookahead()
	var ev := BoardEvaluator.new(w)
	var well: int = w.well_column
	var col := 4

	# A 3-deep notch: only a vertical I fits. An L or J bar is 3 tall, but its foot
	# would clash with the neighbour that makes the notch deep.
	var deep := notch_board(col, 3, 3)
	var no_help := ev.structural(deep, well, {}, false)
	var with_l := ev.structural(deep, well, {&"L": [1]}, false)
	var with_i := ev.structural(deep, well, {&"I": [1]}, false)
	eq(with_l, no_help, "3-deep notch: an L does not help")
	check(with_i > no_help, "3-deep notch: an I does (%.1f -> %.1f)" % [no_help, with_i])

	# A 2-deep notch with the LEFT neighbour exactly 2 above: L fits (its foot sits
	# top-left, resting on that neighbour), J does not.
	var left2 := notch_board(col, 2, 5)
	var base_l := ev.structural(left2, well, {}, false)
	check(ev.structural(left2, well, {&"L": [1]}, false) > base_l,
		"2-deep, left neighbour 2 above: L relieves it")
	eq(ev.structural(left2, well, {&"J": [1]}, false), base_l,
		"...and J does not")

	# Mirrored.
	var right2 := notch_board(col, 5, 2)
	var base_r := ev.structural(right2, well, {}, false)
	check(ev.structural(right2, well, {&"J": [1]}, false) > base_r,
		"2-deep, right neighbour 2 above: J relieves it")
	eq(ev.structural(right2, well, {&"L": [1]}, false), base_r,
		"...and L does not")

	# Hold beats the queue, and nearer beats further.
	var in_hold := ev.structural(deep, well, {&"I": [0]}, false)
	var next_up := ev.structural(deep, well, {&"I": [1]}, false)
	var far_off := ev.structural(deep, well, {&"I": [5]}, false)
	check(in_hold > next_up, "an I in hold relieves more than one in the queue")
	check(next_up > far_off, "a near I relieves more than a distant one")
	check(far_off > no_help, "even a distant I is better than none")

	# The well takes the first I, so it cannot also fix a notch.
	var claimed := ev.structural(deep, well, {&"I": [1]}, true)
	eq(claimed, no_help, "an I already claimed by the well gives no relief")
	check(ev.structural(deep, well, {&"I": [1, 3]}, true) > no_help,
		"a second I does")

	# Greedy consumption: two notches, one I, only one gets relief.
	var two := notch_board(2, 3, 3)
	for i in 3:
		two.set_cell(7, two.total_height - 5 - i, 0)
	var one_i := ev.structural(two, well, {&"I": [1]}, false)
	var two_i := ev.structural(two, well, {&"I": [1, 2]}, false)
	check(two_i > one_i, "a second I relieves the second notch too")

func test_hole_topology() -> void:
	group("hole topology: sealed vs overhang")
	# The split only exists when overhangs are priced differently from holes -- a
	# spin evaluator's concern. With them equal the cheap naive path runs instead,
	# so this has to ask for the split explicitly.
	var w := EvalWeights.lookahead()
	w.w_overhang = -1.0
	var ev := BoardEvaluator.new(w)
	var well: int = w.well_column

	# A T-spin slot. The cell under the right-hand hang is COVERED, but it reaches
	# open air sideways, so it is an overhang -- not a hole.
	var slot := make_board([
		"XXX..XXXX.",
		"XXX...XXX.",
		"XXXX.XXXX.",
	])
	ev.structural(slot, well)
	eq(ev.holes(), 0, "a T-spin slot contains no sealed holes")
	check(ev.overhangs() > 0, "but it does contain overhangs (%d)" % ev.overhangs())

	# Seal the same shape off and it becomes a genuine hole.
	var sealed := make_board([
		"XXXXXXXXX.",
		"XXX...XXX.",
		"XXXXXXXXX.",
	])
	ev.structural(sealed, well)
	eq(ev.holes(), 3, "a fully enclosed cavity is three sealed holes")
	eq(ev.overhangs(), 0, "and no overhangs")

	# The classic case: a single cell buried under a flat lid.
	var buried := make_board([
		"XXXXXXXXX.",
		"XXXX.XXXX.",
	])
	ev.structural(buried, well)
	eq(ev.holes(), 1, "one cell under a lid is one hole")

	# The reserved well is open from the top, so it must never read as a hole.
	var deep_well := make_board([
		"XXXXXXXXX.",
		"XXXXXXXXX.",
		"XXXXXXXXX.",
		"XXXXXXXXX.",
	])
	ev.structural(deep_well, well)
	eq(ev.holes(), 0, "an open well is not a hole")
	eq(ev.overhangs(), 0, "nor an overhang")

	# Cap the well and the whole column becomes trapped.
	var capped := make_board([
		"XXXXXXXXXX",
		"XXXXXXXXX.",
		"XXXXXXXXX.",
	])
	ev.structural(capped, well)
	eq(ev.holes(), 2, "capping the well traps everything under it")

	# The penalty ordering that makes T-spin setups buildable at all.
	check(absf(w.w_holes) > absf(w.w_overhang) * 4.0,
		"a sealed hole costs far more than an overhang (%.1f vs %.1f)" % [w.w_holes, w.w_overhang])

	# And the quad presets deliberately do NOT take the split: an overhang is a
	# hole-shaped future, and pricing it cheaply cost 30 holes when measured.
	var quad_w := EvalWeights.lookahead()
	eq(quad_w.w_overhang, quad_w.w_holes, "quad presets price overhangs as holes")
	var naive := BoardEvaluator.new(quad_w)
	naive.structural(slot, well)
	eq(naive.holes(), 1, "...so the same slot reads as a hole on the cheap path")

func test_tspin_slots() -> void:
	group("T-spin slot detection")
	var ev := BoardEvaluator.new(EvalWeights.tspin())
	var ps := PieceSet.tetrominoes()

	# Nub at column 3. Lip is the filled cell at column 2 on the top row; columns 3
	# and 4 stay open above so the T can drop in and rotate left.
	#            0123456789
	var ready := make_board([
		"XXX..XXXXX",   # hang row:    col 2 filled, 3 and 4 open for entry
		"XX...XXXXX",   # pocket row: cols 2,3,4 empty
		"XXX.XXXXXX",   # nub row:    col 3 empty
	])
	var s1 := TSpinSlots.scan(ready, ev.column_heights(ready), 9)
	check(bool(s1["found"]), "the slot shape is found")
	eq(s1["nub_col"], 3, "nub column identified")
	check(bool(s1["ready"]), "both rows are otherwise full, so it is cashable")
	eq(s1["fill"], 1.0, "fill reads 1.0")

	# The injected placement must be legal, register as a spin, and clear two rows.
	var p := TSpinSlots.placement_for(ready, ps.get_def(&"T"), s1)
	check(p != null, "a T placement is produced")
	if p != null:
		eq(p.rot, 2, "nub-down rotation")
		var ap := ActivePiece.new()
		ap.def = p.def
		ap.rot = p.rot
		ap.pos = p.pos
		ap.last_action_was_rotation = true
		eq(SpinDetect.detect(ready, ap, RuleSet.competitive()), SpinDetect.Kind.FULL,
			"and it registers as a full T-spin")
		var after := ready.duplicate_board()
		after.lock_cells(p.def.rotations[2], p.pos, 1)
		eq(after.find_full_rows().size(), 2, "and clears two rows -- a TSD")

	# Same slot with an OPEN WELL at column 8. The well is a gap in both rows, so the
	# shape is still there but nothing would clear.
	var blocked := make_board([
		"XXX..XXX.X",
		"XX...XXX.X",
		"XXX.XXXX.X",
	])
	var s2 := TSpinSlots.scan(blocked, ev.column_heights(blocked), 8)
	check(bool(s2["found"]), "the shape is still found over an open well")
	check(not bool(s2["ready"]), "but it is NOT cashable -- the well leaves both rows short")
	check(float(s2["fill"]) > 0.8, "it does score as a near-complete partial (%.2f)" % s2["fill"])

	# The same board as the first, but calling column 3 the well: the nub plugs it,
	# which is why a slot AT the well behaves differently from one anywhere else.
	var s3 := TSpinSlots.scan(ready, ev.column_heights(ready), 3)
	check(bool(s3["ready"]), "a slot whose nub column is the well is cashable")
	check(bool(s3["at_well"]), "and is flagged as being at the well")

	# Two slots sharing a height: neither row can complete, so neither is cashable.
	var two := make_board([
		"XX..XXX..X",
		"X...XX...X",
		"XX.XXXX.XX",
	])
	var s4 := TSpinSlots.scan(two, ev.column_heights(two), 9)
	check(bool(s4["found"]), "two slots at one height are still shapes")
	check(not bool(s4["ready"]),
		"but both are uncashable -- the completeness check catches it with no extra rule")

	# A pocket with no hang is droppable, so it is not a spin slot at all.
	var no_lip := make_board([
		"XX...XXXXX",
		"XXX.XXXXXX",
	])
	var s5 := TSpinSlots.scan(no_lip, ev.column_heights(no_lip), 9)
	check(not bool(s5["found"]), "no hang means no spin needed, so no slot")

# ------------------------------------------------------------------------------
# Garbage
# ------------------------------------------------------------------------------

## Reads a row back as ASCII: "." empty, "G" garbage, "X" anything else.
func _row(b: Board, y: int) -> String:
	var out := ""
	for x in b.width:
		var v := b.get_cell(x, y)
		out += "." if v == 0 else ("G" if v == Board.GARBAGE else "X")
	return out

func test_garbage_push() -> void:
	group("Garbage: push_garbage raises the stack")

	# holes[0] is the BOTTOM-most new row, so the array reads upward from the floor.
	var b := make_board(["XX........", "XXX......."])
	var floor_y := b.total_height - 1
	b.push_garbage([4])
	eq(_row(b, floor_y), "GGGG.GGGGG", "one garbage row lands on the floor with its hole")
	eq(_row(b, floor_y - 1), "XXX.......", "the old floor row moved up one")
	eq(_row(b, floor_y - 2), "XX........", "and the row above it moved with it")

	var b2 := make_board(["X........."])
	b2.push_garbage([0, 9])
	eq(_row(b2, b2.total_height - 1), ".GGGGGGGGG", "holes[0] is the bottom row")
	eq(_row(b2, b2.total_height - 2), "GGGGGGGGG.", "holes[1] sits above it")
	eq(_row(b2, b2.total_height - 3), "X.........", "the stack rose by two")

	# Garbage must be distinguishable from placed cells, or the view cannot colour it.
	check(b2.get_cell(1, b2.total_height - 1) == Board.GARBAGE, "garbage cells carry GARBAGE")
	check(b2.get_cell(0, b2.total_height - 3) != Board.GARBAGE, "placed cells are untouched")

	# A full row of garbage is still a clearable row.
	var b3 := Board.new(10, 20, 20)
	b3.push_garbage([3])
	b3.set_cell(3, b3.total_height - 1, 1)
	eq(b3.find_full_rows(), [b3.total_height - 1], "plugging the hole completes the row")

	# Rows pushed past the top of the buffer are discarded rather than wrapping.
	var tall: Array = []
	for _i in 4:
		tall.append("XXXXXXXXXX")
	var b4 := make_board(tall)
	var before := b4.total_height
	b4.push_garbage([0, 1, 2])
	eq(b4.total_height, before, "the field does not grow")
	eq(_row(b4, b4.total_height - 4), "XXXXXXXXXX", "the stack is three rows higher")

	# An empty push is a no-op, so callers need not special-case zero.
	var b5 := make_board(["XX........"])
	b5.push_garbage([])
	eq(_row(b5, b5.total_height - 1), "XX........", "pushing no rows changes nothing")


# ------------------------------------------------------------------------------
# Spin assist
# ------------------------------------------------------------------------------

## The surface the assist arms on: a flat shelf that steps down by one at the column
## beside the well, with the well itself open. Rows are filled to the left so the
## payoff can be checked -- both of them should complete.
func _shelf() -> Board:
	var b := Board.new(10, 20, 20)
	b.ghost_width = 1
	b.ghost.resize(b.total_height)
	var fy := b.total_height - 1
	for x in b.width:
		b.set_cell(x, fy, 1)              # solid floor
	for x in range(0, 9):
		b.set_cell(x, fy - 1, 1)          # nub row: full to the step, well open
	for x in range(0, 8):
		b.set_cell(x, fy - 2, 1)          # pocket row: full to the shelf
	return b

func test_ghost_pocket() -> void:
	group("Spin assist: a pocket, not an eleventh column")

	var b := _shelf()
	var slot := GhostSlot.new()
	slot.update(b)
	check(slot.armed, "arms on a shelf that steps down one cell at the well")

	var opens := 0
	for y in b.total_height:
		if b.ghost[y] == Board.GHOST_OPEN:
			opens += 1
	eq(opens, 1, "exactly one cell of the column is open")
	eq(b.ghost[slot.nub_row], Board.GHOST_SHOWN, "support under the pocket is solid")
	eq(b.ghost[slot.nub_row - 1], Board.GHOST_OPEN, "the pocket row is the open one")
	eq(b.ghost[slot.nub_row - 2], Board.GHOST_SHOWN, "the hang above the pocket is solid")

	# The phantom column must never be a storable cell: in_bounds gates get/set_cell,
	# which index cells[y * width + x] and would otherwise write into the next row.
	check(not b.in_bounds(b.width, 0), "phantom column is NOT a board cell")
	check(b.in_play(b.width, 0), "but a piece may legally sit there")
	b.set_cell(b.width, 5, 3)
	eq(b.get_cell(0, 6), 0, "writing to the phantom column cannot corrupt the next row")

	var t := PieceSet.tetrominoes().get_def(&"T")
	var pocket := slot.nub_row - 1
	var pos := Vector2i(8, pocket - 1)
	check(not b.can_place(t.rotations[2], pos), "strict placement refuses the pocket")
	check(b.can_place_in_play(t.rotations[2], pos), "rotation may enter the pocket")

	# Nothing may fall in: the overhang has to block the way.
	var droppable := false
	for y in range(0, pos.y):
		if b.can_place_in_play(t.rotations[2], Vector2i(pos.x, y)):
			droppable = true
	check(not droppable, "the pocket cannot be entered by dropping")

	var p := ActivePiece.new()
	p.def = t
	p.rot = 2
	p.pos = pos
	check(SpinDetect.is_immobile(b, p), "a T that rotated in is immobile -- a spin")

	# Locking must drop the phantom cells rather than storing them, and the payoff is
	# a DOUBLE: the step is what lets the pocket row complete alongside the nub row.
	var before := 0
	for i in b.cells.size():
		if b.cells[i] != 0:
			before += 1
	b.lock_cells(t.rotations[2], pos, t.index + 1)
	var after := 0
	for i in b.cells.size():
		if b.cells[i] != 0:
			after += 1
	eq(b.get_cell(9, pocket), t.index + 1, "cells inside the field are kept")
	eq(after - before, 3, "only 3 of the T's 4 cells land -- the outside one is dropped")
	var full := b.full_rows_among([slot.nub_row, pocket])
	eq(full.size(), 2, "both rows complete -- a T-spin double, not a single")
	b.clear_rows(full)
	var top := -1
	var flat := true
	for x in b.width:
		var hx := 0
		for y in b.total_height:
			if b.get_cell(x, y) != 0:
				hx = b.total_height - y
				break
		if top < 0:
			top = hx
		elif hx != top:
			flat = false
	check(flat, "and the stack is flat again afterwards")

	# No step means no double, so the assist must not offer itself.
	var level := _shelf()
	level.set_cell(8, level.total_height - 3, 1)   # fill the step away
	var s2 := GhostSlot.new()
	s2.update(level)
	check(not s2.armed, "does not arm without the one-cell step at the well")

	# A blocked well has nowhere for the nub to go.
	var plugged := _shelf()
	plugged.set_cell(9, plugged.total_height - 2, 1)
	var s2b := GhostSlot.new()
	s2b.update(plugged)
	check(not s2b.armed, "does not arm when the well is not open")

	# Nothing further left has to be flat: the shape is local to three columns.
	var lumpy := _shelf()
	lumpy.set_cell(2, lumpy.total_height - 4, 1)
	lumpy.set_cell(4, lumpy.total_height - 4, 1)
	var s2c := GhostSlot.new()
	s2c.update(lumpy)
	check(s2c.armed, "still arms over an uneven stack further left")

	# The shelf may keep going up past the pocket -- the piece never touches it, so
	# the hang row above the shelf is a don't-care.
	var tall := _shelf()
	for dy in range(3, 7):
		tall.set_cell(7, tall.total_height - dy, 1)
	var s2d := GhostSlot.new()
	s2d.update(tall)
	check(s2d.armed, "arms with a wall left of the pocket, not just a one-cell step")

	# But the shelf must still be filled AT the pocket row, or that row cannot complete.
	var gap := _shelf()
	gap.set_cell(7, gap.total_height - 3, 0)
	var s2e := GhostSlot.new()
	s2e.update(gap)
	check(not s2e.armed, "does not arm when the shelf is missing at the pocket row")

	var off := _shelf()
	var s3 := GhostSlot.new()
	s3.enabled = false
	s3.update(off)
	check(not s3.armed, "disabled assist arms nothing")
	var any := false
	for y in off.total_height:
		if off.ghost[y] != Board.GHOST_WALL:
			any = true
	check(not any, "and leaves the column solid, exactly like the field edge")


# ------------------------------------------------------------------------------
# Shape resources
# ------------------------------------------------------------------------------

func test_shape_resources() -> void:
	group("Shape resources: the scoring piece is the source")

	eq(ShapeGroups.of(&"S"), ShapeGroups.of(&"Z"), "S and Z pay into one group")
	eq(ShapeGroups.of(&"L"), ShapeGroups.of(&"J"), "L and J pay into one group")
	check(ShapeGroups.of(&"I") != ShapeGroups.of(&"O"), "I and O are their own")
	eq(ShapeGroups.of(&"T"), ShapeGroups.T, "so is T")

	# The colour comes from the piece that scored, not from the rubble it destroyed.
	var r := ClearResult.new()
	r.piece_id = &"Z"
	r.lines = 3
	eq(r.resource_group(), ShapeGroups.SZ, "a Z scores into SZ")
	eq(r.resource_amount(), 3, "three lines, three resource")

	r.piece_id = &"I"
	r.lines = 4
	eq(r.resource_group(), ShapeGroups.I, "a quad scores into I")
	eq(r.resource_amount(), 4, "and only I can reach four")

	var none := ClearResult.new()
	none.piece_id = &"T"
	eq(none.resource_amount(), 0, "a spin that cleared nothing pays nothing")

	# The ceiling per colour is set by what the piece can physically clear, which is
	# what lets a player skew their build. Counting destroyed cells instead would
	# converge on the bag distribution and could not be skewed at all.
	var ps := PieceSet.tetrominoes()
	var spans := {}
	for id in ps.ids():
		var d := ps.get_def(StringName(id))
		var best := 0
		for rot in d.rotations.size():
			var rows := {}
			for c in d.rotations[rot]:
				rows[c.y] = true
			best = maxi(best, rows.size())
		spans[ShapeGroups.of(StringName(id))] = maxi(int(spans.get(ShapeGroups.of(StringName(id)), 0)), best)
	eq(spans[ShapeGroups.I], 4, "only I spans four rows, so only I can quad")
	eq(spans[ShapeGroups.O], 2, "O spans two, so two is its ceiling")
	eq(spans[ShapeGroups.T], 3, "T tops out at three -- a spin triple")
	eq(spans[ShapeGroups.SZ], 3, "as do S and Z")
	eq(spans[ShapeGroups.LJ], 3, "and L and J")


# ------------------------------------------------------------------------------
# Structures
# ------------------------------------------------------------------------------

## Lays the pieces of an ASCII block onto a board as real placements, one owner id
## each, anchored to the bottom-left. Letters name piece types; '.' is empty.
func _build(rows: Array, ps: PieceSet) -> Board:
	var b := Board.new(10, 20, 20)
	var base := b.total_height - rows.size()
	var ids := {}
	var next := 1
	for i in rows.size():
		var line: String = rows[i]
		for x in line.length():
			var ch := line[x]
			if ch == ".":
				continue
			if not ids.has(ch):
				ids[ch] = next
				next += 1
			var d := ps.get_def(StringName(ch.to_upper()))
			b.cells[(base + i) * b.width + x] = d.index + 1
			b.owner[(base + i) * b.width + x] = ids[ch]
	return b

## Rotate an ASCII block 90 degrees clockwise. A block with h rows of w becomes w rows
## of h, which is the whole point: half of a rectangle's symmetries transpose it.
func _cw(rows: Array) -> Array:
	var h := rows.size()
	var w := String(rows[0]).length()
	var out: Array = []
	for r in w:
		var line := ""
		for c in h:
			line += String(rows[h - 1 - c])[r]
		out.append(line)
	return out

func _flip_h(rows: Array) -> Array:
	var out: Array = []
	for r in rows:
		out.append(String(r).reverse())
	return out

## All eight symmetries of a block: four rotations, and the same four mirrored. Named so
## a failure says which one broke.
func _symmetries(rows: Array) -> Array:
	var out: Array = []
	var g: Array = rows
	for i in 4:
		out.append(["rot %d" % (90 * i), g])
		g = _cw(g)
	g = _flip_h(rows)
	for i in 4:
		out.append(["mirror %d" % (90 * i), g])
		g = _cw(g)
	return out

func test_structures() -> void:
	group("Structures: rectangles tiled by whole pieces")

	var ps := PieceSet.tetrominoes()

	# The three the design calls for.
	for spec in [["LOOJ", "LOOJ", "LLJJ"], ["LJJJ", "LZZJ", "LLZZ"], ["LLLJ", "LSSJ", "SSJJ"]]:
		var b := _build(spec, ps)
		var found := Structures.scan(b, 4, 3)
		eq(found.size(), 1, "%s tiles a 4x3" % "/".join(spec))
		if found.size() == 1:
			eq((found[0]["owners"] as Array).size(), 3, "   and it is made of three whole pieces")

	# A piece hanging out of the rectangle does not tile it, even though every cell
	# inside is filled: the L below reaches a row further down.
	var over := _build(["LOOJ", "LOOJ", "LLJJ", "L..."], ps)
	over.owner[(over.total_height - 1) * over.width] = 1      # same L as above
	over.cells[(over.total_height - 1) * over.width] = ps.get_def(&"L").index + 1
	eq(Structures.scan(over, 4, 3).size(), 0, "a piece poking out does not count")

	# Garbage is not a placement, so it can never be part of a structure.
	var g := _build(["LOOJ", "LOOJ", "LLJJ"], ps)
	g.cells[(g.total_height - 1) * g.width] = Board.GARBAGE
	g.owner[(g.total_height - 1) * g.width] = 0
	eq(Structures.scan(g, 4, 3).size(), 0, "garbage cannot tile a structure")

	# A hole anywhere breaks it.
	var holed := _build(["LOOJ", "LOOJ", "LLJJ"], ps)
	holed.cells[(holed.total_height - 2) * holed.width + 1] = 0
	eq(Structures.scan(holed, 4, 3).size(), 0, "a gap breaks the rectangle")

	# A piece that lost cells to a line clear no longer has four, so it cannot tile.
	var cut := _build(["LOOJ", "LOOJ", "LLJJ"], ps)
	cut.clear_rows([cut.total_height - 1])
	eq(Structures.scan(cut, 4, 3).size(), 0, "a clear breaks the structures it cut")

	# Identity is the owner set, so the same structure is recognisable across scans.
	var b2 := _build(["LOOJ", "LOOJ", "LLJJ"], ps)
	var f1 := Structures.scan(b2, 4, 3)
	var f2 := Structures.scan(b2, 4, 3)
	eq(Structures.key(f1[0]), Structures.key(f2[0]), "the same structure keeps the same identity")

	# Sizes that cannot hold whole pieces are rejected outright.
	eq(Structures.scan(b2, 3, 3).size(), 0, "9 cells cannot be tiled by tetrominoes")
	eq(Structures.scan(b2, 5, 3).size(), 0, "nor 15")

	# Each of the three LJ+X compositions tiles a 4x3 in exactly TWO ways, and the pair is
	# the rectangle turned 180 degrees -- reverse the row order and reverse each row and
	# one becomes the other. Both must be found, which is the point of testing what fills
	# the area instead of matching a template. tools/tilings.py 4 3 -v lists them.
	# Every one of the eight symmetries of a structure must be found. Four of them keep the
	# 4x3 footprint and four transpose it to 3x4 -- a 90 degree turn of the whole block --
	# which is why Game scans both sizes and not just the one the design was drawn at.
	# The images are derived here rather than written out, so this cannot quietly agree
	# with a mistake in the list.
	group("Structures: all eight symmetries, at whichever footprint they land on")
	for spec in [["LOOJ", "LOOJ", "LLJJ"], ["LLLJ", "LSSJ", "SSJJ"], ["LJJJ", "LZZJ", "LLZZ"]]:
		var want_pay := Structures.payout(Structures.scan(_build(spec, ps), 4, 3)[0], ps)
		var seen_fp := {}
		for entry in _symmetries(spec):
			var label: String = entry[0]
			var img: Array = entry[1]
			var gw := String(img[0]).length()
			var gh := img.size()
			seen_fp["%dx%d" % [gw, gh]] = true
			var bd := _build(img, ps)
			var f := Structures.scan(bd, gw, gh)
			eq(f.size(), 1, "%s of %s is found at %dx%d" % [label, "/".join(spec), gw, gh])
			if f.size() == 1:
				# Turning a rectangle cannot change what it is made of.
				eq(Structures.payout(f[0], ps), want_pay, "   and pays the same")
			# The transpose is genuinely a different scan: this image fills one footprint
			# and not the other, which is exactly why both sizes are needed.
			eq(Structures.scan(bd, gh, gw).size(), 0 if gw != gh else 1,
				"   and does not fill %dx%d" % [gh, gw])
		eq(seen_fp.size(), 2, "   across both footprints, 4x3 and 3x4")

		# And Game, scanning both sizes, picks up every image.
		for entry2 in _symmetries(spec):
			var g2: Array = entry2[1]
			var gg := make_game(_build(g2, ps))
			gg.start()
			gg.pieces_placed = 100
			var evg: Array = []
			gg._scan_structures(evg)
			eq(evg.size(), 1, "   Game finds %s of %s" % [entry2[0], "/".join(spec)])

	group("Structures: both tilings of each composition")
	for pair in [
			[["LOOJ", "LOOJ", "LLJJ"], ["JJLL", "JOOL", "JOOL"]],
			[["LLLJ", "LSSJ", "SSJJ"], ["JJSS", "JSSL", "JLLL"]],
			[["LJJJ", "LZZJ", "LLZZ"], ["ZZLL", "JZZL", "JJJL"]],
		]:
		var a: Array = pair[0]
		var bb: Array = pair[1]
		# The pair really is one rotated 180 degrees, not two unrelated shapes.
		var spun: Array = []
		for i in range(a.size() - 1, -1, -1):
			spun.append(String(a[i]).reverse())
		eq(spun, bb, "%s is %s turned 180" % ["/".join(bb), "/".join(a)])
		for spec in [a, bb]:
			var fp := Structures.scan(_build(spec, ps), 4, 3)
			eq(fp.size(), 1, "   %s tiles it" % "/".join(spec))
			if fp.size() == 1:
				# Turning the rectangle cannot change what it is made of, so the payout
				# has to match across the pair.
				eq(Structures.payout(fp[0], ps), Structures.payout(Structures.scan(_build(a, ps), 4, 3)[0], ps),
					"      and pays the same either way")


# ------------------------------------------------------------------------------
# Resources
# ------------------------------------------------------------------------------

func test_resources() -> void:
	group("Resources: the bank the minigame spends from")

	var r := Resources.new()
	for g in ShapeGroups.all():
		eq(r.amount(g), 0, "%s starts empty" % g)

	r.add(ShapeGroups.I, 4, Resources.CLEARS)
	r.add(ShapeGroups.O, 2, Resources.STRUCTURES)
	r.add(ShapeGroups.I, 1, Resources.STRUCTURES)
	eq(r.amount(ShapeGroups.I), 5, "a group totals across sources")
	eq(r.earned(Resources.CLEARS, ShapeGroups.I), 4, "   but the split is kept")
	eq(r.earned(Resources.STRUCTURES, ShapeGroups.I), 1, "   on both sides")
	eq(r.total(), 7, "and the whole bank adds up")

	r.add_table({ShapeGroups.LJ: 2, ShapeGroups.SZ: 1}, Resources.STRUCTURES)
	eq(r.amount(ShapeGroups.LJ), 2, "add_table credits every group in one call")
	eq(r.amount(ShapeGroups.SZ), 1, "   including the second")

	# Spending is atomic: a cost that is only partly covered takes nothing.
	check(not r.can_afford({ShapeGroups.I: 5, ShapeGroups.T: 1}), "an uncovered cost is unaffordable")
	check(not r.spend({ShapeGroups.I: 5, ShapeGroups.T: 1}), "   and spending it fails")
	eq(r.amount(ShapeGroups.I), 5, "   leaving the covered group untouched")
	check(r.spend({ShapeGroups.I: 5, ShapeGroups.LJ: 2}), "a covered cost is paid")
	eq(r.amount(ShapeGroups.I), 0, "   deducting each group")
	eq(r.amount(ShapeGroups.LJ), 0, "   in full")
	eq(r.earned(Resources.CLEARS, ShapeGroups.I), 4, "spending does not rewrite history")

	# Structures pay per contributing piece, into that piece's own group. LOOJ is
	# where O comes from: no line clear can pay O more than 2 at once.
	group("Resources: structure payouts")
	var ps := PieceSet.tetrominoes()
	var b := _build(["LOOJ", "LOOJ", "LLJJ"], ps)
	var pay := Structures.payout(Structures.scan(b, 4, 3)[0], ps)
	eq(int(pay.get(ShapeGroups.O, 0)), 1, "LOOJ pays O once -- one O piece in it")
	eq(int(pay.get(ShapeGroups.LJ, 0)), 2, "   and LJ twice, for the L and the J")
	var pay2 := Structures.payout(Structures.scan(_build(["LJJJ", "LZZJ", "LLZZ"], ps), 4, 3)[0], ps)
	eq(int(pay2.get(ShapeGroups.LJ, 0)), 2, "LJJJ/LZZJ/LLZZ pays LJ twice")
	eq(int(pay2.get(ShapeGroups.SZ, 0)), 1, "   and SZ once")

	# In Game: a clear credits the SCORING piece, and a structure is paid exactly once.
	group("Resources: Game credits both sources")
	var g := make_game(Board.new(10, 20, 20))
	g.start()
	var built := _build(["LOOJ", "LOOJ", "LLJJ"], ps)
	g.board.cells = built.cells.duplicate()
	g.board.owner = built.owner.duplicate()
	var ev: Array = []
	g._scan_structures(ev)
	eq(ev.size(), 1, "the structure is reported once")
	eq(g.resources.amount(ShapeGroups.O), 1, "   and paid into O")
	ev.clear()
	g._scan_structures(ev)
	eq(ev.size(), 0, "a second scan does not report it again")
	eq(g.resources.amount(ShapeGroups.O), 1, "   and pays nothing more")

	# Cut it, and the claim is forgotten rather than kept forever.
	g.board.clear_rows([g.board.total_height - 1])
	ev.clear()
	g._scan_structures(ev)
	eq(g._structure_claims.size(), 0, "a destroyed structure is forgotten")

	# A clear pays the group of the piece that scored it, not of the cells destroyed.
	var g2 := make_game(Board.new(10, 20, 20))
	g2.start()
	var y: int = g2.board.total_height - 1
	for x in 9:
		g2.board.set_cell(x, y, PieceSet.tetrominoes().get_def(&"Z").index + 1)
	var res := ClearResult.new()
	eq(ShapeGroups.of(&"O"), ShapeGroups.O, "an O clear pays O")
	res.lines = 2
	res.piece_id = &"O"
	eq(res.resource_group(), ShapeGroups.O, "   whatever the rows were made of")
	eq(res.resource_amount(), 2, "   and pays one per line")

	# End to end: locking the last piece of a rectangle emits STRUCTURE_BUILT through
	# the ordinary event stream, which is the seam the minigame reads.
	group("Resources: a lock completes a structure")
	var b3 := _build(["LOOJ", "LOOJ", "LLJJ"], ps)
	var jcells := []                                   # empty the J back out again
	for yy in 3:
		for xx in 4:
			var i: int = (b3.total_height - 3 + yy) * b3.width + xx
			if b3.owner[i] == 3:                       # J is the third letter seen
				jcells.append(Vector2i(xx, b3.total_height - 3 + yy))
				b3.cells[i] = 0
				b3.owner[i] = 0
	eq(jcells.size(), 4, "the J's four cells are cleared out")
	var g3 := make_game(b3)
	g3.start()
	# Owner ids must be unique board-wide or the whole-piece test miscounts: the J would
	# be stamped 1, the L is already 1, and together they look like one eight-cell piece.
	# Real play cannot collide -- every id comes from pieces_placed -- but _build numbers
	# from 1 too, so move the counter past it.
	g3.pieces_placed = 100
	eq(Structures.scan(b3, 4, 3).size(), 0, "with the J gone there is no structure")

	# Seat a real J on exactly that footprint, then lock it the normal way.
	var want := {}
	for c in jcells:
		want[c] = true
	var placed := false
	for rot in 4:
		for py in range(b3.total_height - 4, b3.total_height):
			for px in range(-1, 5):
				var ap := ActivePiece.new()
				ap.def = ps.get_def(&"J")
				ap.rot = rot
				ap.pos = Vector2i(px, py)
				var got := ap.board_cells()
				if got.size() != 4:
					continue
				var same := true
				for c in got:
					if not want.has(c):
						same = false
						break
				if not same or not b3.can_place(ap.cells(), ap.pos):
					continue
				g3.active = ap
				placed = true
				break
			if placed:
				break
		if placed:
			break
	check(placed, "a real J fits the hole exactly")
	var ev3: Array = []
	g3.lock_piece(ev3)
	var msgs: Array = ev3.filter(func(e): return e.type == GameEvent.Type.STRUCTURE_BUILT)
	eq(msgs.size(), 1, "locking it emits STRUCTURE_BUILT once")
	if msgs.size() == 1:
		eq(int(msgs[0].data["payout"].get(ShapeGroups.O, 0)), 1, "   carrying the O payout")
		eq(msgs[0].data["rect"].size, Vector2i(4, 3), "   and the rectangle it names")
	eq(g3.resources.earned(Resources.STRUCTURES, ShapeGroups.LJ), 2,
		"the bank is credited from the structure source")

	# The scan is not limited to the three shapes the design named: it accepts any
	# tiling. IOO is the one that matters, being the best O source in the game.
	# Lower-case letters are a SECOND piece of the same type, so the two O pieces get
	# distinct owner ids -- as one letter they would read as a single eight-cell piece.
	group("Resources: any tiling counts, not just the named three")
	var ioo := _build(["IIII", "ooOO", "ooOO"], ps)
	var fi := Structures.scan(ioo, 4, 3)
	eq(fi.size(), 1, "IIII/OOOO/OOOO tiles a 4x3 too")
	if fi.size() == 1:
		var pi := Structures.payout(fi[0], ps)
		eq(int(pi.get(ShapeGroups.O, 0)), 2, "   paying O twice -- matching its best clear")
		eq(int(pi.get(ShapeGroups.I, 0)), 1, "   and I once")
	# T tops out at 2 here, so spinning stays the better route to T.
	var ltt := _build(["Tttt", "TTtL", "TLLL"], ps)
	var fl := Structures.scan(ltt, 4, 3)
	eq(fl.size(), 1, "LTT tiles a 4x3")
	if fl.size() == 1:
		eq(int(Structures.payout(fl[0], ps).get(ShapeGroups.T, 0)), 2, "   paying T twice, not three")

	# The same three pieces stood on end fill a 3x4 and NO 4x3, so the transpose is a
	# separate scan size. L, J and O are exactly the case: the design's LOOJ/LOOJ/LLJJ
	# lying down, and JJJ/OOJ/OOL/LLL standing up.
	group("Resources: a rectangle is not rotation-invariant")
	var tall := _build(["JJJ", "OOJ", "OOL", "LLL"], ps)
	eq(Structures.scan(tall, 4, 3).size(), 0, "a 3x4 build fills no 4x3")
	var ft := Structures.scan(tall, 3, 4)
	eq(ft.size(), 1, "   but it does fill a 3x4")
	if ft.size() == 1:
		var pt := Structures.payout(ft[0], ps)
		eq(int(pt.get(ShapeGroups.LJ, 0)), 2, "   paying the same as lying down: LJ2")
		eq(int(pt.get(ShapeGroups.O, 0)), 1, "   and O1")

	# Game scans both, so the standing build pays -- and a wide one still pays once.
	var g4 := make_game(_build(["JJJ", "OOJ", "OOL", "LLL"], ps))
	g4.start()
	g4.pieces_placed = 100
	var ev4: Array = []
	g4._scan_structures(ev4)
	eq(ev4.size(), 1, "Game finds the standing structure")
	check(Vector2i(4, 3) in g4.structure_sizes and Vector2i(3, 4) in g4.structure_sizes,
		"both orientations are scanned")
	# 2x4 is deliberately absent: two stacked O pieces would pay O2 for no skill.
	check(not (Vector2i(2, 4) in g4.structure_sizes), "2x4 is deliberately not scanned")

	# Scanning two sizes must not pay one structure twice. Keying on the owner set is
	# what prevents it, so a structure that somehow matched both would still pay once.
	var g5 := make_game(_build(["LOOJ", "LOOJ", "LLJJ"], ps))
	g5.start()
	g5.pieces_placed = 100
	g5.structure_sizes = [Vector2i(4, 3), Vector2i(4, 3)]   # same size twice
	var ev5: Array = []
	g5._scan_structures(ev5)
	eq(ev5.size(), 1, "one structure pays once however many scans find it")
	eq(g5.resources.earned(Resources.STRUCTURES, ShapeGroups.LJ), 2, "   crediting LJ twice, not four")


# ------------------------------------------------------------------------------
# PaneTab
# ------------------------------------------------------------------------------

func test_pane_tab() -> void:
	group("PaneTab: the right pane's pages")

	eq(PaneTab.next(PaneTab.GAMEPLAY), PaneTab.DEBUG, "F8 goes gameplay -> debug")
	eq(PaneTab.next(PaneTab.DEBUG), PaneTab.OFF, "   then off")
	eq(PaneTab.next(PaneTab.OFF), PaneTab.GAMEPLAY, "   then back round")
	eq(PaneTab.pages().size(), 2, "OFF is a state, not a page, so the strip has two tabs")
	check(not (PaneTab.OFF in PaneTab.pages()), "   and OFF is not one of them")
	eq(PaneTab.name_of(PaneTab.GAMEPLAY), "GAMEPLAY", "pages are named for the strip")
	eq(PaneTab.name_of(PaneTab.DEBUG), "DEBUG", "   both of them")
	# F8 is Godot's "Stop Running Project" and the editor honours it even while the game
	# has focus, so a meta key bound there quits instead of firing. Nothing may claim it.
	var meta: Dictionary = InputReader.META
	for action in meta:
		check(not (KEY_F8 in meta[action]),
			"%s does not use F8, which Godot reserves for Stop" % action)
	check(KEY_TAB in meta["cycle_pane"], "the pane tab is on TAB instead")
	# And TAB must not collide with a game action in any profile.
	for pname in ControlProfiles.names():
		var prof: Dictionary = ControlProfiles.get_profile(pname)
		for act in prof:
			check(not (KEY_TAB in prof[act]), "%s/%s leaves TAB free" % [pname, act])

	group("RuleSet: soft drop factor per preset")
	eq(RuleSet.guideline().sdf, 20.0, "guideline soft drop is 20x gravity, as specified")
	eq(RuleSet.competitive().sdf, 60.0, "competitive is 60x -- one cell per tick at level 1")
	# The two presets must actually differ here, or the competitive override is dead code.
	check(RuleSet.competitive().sdf > RuleSet.guideline().sdf,
		"competitive soft drop is the faster of the two")
	# One cell per tick at level 1 is the whole point of 60; check the arithmetic, since
	# sdf multiplies gravity and the relationship is easy to break.
	var rc := RuleSet.competitive()
	var cells_per_tick := rc.gravity_cells_per_tick(1, 60.0) * rc.sdf
	check(absf(cells_per_tick - 1.0) < 0.0001,
		"at level 1 soft drop moves exactly one cell per tick")

	group("AttackTable: back-to-back is floor(sqrt(b2b))")
	var at := AttackTable.new()
	# No chain pays nothing: -1 is "no b2b", 0 is the first difficult clear, which starts
	# the chain rather than being rewarded by it.
	eq(at.b2b_bonus_for(-1), 0, "no chain pays nothing")
	eq(at.b2b_bonus_for(0), 0, "the first difficult clear starts the chain, no bonus")
	for pair in [[1, 1], [2, 1], [3, 1], [4, 2], [8, 2], [9, 3], [15, 3], [16, 4],
			[24, 4], [25, 5], [27, 5], [35, 5], [36, 6], [48, 6], [49, 7]]:
		eq(at.b2b_bonus_for(int(pair[0])), int(pair[1]),
			"b2b %d pays +%d" % [int(pair[0]), int(pair[1])])
	# Each step up costs quadratically more than the last, which is the point of sqrt.
	var steps: Array = []
	var last := 0
	for n in range(1, 50):
		var v := at.b2b_bonus_for(n)
		if v != last:
			steps.append(n)
			last = v
	eq(steps, [1, 4, 9, 16, 25, 36, 49], "the bonus steps at the perfect squares")

	# And it reaches attack_for: a quad on a 27-chain is 4 + 5.
	var qr := ClearResult.new()
	qr.lines = 4
	qr.piece_id = &"I"
	qr.b2b = 27
	eq(at.attack_for(qr), 9, "a quad on a 27-chain sends 4 + 5 = 9")
	qr.b2b = 0
	eq(at.attack_for(qr), 4, "the same quad starting a chain sends just 4")
	# A single is not difficult, so it never collects a chain bonus even if one is passed.
	var sr := ClearResult.new()
	sr.lines = 1
	sr.piece_id = &"I"
	sr.b2b = 27
	eq(at.attack_for(sr), 0, "a single collects no chain bonus -- it is not a difficult clear")

	# docs/replay_format.md claims five fields are enough to view a record. Prove it, by
	# writing a minimal file and loading it -- a contract doc nobody tests is a guess.
	group("ReplayLog: the documented minimal record")
	var blank_row := "."
	blank_row = blank_row.repeat(10)
	var rows: Array = []
	for i in 40:
		rows.append(blank_row)
	# A flat T on the floor: three cells on row 39, the nub on row 38.
	var lines_out: Array = []
	for placement in 2:
		var rec := {
			"replay": "contract_probe",
			"n": placement,
			"board": rows,
			"piece": "t",
			"place": {"type": "t", "cells": [[3, 39], [4, 39], [5, 39], [4, 38]]},
		}
		lines_out.append(JSON.stringify(rec))
	var probe := "user://_contract_probe.jsonl"
	var fh := FileAccess.open(probe, FileAccess.WRITE)
	check(fh != null, "the probe file can be written")
	if fh != null:
		fh.store_string("
".join(lines_out) + "
")
		fh.close()
		var rl := ReplayLog.new(PieceSet.tetrominoes())
		check(rl.load_file(probe), "a file of minimal records loads: %s" % rl.error)
		eq(rl.run_count(), 1, "   and holds one run")
		eq(rl.names[0], "contract_probe", "   named by the replay field")
		var recs := rl.steps_for("contract_probe")
		eq(recs.size(), 2, "   with both records")
		if recs.size() == 2:
			var gp := make_game(Board.new(10, 20, 20))
			gp.start()
			rl.apply(recs[0], gp)
			check(gp.active != null, "   the placed piece is recovered from absolute cells")
			if gp.active != null:
				eq(gp.active.def.id, &"T", "      as a T")
				var got := {}
				for c in gp.active.board_cells():
					got[c] = true
				var want_cells := [Vector2i(3, 39), Vector2i(4, 39), Vector2i(5, 39), Vector2i(4, 38)]
				var all_there := true
				for c in want_cells:
					if not got.has(c):
						all_there = false
				check(all_there, "      at exactly the cells given")
			eq(gp.pieces_placed, 0, "   and n becomes the placement number")
		# Garbage must be uppercase G; lowercase is not a piece letter and reads as empty.
		var grows: Array = rows.duplicate()
		grows[39] = "GGGGGGGGG."
		var g1 := {"replay": "g_probe", "n": 0, "board": grows, "piece": "t",
			"place": {"type": "t", "cells": [[3, 38], [4, 38], [5, 38], [4, 37]]}}
		var fh2 := FileAccess.open(probe, FileAccess.WRITE)
		fh2.store_string(JSON.stringify(g1) + "
")
		fh2.close()
		var rl2 := ReplayLog.new(PieceSet.tetrominoes())
		check(rl2.load_file(probe), "a record with garbage loads")
		var gp2 := make_game(Board.new(10, 20, 20))
		gp2.start()
		rl2.apply(rl2.steps_for("g_probe")[0], gp2)
		eq(gp2.board.get_cell(0, 39), Board.GARBAGE, "uppercase G reads as garbage")
		eq(gp2.board.get_cell(9, 39), 0, "   and the hole stays empty")
		DirAccess.remove_absolute(ProjectSettings.globalize_path(probe))
