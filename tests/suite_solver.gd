extends TestHarness

## Solver: enumeration, evaluation and board reading
##
## Run by tests/run_tests.gd, which owns the totals. See tests/test_harness.gd for
## group/check/eq and the shared board fixtures.

func run() -> void:
	test_placement_enumeration()
	test_quad_evaluator()
	test_well_detection_with_holes()
	test_dependency_attribution()
	test_hole_topology()
	test_tspin_slots()

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
