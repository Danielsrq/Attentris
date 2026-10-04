extends TestHarness

## Scoring: points, chains and attack
##
## Run by tests/run_tests.gd, which owns the totals. See tests/test_harness.gd for
## group/check/eq and the shared board fixtures.

func run() -> void:
	test_scorer_values()
	test_b2b_and_combo_chain()
	test_outcome_names()
	test_zero_line_spin_scoring()
	test_parity_property()
	test_attack_table()
	test_hold_swap()
	test_width_12()

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
	eq(t.attack_for(attack_result(1, SpinDetect.Kind.MINI, &"T", -1)), 0, "T-spin mini single sends nothing")
	eq(t.attack_for(attack_result(2, SpinDetect.Kind.MINI, &"T", -1)), 0, "T-spin mini double sends nothing")
	eq(t.attack_for(attack_result(2, SpinDetect.Kind.MINI, &"S", -1)), 0, "a non-T Mini sends nothing")
	eq(t.attack_for(attack_result(2, full, &"S", -1)), 4, "a full non-T spin (all-spin) still sends 4")
	check(attack_result(1, SpinDetect.Kind.MINI, &"S", -1).is_difficult(), "a Mini that clears keeps B2B alive")
	eq(t.attack_for(attack_result(1, SpinDetect.Kind.MINI, &"S", 3)), 0, "   but earns no B2B bonus itself")

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

	# Burst rules: the spec's worked examples, B2B off so only the doubling shows.
	group("burst attack")
	var bt := AttackTable.new()
	bt.configure(RuleSet.burst())
	var burst_atk := func(lines: int, spin: int, piece: StringName, doubled: bool, chain: int = -1) -> int:
		var r := attack_result(lines, spin, piece, chain)
		r.burst = doubled
		return bt.attack_for(r)
	eq(burst_atk.call(3, full, &"T", false), 10, "TST is 10 under burst rules")
	eq(burst_atk.call(4, none, &"I", false) + burst_atk.call(4, none, &"I", true), 12, "quad -> quad = 4 + 4x2")
	eq(burst_atk.call(2, full, &"T", false) + burst_atk.call(3, full, &"T", true), 24, "TSD -> TST = 4 + 20")
	eq(burst_atk.call(3, full, &"T", false) + burst_atk.call(2, full, &"T", true), 18, "TST -> TSD = 10 + 8")
	eq(burst_atk.call(1, full, &"T", false) + burst_atk.call(2, full, &"T", true), 10, "TSS -> TSD = 2 + 8")
	eq(burst_atk.call(2, full, &"T", false) + burst_atk.call(1, full, &"T", true), 8, "TSD -> TSS = 4 + 4")
	eq(burst_atk.call(2, full, &"T", true, 4), 10, "the B2B bonus is added after, undoubled: 8 + 2")
	eq(t.attack_for(attack_result(3, full, &"T", -1)), 6, "TST stays 6 outside burst rules")
	check(attack_result(2, SpinDetect.Kind.MINI, &"S", -1).qualifies_burst(), "a Mini double qualifies")
	check(attack_result(3, SpinDetect.Kind.MINI, &"T", -1).qualifies_burst(), "a Mini triple qualifies")
	check(not attack_result(1, SpinDetect.Kind.MINI, &"T", -1).qualifies_burst(), "a Mini single does not")
	check(not attack_result(3, none, &"L", -1).qualifies_burst(), "a plain triple does not")

	# Through the lock path: quad, quad (doubled), a non-clearing placement, quad.
	group("burst through Game")
	var bg := make_game(Board.new(10, 20, 20), RuleSet.burst())
	var sent: Array = []
	for step in ["quad", "quad", "drop", "quad"]:
		if step == "quad":
			for y in range(36, 40):
				for x in range(1, 10):
					bg.board.set_cell(x, y, 1)
			bg.board.set_cell(5, 34, 1)
			place(bg, &"I", 1, Vector2i(-2, 36))
		else:
			place(bg, &"O", 0, Vector2i(7, 38))
		var ev: Array = []
		var before := bg.attack_sent
		bg.lock_piece(ev)
		sent.append(bg.attack_sent - before)
		if step == "quad":
			bg.board.clear()
	# 4; then 4x2 + 1 (chain 1); the drop breaks the burst; then 4 + 1 (chain 2).
	eq(sent, [4, 9, 0, 5], "quad 4, doubled quad 9, a drop breaks it, the next quad 5")

	# burst2: the window is 2 placements, so one drop in between keeps it.
	group("burst2 through Game")
	var bg2 := make_game(Board.new(10, 20, 20), RuleSet.burst2())
	var sent2: Array = []
	for step in ["quad", "drop", "quad", "drop", "drop", "quad"]:
		if step == "quad":
			for y in range(36, 40):
				for x in range(1, 10):
					bg2.board.set_cell(x, y, 1)
			bg2.board.set_cell(5, 34, 1)
			place(bg2, &"I", 1, Vector2i(-2, 36))
		else:
			bg2.board.clear()
			place(bg2, &"O", 0, Vector2i(7, 38))
		var ev2: Array = []
		var before2 := bg2.attack_sent
		bg2.lock_piece(ev2)
		sent2.append(bg2.attack_sent - before2)
		bg2.board.clear()
	# 4; drop; 4x2 + 1 (chain 1); two drops end it; 4 + 1 (chain 2).
	eq(sent2, [4, 0, 9, 0, 0, 5], "a drop in between is fine under burst2, two are not")
	var rs2 := RuleSet.burst2()
	var tss := attack_result(1, full, &"T", -1)
	eq(rs2.burst_step(0, tss), [false, 0], "a T-spin single does not arm under burst2")
	eq(rs2.burst_step(2, tss), [true, 1], "   but is doubled when armed, using up a placement")
	eq(RuleSet.burst().burst_step(0, tss), [false, 1], "under burst it still arms")

	# Gate rules: only chains opened by a spin double/triple score.
	group("gate chains")
	var gr := RuleSet.gate_rules()
	var credits := func(seq: Array) -> Array:
		var st: Array = [0, 0, 0]
		var out: Array = []
		for c in seq:
			var res := attack_result(c[0], c[1], &"T" if c[1] != none else &"I", -1)
			var gs: Array = gr.gate_step(st, res)
			out.append(gs[0])
			st = gs[1]
		return out
	var g_tst := [3, full]
	var g_tsd := [2, full]
	var g_quad := [4, none]
	var g_drop := [0, none]
	eq(credits.call([g_tst, g_tsd]), [0, 14], "C-spin: TST then TSD pays 10 + 4")
	eq(credits.call([g_tst, g_drop, g_tsd]), [0, 0, 14], "one placement in between is fine")
	eq(credits.call([g_tst, g_drop, g_drop, g_tsd]), [0, 0, 0, 0], "the window runs out")
	eq(credits.call([g_tsd, g_quad, g_quad]), [0, 0, 0], "quads alone cannot complete a chain")
	eq(credits.call([g_tsd, g_quad, g_tsd]), [0, 0, 9], "but they hold the gate open and add 1")
	eq(credits.call([g_tsd, g_tsd, g_quad]), [0, 8, 1], "and add 1 to a complete chain")
	eq(credits.call([g_quad, g_tsd]), [0, 0], "a quad does not open a gate")
	eq(credits.call([[1, full], g_tsd]), [0, 0], "a T-spin single does not open one either")

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

func attack_result(lines: int, spin: int, piece: StringName, b2b: int) -> ClearResult:
	var r := ClearResult.new()
	r.lines = lines
	r.spin = spin
	r.piece_id = piece
	r.b2b = b2b
	r.combo = -1
	r.level = 1
	return r
