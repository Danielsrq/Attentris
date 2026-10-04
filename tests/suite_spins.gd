extends TestHarness

## Spins: detection, all-spin mode and the assist
##
## Run by tests/run_tests.gd, which owns the totals. See tests/test_harness.gd for
## group/check/eq and the shared board fixtures.

func run() -> void:
	test_all_spin_immobility()
	test_allspin_mini_promotion()
	test_wound_board()
	test_repeat_penalty()
	test_ghost_pocket()

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
	# Normal rules (the default preset): the same S-spin counts, but only as a Mini.
	eq(SpinDetect.detect(b, p, RuleSet.normal()), SpinDetect.Kind.MINI, "normal rules: S-spin is a Mini")
	eq(SpinDetect.detect(b, p, RuleSet.allspin()), SpinDetect.Kind.FULL, "all-spin: the same S-spin is full")

	# Free space is never a spin.
	var empty := Board.new(10, 20, 20)
	var g2 := make_game(empty)
	var p2 := place(g2, &"S", 0, Vector2i(4, 30))
	p2.last_action_was_rotation = true
	check(not SpinDetect.is_immobile(empty, p2), "S in open space is mobile")
	eq(SpinDetect.detect(empty, p2, g2.rules), SpinDetect.Kind.NONE, "open-space rotation is not a spin")

# ------------------------------------------------------------------------------
# All-spin mode: Mini promotion, wound lines, repeat penalty
# ------------------------------------------------------------------------------

func test_allspin_mini_promotion() -> void:
	group("All-spin: Minis promote to full")
	var rs := RuleSet.allspin()
	check(rs.all_spin and rs.allspin_full and rs.repeat_penalty, "allspin preset turns on all three flags")
	eq(rs.wound_clears, 10, "wound lines last 10 clears")
	# The same position test_tss_mini proves is a Mini under competitive rules.
	var b := make_board([
		"...X......",
		"..........",
		"...X.X....",
	])
	var g := make_game(b, rs)
	var p := place(g, &"T", 1, Vector2i(3, 37))
	check(g.try_rotate(0), "R->0 rotation succeeds")
	eq(SpinDetect.detect(b, p, g.rules), SpinDetect.Kind.FULL, "the Mini is promoted to full")

func test_wound_board() -> void:
	group("Wound lines on the board")
	var b := Board.new(10, 20, 20)
	var bottom := b.total_height - 1
	b.set_cell(0, bottom, 1)
	b.add_wound(10, 6)
	eq(b.wound_count(), 1, "one wound")
	eq(b.get_cell(5, bottom), Board.WOUND, "a solid wound line at the very bottom")
	eq(b.get_cell(0, bottom - 1), 1, "   the stack was pushed up, nothing taken")
	check(b.is_blocked(5, bottom), "   pieces cannot enter it")
	check(b.find_full_rows().is_empty(), "   and it is never a clearable full row")
	check(not b.is_empty(), "no perfect clear while a wound line is on the board")
	b.tick_wounds()
	b.add_wound(10, 2)
	eq([b.wound_left[bottom - 1], b.wound_left[bottom]], [9, 10], "a newer wound enters beneath, pushing the older up")
	for i in 8:
		b.tick_wounds()
	eq(b.tick_wounds(), 1, "the older wound turns into garbage on its 10th clear")
	eq(b.get_cell(6, bottom - 1), 0, "   its hole opens at column 6")
	eq(b.get_cell(0, bottom - 1), Board.GARBAGE, "   the rest is ordinary garbage")
	check(not b.is_wound(bottom - 1), "   and it is no longer a wound")
	eq(b.get_cell(2, bottom), Board.WOUND, "the newer one is still a wound")
	b.tick_wounds()
	eq(b.get_cell(2, bottom), 0, "   until its own 10th clear (hole at 2)")

	var e := Board.new(10, 20, 20)
	e.add_wound(10, 3)
	e.push_garbage([4])
	eq(e.get_cell(4, e.total_height - 1), 0, "garbage enters beneath the wound, hole at 4")
	check(e.is_wound(e.total_height - 2), "   the wound line was pushed up with the stack")
	# A clear above the wound drops rows onto it; the wound stays put and stays a wound.
	for x in 10:
		e.set_cell(x, e.total_height - 3, 1)
	e.clear_rows(e.find_full_rows())
	check(e.is_wound(e.total_height - 2), "clearing a row above leaves the wound in place")
	var d := e.duplicate_board()
	eq(d.wound_count(), 1, "duplicate_board copies the wounds")

func test_repeat_penalty() -> void:
	group("Repeat penalty")
	var g := make_game(Board.new(10, 20, 20), RuleSet.allspin())
	g.start()
	var r1 := _o_double(g)
	eq(r1.lines, 2, "first O double clears two rows")
	check(not r1.repeat, "   first clear of the game is never a repeat")
	eq(g.board.wound_count(), 0, "   no wound")
	var r2 := _o_double(g)
	check(r2.repeat, "double -> double is a repeat")
	eq(g.board.wound_count(), 1, "   and inserts a wound line")
	# Both doubles empty the board, so both are perfect clears (1 + 10).
	eq(r2.attack, r1.attack, "   the repeat still sends its full attack")
	var added := false
	for ev in _last_events:
		if ev.type == GameEvent.Type.WOUND_ADDED:
			added = true
	check(added, "   WOUND_ADDED fires")
	check(r2.tags().has("Repeat - wounded"), "   and the result is tagged")
	eq(g.board.get_cell(9, g.board.total_height - 1), Board.WOUND, "the floor row is the wound line")
	eq(g.board.wound_left[g.board.total_height - 1], 10, "   with 10 clears to go")
	check(r2.wound_hole >= 0 and r2.wound_hole < 10, "   and a hole chosen for later")

	var c := ClearResult.new()
	c.lines = 1
	c.piece_id = &"I"
	var dbl := ClearResult.new()
	dbl.lines = 2
	dbl.piece_id = &"T"
	check(c.clear_type() != dbl.clear_type(), "single -> double is not a repeat")
	var tsd := ClearResult.new()
	tsd.lines = 2
	tsd.piece_id = &"T"
	tsd.spin = SpinDetect.Kind.FULL
	var zsd := ClearResult.new()
	zsd.lines = 2
	zsd.piece_id = &"Z"
	zsd.spin = SpinDetect.Kind.FULL
	var tst := ClearResult.new()
	tst.lines = 3
	tst.piece_id = &"T"
	tst.spin = SpinDetect.Kind.FULL
	check(tsd.clear_type() != zsd.clear_type(), "TSD -> ZSD is not a repeat")
	check(tsd.clear_type() != tst.clear_type(), "TSD -> TST is not a repeat")
	check(tsd.clear_type() != dbl.clear_type(), "TSD -> plain double is not a repeat")
	var q1 := ClearResult.new()
	q1.lines = 4
	q1.piece_id = &"I"
	var q2 := ClearResult.new()
	q2.lines = 4
	q2.piece_id = &"I"
	eq(q1.clear_type(), q2.clear_type(), "quad -> quad is a repeat")

	var off := make_game(Board.new(10, 20, 20), RuleSet.competitive())
	off.start()
	_o_double(off)
	_o_double(off)
	eq(off.board.wound_count(), 0, "competitive rules: no repeat penalty")

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

## Fills the bottom two rows except columns 0-1, locks an O there: a plain double.
func _o_double(g: Game) -> ClearResult:
	var b := g.board
	for y in [b.total_height - 2, b.total_height - 1]:
		for x in range(2, 10):
			b.set_cell(x, y - b.wound_count(), 9)
	var p := place(g, &"O", 0, Vector2i(0, b.total_height - 2 - b.wound_count()))
	p.last_action_was_rotation = false
	var ev: Array = []
	g.lock_piece(ev)
	_last_events = ev
	return g.last_result

var _last_events: Array = []
