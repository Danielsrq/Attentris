extends TestHarness

## Board: clears, the randomizer and garbage
##
## Run by tests/run_tests.gd, which owns the totals. See tests/test_harness.gd for
## group/check/eq and the shared board fixtures.

func run() -> void:
	test_board_clears()
	test_board_perfect_clear_and_remove_row()
	test_randomizer_bag()
	test_randomizer_seed()
	test_garbage_push()
	test_receive_garbage()
	test_game_advancing()

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
# Incoming garbage
# ------------------------------------------------------------------------------

func test_receive_garbage() -> void:
	group("receive_garbage: garbage from outside the board can kill")

	var g := make_game(Board.new(10, 20, 20))
	g.start()
	var floor_y: int = g.board.total_height - 1
	var ev: Array = []
	g.receive_garbage([4], ev)
	eq(_row(g.board, floor_y), "GGGG.GGGGG", "one row lands on the floor with its hole")
	var got: Array = ev.filter(func(e): return e.type == GameEvent.Type.GARBAGE_RECEIVED)
	eq(got.size(), 1, "   and reports itself once")
	if got.size() == 1:
		eq(int(got[0].data["rows"]), 1, "      carrying the row count")
	check(not g.topped_out, "   without topping out")

	# Nothing to do is not an error -- the caller never has to special-case a quiet tick.
	ev.clear()
	g.receive_garbage([], ev)
	eq(ev.size(), 0, "an empty push is a silent no-op")

	# Bury: the stack already reaches the top of the buffer, so this push would discard
	# real cells. push_garbage would do that silently; receive_garbage must not.
	var full := make_game(Board.new(10, 20, 20))
	full.start()
	full.active = null                       # take the piece out of it; this is about the stack
	for x in 10:
		full.board.set_cell(x, 0, 1)         # row 0 is the very top of the buffer
	ev.clear()
	full.receive_garbage([3], ev)
	check(full.topped_out, "a push that would bury the stack tops out")
	eq(full.top_out_reason, "garbage_bury", "   named garbage_bury")
	eq(_row(full.board, 0), "XXXXXXXXXX", "   and the board is left untouched, not half-pushed")
	var outs: Array = ev.filter(func(e): return e.type == GameEvent.Type.TOP_OUT)
	eq(outs.size(), 1, "   reporting a TOP_OUT")
	# Already dead stays dead, and stops mutating the board.
	ev.clear()
	full.receive_garbage([3], ev)
	eq(ev.size(), 0, "a topped-out game ignores further garbage")

	# The active piece rides up with the stack rather than being left embedded in it.
	var lift := make_game(Board.new(10, 20, 20))
	lift.start()
	lift.active.pos = Vector2i(3, lift.board.total_height - 2)
	var before: int = lift.active.pos.y
	ev.clear()
	lift.receive_garbage([0, 1], ev)
	check(not lift.topped_out, "a piece sitting low does not die to garbage under it")
	eq(lift.active.pos.y, before - 2, "   it is lifted by exactly the rows pushed")
	check(lift.board.can_place(lift.active.cells(), lift.active.pos),
		"   and ends somewhere it actually fits")

	# Block-out: the piece cannot fit anywhere within the lift, because the buffer above
	# it is solid. Rows 0..n-1 stay clear so this is a block-out and not a bury.
	var blocked := make_game(Board.new(10, 20, 20))
	blocked.start()
	for y in range(1, blocked.board.total_height):
		for x in 10:
			blocked.board.set_cell(x, y, 1)
	blocked.active.pos = Vector2i(3, 0)
	ev.clear()
	blocked.receive_garbage([5], ev)
	check(blocked.topped_out, "a piece with nowhere to rise blocks out")
	eq(blocked.top_out_reason, "garbage_block_out", "   named garbage_block_out")


# ------------------------------------------------------------------------------
# Game: freezing and gravity scale
# ------------------------------------------------------------------------------

## Rehomed from the deleted defence suite. `advancing` and `gravity_scale` were added for
## that minigame's focus switching and lost their only writer when it went, but they are
## general Game mechanics and still work -- so the coverage moves here rather than being
## deleted with the feature that happened to motivate it.
func test_game_advancing() -> void:
	group("Game.advancing: freezing stops everything that advances on its own")
	# The trap, and the reason `advancing` exists at all: scaling gravity to zero stops the
	# FALL but not the LOCK. A piece already resting on the stack is governed by
	# lock_delay_sec, so with gravity alone it locks itself half a second later. Whenever
	# something wants to pause Tetris, that lock is what reads as "it carried on without me".
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

	group("Game.gravity_scale: passive gravity only")
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
	# Zero scale must behave exactly like gravity off, or the two ways of stopping the
	# piece would disagree.
	var g4 := make_game(Board.new(10, 20, 20))
	g4.start()
	g4.gravity_scale = 0.0
	var held: int = g4.active.pos.y
	for _i in 300:
		g4.tick(inp)
	eq(g4.active.pos.y, held, "a zero scale holds the piece exactly like gravity off")
