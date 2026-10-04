extends TestHarness

## Economy: shape resources and structures
##
## Run by tests/run_tests.gd, which owns the totals. See tests/test_harness.gd for
## group/check/eq and the shared board fixtures.

func run() -> void:
	test_shape_resources()
	test_structures()
	test_resources()

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
