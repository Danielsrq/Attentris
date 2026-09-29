class_name Structures
extends RefCounted

## Finds rectangles in the stack that are tiled EXACTLY by whole pieces.
##
##     LOOJ        LJJJ        LLLJ
##     LOOJ        LZZJ        LSSJ
##     LLJJ        LLZZ        SSJJ
##
## Three tetrominoes filling a 4x3 with no gaps and nothing hanging out. A second way
## to earn resource that rewards deliberate construction rather than line clears, so a
## piece the clear economy barely pays for -- O especially -- can still be worth
## placing well.
##
## "Whole piece" is why Board carries an owner id. Cell VALUES only record the type, so
## two L pieces side by side are eight indistinguishable L cells; owner ids keep each
## placement identifiable, and a piece that has lost cells to a line clear no longer
## has four of them and so cannot complete a structure.
##
## Rotations and reflections need no special handling INSIDE a given rectangle: the area
## is checked by what fills it, not matched against a template, so every arrangement that
## tiles a 4x3 is found by the same test and none has to be written down.
##
## But a rectangle has eight symmetries and only four of them keep its footprint. Turning
## a 4x3 block 90 degrees makes it 3x4, and a 3x4 build fills no 4x3 at all:
##
##     identity   4x3   LOOJ / LOOJ / LLJJ      flip H   4x3   JOOL / JOOL / JJLL
##     rot 180    4x3   JJLL / JOOL / JOOL      flip V   4x3   LLJJ / LOOJ / LOOJ
##     rot 90     3x4   LLL / LOO / JOO / JJJ   diag     3x4   JJJ / JOO / LOO / LLL
##     rot 270    3x4   JJJ / OOJ / OOL / LLL   anti     3x4   LLL / OOL / OOJ / JJJ
##
## So covering all eight is not a property of this function -- it is a property of the
## SIZE LIST. Game scans 4x3 and 3x4 for exactly that reason; with only the size the
## design was drawn at, half of every structure's orientations would score nothing. All
## eight pay identically, since turning a rectangle cannot change what it is made of.
##
## A 4x3 has 23 distinct tilings, in 13 compositions. With payout() paying one per
## piece into its own group, that is the whole structure economy:
##
##     IOO -> I1 O2      JLS JLZ LLS JJZ -> LJ2 SZ1
##     IJJ ILL -> I1 LJ2  JTT LTT -> LJ1 T2
##     JJO JLO LLO -> LJ2 O1
##     III -> I3
##
## Three things follow, and they are why this mechanic is worth having:
##   - O caps at 2 here, the same as its best line clear, so structures roughly double
##     the only supply O has.
##   - T caps at 2, against the 3 a T-spin triple pays, so spinning stays the better T
##     route and this is just a floor.
##   - SZ caps at 1. S and Z tile badly, so the skew resources stay clear-led -- which
##     is the intent: a Z-spin triple should be the thing worth learning.
##   - LJ appears in 11 of the 13, because L and J are the only pieces whose 3+1
##     footprint fills a corner, so LJ is the abundant structure resource.
## III pays I3 and is trivial to build, but a single quad pays I4, so it undercuts
## nothing.

## A find is a Dictionary, not an object -- matching TSpinSlots, and avoiding an inner
## class for what is only ever a few fields:
##
##   rect    Rect2i of the tiled area
##   owners  placement ids that tile it, ascending -- a stable identity, so the same
##           structure is not paid for twice
##   pieces  cell value (PieceDef.index + 1) per owner, for whatever the economy pays on

const CELLS_PER_PIECE := 4

## Scans for every rectangle of the given size tiled by whole pieces.
##
## Cost is one pass per top-left position, each reading w*h cells, so a 4x3 over a
## 10x40 board is ~266 positions of 12 reads. Cheap enough to run on every lock.
static func scan(board: Board, w: int = 4, h: int = 3) -> Array:
	var out: Array = []
	if w <= 0 or h <= 0 or (w * h) % CELLS_PER_PIECE != 0:
		return out
	var want := (w * h) / CELLS_PER_PIECE
	for top in range(0, board.total_height - h + 1):
		for left in range(0, board.width - w + 1):
			var f = _at(board, left, top, w, h, want)
			if f != null:
				out.append(f)
	return out

## Returns the find Dictionary, or null when this rectangle is not tiled.
static func _at(board: Board, left: int, top: int, w: int, h: int, want: int):
	var inside := {}
	for y in range(top, top + h):
		for x in range(left, left + w):
			var i: int = y * board.width + x
			if board.cells[i] == 0:
				return null                       # a gap: not a solid rectangle
			var o: int = board.owner[i]
			if o == 0:
				return null                       # garbage, or a cell with no placement
			inside[o] = int(inside.get(o, 0)) + 1
	if inside.size() != want:
		return null
	# Every piece must be WHOLLY inside: four cells here and four on the board.
	for o in inside:
		if int(inside[o]) != CELLS_PER_PIECE:
			return null
	if _cells_owned(board, inside.keys()) != want * CELLS_PER_PIECE:
		return null

	var owners: Array = inside.keys()
	owners.sort()
	var pieces: Array = []
	for o in owners:
		pieces.append(_value_of(board, o))
	return {"rect": Rect2i(left, top, w, h), "owners": owners, "pieces": pieces}

## Total cells on the whole board belonging to these owners. Larger than the rectangle
## holds means a piece pokes out of it, which does not count as tiling.
##
## Relies on owner ids being unique board-wide, which Game guarantees by stamping
## pieces_placed. Two placements sharing an id would read as one eight-cell piece.
static func _cells_owned(board: Board, owners: Array) -> int:
	var want := {}
	for o in owners:
		want[o] = true
	var n := 0
	for i in board.owner.size():
		if board.owner[i] != 0 and want.has(board.owner[i]):
			n += 1
	return n

static func _value_of(board: Board, owner_id: int) -> int:
	for i in board.owner.size():
		if board.owner[i] == owner_id:
			return board.cells[i]
	return 0

## Stable identity for a find, so a structure is only ever paid for once.
static func key(found: Dictionary) -> String:
	var parts: Array = []
	for o in found["owners"]:
		parts.append(str(o))
	return ",".join(parts)

## What a structure pays, as { group -> amount }: one per contributing piece, into that
## piece's own group.
##
## Spreading the payout by composition is what makes this the O supply. A 4x3 pays 3 in
## total -- about a triple -- but LOOJ pays O 2 and LJ 1, and no line clear can pay O
## more than 2 at once even in the best case. So a player who wants O builds for it.
static func payout(found: Dictionary, piece_set: PieceSet) -> Dictionary:
	var out := {}
	for v in found["pieces"]:
		var g := ShapeGroups.of(piece_set.id_for_value(int(v)))
		out[g] = int(out.get(g, 0)) + 1
	return out
