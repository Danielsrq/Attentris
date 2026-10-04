class_name TowerSet
extends RefCounted

## The tower factory. Takes a spec dictionary (see TowerData), compiles each mask into a
## TowerDef with all four footprint rotations and their cannon cells cached, and hands out
## the result. Mirrors PieceSet, down to the registry fields.

## The cannon marker in a mask. Any other non-blank character is an ordinary body cell.
const CANNON_CHAR := "C"

var defs: Array = []        ## Array[TowerDef], in declaration order
var by_id: Dictionary = {}  ## StringName -> TowerDef

static func standard() -> TowerSet:
	return TowerSet.new(TowerData.towers())

func _init(specs: Dictionary) -> void:
	var i := 0
	for id in specs.keys():
		var d := _compile(id, specs[id], i)
		defs.append(d)
		by_id[id] = d
		i += 1

func get_def(id: StringName) -> TowerDef:
	return by_id.get(id, null)

func size() -> int:
	return defs.size()

func ids() -> Array:
	return by_id.keys()

## Compiles one mask.
##
## The rotation is PieceSet.rotate_cw, reused rather than reimplemented -- its own doc
## calls it "the one rotation primitive", and two copies of a rotation are how two
## rotations drift apart.
##
## The loop rotates in RAW box coordinates and normalises only the copy it stores. Keeping
## the working cells un-normalised is what makes four rotations return exactly to the
## start: rotate_cw is (x, y) -> (box - 1 - y, x), which is the identity after four
## applications only while the pivot domain stays put.
static func _compile(id: StringName, spec: Dictionary, index: int) -> TowerDef:
	var d := TowerDef.new()
	d.id = id
	d.color = spec["color"]
	d.index = index

	var rows: Array = spec["rows"]
	var base: Array = []
	var cannon := Vector2i(-1, -1)
	var cannon_count := 0
	var width := 0
	for y in rows.size():
		var line: String = rows[y]
		width = maxi(width, line.length())
		for x in line.length():
			var ch := line[x]
			if ch == "." or ch == " ":
				continue
			base.append(Vector2i(x, y))
			if ch == CANNON_CHAR:
				cannon = Vector2i(x, y)
				cannon_count += 1
	assert(not base.is_empty(), "tower %s has no cells" % id)
	assert(cannon_count == 1, "tower %s needs exactly one %s, found %d" % [
		id, CANNON_CHAR, cannon_count])

	# Derived, not authored: a wrong box silently mis-centres a piece in PieceSet, and
	# there is no kick table here that would justify letting anyone choose it.
	d.box = maxi(width, rows.size())

	var cells: Array = base
	var cc := cannon
	for r in 4:
		if r > 0:
			cells = PieceSet.rotate_cw(cells, d.box)
			cc = PieceSet.rotate_cw([cc], d.box)[0]
		var off := _min_corner(cells)
		d.rotations.append(_sorted(_shifted(cells, -off)))
		d.cannons.append(cc - off)          # the SAME delta, so it stays inside the shape
		d.extents.append(_span(cells))
	# After the rotations exist, because it walks them.
	d.muzzles = Muzzle.table(d)
	return d

## Top-left corner of the tight bounding box.
static func _min_corner(cells: Array) -> Vector2i:
	var lo := Vector2i(0x7FFFFFFF, 0x7FFFFFFF)
	for c in cells:
		lo = Vector2i(mini(lo.x, c.x), mini(lo.y, c.y))
	return lo

## Width and height of the tight bounding box, in cells.
static func _span(cells: Array) -> Vector2i:
	var lo := _min_corner(cells)
	var hi := Vector2i(-0x7FFFFFFF, -0x7FFFFFFF)
	for c in cells:
		hi = Vector2i(maxi(hi.x, c.x), maxi(hi.y, c.y))
	return hi - lo + Vector2i.ONE

static func _shifted(cells: Array, delta: Vector2i) -> Array[Vector2i]:
	var out: Array[Vector2i] = []
	for c in cells:
		out.append(c + delta)
	return out

## Row-major canonical order, so two states with the same occupied set compare equal and
## iteration is deterministic. Copies first: the caller's array is never mutated.
## Typed on the way out, so TowerDef.cells() can promise Array[Vector2i] and callers can
## use the elements as Vector2i without a cast.
static func _sorted(cells: Array) -> Array[Vector2i]:
	var out: Array[Vector2i] = []
	for c in cells:
		out.append(c)
	out.sort_custom(func(a: Vector2i, b: Vector2i) -> bool:
		if a.y != b.y:
			return a.y < b.y
		return a.x < b.x)
	return out
