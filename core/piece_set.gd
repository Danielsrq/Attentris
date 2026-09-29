class_name PieceSet
extends RefCounted

## The piece factory. Takes a spec dictionary (see PieceData), compiles each entry
## into a PieceDef with all four rotation states cached, and hands out the result.
##
## Nothing downstream assumes seven pieces or a particular box size, so a tromino
## or pentomino set is a pure data addition.

var defs: Array = []        ## Array[PieceDef], in declaration order
var by_id: Dictionary = {}  ## StringName -> PieceDef

static func tetrominoes() -> PieceSet:
	return PieceSet.new(PieceData.tetrominoes())

func _init(specs: Dictionary) -> void:
	var i := 0
	for id in specs.keys():
		var d := _compile(id, specs[id], i)
		defs.append(d)
		by_id[id] = d
		i += 1

func get_def(id: StringName) -> PieceDef:
	return by_id.get(id, null)

func size() -> int:
	return defs.size()

func ids() -> Array:
	return by_id.keys()

## Piece id for a board cell value (1-based; 0 means empty). Garbage has no piece,
## so it comes back empty and ShapeGroups.of maps that to the GARBAGE group.
func id_for_value(v: int) -> StringName:
	if v <= 0 or v > defs.size():
		return &""
	return defs[v - 1].id

## Colour for a board cell value (1-based; 0 means empty).
func color_for_value(v: int) -> Color:
	if v == Board.GARBAGE:
		return Color(0.42, 0.44, 0.48)
	if v == Board.WOUND:
		return Color(0.22, 0.23, 0.27)
	if v <= 0 or v > defs.size():
		return Color.WHITE
	return defs[v - 1].color

static func _compile(id: StringName, spec: Dictionary, index: int) -> PieceDef:
	var d := PieceDef.new()
	d.id = id
	d.box = spec["box"]
	d.color = spec["color"]
	d.kick_class = spec["kick"]
	d.index = index

	var base: Array = []
	var rows: Array = spec["rows"]
	for y in rows.size():
		var line: String = rows[y]
		for x in line.length():
			if line[x] != "." and line[x] != " ":
				base.append(Vector2i(x, y))
	assert(not base.is_empty(), "piece %s has no cells" % id)

	d.first_filled_row = 0x7FFFFFFF
	for c in base:
		d.first_filled_row = mini(d.first_filled_row, c.y)

	var cur: Array = base
	d.rotations = [_sorted(cur)]
	for _r in 3:
		cur = rotate_cw(cur, d.box)
		d.rotations.append(_sorted(cur))
	return d

## The one rotation primitive. Exact integer maths for any box size, which is why
## trominoes (3), tetrominoes (3 and 4) and pentominoes (5) all share it.
## Board space is y-down, so this is a visually clockwise turn.
static func rotate_cw(cells: Array, box: int) -> Array:
	var out: Array = []
	for c in cells:
		out.append(Vector2i(box - 1 - c.y, c.x))
	return out

static func _sorted(cells: Array) -> Array:
	var out := cells.duplicate()
	out.sort_custom(func(a: Vector2i, b: Vector2i) -> bool:
		if a.y != b.y:
			return a.y < b.y
		return a.x < b.x)
	return out
