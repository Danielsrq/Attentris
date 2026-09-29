class_name KickTables
extends RefCounted

## SRS wall-kick data.
##
## IMPORTANT: the tables below are written in the conventional Tetris-wiki
## orientation, where +y points UP. Godot's board space has +y pointing DOWN.
## `_flip_y()` is the ONE AND ONLY place that conversion happens -- if kicks ever
## look mirrored vertically, the bug is here and nowhere else.
##
## Lookup is `kick_class -> from_rot*4 + to_rot -> Array[Vector2i]`, tried in order;
## the first offset that places the rotated piece without collision wins, and if all
## of them fail the rotation is rejected outright.
##
## Rotation states: 0 = spawn, 1 = R (cw), 2 = 180, 3 = L (ccw).

const ROT_0 := 0
const ROT_R := 1
const ROT_2 := 2
const ROT_L := 3

static var _tables: Dictionary = {}

## Kick offsets for a rotation, already in Godot's y-down board space.
static func get_kicks(kick_class: StringName, from_rot: int, to_rot: int) -> Array:
	if _tables.is_empty():
		_build()
	var table: Dictionary = _tables.get(kick_class, _tables[&"o"])
	return table.get(_key(from_rot, to_rot), [Vector2i.ZERO])

static func has_class(kick_class: StringName) -> bool:
	if _tables.is_empty():
		_build()
	return _tables.has(kick_class)

static func _key(from_rot: int, to_rot: int) -> int:
	return from_rot * 4 + to_rot

static func _build() -> void:
	# --- J, L, S, T, Z: the standard SRS table (y-up) ---
	var jlstz := {
		_key(0, 1): [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
		_key(1, 0): [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
		_key(1, 2): [[0, 0], [1, 0], [1, -1], [0, 2], [1, 2]],
		_key(2, 1): [[0, 0], [-1, 0], [-1, 1], [0, -2], [-1, -2]],
		_key(2, 3): [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
		_key(3, 2): [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
		_key(3, 0): [[0, 0], [-1, 0], [-1, -1], [0, 2], [-1, 2]],
		_key(0, 3): [[0, 0], [1, 0], [1, 1], [0, -2], [1, -2]],
	}
	# --- I: its own table. This is what makes I-spins possible. (y-up) ---
	var i_piece := {
		_key(0, 1): [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
		_key(1, 0): [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
		_key(1, 2): [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
		_key(2, 1): [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
		_key(2, 3): [[0, 0], [2, 0], [-1, 0], [2, 1], [-1, -2]],
		_key(3, 2): [[0, 0], [-2, 0], [1, 0], [-2, -1], [1, 2]],
		_key(3, 0): [[0, 0], [1, 0], [-2, 0], [1, -2], [-2, 1]],
		_key(0, 3): [[0, 0], [-1, 0], [2, 0], [-1, 2], [2, -1]],
	}
	# --- 180 degree kicks ---
	# NOT part of the Tetris Guideline; this is the widely used SRS+ table. Tune
	# freely -- nothing else depends on these values.
	var jlstz_180 := {
		_key(0, 2): [[0, 0], [0, 1], [1, 1], [-1, 1], [1, 0], [-1, 0]],
		_key(2, 0): [[0, 0], [0, -1], [-1, -1], [1, -1], [-1, 0], [1, 0]],
		_key(1, 3): [[0, 0], [1, 0], [1, 2], [1, 1], [0, 2], [0, 1]],
		_key(3, 1): [[0, 0], [-1, 0], [-1, 2], [-1, 1], [0, 2], [0, 1]],
	}
	var i_180 := {
		_key(0, 2): [[0, 0], [0, 1], [0, -1]],
		_key(2, 0): [[0, 0], [0, -1], [0, 1]],
		_key(1, 3): [[0, 0], [1, 0], [-1, 0]],
		_key(3, 1): [[0, 0], [-1, 0], [1, 0]],
	}
	for k in jlstz_180:
		jlstz[k] = jlstz_180[k]
	for k in i_180:
		i_piece[k] = i_180[k]

	# --- O: never kicks. With a 2x2 box the rotation is a no-op anyway. ---
	var o_piece := {}
	for a in 4:
		for b in 4:
			if a != b:
				o_piece[_key(a, b)] = [[0, 0]]

	_tables = {
		&"jlstz": _flip_y(jlstz),
		&"i": _flip_y(i_piece),
		&"o": _flip_y(o_piece),
	}

## The single y-up -> y-down conversion for the whole project.
static func _flip_y(src: Dictionary) -> Dictionary:
	var out := {}
	for key in src:
		var offsets: Array = []
		for pair in src[key]:
			offsets.append(Vector2i(pair[0], -pair[1]))
		out[key] = offsets
	return out
