class_name PieceDef
extends RefCounted

## An immutable, compiled piece template. One instance per piece kind, shared by
## every ActivePiece of that kind. Built by PieceSet from an ASCII spec.
##
## `rotations` holds all four states precomputed at load time, each as an Array of
## Vector2i offsets inside the piece's NxN bounding box. Collision therefore walks
## only the occupied cells (4 for a tetromino, 5 for a pentomino) and never the box.

var id: StringName = &""
var box: int = 3
var color: Color = Color.WHITE
var kick_class: StringName = &"jlstz"

## Index into PieceSet.defs. Board cells store index + 1 (0 means empty).
var index: int = 0

## Array[Array[Vector2i]], one entry per rotation state (0, R, 2, L).
var rotations: Array = []

## Topmost occupied row within the spawn-state box. Used so every piece spawns with
## its first filled row on the same board row regardless of box size (I sits on row
## 1 of a 4x4 box, T on row 0 of a 3x3 box).
var first_filled_row: int = 0

func cells(rot: int) -> Array:
	return rotations[rot]

func cell_count() -> int:
	return rotations[0].size()

func _to_string() -> String:
	return "PieceDef(%s box=%d kick=%s)" % [id, box, kick_class]
