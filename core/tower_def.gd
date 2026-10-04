class_name TowerDef
extends RefCounted

## An immutable, compiled tower template. One instance per tower kind, shared by every
## placed tower of that kind. Built by TowerSet from an ASCII mask, exactly as PieceDef is
## built from a piece spec -- and like PieceDef it is RefCounted, not a Resource: tunable
## data goes in DefenceRules, compiled products go here.
##
## `rotations` holds all four footprint states precomputed, and `cannons` holds the cannon
## cell for each, parallel-indexed. Keeping them parallel is the point: the cannon is
## rotated by the same call as the footprint and then shifted by the same normalising
## delta, so it cannot drift out of the shape it belongs to. Deriving it by remembering an
## index into the cell array would NOT survive, because sorting reorders the cells.
##
## Each rotation is normalised to its own tight bounding box, which is a deliberate
## divergence from PieceDef. PieceDef keeps raw box coordinates because `box` is the SRS
## pivot domain and kicks depend on it; towers have no kicks, so there is nothing to
## preserve, and normalising makes the placement origin mean "top-left of the footprint" --
## so the ghost lands exactly under the cursor and a 3x2 mask does not drift inside a 3x3
## box as it turns.

var id: StringName = &""
## Rotation pivot domain: max(width, height) of the authored mask. Derived by TowerSet
## rather than authored, because unlike SRS there is no reason a human should be able to
## get it wrong.
var box: int = 2
var color: Color = Color.WHITE

## Index into TowerSet.defs.
var index: int = 0

## Array[Array[Vector2i]], one entry per rotation state (0, R, 2, L).
var rotations: Array = []
## Array[Vector2i], the cannon cell in the same coordinates as `rotations[rot]`.
var cannons: Array = []
## Array[Vector2i], the footprint's span in cells per rotation. A 3x2 mask spans (3,2) at
## rotation 0 and (2,3) at rotation 1, which is what the placement bounds check needs.
var extents: Array = []

## Array of {ok, offset} indexed [rot * 16 + dir], from Muzzle.table. Precomputed so
## firing is a lookup, in the same spirit as caching all four rotations at load time.
## `offset` is in CELL units from the footprint's top-left.
var muzzles: Array = []

## Where a shot leaves the hull for this rotation and aim, in cell units. `ok` is false
## only for a mask that is not solid -- see Muzzle for why "walled in" cannot happen.
func muzzle(rot: int, dir: int) -> Dictionary:
	return muzzles[(rot & 3) * Aim.COUNT + (dir & 15)]

## Typed: every entry of `rotations` is built as Array[Vector2i] by TowerSet._sorted.
func cells(rot: int) -> Array[Vector2i]:
	return rotations[rot & 3]

func cannon(rot: int) -> Vector2i:
	return cannons[rot & 3]

func extent(rot: int) -> Vector2i:
	return extents[rot & 3]

func cell_count() -> int:
	return rotations[0].size()

## True when `cell` is part of the footprint at this rotation.
func covers(rot: int, cell: Vector2i) -> bool:
	return cell in rotations[rot & 3]

func _to_string() -> String:
	return "TowerDef(%s box=%d cells=%d)" % [id, box, cell_count()]
