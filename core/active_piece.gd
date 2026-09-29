class_name ActivePiece
extends RefCounted

## The mutable half of a piece: which template, where it is, and the bookkeeping
## the lock and spin-detection paths need.

var def: PieceDef
var pos: Vector2i = Vector2i.ZERO    ## board coord of the bounding box's top-left
var rot: int = 0                     ## 0 = spawn, 1 = R, 2 = 180, 3 = L

var lock_timer: float = 0.0          ## seconds grounded
var move_resets: int = 0             ## lock-delay resets used at the current depth
var lowest_row: int = -0x7FFFFFFF    ## deepest row reached; resets the reset counter

## Spin detection needs to know that the last successful action was a rotation, and
## which kick offset got used (offset index 4 is the 1x2 tuck that upgrades a
## T-spin Mini to a full T-spin).
var last_action_was_rotation: bool = false
var last_kick_index: int = -1

func cells() -> Array:
	return def.rotations[rot]

func cells_for(r: int) -> Array:
	return def.rotations[r]

## Absolute board coords of the occupied cells.
func board_cells() -> Array:
	var out: Array = []
	for c in def.rotations[rot]:
		out.append(pos + c)
	return out
