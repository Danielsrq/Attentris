class_name Placement
extends RefCounted

## One candidate final resting position for a piece, plus how the player gets there.

var def: PieceDef
var rot: int = 0
var pos: Vector2i = Vector2i.ZERO
## True when this placement requires pressing hold first.
var uses_hold: bool = false
## Lines this placement would clear on the board it was enumerated against.
var lines: int = 0
## Absolute board cells the piece would occupy.
var cells: Array = []
## The final move was a rotation. Spin detection only applies then, so a placement
## that omits this is silently un-spinnable no matter what shape it lands in.
var via_rotation: bool = false
## Kick offset used, for the mini-to-full T-spin upgrade rule.
var kick_index: int = -1
## Spin the placement would register, resolved when it was produced.
var spin: int = SpinDetect.Kind.NONE

func _to_string() -> String:
	return "%s r%d @%s%s%s" % [def.id, rot, pos,
		"  hold" if uses_hold else "",
		"  x%d" % lines if lines > 0 else ""]
