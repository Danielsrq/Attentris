class_name InputState
extends RefCounted

## One tick of player intent. Game never touches Input directly -- it only ever sees
## this struct, which is what keeps replays, undo and headless tests possible.
##
## `left`, `right` and `soft_drop` are held states; the rest are just-pressed edges.

var left: bool = false
var right: bool = false
var soft_drop: bool = false
var hard_drop: bool = false
var rotate_cw: bool = false
var rotate_ccw: bool = false
var rotate_180: bool = false
var hold: bool = false

func clear() -> void:
	left = false
	right = false
	soft_drop = false
	hard_drop = false
	rotate_cw = false
	rotate_ccw = false
	rotate_180 = false
	hold = false
