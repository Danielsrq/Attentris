class_name TowerData
extends RefCounted

## Tower specs as ASCII masks. This is the only place tower shapes and colours are
## declared; TowerSet compiles them into TowerDefs. The same split PieceData/PieceSet
## makes, for the same reason: adding a tower is a data change.
##
## `X` is a body cell and `C` is the cannon -- the cell a shot's ray starts from. `C` is
## ALSO a body cell: the mask is solid wherever a letter sits. That matters because a
## non-solid cannon would be a hole in the footprint, leaving both the muzzle exit and a
## shot's bounce against the tower ambiguous. It costs nothing to get for free, because
## TowerSet's cell test is a blacklist -- anything that is not `.` or a space fills.
##
## Exactly one `C` per mask, enforced at compile time.
##
## Masks need not be square and need not be padded. TowerSet derives `box` as
## max(width, height) and normalises every rotation to its own tight bounding box, so what
## you draw is what gets placed.

## The first pass. `basic` is the 2x2 the design calls for; `angled` is the five-cell
## mask from the same discussion, kept because it is the case that proves rotation --
## a square mask cannot show a footprint whose extent swaps between 3x2 and 2x3.
static func towers() -> Dictionary:
	return {
		&"basic": {
			"rows": ["XC", "XX"],
			"color": Color8(120, 170, 240),
		},
		&"angled": {
			"rows": ["XXC", "XX."],
			"color": Color8(240, 170, 110),
		},
	}
