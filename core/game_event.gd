class_name GameEvent
extends RefCounted

## What Game.tick() returns. The view renders state; consumers (the HUD now, the
## right-hand minigame later) read these.

## Emitted once per lock, in this order:
##   PIECE_LOCKED -> [SPIN_DETECTED] -> [LINES_CLEARED] -> [LEVEL_UP]
##   -> [STRUCTURE_BUILT ...] -> PIECE_SPAWNED
## PIECE_LOCKED always fires and carries the complete ClearResult, so a consumer
## that wants to branch on the outcome (audio, the minigame economy) needs to
## handle that one event and nothing else.
enum Type {
	PIECE_SPAWNED,   ## data: piece
	PIECE_MOVED,     ## data: dir (-1/1), cells -- horizontal only, one per tick
	PIECE_ROTATED,   ## data: piece, to_rot, kick, spin -- spin is the state NOW
	PIECE_LOCKED,    ## data: piece, spin, kick, cells, result, points, attack, cause, combo_broken
	SPIN_DETECTED,   ## data: piece, spin, kick
	LINES_CLEARED,   ## data: result (ClearResult), points, attack -- only when lines > 0
	STRUCTURE_BUILT, ## data: rect, owners, pieces, payout -- one per newly completed structure
	LEVEL_UP,        ## data: level
	HOLD_USED,       ## data: held, swapped_in
	TOP_OUT,         ## data: reason
}

## PIECE_MOVED is coalesced: with ARR 0 a wall slide is many cells inside a single
## tick, and it fires once with the total in `cells` rather than once per cell.
##
## PIECE_ROTATED carries the spin state *at the moment of the rotation*, not at
## lock. That is what a spin-confirm cue needs -- it answers "if this cleared right
## now, would it count as a spin?" It follows the active RuleSet, so with all_spin
## off it only ever reports T-spins.

var type: int
var data: Dictionary

func _init(t: int, d: Dictionary = {}) -> void:
	type = t
	data = d

func type_name() -> String:
	return Type.keys()[type]

func _to_string() -> String:
	return "%s %s" % [type_name(), data]

## STRUCTURE_BUILT fires after the clear, not before: dropping rows changes which pieces
## are adjacent, so a clear can complete a rectangle by itself and the scan has to see
## the settled board. It fires once per structure and never again for the same one.
