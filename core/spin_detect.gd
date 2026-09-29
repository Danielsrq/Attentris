class_name SpinDetect
extends RefCounted

## Two independent spin rules, both implemented, selected by RuleSet flags.
##
## Guideline 3-corner rule (T only): at least three of the four cells diagonally
## adjacent to the T's centre are occupied, walls and floor counting as occupied.
## Full if both *front* corners (the side the nub points toward) are filled,
## otherwise Mini -- upgraded to full when the 1x2 tuck kick (offset index 4) was
## the one that succeeded.
##
## All-spin (the TETR.IO/Jstris rule): after the rotation the piece cannot move
## left, right, up or down. Governs S, Z, L, J and I.
##
## Either way, nothing counts unless the last successful action was a rotation.

enum Kind { NONE, MINI, FULL }

static func kind_name(k: int) -> String:
	match k:
		Kind.MINI: return "MINI"
		Kind.FULL: return "FULL"
		_: return "NONE"

static func detect(board: Board, p: ActivePiece, rules: RuleSet) -> int:
	if p == null or not p.last_action_was_rotation:
		return Kind.NONE
	if p.def.id == &"T" and rules.t_spin:
		var k := t_spin(board, p, rules)
		if k != Kind.NONE:
			return k
	if rules.all_spin and is_immobile(board, p):
		if rules.non_t_spin_is_mini and p.def.id != &"T":
			return Kind.MINI
		return Kind.FULL
	return Kind.NONE

static func t_spin(board: Board, p: ActivePiece, rules: RuleSet) -> int:
	# Centre cell of the T's 3x3 box.
	var centre := p.pos + Vector2i(1, 1)
	var filled := 0
	for c in [Vector2i(-1, -1), Vector2i(1, -1), Vector2i(-1, 1), Vector2i(1, 1)]:
		if board.is_blocked(centre.x + c.x, centre.y + c.y):
			filled += 1
	if filled < 3:
		return Kind.NONE
	var front_filled := 0
	for c in front_corners(p.rot):
		if board.is_blocked(centre.x + c.x, centre.y + c.y):
			front_filled += 1
	if front_filled == 2:
		return Kind.FULL
	if rules.mini_upgrade_on_kick5 and p.last_kick_index == 4:
		return Kind.FULL
	return Kind.MINI

## The two corners on the side the T's nub points toward, relative to the centre
## cell, in y-down board space.
static func front_corners(rot: int) -> Array:
	match rot:
		0: return [Vector2i(-1, -1), Vector2i(1, -1)]   # nub up
		1: return [Vector2i(1, -1), Vector2i(1, 1)]     # nub right
		2: return [Vector2i(-1, 1), Vector2i(1, 1)]     # nub down
		_: return [Vector2i(-1, -1), Vector2i(-1, 1)]   # nub left

static func is_immobile(board: Board, p: ActivePiece) -> bool:
	var cells := p.cells()
	for d in [Vector2i(-1, 0), Vector2i(1, 0), Vector2i(0, -1), Vector2i(0, 1)]:
		if board.can_place(cells, p.pos + d):
			return false
	return true
