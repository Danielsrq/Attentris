class_name ShapeGroups
extends RefCounted

## Which resource a cleared cell pays into.
##
## Mirror pairs share a group because they are the same shape reflected: S/Z are one
## skew, L/J one corner. I, O and T are their own. That leaves five piece groups plus
## garbage, which is counted separately so the economy can treat dug-out garbage as its
## own thing -- spoil, scrap, whatever it becomes -- rather than as free resource.
##
## Deliberately a mapping and nothing else. Rates, multipliers and what the resources
## MEAN belong to the minigame; Game only reports what was destroyed.

const SZ := &"SZ"
const LJ := &"LJ"
const I := &"I"
const O := &"O"
const T := &"T"
const GARBAGE := &"GARBAGE"

const OF_PIECE := {
	&"S": SZ, &"Z": SZ,
	&"L": LJ, &"J": LJ,
	&"I": I, &"O": O, &"T": T,
}

## A representative piece per group, for anything that needs a colour or an icon. S and
## L are arbitrary picks from their mirror pairs -- the group is the shape, so either
## member stands for it. GARBAGE has no piece and comes back empty.
const ICON := {SZ: &"Z", LJ: &"L", I: &"I", O: &"O", T: &"T"}

static func icon_of(group: StringName) -> StringName:
	return ICON.get(group, &"")

## Every group, in a fixed order, so callers can build tables without guessing.
static func all() -> Array:
	return [SZ, LJ, I, O, T, GARBAGE]

static func of(piece_id: StringName) -> StringName:
	return OF_PIECE.get(piece_id, GARBAGE)
