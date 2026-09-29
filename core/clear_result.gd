class_name ClearResult
extends RefCounted

## Everything that happened when a piece locked, as structured facts.
##
## Game emits this in its event stream regardless of which Scorer is installed, so
## the right-hand minigame can define its own economy off these fields rather than
## off a single points number.

var lines: int = 0
var rows: Array = []                      ## board rows that were cleared
var spin: int = SpinDetect.Kind.NONE
var piece_id: StringName = &""
var b2b: int = -1                         ## -1 none, 0 first difficult clear, >=1 chain
var combo: int = -1                       ## -1 none, 0 first clear, >=1 combo active
var perfect_clear: bool = false
var level: int = 1
## Lines of garbage this clear sends. Filled in by Game from its AttackTable.
var attack: int = 0
## All-spin repeat penalty: this clear had the same type as the previous clear and
## inserted a wound line; `healed` wound lines turned into garbage on it.
var repeat: bool = false
var healed: int = 0
## Hole of the wound line this repeat inserted (it opens when the wound turns to garbage).
var wound_hole: int = -1

## Type for the repeat penalty: piece + lines + spin. Non-spin clears of one size share a
## type whatever the piece (double -> double repeats, single -> double does not); spins
## are per piece (TSD -> ZSD does not). Empty when nothing cleared.
func clear_type() -> String:
	if lines <= 0:
		return ""
	if spin == SpinDetect.Kind.NONE:
		return "plain:%d" % lines
	return "%s:%d:%d" % [piece_id, spin, lines]

## Which resource this clear pays into: the group of the piece that SCORED it, not
## of the cells destroyed.
##
## That distinction is the whole design. Counting destroyed cells converges on the bag
## distribution -- every piece placed is eventually cleared, so the mix is fixed at
## roughly 2/7 SZ, 2/7 LJ, 1/7 each of I, O, T no matter how anyone plays, and cannot
## be skewed. Crediting the scoring piece instead makes the mix a consequence of
## strategy, and caps each colour by what its piece can physically clear: 4 for I,
## 3 for a spin triple, 2 for O.
func resource_group() -> StringName:
	return ShapeGroups.of(piece_id)

## Resource earned, before any economy multiplier. One per line cleared, which is what
## puts the ceiling on each colour.
func resource_amount() -> int:
	return lines

const LINE_NAMES := {0: "Zero", 1: "Single", 2: "Double", 3: "Triple", 4: "Quad"}

## "T-Spin", "T-Spin Mini", "S-Spin", ... or "" when no spin was detected.
func spin_name() -> String:
	if spin == SpinDetect.Kind.NONE:
		return ""
	var out := "%s-Spin" % piece_id
	if spin == SpinDetect.Kind.MINI:
		out += " Mini"
	return out

## "Single", "Quad", "Zero", or "N Lines" past four.
func count_name() -> String:
	return LINE_NAMES.get(lines, "%d Lines" % lines)

## The full name for this placement: "T-Spin Double", "Quad", "L-Spin Zero".
## Empty for an ordinary placement that neither spun nor cleared.
## Displays with a narrow column can draw spin_name() and count_name() on separate
## lines instead.
func describe() -> String:
	var sn := spin_name()
	if sn != "":
		return "%s %s" % [sn, count_name()]
	if lines > 0:
		return count_name()
	return ""

## True when this placement is worth announcing at all.
func is_named() -> bool:
	return lines > 0 or spin != SpinDetect.Kind.NONE

## Modifiers, kept apart from the name so the display can style them separately.
func tags() -> Array:
	var out: Array = []
	if b2b_active():
		out.append("B2B x%d" % b2b)
	if combo >= 1:
		out.append("Combo %d" % combo)
	if perfect_clear:
		out.append("Perfect Clear")
	if repeat:
		out.append("Repeat - wounded")
	return out

func b2b_active() -> bool:
	return b2b >= 1

## "Difficult" clears are the ones that keep a back-to-back chain alive: tetrises
## and any line clear that came from a spin.
func is_difficult() -> bool:
	return lines == 4 or (lines > 0 and spin != SpinDetect.Kind.NONE)

func _to_string() -> String:
	return "ClearResult(lines=%d spin=%s piece=%s b2b=%d combo=%d pc=%s)" % [
		lines, SpinDetect.kind_name(spin), piece_id, b2b, combo, perfect_clear]
