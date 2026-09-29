class_name Resources
extends RefCounted

## The bank the right-hand minigame spends from. One running total per ShapeGroups
## group, plus the same split by where it came from.
##
## Two sources pay in, and they are deliberately different in character:
##
##   clears      fast, capped by what a piece can physically clear -- 4 for I, 3 for a
##               spin triple, 2 for O -- so the mix follows how the player chooses to
##               clear rather than the bag distribution
##   structures  slow, chosen, and the only way to earn much O, since O is the piece
##               the clear economy pays worst
##
## Keeping the split means the two can be balanced against each other later without
## re-deriving it from the event log. The ledger stores counts and nothing else: rates,
## multipliers and what a resource BUYS belong to the minigame.

const CLEARS := &"clears"
const STRUCTURES := &"structures"

var bank: Dictionary = {}       ## group -> total held
var by_source: Dictionary = {}  ## source -> { group -> total earned }

func _init() -> void:
	clear()

func clear() -> void:
	bank = {}
	by_source = {}
	for g in ShapeGroups.all():
		bank[g] = 0
	for s in [CLEARS, STRUCTURES]:
		var d := {}
		for g in ShapeGroups.all():
			d[g] = 0
		by_source[s] = d

func add(group: StringName, amount: int, source: StringName = CLEARS) -> void:
	if amount == 0:
		return
	bank[group] = int(bank.get(group, 0)) + amount
	var d: Dictionary = by_source.get(source, {})
	d[group] = int(d.get(group, 0)) + amount
	by_source[source] = d

## Credits a whole { group -> amount } table in one call, which is the shape both a
## structure payout and a multi-group bonus come in.
func add_table(table: Dictionary, source: StringName = CLEARS) -> void:
	for g in table:
		add(g, int(table[g]), source)

func amount(group: StringName) -> int:
	return int(bank.get(group, 0))

func earned(source: StringName, group: StringName) -> int:
	var d: Dictionary = by_source.get(source, {})
	return int(d.get(group, 0))

## True only if every group in the cost is covered. Costs are { group -> amount }.
func can_afford(cost: Dictionary) -> bool:
	for g in cost:
		if amount(g) < int(cost[g]):
			return false
	return true

## Spends a cost atomically: nothing is deducted unless the whole cost is covered.
func spend(cost: Dictionary) -> bool:
	if not can_afford(cost):
		return false
	for g in cost:
		bank[g] = amount(g) - int(cost[g])
	return true

func total() -> int:
	var n := 0
	for g in bank:
		n += int(bank[g])
	return n
