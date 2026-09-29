class_name TetrioRandomizer
extends Randomizer

## TETR.IO's bag generator, so a replay's seed reproduces its piece sequence.
##
## Ours and theirs are both "7-bag", but that only names the *shape* of the
## guarantee. The actual sequence depends on three things that have to match
## exactly, and getting any one wrong desyncs from the first piece:
##
##   1. the PRNG            -- Lehmer / MINSTD, t = 16807 * t mod 2147483647
##   2. the pre-shuffle bag order -- ZLOSIJT, not IJLOSTZ
##   3. the shuffle         -- Fisher-Yates walking DOWN from the last index
##
## Our own Randomizer uses Godot's PCG32 and its own ordering, which is fine for
## play and useless for reconstruction.
##
## Algorithm per the community bot documentation; verified here by replaying a real
## TETR.IO 40L replay and checking the outcome against its recorded stats.

## The "rainbow" order TETR.IO fills the bag in before shuffling.
const BAG_ORDER := ["Z", "L", "O", "S", "I", "J", "T"]

const MODULUS := 2147483647      ## 2^31 - 1
const MULTIPLIER := 16807

var _t: int = 1

func _init(ps: PieceSet, seed_value: int = 0, exclude: Array = []) -> void:
	super(ps, seed_value, exclude)
	_t = seed_value % MODULUS
	if _t <= 0:
		_t += MODULUS - 1

func _next() -> int:
	# 16807 * t peaks around 3.6e13, comfortably inside a 64-bit int.
	_t = MULTIPLIER * _t % MODULUS
	return _t

func _next_float() -> float:
	return float(_next() - 1) / float(MODULUS - 1)

func _refill() -> void:
	var arr: Array = []
	for id in BAG_ORDER:
		var d := piece_set.get_def(StringName(id))
		if d != null:
			arr.append(d)
	# Fisher-Yates from the last index down to 1, exactly as TETR.IO does it.
	# Walking the other direction produces a different permutation from the same
	# seed, which is the sort of detail that desyncs silently.
	for i in range(arr.size() - 1, 0, -1):
		var r := int(floor(_next_float() * float(i + 1)))
		var tmp = arr[i]
		arr[i] = arr[r]
		arr[r] = tmp
	_bag = arr
