class_name Randomizer
extends RefCounted

## 7-bag (more precisely: N-bag, where N is the size of the piece set).
##
## Seeded from its own RandomNumberGenerator so a seed reproduces an entire run --
## which is what makes deterministic replays and the planned undo powerup viable.
## Swap this class out for a TGM-style history randomizer or pure random without
## touching anything else.

var piece_set: PieceSet
var rng: RandomNumberGenerator
var exclude_first: Array = []   ## Array[String] of piece ids barred from the opening draw

var _bag: Array = []
var _drawn: int = 0

func _init(ps: PieceSet, seed_value: int = 0, exclude: Array = []) -> void:
	piece_set = ps
	exclude_first = exclude
	rng = RandomNumberGenerator.new()
	rng.seed = seed_value

func next() -> PieceDef:
	if _bag.is_empty():
		_refill()
	_drawn += 1
	return _bag.pop_front()

func total_drawn() -> int:
	return _drawn

func _refill() -> void:
	_bag = piece_set.defs.duplicate()
	# Fisher-Yates using our own RNG, not Array.shuffle(), so the sequence depends
	# only on the seed and not on the global RNG state.
	for i in range(_bag.size() - 1, 0, -1):
		var j := rng.randi_range(0, i)
		var t = _bag[i]
		_bag[i] = _bag[j]
		_bag[j] = t
	if _drawn == 0 and not exclude_first.is_empty():
		_fix_opening_piece()

## Guideline: the opening piece is never S, Z or O. Swap it with the first eligible
## piece further back in the bag rather than re-rolling, so the bag stays a
## permutation and the distribution guarantee holds.
func _fix_opening_piece() -> void:
	if not exclude_first.has(String(_bag[0].id)):
		return
	for k in range(1, _bag.size()):
		if not exclude_first.has(String(_bag[k].id)):
			var t = _bag[0]
			_bag[0] = _bag[k]
			_bag[k] = t
			return
