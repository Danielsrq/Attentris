class_name HitLedger
extends RefCounted

## One per VICTIM, keyed by attacker: who has hit me recently, and for how much longer they
## cannot hit me again.
##
## Living on the victim is what makes "per (attacker, target) pair" structural rather than a
## convention. There is no global table to key wrongly, and no way to accidentally implement
## per-victim immunity instead.
##
## That distinction is load-bearing, not fussiness. Per-victim immunity would cap incoming
## damage at `damage / window` no matter how many attackers -- and because allies do not
## block each other here, several units reach the same target from the first playtest. A
## stack of five would kill no faster than one, so clearing more lines would stop helping,
## which would break the only coupling this minigame has to Tetris.
##
## Extracted from the unit so it can be tested with no physics and no scene tree: it is the
## only part of combat with real invariants, and they are pure dictionary arithmetic.

var window: float = 0.3
## attacker instance id -> seconds remaining
var _cooldown: Dictionary = {}

func _init(window_sec: float = 0.3) -> void:
	window = window_sec

func can_hit(attacker_id: int) -> bool:
	return float(_cooldown.get(attacker_id, 0.0)) <= 0.0

func register(attacker_id: int) -> void:
	_cooldown[attacker_id] = window

## Advances every cooldown and drops the expired ones, so the dictionary cannot grow without
## bound over a long fight.
func tick(delta: float) -> void:
	if _cooldown.is_empty():
		return
	for id in _cooldown.keys():
		var left := float(_cooldown[id]) - delta
		if left <= 0.0:
			_cooldown.erase(id)
		else:
			_cooldown[id] = left

## Forgets an attacker. Called when one dies, because Godot recycles instance ids after a
## free -- without this, a dead attacker's lingering entry could refuse a brand new unit's
## first hit. Bounded rather than a leak, since entries expire anyway, but cheap to be right.
func forget(attacker_id: int) -> void:
	_cooldown.erase(attacker_id)

func tracked() -> int:
	return _cooldown.size()
