class_name GarbageBuffer
extends RefCounted

## The window between an enemy reaching the left wall and garbage landing in the Tetris
## board.
##
## The first breach opens a window; every breach while it is open adds a row; when it
## closes the whole buffer is sent at once. That shape is also the telegraph -- the player
## watches bodies pile against the wall and knows what is coming and roughly when -- which
## is why no separate warning UI is needed.
##
## Pure logic and no nodes, so the whole garbage path is testable without a scene. It owns
## its own seeded RandomNumberGenerator: the defence side does not need determinism in
## general, but the hole column walk is the one part worth being able to assert exactly.
##
## Rows beyond `flush_cap` are CARRIED, not dropped, so a disastrous wave stays survivable
## while still hurting. With `drain_carry` the window re-opens to send them -- without it,
## three rows carried from a bad wave would sit forever if nothing ever breached again.

var rules: DefenceRules
var rng := RandomNumberGenerator.new()

## Rows waiting to be sent, including anything carried from a capped flush.
var pending: int = 0
## Seconds left before the window closes. 0 means closed.
var window_left: float = 0.0
## The current hole column. Walks by at most one per flush.
var hole: int = 0
## Flushes sent, for the readout.
var flushes: int = 0

func _init(r: DefenceRules, seed_v: int = 0, width: int = 10) -> void:
	rules = r
	rng.seed = seed_v
	hole = rng.randi_range(0, maxi(0, width - 1))

func is_open() -> bool:
	return window_left > 0.0

## An enemy reached the wall. The first one opens the window; the rest just add to it.
func touch() -> void:
	pending += 1
	if window_left <= 0.0:
		window_left = rules.window_sec

## Advances the window. Returns the rows to send THIS tick, 0 for nothing.
##
## Returning a count rather than pushing to a Board keeps this unit free of the Tetris
## side; the caller pairs it with `holes_for` and hands the result to
## Game.receive_garbage, which is where a top-out gets decided.
func tick(delta: float) -> int:
	if window_left <= 0.0:
		return 0
	window_left -= delta
	if window_left > 0.0:
		return 0
	window_left = 0.0
	return _flush()

func _flush() -> int:
	var send := mini(pending, rules.flush_cap)
	pending -= send
	if send <= 0:
		return 0
	flushes += 1
	_walk_hole()
	# Carried rows re-open the window so they drain on their own rather than waiting for
	# an enemy that may never come.
	if pending > 0 and rules.drain_carry:
		window_left = rules.window_sec
	return send

## Steps the hole column by at most one, clamped to the field.
##
## 0.1 is the `messiness_change` measured in real Zenith replays, so the drift is not
## invented. Clamping rather than wrapping makes columns 0 and 9 slightly sticky, which is
## acceptable and worth stating -- it also happens to match the edge suppression seen in
## that same data.
func _walk_hole(width: int = 10) -> void:
	if rng.randf() >= rules.hole_shift_chance:
		return
	hole = clampi(hole + (1 if rng.randf() < 0.5 else -1), 0, width - 1)

## The array Board.push_garbage wants: one hole column per row.
##
## CLEAN repeats one column, which is effectively a free well. CHEESE walks it per row,
## which is the case that actually tests downstacking. Note push_garbage reads holes[0] as
## the BOTTOM-most new row -- irrelevant while every entry is equal, but it has to stay
## right for the day cheese is switched on.
func holes_for(rows: int, width: int = 10) -> Array:
	var out: Array = []
	var h := hole
	for _i in rows:
		out.append(h)
		if rules.style == DefenceRules.GarbageStyle.CHEESE:
			h = clampi(h + rng.randi_range(-1, 1), 0, width - 1)
	return out

## Rows that would be sent if the window closed right now, for the warning readout.
func incoming() -> int:
	return mini(pending, rules.flush_cap)
