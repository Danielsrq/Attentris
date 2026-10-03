class_name Game
extends RefCounted

## The simulation. Pure logic: no Node, no Input, no rendering.
##
## `tick(InputState) -> Array[GameEvent]` is called once per fixed 60 Hz step, which
## is what makes DAS and ARR frame counts exact and a seed reproduce a whole run.

const TICKS_PER_SECOND := 60.0
const MS_PER_TICK := 1000.0 / TICKS_PER_SECOND

var rules: RuleSet
var piece_set: PieceSet
var board: Board
var scorer: Scorer
var randomizer: Randomizer

var active: ActivePiece = null
var next_queue: Array = []          ## Array[PieceDef]
var hold_def: PieceDef = null
var hold_used: bool = false

var score: int = 0
var lines: int = 0
var level: int = 1
var combo: int = -1
var b2b: int = -1
var pieces_placed: int = 0
## Cumulative garbage sent. The currency the right-hand minigame will spend, kept
## separate from `score` because they reward different play.
var attack_sent: int = 0
var attack_table: AttackTable = AttackTable.new()

## Coloured/shape resources, the other currency the minigame spends. Line clears pay
## the group of the SCORING piece; completed structures pay per contributing piece.
var resources := Resources.new()
## Rectangle sizes the structure scan looks for.
##
## 4x3 and its transpose 3x4 are the smallest area three whole tetrominoes can tile, and
## BOTH are needed to cover one structure's orientations. A rectangle has eight
## symmetries; four keep a 4x3 footprint and four turn it into a 3x4. scan() finds every
## arrangement that fills the size it is given, so the size list is what decides whether a
## 90-degree turn of a structure still counts -- with 4x3 alone, half of them would score
## nothing. The transpose costs nothing in balance: tools/tilings.py gives 3x4 the same 13
## compositions and the same payouts.
##
## Deliberately NOT here: 2x4. It has four tilings and OO is one of them, so two O pieces
## stacked would pay O2 for no skill and collapse the only resource structures exist to
## supply. Check a new size with tools/tilings.py before adding it.
var structure_sizes: Array = [Vector2i(4, 3), Vector2i(3, 4)]
var structures_enabled: bool = true
## Structure keys already paid for. Rebuilt from each scan rather than only added to:
## a structure cut by a line clear drops out and can never come back, because its
## pieces have permanently lost cells, so pruning both bounds this and costs nothing.
var _structure_claims: Dictionary = {}
var topped_out: bool = false
var top_out_reason: String = ""

## Debug affordance: with gravity off, soft drop still works, which makes it easy to
## walk a piece into a slot and verify kicks by hand.
var gravity_enabled: bool = true

var seed_value: int = 0
var last_spin: int = SpinDetect.Kind.NONE
var last_kick: int = -1
## Result of the most recent placement, clearing or not. Ask it for describe()
## and tags() rather than reassembling the outcome from counters.
var last_result: ClearResult = null
var last_points: int = 0
## Type of the previous line clear (ClearResult.clear_type()), for the all-spin repeat
## penalty. Placements that clear nothing leave it alone.
var last_clear_type: String = ""
## Hole of the most recent wound line, -1 before the first. RuleSet.wound_messiness is
## the chance the next one moves; its own RNG so the piece sequence is unaffected.
var last_wound_hole: int = -1
var _wound_rng := RandomNumberGenerator.new()
## Burst rules: the previous placement was a qualifying clear (any placement updates it).
var burst_armed: bool = false

## Hole column for a new wound line: the previous wound's column, or with probability
## `wound_messiness` (and always for the first) a different one.
func _next_wound_hole() -> int:
	if _wound_rng.seed == 0:
		_wound_rng.seed = seed_value * 2654435761 + 97
	var hole := last_wound_hole
	if hole < 0 or _wound_rng.randf() < rules.wound_messiness:
		var h := _wound_rng.randi_range(0, board.width - 2)
		if hole >= 0 and h >= hole:
			h += 1
		hole = h
	last_wound_hole = hole
	return hole

var _das_left: float = 0.0
var _arr_left: float = 0.0
var _dir: int = 0
var _grav_accum: float = 0.0
var _are_left: float = 0.0

func _init(rs: RuleSet, ps: PieceSet, b: Board, sc: Scorer, seed_v: int = 0) -> void:
	rules = rs
	piece_set = ps
	board = b
	scorer = sc
	seed_value = seed_v
	attack_table.configure(rules)
	randomizer = Randomizer.new(ps, seed_v, rules.first_piece_excludes)
	for _i in rules.next_count:
		next_queue.append(randomizer.next())

## Swap in a different piece generator and rebuild the preview queue. Must be called
## before start(), since _init has already drawn the first `next_count` pieces.
## Used to reconstruct replays with TETR.IO's own bag rather than ours.
func set_randomizer(r: Randomizer) -> void:
	randomizer = r
	next_queue.clear()
	for _i in rules.next_count:
		next_queue.append(randomizer.next())

## Spawns the first piece. Separate from _init so the caller gets the events.
func start() -> Array:
	var events: Array = []
	_spawn_from_queue(events)
	return events

func tick(input: InputState) -> Array:
	var events: Array = []
	if topped_out:
		return events

	if _are_left > 0.0:
		_are_left -= MS_PER_TICK
		if _are_left > 0.0:
			return events
		_spawn_from_queue(events)
		if topped_out:
			return events
	if active == null:
		return events

	_handle_hold(input, events)
	_handle_rotation(input, events)
	_handle_horizontal(input, events)
	if input.hard_drop:
		_hard_drop(events)
		return events
	_handle_gravity(input)
	_handle_lock(events)
	return events

# ------------------------------------------------------------------------------
# Horizontal movement: DAS / ARR / DCD
# ------------------------------------------------------------------------------

func _handle_horizontal(input: InputState, events: Array = []) -> void:
	var dir := 0
	if input.left and not input.right:
		dir = -1
	elif input.right and not input.left:
		dir = 1

	if dir == 0:
		_dir = 0
		_das_left = 0.0
		_arr_left = 0.0
		return

	# Cells moved this tick, reported as a single coalesced event at the end. An
	# ARR 0 wall slide is many cells in one tick and must not fire many events.
	var moved := 0

	if dir != _dir:
		# Fresh tap: move once, then start charging DAS.
		_dir = dir
		_das_left = rules.das_ms
		_arr_left = 0.0
		if _try_move(Vector2i(dir, 0)):
			moved += 1
		_emit_moved(events, dir, moved)
		return

	if _das_left > 0.0:
		_das_left -= MS_PER_TICK
		if _das_left > 0.0:
			return

	if rules.arr_ms <= 0.0:
		# ARR 0: slide all the way to the wall within this tick.
		while _try_move(Vector2i(dir, 0)):
			moved += 1
	else:
		_arr_left -= MS_PER_TICK
		if _arr_left <= 0.0:
			_arr_left += rules.arr_ms
			if _try_move(Vector2i(dir, 0)):
				moved += 1
	_emit_moved(events, dir, moved)

func _emit_moved(events: Array, dir: int, cells: int) -> void:
	if cells > 0:
		events.append(GameEvent.new(GameEvent.Type.PIECE_MOVED, {"dir": dir, "cells": cells}))

# ------------------------------------------------------------------------------
# Rotation
# ------------------------------------------------------------------------------

func _handle_rotation(input: InputState, events: Array = []) -> void:
	var to_rot := -1
	if input.rotate_cw:
		to_rot = (active.rot + 1) % 4
	elif input.rotate_ccw:
		to_rot = (active.rot + 3) % 4
	elif input.rotate_180 and rules.allow_180:
		to_rot = (active.rot + 2) % 4
	if to_rot < 0:
		return
	if try_rotate(to_rot, events) and rules.dcd_ms > 0.0 and _dir != 0:
		_das_left = maxf(_das_left, minf(rules.dcd_ms, rules.das_ms))

## Walks the kick table in order and takes the first offset that fits. Public so the
## test scene can drive individual rotations.
##
## On success the spin state is evaluated immediately and reported on the event,
## which is what a spin-confirm cue needs: the answer to "would this count as a spin
## if it cleared right now?" A failed rotation emits nothing.
func try_rotate(to_rot: int, events: Array = []) -> bool:
	if active == null:
		return false
	var kicks := KickTables.get_kicks(active.def.kick_class, active.rot, to_rot)
	var target: Array = active.def.rotations[to_rot]
	for i in kicks.size():
		var p: Vector2i = active.pos + kicks[i]
		# Rotation is the ONE path allowed into the phantom assist column. Movement and
		# gravity use the strict check, so a piece can turn into the pocket beside the
		# last column but can never simply be slid out of the field.
		if board.can_place_in_play(target, p):
			active.rot = to_rot
			active.pos = p
			active.last_kick_index = i
			active.last_action_was_rotation = true
			last_kick = i
			_reset_lock_on_action()
			events.append(GameEvent.new(GameEvent.Type.PIECE_ROTATED, {
				"piece": active.def.id,
				"to_rot": to_rot,
				"kick": i,
				"spin": SpinDetect.detect(board, active, rules),
			}))
			return true
	return false

# ------------------------------------------------------------------------------
# Hold
# ------------------------------------------------------------------------------

func _handle_hold(input: InputState, events: Array) -> void:
	if not input.hold:
		return
	if hold_used and not rules.infinite_hold:
		return
	var prev := hold_def
	hold_def = active.def
	hold_used = true
	var swapped: StringName = prev.id if prev != null else &""
	events.append(GameEvent.new(GameEvent.Type.HOLD_USED, {
		"held": hold_def.id,
		"swapped_in": swapped,
	}))
	if prev == null:
		_spawn_from_queue(events)
	else:
		_spawn_def(prev, events)

# ------------------------------------------------------------------------------
# Gravity, lock and hard drop
# ------------------------------------------------------------------------------

func _handle_gravity(input: InputState) -> void:
	var g := current_gravity()
	var step := 0.0
	if input.soft_drop:
		step = g * rules.sdf
	elif gravity_enabled:
		step = g
	if step <= 0.0:
		return
	_grav_accum += step
	while _grav_accum >= 1.0:
		_grav_accum -= 1.0
		if not _try_move(Vector2i(0, 1), false):
			_grav_accum = 0.0
			return
		if input.soft_drop:
			score += scorer.table.soft_drop_per_cell

func _handle_lock(events: Array) -> void:
	if board.can_place(active.cells(), active.pos + Vector2i(0, 1)):
		active.lock_timer = 0.0
		return
	active.lock_timer += 1.0 / TICKS_PER_SECOND
	if active.lock_timer >= rules.lock_delay_sec:
		lock_piece(events)

func _hard_drop(events: Array) -> void:
	var dropped := 0
	while board.can_place(active.cells(), active.pos + Vector2i(0, 1)):
		active.pos.y += 1
		dropped += 1
	if dropped > 0:
		score += dropped * scorer.table.hard_drop_per_cell
		# The piece moved, so whatever rotation preceded it no longer counts. A piece
		# already seated in a spin slot cannot fall, so real spins survive this.
		active.last_action_was_rotation = false
	lock_piece(events, "hard_drop")

func lock_piece(events: Array, cause: String = "lock_delay") -> void:
	var spin := SpinDetect.detect(board, active, rules)
	var kick := active.last_kick_index
	last_spin = spin
	var piece_id := active.def.id
	var locked_cells := active.board_cells()

	pieces_placed += 1
	# Placement number doubles as the owner id; 0 stays reserved for empty and garbage.
	board.lock_cells(active.cells(), active.pos, active.def.index + 1, pieces_placed)

	# Lock-out: the piece came to rest entirely inside the buffer above the field.
	var all_in_buffer := true
	for c in locked_cells:
		if c.y >= board.buffer_height:
			all_in_buffer = false
			break

	var full := board.find_full_rows()
	var r := ClearResult.new()
	r.lines = full.size()
	r.rows = full
	r.spin = spin
	r.piece_id = piece_id
	# Captured before the reset below, so a cue can fire on the chain ending.
	var combo_broken := combo >= 1 and r.lines == 0
	if r.lines > 0:
		board.clear_rows(full)
		r.perfect_clear = board.is_empty()
		if r.is_difficult():
			b2b += 1
		else:
			b2b = -1
		combo += 1
	else:
		combo = -1
	r.b2b = b2b
	if rules.burst_doubling:
		r.burst = burst_armed and r.qualifies_burst()
		burst_armed = r.qualifies_burst()
	r.combo = combo
	r.level = level
	# All-spin repeat penalty. Order: the lines are gone (and the perfect clear judged),
	# every wound counts down and turns into garbage at 0, then a repeat inserts a new
	# wound line at the bottom. The repeat still sends its full attack.
	var wound_events: Array = []
	if rules.repeat_penalty and r.lines > 0:
		r.healed = board.tick_wounds()
		if r.healed > 0:
			wound_events.append(GameEvent.new(GameEvent.Type.WOUND_HEALED, {
				"healed": r.healed, "count": board.wound_count()}))
		var t := r.clear_type()
		if t == last_clear_type:
			r.repeat = true
			r.wound_hole = _next_wound_hole()
			board.add_wound(rules.wound_clears, r.wound_hole)
			wound_events.append(GameEvent.new(GameEvent.Type.WOUND_ADDED, {
				"row": board.total_height - 1, "left": rules.wound_clears,
				"hole": r.wound_hole, "count": board.wound_count()}))
		last_clear_type = t

	# A spin that clears nothing still scores (guideline: a T-Spin is worth 400), so
	# this deliberately is not gated on r.lines.
	var points := 0
	if r.lines > 0 or r.spin != SpinDetect.Kind.NONE:
		points = scorer.score_clear(r)
		score += points
	r.attack = attack_table.attack_for(r)
	attack_sent += r.attack
	last_result = r
	last_points = points

	# Every placement carries the full result, so a consumer that cares about spins
	# with no line clear does not have to reassemble one.
	events.append(GameEvent.new(GameEvent.Type.PIECE_LOCKED, {
		"piece": piece_id, "spin": spin, "kick": kick,
		"cells": locked_cells, "result": r, "points": points, "attack": r.attack,
		"cause": cause, "combo_broken": combo_broken,
	}))
	if spin != SpinDetect.Kind.NONE:
		events.append(GameEvent.new(GameEvent.Type.SPIN_DETECTED, {
			"piece": piece_id, "spin": spin, "kick": kick,
		}))
	if r.lines > 0:
		lines += r.lines
		resources.add(r.resource_group(), r.resource_amount(), Resources.CLEARS)
		events.append(GameEvent.new(GameEvent.Type.LINES_CLEARED, {
			"result": r, "points": points, "attack": r.attack}))
		var new_level := 1 + lines / rules.lines_per_level
		if new_level != level:
			level = new_level
			events.append(GameEvent.new(GameEvent.Type.LEVEL_UP, {"level": level}))
	events.append_array(wound_events)

	# After the clear, so the scan sees the settled board: dropping rows changes which
	# pieces are adjacent, so a clear can complete a rectangle by itself.
	_scan_structures(events)

	active = null
	hold_used = false

	if all_in_buffer and r.lines == 0:
		_top_out("lock_out", events)
		return
	_are_left = rules.are_ms
	if _are_left <= 0.0:
		_spawn_from_queue(events)

## Pays for every structure standing on the board that has not been paid for yet, and
## forgets the ones that no longer stand.
func _scan_structures(events: Array) -> void:
	if not structures_enabled:
		return
	var seen := {}
	for size in structure_sizes:
		for f in Structures.scan(board, size.x, size.y):
			var k := Structures.key(f)
			# `seen` as well as the old claims: two scan sizes can both turn up the same
			# structure, and one pass must not pay for it twice. The two sizes in use
			# cannot overlap -- a set of pieces has one bounding box -- but that is a
			# property of the list, not of this loop.
			if _structure_claims.has(k) or seen.has(k):
				seen[k] = true
				continue
			seen[k] = true
			var pay := Structures.payout(f, piece_set)
			resources.add_table(pay, Resources.STRUCTURES)
			events.append(GameEvent.new(GameEvent.Type.STRUCTURE_BUILT, {
				"rect": f["rect"], "owners": f["owners"], "pieces": f["pieces"],
				"payout": pay,
			}))
	_structure_claims = seen

# ------------------------------------------------------------------------------
# Spawning
# ------------------------------------------------------------------------------

func _spawn_from_queue(events: Array) -> void:
	var d: PieceDef = next_queue.pop_front()
	next_queue.append(randomizer.next())
	_spawn_def(d, events)

func _spawn_def(d: PieceDef, events: Array) -> void:
	active = ActivePiece.new()
	active.def = d
	active.rot = 0
	active.pos = spawn_pos_for(d)
	active.lowest_row = active.pos.y
	_grav_accum = 0.0
	if not board.can_place(active.cells(), active.pos):
		_top_out("block_out", events)
		return
	events.append(GameEvent.new(GameEvent.Type.PIECE_SPAWNED, {"piece": d.id}))

## Horizontally centred for any box size (also correct for a 12-wide field), and
## vertically aligned so every piece's topmost filled row lands on the first visible
## row whether its box is 2, 3, 4 or 5 tall.
func spawn_pos_for(d: PieceDef) -> Vector2i:
	return Vector2i((board.width - d.box) / 2, board.buffer_height - d.first_filled_row)

func _top_out(reason: String, events: Array) -> void:
	topped_out = true
	top_out_reason = reason
	events.append(GameEvent.new(GameEvent.Type.TOP_OUT, {"reason": reason}))

# ------------------------------------------------------------------------------
# Movement helper
# ------------------------------------------------------------------------------

func _try_move(delta: Vector2i, player: bool = true) -> bool:
	if active == null:
		return false
	var np := active.pos + delta
	if not board.can_place(active.cells(), np):
		return false
	active.pos = np
	active.last_action_was_rotation = false
	if active.pos.y > active.lowest_row:
		# Reached new depth: the move-reset budget refills.
		active.lowest_row = active.pos.y
		active.move_resets = 0
		active.lock_timer = 0.0
	if player:
		_reset_lock_on_action()
	return true

## Extended placement lockdown: a move or rotation resets the lock timer, but only
## while grounded and only up to max_move_resets times at the current depth.
func _reset_lock_on_action() -> void:
	if active == null:
		return
	if board.can_place(active.cells(), active.pos + Vector2i(0, 1)):
		return
	if active.move_resets < rules.max_move_resets:
		active.move_resets += 1
		active.lock_timer = 0.0

# ------------------------------------------------------------------------------
# Queries for the view and debug overlay
# ------------------------------------------------------------------------------

func ghost_pos() -> Vector2i:
	if active == null:
		return Vector2i.ZERO
	var p := active.pos
	while board.can_place(active.cells(), p + Vector2i(0, 1)):
		p.y += 1
	return p

func current_gravity() -> float:
	return rules.gravity_cells_per_tick(level, TICKS_PER_SECOND)

func das_remaining() -> float:
	return _das_left

func arr_remaining() -> float:
	return _arr_left

func gravity_accum() -> float:
	return _grav_accum

func are_remaining() -> float:
	return _are_left
