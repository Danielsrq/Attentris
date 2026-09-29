class_name BeamSolver
extends RefCounted

## Beam search over the visible queue. Returns a plan: an ordered list of placements
## for the current piece and as many lookahead pieces as it searched.
##
## The search is over YOUR CHOICES ONLY. 7-bag means the next pieces are known
## exactly, so there is no branching on randomness inside the queue -- which is what
## makes 4-5 plies affordable.
##
## Hold doubles the branching factor at each ply (place the queue piece, or swap),
## but the beam caps total nodes regardless, so it costs 2x per node rather than
## 2^depth overall.

var evaluator: BoardEvaluator
## Nodes kept between plies. Higher is better and slower, roughly linearly.
var beam_width: int = 24
## Plies to search, capped by the queue length.
var depth: int = 4
var use_hold: bool = true
## Add the T placement for a detected spin slot to the candidate list. Off for the
## quad solver, which cannot use it.
var inject_spins: bool = false

## Filled in by solve() so a benchmark can report cost without timing externally.
var last_nodes_expanded: int = 0
var last_placements_considered: int = 0

var _queue: Array = []

func _init(ev: BoardEvaluator = null) -> void:
	evaluator = ev if ev != null else BoardEvaluator.new()

## `queue` is [current, next1, next2, ...]. `hold` may be null.
## Returns an Array[Placement]; empty if nothing is reachable (topped out).
func solve(board: Board, queue: Array, hold: PieceDef, spawn_for: Callable, b2b: int = -1) -> Array:
	last_nodes_expanded = 0
	last_placements_considered = 0
	if queue.is_empty():
		return []

	_queue = queue
	var heights := evaluator.column_heights(board)
	var well: int = evaluator.weights.well_column
	if well < 0:
		well = BoardEvaluator.pick_well(board, heights)
	# A fixed well column must survive a wider board; 9 on a 12-wide field would
	# reserve the middle and wreck the stack.
	well = clampi(well, 0, board.width - 1)

	var root := {
		"board": board,
		"hold": hold,
		"qi": 0,
		"score": 0.0,
		"events": 0.0,
		"b2b": b2b,
		"path": [],
	}
	var beam: Array = [root]
	var plies: int = mini(depth, queue.size())

	for _ply in plies:
		var children: Array = []
		for node in beam:
			_expand(node, queue, well, children)
		if children.is_empty():
			break
		children.sort_custom(func(a, b): return a["score"] > b["score"])
		beam = children.slice(0, mini(beam_width, children.size()))

	if beam.is_empty() or beam[0]["path"].is_empty():
		return []
	return beam[0]["path"]

func _expand(node: Dictionary, queue: Array, well: int, out: Array) -> void:
	var qi: int = node["qi"]
	if qi >= queue.size():
		return
	last_nodes_expanded += 1
	var board: Board = node["board"]

	# Option A: place the piece at the front of the queue.
	var current: PieceDef = queue[qi]
	_try_piece(node, current, false, node["hold"], qi + 1, well, out)

	if not use_hold:
		return
	# Option B: swap. With an empty hold the queue also advances, because holding
	# stashes the current piece and pulls the next one.
	var held: PieceDef = node["hold"]
	if held != null:
		_try_piece(node, held, true, current, qi + 1, well, out)
	elif qi + 1 < queue.size():
		_try_piece(node, queue[qi + 1], true, current, qi + 2, well, out)

## Pieces of a given kind still reachable from this node: whatever remains in the
## visible queue, plus whatever is sitting in hold.
## Which pieces are still reachable from this node, and how far off.
## Distance 0 = in hold (summonable at any time), 1 = the piece now in play, 2 the
## one after it, and so on. Lists come out ascending.
static func _count(avail: Dictionary, id: StringName) -> int:
	var list: Array = avail.get(id, [])
	return list.size()

static func _availability(queue: Array, from_qi: int, hold: PieceDef) -> Dictionary:
	var out := {}
	if hold != null:
		out[hold.id] = [0]
	for i in range(from_qi, queue.size()):
		var id: StringName = queue[i].id
		if not out.has(id):
			out[id] = []
		out[id].append(i - from_qi + 1)
	return out

func _try_piece(node: Dictionary, def: PieceDef, uses_hold: bool, new_hold: PieceDef,
		next_qi: int, well: int, out: Array) -> void:
	var board: Board = node["board"]
	var spawn := Vector2i((board.width - def.box) / 2, board.buffer_height - def.first_filled_row)
	var placements := PlacementEnumerator.placements_for(board, def, spawn, uses_hold)
	# Straight-drop enumeration cannot reach into a spin slot by construction. When a
	# slot is detected the pattern pins down the placement exactly -- position and
	# rotation both -- so it can be injected directly instead of needing a general
	# reachability search.
	if inject_spins and def.id == &"T":
		var slot := TSpinSlots.scan(board, evaluator.column_heights(board), well)
		if bool(slot["found"]):
			var sp := TSpinSlots.placement_for(board, def, slot)
			if sp != null:
				sp.uses_hold = uses_hold
				placements.append(sp)
	last_placements_considered += placements.size()

	var queue: Array = _queue
	# Built once per node, not per placement -- the queue does not change between
	# candidate placements of the same piece.
	var avail := _availability(queue, next_qi, new_hold)
	for p in placements:
		var next_board := board.duplicate_board()
		next_board.lock_cells(p.cells, Vector2i.ZERO, def.index + 1)
		# Only the rows this piece touched can have become full.
		var touched := {}
		for c in p.cells:
			touched[c.y] = true
		var full := next_board.full_rows_among(touched.keys())
		p.lines = full.size()
		if p.lines > 0:
			next_board.clear_rows(full)

		var landing: int = p.pos.y
		var b2b_before: int = node["b2b"]
		# Board score is taken FRESH each ply, not accumulated. Only the per-placement
		# events accumulate. Summing board scores made deep searches prefer clearing
		# early, because an early clear had its lower board counted once per remaining
		# ply -- an artefact of the accounting rather than a real gain.
		# Does the well already have a claim on the next I? Asked per placement,
		# because a placement can be what makes the well ready.
		var well_claims_i := evaluator.clean_well_rows(next_board, well) >= 3
		var structural := evaluator.structural(next_board, well, avail, well_claims_i)
		var event_gain := evaluator.events(p.lines, landing, b2b_before, p.spin,
			def.id, _count(avail, &"I"))
		var events_total: float = float(node["events"]) + event_gain
		# Same bookkeeping as Game.lock_piece: only a clear moves the chain.
		var b2b_after := b2b_before
		if p.lines > 0:
			var difficult: bool = p.lines >= 4 or p.spin != SpinDetect.Kind.NONE
			b2b_after = (b2b_before + 1) if difficult else -1
		var path: Array = node["path"].duplicate()
		path.append(p)
		out.append({
			"board": next_board,
			"hold": new_hold,
			"qi": next_qi,
			"score": events_total + structural,
			"events": events_total,
			"b2b": b2b_after,
			"path": path,
		})
