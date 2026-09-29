extends Node

## Headless solver benchmark. Compares evaluator presets across SEVERAL SEEDS,
## because single-seed runs have twice now produced conclusions that did not survive
## better measurement.
##
##   godot --headless --path . res://tests/bench_solver.tscn

const SEEDS := [11, 22, 33, 44, 55]
const MAX_PIECES := 40
## Wall-clock ceiling per run. A run that hits it shows up as a short piece count.
const BUDGET_MS := 120000.0

## [label, weights preset, depth, beam, garbage mode]
## Garbage modes: "none", "clean" (all rows share a hole column -- hands you a ready
## made well, which is what real attacks usually look like), "cheese" (a different
## hole per row, the hostile case).
## Beam sweep on clean garbage -- the mode where the T-spin solver has the best
## chance. If a wider beam finds spins where a narrow one does not, the problem is
## search. If it changes nothing, the slot reward is not steering and the weights are.
## Confirming the b32 result at 5 seeds -- AND controlling for beam width, which
## the previous run did not: it compared tspin@b32 against quad@b8, so part of the
## gap could have been the wider beam rather than the spins.
const RUNS := [
	["Iw flat  b32 cln", "lookahead_i", 3, 32, "clean"],
	["Iw <h/2  b32 cln", "lookahead_ih", 3, 32, "clean"],
	["Iw flat  b32 chs", "lookahead_i", 3, 32, "cheese"],
	["Iw <h/2  b32 chs", "lookahead_ih", 3, 32, "cheese"],
]

func _ready() -> void:
	print("solver benchmark -- %d seeds x up to %d pieces per run" % [SEEDS.size(), MAX_PIECES])
	print("")
	print("preset            | ms   | pieces | clears   Q  spin | slots seen/took | T-eff I-eff | atk/pc | holes | out")
	print("                  |      |        |                  |                 | spin  quad  |        |       |")
	print("------------------+------+--------+------------------+-----------------+-------------+--------+-------+----")
	for r in RUNS:
		_run(r[0], r[1], r[2], r[3], r[4])
	print("")
	print("S/D/T/Q = single / double / triple / quad CLEAR EVENTS.")
	print("quad%% = Q / clears -- events, not lines: weighting by lines flatters quads 4x.")
	print("slots seen = placements where a CASHABLE slot already existed on the board.")
	print("slots took = placements where the solver actually played the spin.")
	print("T-eff = T pieces that produced a spin clear / T pieces placed.")
	print("I-eff = I pieces that produced a quad / I pieces placed.")
	print("atk = garbage sent. Single 0, double 1, triple 2, quad 4, +floor(sqrt(b2b)) on a chain.")
	print("atk/pc = attack per piece -- the efficiency the minigame economy will spend.")
	get_tree().quit(0)

static func weights_for(preset: String) -> EvalWeights:
	match preset:
		"quad_v2": return EvalWeights.quad_v2()
		"oh1", "oh3", "oh6", "oh8":
			var o := EvalWeights.lookahead()
			o.w_overhang = -float(preset.substr(2).to_int())
			return o
		"tspin": return EvalWeights.tspin()
		"tspin_flat": return EvalWeights.tspin_flat()
		"distilled": return EvalWeights.distilled()
		"lookahead": return EvalWeights.lookahead()
		"lookahead_i": return EvalWeights.lookahead_i()
		"lookahead_ih": return EvalWeights.lookahead_ih()
		"distilled_no_small_penalty": return EvalWeights.distilled_no_small_penalty()
		"minimal": return EvalWeights.minimal()
		"minimal_sum_only": return EvalWeights.minimal_sum_only()
		"minimal_count_only": return EvalWeights.minimal_count_only()
		"minimal_no_parity": return EvalWeights.minimal_no_parity()
		"minimal_no_height": return EvalWeights.minimal_no_height()
		"line_greedy": return EvalWeights.line_greedy()
		"quad_well0":
			var a := EvalWeights.quad()
			a.well_column = 0
			return a
		"quad_dynamic":
			# -1 re-picks the emptiest edge column on every solve. Claimed earlier to
			# be bad because the well never gets dug; never isolated until now.
			var b := EvalWeights.quad()
			b.well_column = -1
			return b
		_: return EvalWeights.quad()

func _run(label: String, preset: String, depth: int, beam: int, garbage: String) -> void:
	var times: Array = []
	var pieces := 0
	var lines := 0
	var holes := 0
	var best_b2b := 0
	var topouts := 0
	var attack := 0
	var spin_clears := 0
	var t_used := 0
	var t_spins := 0
	var i_used := 0
	var i_quads := 0
	var slots_seen := 0
	var slots_taken := 0
	var by_size := PackedInt32Array()
	by_size.resize(9)

	var started := Time.get_ticks_msec()
	for seed_v in SEEDS:
		if float(Time.get_ticks_msec() - started) > BUDGET_MS:
			break
		var res := _play(seed_v, preset, depth, beam, started, garbage)
		times.append_array(res["times"])
		pieces += res["pieces"]
		lines += res["lines"]
		holes += res["holes"]
		attack += res["attack"]
		spin_clears += res["spin_clears"]
		t_used += res["t_used"]
		t_spins += res["t_spins"]
		i_used += res["i_used"]
		i_quads += res["i_quads"]
		slots_seen += res["slots_seen"]
		slots_taken += res["slots_taken"]
		best_b2b = maxi(best_b2b, res["best_b2b"])
		if res["topped_out"]:
			topouts += 1
		var rs: PackedInt32Array = res["by_size"]
		for i in by_size.size():
			by_size[i] += rs[i]

	times.sort()
	var avg := 0.0
	for t in times:
		avg += t
	avg /= maxf(1.0, float(times.size()))
	var mx: float = times[times.size() - 1] if not times.is_empty() else 0.0

	var quads := 0
	for i in range(4, by_size.size()):
		quads += by_size[i]
	var clears := 0
	for i in range(1, by_size.size()):
		clears += by_size[i]

	print("%-17s | %4.0f | %6d | %6d %3d %5d | %8d %6d | %5.2f %5.2f | %6.3f | %5d | %d/%d" % [
		label, avg, pieces,
		clears, quads, spin_clears,
		slots_seen, slots_taken,
		float(t_spins) / maxf(1.0, float(t_used)),
		float(i_quads) / maxf(1.0, float(i_used)),
		float(attack) / maxf(1.0, float(pieces)),
		holes, topouts, SEEDS.size()])
	print("                  |      | atk %d over %d pieces, %d clears" % [attack, pieces, clears])

func _play(seed_v: int, preset: String, depth: int, beam: int, started: int, garbage: String) -> Dictionary:
	var rules := RuleSet.competitive()
	var piece_set := PieceSet.tetrominoes()
	var board := Board.new(10, 20, 20)
	var game := Game.new(rules, piece_set, board, Scorer.new(ScoreTable.new()), seed_v)
	if garbage != "none":
		var grng := RandomNumberGenerator.new()
		grng.seed = seed_v * 7919
		var holes: Array = []
		if garbage == "clean":
			# One shared hole column: a ready-made well, which is what most real
			# attacks look like.
			var h := grng.randi_range(0, board.width - 1)
			for i in 4:
				holes.append(h)
		else:
			# Cheese: a different hole per row, the case that tests downstacking.
			var last := -1
			for i in 4:
				var h := grng.randi_range(0, board.width - 1)
				if h == last:
					h = (h + 1) % board.width
				last = h
				holes.append(h)
		board.push_garbage(holes)
	game.start()

	var weights := weights_for(preset)
	var is_spin := preset.begins_with("tspin")
	var ev: BoardEvaluator = TSpinEvaluator.new(weights) if is_spin else BoardEvaluator.new(weights)
	var solver := BeamSolver.new(ev)
	solver.inject_spins = is_spin
	solver.depth = depth
	solver.beam_width = beam

	var times: Array = []
	var pieces := 0
	var best_b2b := 0
	var spin_clears := 0
	var t_used := 0
	var t_spins := 0
	var i_used := 0
	var i_quads := 0
	var slots_seen := 0
	var slots_taken := 0
	var well_col: int = clampi(weights.well_column, 0, board.width - 1)
	var scan_ev := BoardEvaluator.new(weights)
	var by_size := PackedInt32Array()
	by_size.resize(9)

	while pieces < MAX_PIECES and not game.topped_out and game.active != null:
		if float(Time.get_ticks_msec() - started) > BUDGET_MS:
			break
		var queue: Array = [game.active.def]
		queue.append_array(game.next_queue)

		# Counted before the move, so "seen" means the opportunity existed.
		if TSpinSlots.scan(game.board, scan_ev.column_heights(game.board), well_col)["ready"]:
			slots_seen += 1

		var t0 := Time.get_ticks_usec()
		var plan := solver.solve(game.board, queue, game.hold_def, Callable(), game.b2b)
		times.append(float(Time.get_ticks_usec() - t0) / 1000.0)

		if plan.is_empty():
			break
		var p: Placement = plan[0]
		if not _apply(game, p):
			break
		# Per-piece accounting, read from the real result rather than the plan, so a
		# placement that failed to register as a spin shows up as one that did not.
		var r: ClearResult = game.last_result
		var was_spin: bool = r != null and r.lines > 0 and r.spin != SpinDetect.Kind.NONE
		var was_quad: bool = r != null and r.lines >= 4
		if p.def.id == &"T":
			t_used += 1
			if was_spin:
				t_spins += 1
		elif p.def.id == &"I":
			i_used += 1
			if was_quad:
				i_quads += 1
		if was_spin:
			spin_clears += 1
		if p.spin != SpinDetect.Kind.NONE:
			slots_taken += 1
		if p.lines > 0:
			by_size[mini(p.lines, by_size.size() - 1)] += 1
		best_b2b = maxi(best_b2b, game.b2b)
		pieces += 1

	# Board shape at the end, so the new features can be seen doing something rather
	# than assumed to be.
	var well: int = clampi(weights.well_column, 0, board.width - 1)
	var probe := BoardEvaluator.new(weights)
	probe.structural(board, well)
	var i_deps := 0
	for x in board.width:
		if x != well and probe.notch_depth(board.width, x) >= 3:
			i_deps += 1

	return {
		"times": times, "pieces": pieces, "lines": game.lines, "by_size": by_size,
		"holes": _count_holes(board), "topped_out": game.topped_out,
		"best_b2b": best_b2b,
		"attack": game.attack_sent,
		"spin_clears": spin_clears,
		"t_used": t_used, "t_spins": t_spins,
		"i_used": i_used, "i_quads": i_quads,
		"slots_seen": slots_seen, "slots_taken": slots_taken,
	}

## Drives the real Game rather than a private board, so the benchmark exercises the
## same lock / clear / scoring path the player does.
func _apply(game: Game, p: Placement) -> bool:
	if p.uses_hold:
		var inp := InputState.new()
		inp.hold = true
		game.tick(inp)
		if game.active == null or game.active.def != p.def:
			return false
	elif game.active.def != p.def:
		return false
	game.active.rot = p.rot
	game.active.pos = p.pos
	# Without this an injected slot placement lands in a spin shape and registers as
	# an ordinary clear -- SpinDetect only fires when the last action was a rotation.
	game.active.last_action_was_rotation = p.via_rotation
	game.active.last_kick_index = p.kick_index
	if not game.board.can_place(game.active.cells(), game.active.pos):
		return false
	var events: Array = []
	game.lock_piece(events)
	return true

func _count_holes(board: Board) -> int:
	var n := 0
	for x in board.width:
		var seen := false
		for y in board.total_height:
			if board.get_cell(x, y) != 0:
				seen = true
			elif seen:
				n += 1
	return n
