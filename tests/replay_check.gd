extends Node

## Validates the harvested dataset through OUR engine, not the exporter's.
##
## The dataset stores each placement as absolute cells, so nothing depends on matching
## the exporting engine's pivot or rotation tables. This check proves the conversion
## back to our own representation is exact, for every sample in the file:
##
##   1. every placement resolves to one of our four rotations, at a definite position
##   2. the piece fits the recorded board -- no overlap, nothing out of bounds
##   3. placing it completes exactly the number of rows the dataset says it cleared
##
## (3) is the one that matters. It ties board, piece geometry and outcome together, so
## an off-by-one in rotation, position or row indexing cannot pass.

func _ready() -> void:
	var t0 := Time.get_ticks_usec()
	var ps := PieceSet.tetrominoes()
	var log := ReplayLog.new(ps)
	if not log.load_file():
		printerr("FAILED to load: %s" % log.error)
		get_tree().quit(1)
		return
	var load_ms := float(Time.get_ticks_usec() - t0) / 1000.0

	print("dataset : %d samples, %d runs, loaded in %.0f ms" % [
		log.total_samples(), log.run_count(), load_ms])

	var board := Board.new(10, 20, 20)
	var game := Game.new(RuleSet.competitive(), ps, board, Scorer.new(ScoreTable.new()), 0)

	var checked := 0
	var unmatched := 0
	var overlap := 0
	var wrong_clears := 0
	var broken_chain := 0
	var lost_hold := 0
	var alien_piece := 0
	var pairs := 0
	var tiers := {}
	var users := {}
	var first_fail := ""

	# Optional cap so the check can be run over a few runs while iterating:
	#   godot --headless --path . res://tests/replay_check.tscn -- 3
	var limit := log.run_count()
	for arg in OS.get_cmdline_user_args():
		if arg.is_valid_int():
			limit = mini(limit, maxi(1, arg.to_int()))

	var t1 := Time.get_ticks_usec()
	for run_name in log.names.slice(0, limit):
		var run_steps := log.steps_for(run_name)

		# Continuity across placements. These are what catch a viewer showing the
		# piece on the board while the NEXT panel still holds it: the queue and hold
		# a sample displays must be the ones in force when that piece LANDED, not
		# when it spawned, because a hold swap happens in between.
		for i in maxi(0, run_steps.size() - 1):
			var a: Dictionary = run_steps[i]
			var b: Dictionary = run_steps[i + 1]
			pairs += 1
			var q: Array = a.get("lock_queue", [])
			if q.is_empty() or String(b.get("piece", "")) != String(q[0]):
				broken_chain += 1
				if first_fail == "":
					first_fail = "%s n=%d: next piece is %s, queue head was %s" % [
						run_name, int(a.get("n", -1)), b.get("piece", "?"),
						q[0] if not q.is_empty() else "-"]
			if str(b.get("hold", null)) != str(a.get("lock_hold", null)):
				lost_hold += 1
			# The piece that locks is either the one that spawned, or the one that
			# was sitting in hold.
			var want = a.get("hold", null) if bool(a.get("used_hold", false)) \
				else a.get("piece", null)
			var got = a.get("place", {}).get("type", null)
			if want != null and got != want:
				alien_piece += 1

		for step in run_steps:
			checked += 1
			var tier := String(step.get("tier", "?"))
			tiers[tier] = int(tiers.get(tier, 0)) + 1
			var u := String(step.get("user", "?"))
			users[u] = int(users.get(u, 0)) + 1

			log.apply(step, game)
			var piece: ActivePiece = game.active
			if piece == null:
				unmatched += 1
				if first_fail == "":
					first_fail = "%s n=%d: no rotation matches the recorded cells" % [
						run_name, int(step.get("n", -1))]
				continue
			if not board.can_place(piece.cells(), piece.pos):
				overlap += 1
				if first_fail == "":
					first_fail = "%s n=%d: %s does not fit at %s" % [
						run_name, int(step.get("n", -1)), piece.def.id, piece.pos]
				continue
			board.lock_cells(piece.cells(), piece.pos, piece.def.index + 1)
			var full := board.find_full_rows().size()
			var want := int(step.get("cleared", 0))
			if full != want:
				wrong_clears += 1
				if first_fail == "":
					first_fail = "%s n=%d: completes %d rows, dataset says %d" % [
						run_name, int(step.get("n", -1)), full, want]
	var check_ms := float(Time.get_ticks_usec() - t1) / 1000.0

	print("checked : %d samples in %.0f ms" % [checked, check_ms])
	print("tiers   : %s" % tiers)
	print("players : %s" % users)
	print("chain   : %d consecutive pairs" % pairs)
	var bad := unmatched + overlap + wrong_clears + broken_chain + lost_hold + alien_piece
	if bad == 0:
		print("\nOK  all %d placements resolve, fit, and clear exactly what is recorded," % checked)
		print("    and queue/hold stay continuous across all %d transitions" % pairs)
		get_tree().quit(0)
		return
	printerr("\nFAILED  unmatched=%d overlap=%d wrong_clears=%d" % [
		unmatched, overlap, wrong_clears])
	printerr("        broken_chain=%d lost_hold=%d alien_piece=%d" % [
		broken_chain, lost_hold, alien_piece])
	printerr("first : %s" % first_fail)
	get_tree().quit(1)
