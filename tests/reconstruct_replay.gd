extends Node

## Replay reconstruction diagnostic.
##   godot --headless --path . res://tests/reconstruct_replay.tscn -- <path-to.ttr>
##
## Reports where our engine stops matching TETR.IO's, rather than asserting it does.

const DEFAULT_PATH := "user://replay.ttr"

func _ready() -> void:
	var path := DEFAULT_PATH
	for a in OS.get_cmdline_user_args():
		if a.ends_with(".ttr") or a.ends_with(".ttrm"):
			path = a
	print("loading %s" % path)
	var r := TTRReplay.load_file(path)
	if not r.ok:
		printerr("parse failed: %s" % r.error)
		print("parse failed: %s" % r.error)
		get_tree().quit(1)
		return

	print("")
	print("--- replay ---")
	print("  mode %s   version %d   frames %d" % [r.gamemode, r.version, r.frames])
	print("  seed %d   bagtype %s   180 %s   spins %s   garbage %s" % [
		r.seed_value, r.bagtype, r.allow180, r.spinbonuses, r.has_garbage])
	print("  handling: das %.1ff (%.0fms)  arr %.1ff  sdf %.0f  dcd %.1ff" % [
		r.das_frames, r.das_frames * 1000.0 / 60.0, r.arr_frames, r.sdf, r.dcd_frames])

	# Parse check before trusting anything downstream: these must match the file's
	# own recorded totals, or we are reading the event stream wrong.
	var hist := r.keydown_histogram()
	var total := 0
	for k in hist:
		total += int(hist[k])
	print("")
	print("--- parse check ---")
	print("  keydowns %d vs recorded inputs %d   %s" % [
		total, r.stat_inputs, "OK" if total == r.stat_inputs else "MISMATCH"])
	print("  holds    %d vs recorded holds  %d   %s" % [
		int(hist.get("hold", 0)), r.stat_holds,
		"OK" if int(hist.get("hold", 0)) == r.stat_holds else "MISMATCH"])
	print("  hardDrop %d vs recorded pieces %d   %s" % [
		int(hist.get("hard_drop", 0)), r.stat_pieces,
		"OK" if int(hist.get("hard_drop", 0)) == r.stat_pieces else "MISMATCH"])
	var keys: Array = hist.keys()
	keys.sort()
	var parts: Array = []
	for k in keys:
		parts.append("%s=%d" % [k, hist[k]])
	print("  keys: %s" % ", ".join(parts))

	print("")
	print("--- reconstruction ---")
	var rules := r.to_rules()
	var ps := PieceSet.tetrominoes()
	var board := Board.new(10, 20, 20)
	var game := Game.new(rules, ps, board, Scorer.new(ScoreTable.new()), r.seed_value)
	# TETR.IO's bag, not ours -- same guarantee, different sequence.
	# "zenith" is an undocumented bagtype. Hypothesis: it is ordinary 7-bag with a
	# constrained opening (openerphase 14 = two bags). Forcing the standard generator
	# tests that -- if it survives far longer than our own bag did, the steady state
	# really is 7-bag; if it dies just as fast, the generator is genuinely different.
	if r.bagtype == "7-bag" or r.bagtype == "zenith":
		game.set_randomizer(TetrioRandomizer.new(ps, r.seed_value))
		print("  using TetrioRandomizer (MINSTD, ZLOSIJT, Fisher-Yates down)%s" % (
			"  [forced -- bagtype is 'zenith']" if r.bagtype == "zenith" else ""))
	else:
		print("  bagtype '%s' unimplemented -- using our 7-bag, expect desync" % r.bagtype)
	game.start()
	var player := ReplayPlayer.new(r, game)
	var t0 := Time.get_ticks_msec()
	player.run()
	var ms := Time.get_ticks_msec() - t0

	print("  ran %d of %d frames in %d ms" % [player.frames_run, r.frames, ms])
	print("  pieces %d (recorded %d)" % [player.pieces_placed, r.stat_pieces])
	print("  lines  %d (recorded %d)" % [player.lines_cleared, r.stat_lines])
	print("  verified: %s" % player.verified())
	print("")
	print("--- divergences ---")
	if player.notes.is_empty():
		print("  none")
	for n in player.notes:
		print("  * %s" % n)
	get_tree().quit(0)
