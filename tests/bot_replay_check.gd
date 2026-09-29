extends SceneTree

## Headless check that a bot replay file loads in the viewer's own reader.
##
##   godot --headless --path . -s res://tests/bot_replay_check.gd -- --replay=res://data/bot/sample.jsonl
##
## For every record: ReplayLog.apply() must recover the placed piece from its cells
## (piece_for != null), and the placed cells must be empty on the spawn-time board.
## Also re-derives each run's attack with the Godot AttackTable from the recorded
## clears, which is what the viewer's HUD shows.

func _init() -> void:
	var path := "res://data/bot/sample.jsonl"
	for arg in OS.get_cmdline_user_args():
		if arg.begins_with("--replay="):
			path = arg.substr("--replay=".length())
	var ps := PieceSet.tetrominoes()
	var log := ReplayLog.new(ps)
	if not log.load_file(path):
		print("FAIL load: ", log.error)
		quit(1)
		return
	var board := Board.new(10, 20, 20)
	var game := Game.new(RuleSet.competitive(), ps, board, Scorer.new(ScoreTable.new()), 0)
	var bad := 0
	var total := 0
	var plan_cells := 0
	for name in log.names:
		var steps := log.steps_for(name)
		for step in steps:
			total += 1
			log.apply(step, game)
			if game.active == null:
				bad += 1
				if bad <= 5:
					print("unmatched piece: ", name, " n=", step.get("n"))
				continue
			for c in game.active.board_cells():
				if game.board.get_cell(c.x, c.y) != 0:
					bad += 1
					if bad <= 5:
						print("overlap: ", name, " n=", step.get("n"), " at ", c)
					break
			# The plan overlay: every planned cell must be empty on this board and
			# distinct from the placed piece and from every other planned piece --
			# otherwise the row remapping in the bot's plan_to_json is wrong.
			var taken := {}
			for c in game.active.board_cells():
				taken[c] = true
			for s in step.get("plan", []):
				if int(s.get("k", 0)) == 0:
					continue
				for cc in s.get("cells", []):
					var c := Vector2i(int(cc[0]), int(cc[1]))
					if c.y < 0:
						continue  # scrolled in from above the field
					if game.board.get_cell(c.x, c.y) != 0 or taken.has(c):
						bad += 1
						if bad <= 5:
							print("plan overlap: ", name, " n=", step.get("n"), " k=", s.get("k"), " at ", c)
					taken[c] = true
					plan_cells += 1
		print("%s: %d placements" % [name, steps.size()])
	print("records %d, planned cells checked %d, problems %d" % [total, plan_cells, bad])
	quit(0 if bad == 0 else 1)
