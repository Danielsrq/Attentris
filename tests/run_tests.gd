extends Node

## Headless assertions over the pure sim. Run with:
##   godot --headless --path . res://tests/run_tests.tscn
## Exits 0 when everything passes, 1 otherwise.
##
## This file used to hold every test, and had grown past 3,600 lines -- a quarter of it
## under one `Scoring` banner that had accumulated audio cues, control profiles and the
## solver's well detection. It is now just the runner: each suite is a file extending
## TestHarness, and the totals are summed here so the whole set stays one command and
## one exit code.
##
## Adding a test means adding it to a suite's `run()`. Adding a SUITE means adding a
## line below.

const SUITES := [
	"res://tests/suite_pieces.gd",
	"res://tests/suite_board.gd",
	"res://tests/suite_spins.gd",
	"res://tests/suite_scoring.gd",
	"res://tests/suite_solver.gd",
	"res://tests/suite_audio.gd",
	"res://tests/suite_input.gd",
	"res://tests/suite_economy.gd",
	"res://tests/suite_defence.gd",
]

func _ready() -> void:
	var passed := 0
	var failed := 0
	for path in SUITES:
		var suite: TestHarness = load(path).new()
		# Parented so a suite that needs the tree (the defence view does) can use it.
		add_child(suite)
		suite.run()
		passed += suite._pass
		failed += suite._fail
		suite.queue_free()

	print("")
	print("%d passed, %d failed" % [passed, failed])
	get_tree().quit(1 if failed > 0 else 0)
