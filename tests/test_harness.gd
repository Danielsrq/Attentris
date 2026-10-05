class_name TestHarness
extends Node

## Shared assertion vocabulary and board fixtures for the test suites.
##
## Suites extend this and implement `run()`; tests/run_tests.gd instantiates each one
## and sums the counters, so the whole set is still one command and one exit code.
##
## A helper lives here when its callers span more than one suite, and in the suite
## otherwise. That rule is applied mechanically rather than by taste, so this file does
## not quietly become the dumping ground that the old `Scoring` section became -- 1005
## lines holding audio cues, control profiles and the solver's well detection.

var _pass := 0
var _fail := 0
var _group := ""

## Each suite overrides this and calls its own test functions. May be a coroutine: the
## runner awaits it, so a suite that needs physics frames can await them.
func run() -> void:
	pass

## Waits n physics frames. TWO is the minimum anywhere a freshly added Area2D is involved:
## the first frame registers it, the second runs the step that populates the overlap list.
## One frame is the classic flake.
func physics_frames(n: int) -> void:
	for _i in n:
		await get_tree().physics_frame

# ------------------------------------------------------------------------------
# Garbage
# ------------------------------------------------------------------------------

## Reads a row back as ASCII: "." empty, "G" garbage, "X" anything else.
func _row(b: Board, y: int) -> String:
	var out := ""
	for x in b.width:
		var v := b.get_cell(x, y)
		out += "." if v == 0 else ("G" if v == Board.GARBAGE else "X")
	return out

func check(cond: bool, what: String) -> void:
	if cond:
		_pass += 1
		print("   ok    %s" % what)
	else:
		_fail += 1
		printerr("   FAIL  [%s] %s" % [_group, what])
		print("   FAIL  %s" % what)

func eq(got, want, what: String) -> void:
	if typeof(got) == typeof(want) and got == want:
		_pass += 1
		print("   ok    %s" % what)
	else:
		_fail += 1
		printerr("   FAIL  [%s] %s -- got %s, want %s" % [_group, what, got, want])
		print("   FAIL  %s -- got %s, want %s" % [what, got, want])

func first_clear(events: Array) -> ClearResult:
	for e in events:
		if e.type == GameEvent.Type.LINES_CLEARED:
			return e.data["result"]
	return null

# ------------------------------------------------------------------------------
# Harness
# ------------------------------------------------------------------------------

func group(name: String) -> void:
	_group = name
	print("")
	print("-- %s" % name)

## Build a board from ASCII rows. Rows are given top-to-bottom and anchored to the
## BOTTOM of the field, so the last string is the floor row.
func make_board(rows: Array, w: int = 10) -> Board:
	var b := Board.new(w, 20, 20)
	var base := b.total_height - rows.size()
	for i in rows.size():
		var line: String = rows[i]
		for x in line.length():
			if line[x] == "X":
				b.set_cell(x, base + i, 1)
	return b

func make_game(b: Board, rs: RuleSet = null) -> Game:
	var rules := rs if rs != null else RuleSet.competitive()
	return Game.new(rules, PieceSet.tetrominoes(), b, Scorer.new(ScoreTable.new()), 12345)

# ------------------------------------------------------------------------------
# Scoring
# ------------------------------------------------------------------------------

func make_result(lines: int, spin: int, piece: StringName, b2b: int, combo: int) -> ClearResult:
	var r := ClearResult.new()
	r.lines = lines
	r.spin = spin
	r.piece_id = piece
	r.b2b = b2b
	r.combo = combo
	r.level = 1
	return r

## Place a piece by hand without spawning, for kick tests.
func place(g: Game, id: StringName, rot: int, pos: Vector2i) -> ActivePiece:
	var p := ActivePiece.new()
	p.def = g.piece_set.get_def(id)
	p.rot = rot
	p.pos = pos
	p.lowest_row = pos.y
	g.active = p
	return p
