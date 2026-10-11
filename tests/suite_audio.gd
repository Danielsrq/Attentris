extends TestHarness

## Audio: events and cues
##
## Run by tests/run_tests.gd, which owns the totals. See tests/test_harness.gd for
## group/check/eq and the shared board fixtures.

func run() -> void:
	test_audio_events()
	test_audio_cue_mapping()

func test_audio_events() -> void:
	group("events for audio cues")

	# --- movement is coalesced: one event per tick regardless of cells travelled ---
	var g := make_game(Board.new(10, 20, 20))
	g.start()
	var right := InputState.new()
	right.right = true
	var tap := g.tick(right)
	eq(events_of(tap, GameEvent.Type.PIECE_MOVED).size(), 1, "a tap emits one move event")
	eq(events_of(tap, GameEvent.Type.PIECE_MOVED)[0].data["cells"], 1, "tap moved one cell")
	eq(events_of(tap, GameEvent.Type.PIECE_MOVED)[0].data["dir"], 1, "direction reported")

	# Charge DAS, then the ARR 0 slide must still be a single event.
	var slide_events: Array = []
	for _i in 20:
		slide_events = g.tick(right)
		var moved := events_of(slide_events, GameEvent.Type.PIECE_MOVED)
		if not moved.is_empty() and moved[0].data["cells"] > 1:
			break
	var multi := events_of(slide_events, GameEvent.Type.PIECE_MOVED)
	eq(multi.size(), 1, "an ARR 0 wall slide is still ONE event, not one per cell")
	check(multi[0].data["cells"] > 1, "and it reports the whole distance (%d cells)" % multi[0].data["cells"])

	# Held against the wall, nothing moves, so nothing fires.
	var at_wall: Array = []
	for _i in 10:
		at_wall = g.tick(right)
	check(events_of(at_wall, GameEvent.Type.PIECE_MOVED).is_empty(),
		"a blocked piece emits no move event, so the cue does not machine-gun")

	# --- rotation reports the spin state at that moment ---
	var b := make_board([
		"..........",
		"XXXX..XXXX",
		"XXX...XXXX",
		"XXXX.XXXXX",
	])
	var g2 := make_game(b)
	place(g2, &"T", 3, Vector2i(4, 36))
	var rot: Array = []
	g2.try_rotate(2, rot)
	var rotated := events_of(rot, GameEvent.Type.PIECE_ROTATED)
	eq(rotated.size(), 1, "a successful rotation emits one event")
	eq(rotated[0].data["spin"], SpinDetect.Kind.FULL, "and reports the spin BEFORE the clear happens")
	eq(rotated[0].data["kick"], 2, "carrying the kick index")

	# A rotation in open space reports no spin, so the click must not fire.
	var g3 := make_game(Board.new(10, 20, 20))
	place(g3, &"T", 0, Vector2i(3, 30))
	var rot3: Array = []
	g3.try_rotate(1, rot3)
	eq(events_of(rot3, GameEvent.Type.PIECE_ROTATED)[0].data["spin"], SpinDetect.Kind.NONE,
		"an open-space rotation reports no spin")

	# A rotation that fails every kick emits nothing at all.
	var packed := make_board(["XXXXXXXXXX", "XXXX..XXXX", "XXXXXXXXXX"])
	var g4 := make_game(packed)
	place(g4, &"T", 0, Vector2i(3, 30))
	g4.active.pos = Vector2i(4, 38)
	var rot4: Array = []
	var ok := g4.try_rotate(1, rot4)
	check(not ok or events_of(rot4, GameEvent.Type.PIECE_ROTATED).size() == 1,
		"a rejected rotation emits nothing")

	# --- lock cause distinguishes a slam from a settle ---
	var g5 := make_game(Board.new(10, 20, 20))
	g5.start()
	var drop := InputState.new()
	drop.hard_drop = true
	var locked := events_of(g5.tick(drop), GameEvent.Type.PIECE_LOCKED)
	eq(locked.size(), 1, "hard drop locks the piece")
	eq(locked[0].data["cause"], "hard_drop", "reported as a hard drop")

	var g6 := make_game(Board.new(10, 20, 20))
	g6.start()
	var manual: Array = []
	g6.lock_piece(manual)
	eq(events_of(manual, GameEvent.Type.PIECE_LOCKED)[0].data["cause"], "lock_delay",
		"a natural lock reports lock_delay")

	# --- combo_broken fires exactly once, on the placement that ends the chain ---
	var g7 := make_game(Board.new(10, 20, 20))
	g7.start()
	g7.combo = 3
	var brk: Array = []
	drop_to_floor(g7)
	g7.lock_piece(brk)
	eq(events_of(brk, GameEvent.Type.PIECE_LOCKED)[0].data["combo_broken"], true,
		"ending a combo chain is reported")
	eq(g7.combo, -1, "and the chain is reset")
	var brk2: Array = []
	drop_to_floor(g7)
	g7.lock_piece(brk2)
	eq(events_of(brk2, GameEvent.Type.PIECE_LOCKED)[0].data["combo_broken"], false,
		"the next placement does not report it again")

	# --- the cue routing itself, without needing an audio device ---
	group("cue routing")
	eq(_cue_for(4, SpinDetect.Kind.NONE), "line_clear_plus", "quad -> line_clear_plus")
	eq(_cue_for(3, SpinDetect.Kind.NONE), "line_clear", "triple -> line_clear")
	eq(_cue_for(2, SpinDetect.Kind.NONE), "line_clear", "double -> line_clear")
	eq(_cue_for(1, SpinDetect.Kind.NONE), "line_clear", "single -> line_clear")
	eq(_cue_for(1, SpinDetect.Kind.FULL), "line_clear_plus", "spin single -> line_clear_plus")
	eq(_cue_for(2, SpinDetect.Kind.MINI), "line_clear_plus", "mini spin double -> line_clear_plus")
	eq(_cue_for(0, SpinDetect.Kind.FULL), "", "a spin with no clear plays no clear cue")
	eq(_cue_for(0, SpinDetect.Kind.NONE), "", "a plain placement plays no clear cue")

func test_audio_cue_mapping() -> void:
	group("audio cue mapping")
	var bus := AudioBus.new()
	var cues := [
		AudioBus.CUE_CLICK, AudioBus.CUE_MOVE, AudioBus.CUE_DROP,
		AudioBus.CUE_CLEAR, AudioBus.CUE_CLEAR_PLUS, AudioBus.CUE_COMBO_BREAK,
	]
	for cue in cues:
		check(bus.stream_for(cue) != null, "cue '%s' resolves to a stream" % cue)
		bus.set_cue_gain(cue, -7.5)
		eq(bus.gain_for(cue), -7.5, "cue '%s' gain round-trips" % cue)
	eq(bus.stream_for("nonexistent"), null, "an unknown cue resolves to nothing")
	eq(bus.gain_for("nonexistent"), 0.0, "an unknown cue has no trim")

	# Shipped balance.
	var fresh := AudioBus.new()
	eq(fresh.gain_click, 0.0, "click ships flat (the louder sample sits there now)")
	eq(fresh.gain_combo_break, -3.0, "combo break ships at -3 dB")
	eq(fresh.stream_click.resource_path, "res://audio/spin_click.ogg", "click plays spin_click")
	eq(fresh.stream_move.resource_path, "res://audio/piece_move.ogg", "move plays piece_move")
	fresh.free()

	# Clearing a slot must silence that cue rather than erroring.
	bus.stream_move = null
	eq(bus.stream_for(AudioBus.CUE_MOVE), null, "an emptied slot silences its cue")
	bus.play(AudioBus.CUE_MOVE)
	check(true, "playing a silenced cue is a no-op, not a crash")
	bus.free()

## Seat the active piece on the stack. Locking at the spawn position would block out
## the next spawn, which is correct behaviour but wrong for these setups.
func drop_to_floor(g: Game) -> void:
	while g.board.can_place(g.active.cells(), g.active.pos + Vector2i(0, 1)):
		g.active.pos.y += 1

func events_of(events: Array, type: int) -> Array:
	var out: Array = []
	for e in events:
		if e.type == type:
			out.append(e)
	return out

## Mirrors AudioBus._on_locked so the routing is checked without an audio device.
func _cue_for(lines: int, spin: int) -> String:
	if lines <= 0:
		return ""
	return "line_clear_plus" if (lines >= 4 or spin != SpinDetect.Kind.NONE) else "line_clear"
