extends TestHarness

## Input and layout
##
## Run by tests/run_tests.gd, which owns the totals. See tests/test_harness.gd for
## group/check/eq and the shared board fixtures.

func run() -> void:
	test_control_profiles()
	test_input_map()
	test_layout_regions()

func test_control_profiles() -> void:
	group("control profiles")
	var meta_keys := {}
	for action in InputReader.META:
		for k in InputReader.META[action]:
			meta_keys[k] = action

	for profile_name in ControlProfiles.names():
		var p := ControlProfiles.get_profile(profile_name)
		var missing: Array = []
		for action in ControlProfiles.ACTIONS:
			if not p.has(action):
				missing.append(action)
		check(missing.is_empty(), "%s defines every action%s" % [
			profile_name, "" if missing.is_empty() else " (missing %s)" % str(missing)])

		# No key may drive two actions inside one profile, and no key may collide
		# with a meta key, or restart/debug would fire mid-game.
		var seen := {}
		var dupes: Array = []
		var clashes: Array = []
		for action in ControlProfiles.ACTIONS:
			for k in p.get(action, []):
				if seen.has(k):
					dupes.append("%s uses the same key as %s" % [action, seen[k]])
				seen[k] = action
				if meta_keys.has(k):
					clashes.append("%s clashes with meta %s" % [action, meta_keys[k]])
		check(dupes.is_empty(), "%s has no duplicate bindings%s" % [
			profile_name, "" if dupes.is_empty() else " (%s)" % str(dupes)])
		check(clashes.is_empty(), "%s does not collide with the meta keys%s" % [
			profile_name, "" if clashes.is_empty() else " (%s)" % str(clashes)])

		# Every action except 180 must actually be bound to something.
		var unbound: Array = []
		for action in ControlProfiles.ACTIONS:
			if action != "rotate_180" and p.get(action, []).is_empty():
				unbound.append(action)
		check(unbound.is_empty(), "%s binds every essential action%s" % [
			profile_name, "" if unbound.is_empty() else " (unbound %s)" % str(unbound)])

	# The personal profile, spelled out.
	var w := ControlProfiles.wasd_numpad()
	eq(w["left"], [KEY_A], "wasd_numpad: left is A")
	eq(w["right"], [KEY_D], "wasd_numpad: right is D")
	eq(w["soft_drop"], [KEY_W], "wasd_numpad: soft drop is W")
	eq(w["hard_drop"], [KEY_S], "wasd_numpad: hard drop is S")
	eq(w["rotate_ccw"], [KEY_KP_7], "wasd_numpad: ccw is numpad 7")
	eq(w["rotate_cw"], [KEY_KP_9], "wasd_numpad: cw is numpad 9")

	# Cycling wraps and always lands on a real profile.
	var reader := InputReader.new("wasd_numpad")
	eq(reader.profile_name, "wasd_numpad", "reader starts on the requested profile")
	var visited: Array = []
	for _i in ControlProfiles.names().size():
		visited.append(reader.profile_name)
		reader.next_profile()
	eq(visited.size(), ControlProfiles.names().size(), "cycling visits every profile")
	eq(reader.profile_name, "wasd_numpad", "cycling wraps back to the start")
	check(reader.binding_text().size() > 0, "binding text renders for the overlay")
	# keys_text now reads the InputMap, so it reports what is actually live. guideline
	# leaves rotate_180 registered but unbound, which must read as a dash rather than
	# as a missing action.
	# keys_text reads the InputMap, so it can only be checked once the actions exist.
	if InputMap.has_action("rotate_180"):
		reader.set_profile("guideline")
		eq(ControlProfiles.keys_text("rotate_180"), "-", "an unbound action reads as a dash")
		reader.set_profile("wasd_numpad")
		eq(ControlProfiles.keys_text("rotate_180"), "Kp 8", "and a bound one reads its key")

func test_input_map() -> void:
	group("InputMap: every action registered and bound physically")

	# Constructing a reader is what populates the map, so nothing here depends on
	# main.gd having run.
	var reader := InputReader.new("wasd_numpad")

	var owned: Array = []
	for a in ControlProfiles.ACTIONS:
		owned.append(a)
	for a in InputReader.META:
		owned.append(a)

	# Godot ships 92 built-in actions and every one is ui_-prefixed, so that prefix is the
	# whole rule: stay off it and nothing can be shadowed. Checked by name rather than
	# against a snapshot of the list, because once the 35 are declared they appear in
	# InputMap.get_actions() too and the two sets stop being distinguishable.
	var shadowing: Array = []
	for a in owned:
		if a.begins_with("ui_"):
			shadowing.append(a)
	check(shadowing.is_empty(), "no action shadows a Godot built-in (ui_*)%s" % [
		"" if shadowing.is_empty() else " (%s)" % str(shadowing)])

	# Nothing is created at runtime any more, so these only exist once they are declared
	# in project.godot. Report the gap as ONE line naming every absent action and stop:
	# twenty cascading failures would bury the only fact that matters.
	var unregistered: Array = reader.missing_actions()
	check(unregistered.is_empty(), "every expected action is declared in project.godot%s" % [
		"" if unregistered.is_empty() else " -- %d missing, see docs/input_map.md" % unregistered.size()])
	if not unregistered.is_empty():
		print("   .... %s" % str(unregistered))
		print("   .... declare these in Project Settings > Input Map, then re-run")
		return
	# Not a failure: tower selection cannot matter until there is more than one tower type.
	var later: Array = reader.missing_optional_actions()
	if not later.is_empty():
		print("   note  %d optional action(s) not declared yet: %s" % [later.size(), str(later)])

	# Physical keycodes, not labels. If this regresses, WASD moves on AZERTY and the
	# numpad stops working with NumLock off -- silently, and only for other people.
	var mislabelled: Array = []
	for a in owned:
		if not InputMap.has_action(a):
			continue
		for e in InputMap.action_get_events(a):
			if e is InputEventKey and (e.physical_keycode == 0 or e.keycode != 0):
				mislabelled.append(a)
	check(mislabelled.is_empty(), "every key event is physical, with keycode left at 0%s" % [
		"" if mislabelled.is_empty() else " (%s)" % str(mislabelled)])

	# F8 is Godot's "Stop Running Project" and the editor honours it even while the game
	# has focus, so anything bound there quits instead of firing.
	var on_f8: Array = []
	for a in owned:
		if KEY_F8 in _mapped_keys(a):
			on_f8.append(a)
	check(on_f8.is_empty(), "nothing is bound to F8, which Godot reserves for Stop%s" % [
		"" if on_f8.is_empty() else " (%s)" % str(on_f8)])

	# TAB now belongs to switch_board, authored by hand in project.godot. The pane tab
	# moved to F12 to get out of its way -- that is the whole "work around TAB" fix.
	check(InputMap.has_action("switch_board"), "switch_board exists, from project.godot")
	check(KEY_TAB in _mapped_keys("switch_board"), "   and holds TAB")
	# cycle_pane is gone entirely: the pane tabs it cycled went with the defence board
	# taking the pane, which is also what freed TAB. No workaround was needed in the end.
	var tab_clashes: Array = []
	for a in owned:
		if KEY_TAB in _mapped_keys(a):
			tab_clashes.append(a)
	check(tab_clashes.is_empty(), "no owned action competes with switch_board for TAB%s" % [
		"" if tab_clashes.is_empty() else " (%s)" % str(tab_clashes)])

	# Defence keys reuse the Tetris keys on purpose -- focus is exclusive -- but the meta
	# keys are global, so a collision there would fire mid-build.
	# A profile key must not collide with a meta key, or restart or a debug toggle fires
	# mid-game. Meta wins any tie by being bound last, which would be silent.
	var meta_keys := {}
	for a in InputReader.META:
		for k in InputReader.META[a]:
			meta_keys[k] = a
	var meta_clashes: Array = []
	for pname in ControlProfiles.names():
		var prof: Dictionary = ControlProfiles.get_profile(pname)
		for a in prof:
			for k in prof[a]:
				if meta_keys.has(k):
					meta_clashes.append("%s/%s vs meta %s" % [pname, a, meta_keys[k]])
	check(meta_clashes.is_empty(), "no profile key collides with a meta key%s" % [
		"" if meta_clashes.is_empty() else " (%s)" % str(meta_clashes)])

	group("InputMap: profile switching rewrites the bindings")
	reader.set_profile("wasd_numpad")
	eq(_mapped_keys("left"), [KEY_A], "wasd_numpad puts left on A")
	reader.set_profile("competitive")
	eq(_mapped_keys("left"), [KEY_LEFT], "switching to competitive moves it to Left")
	eq(_mapped_keys("rotate_180"), [KEY_A], "   and 180 onto A")
	reader.set_profile("guideline")
	eq(_mapped_keys("rotate_180"), [], "guideline leaves 180 registered but unbound")
	check(InputMap.has_action("rotate_180"), "   the action still exists, so callers need no guard")
	reader.set_profile("wasd_numpad")
	eq(_mapped_keys("left"), [KEY_A], "and switching back restores it")


	# Edge detection, assertable for the first time. is_physical_key_pressed could not be
	# faked, so nothing ever proved that pressed() was an edge and not a level.
	# Input.action_press writes action state directly and works headless.
	group("InputMap: pressed() is an edge, held() is a level")
	reader.set_profile("wasd_numpad")
	Input.action_press("hard_drop")
	reader.poll()
	check(reader.pressed("hard_drop"), "a fresh press reads as an edge")
	check(reader.held("hard_drop"), "   and as held")
	reader.poll()
	check(not reader.pressed("hard_drop"), "the same hold is not an edge on the next tick")
	check(reader.held("hard_drop"), "   but is still held")
	Input.action_release("hard_drop")
	reader.poll()
	check(not reader.held("hard_drop"), "releasing clears the level")
	check(not reader.pressed("hard_drop"), "   and does not register as a press")
	# Re-pressing after a release is a new edge, which is what repeat-tapping depends on.
	Input.action_press("hard_drop")
	reader.poll()
	check(reader.pressed("hard_drop"), "pressing again is a fresh edge")
	Input.action_release("hard_drop")
	reader.poll()

	# to_input_state maps held for continuous actions and edges for discrete ones.
	Input.action_press("left")
	Input.action_press("rotate_cw")
	reader.poll()
	var st := InputState.new()
	reader.to_input_state(st)
	check(st.left, "to_input_state carries a held direction")
	check(st.rotate_cw, "   and a rotation edge")
	reader.poll()
	reader.to_input_state(st)
	check(st.left, "the direction stays held on the next tick")
	check(not st.rotate_cw, "   while the rotation does not repeat")
	Input.action_release("left")
	Input.action_release("rotate_cw")
	reader.poll()

# ------------------------------------------------------------------------------
# Layout
# ------------------------------------------------------------------------------

func test_layout_regions() -> void:
	group("ViewLayout: the minigame's share of the screen")

	var b := Board.new(10, 20, 20)
	var l := ViewLayout.new(b, Vector2(1280, 720), 30)

	# One lane, vertically centred in the pane, deep enough to move around in.
	eq(l.lane_rect.size.y, float(l.lane_h), "the lane is lane_h deep")
	eq(l.lane_rect.size.x, l.right_pane.size.x, "   and spans the pane's full width")
	eq(l.lane_rect.position.x, l.right_pane.position.x, "   starting at its left edge")
	var above := l.lane_rect.position.y - l.right_pane.position.y
	var below := l.right_pane.end.y - l.lane_rect.end.y
	check(absf(above - below) < 1.0, "   and is centred, with equal room above and below")

	# The depth IS the number of bodies that seals the lane, so it is worth pinning against
	# the body size rather than left as a bare 256. Five would be sealed constantly.
	var bodies := float(l.lane_h) / (BattlerRules.new().body_radius * 2.0)
	eq(bodies, 8.0, "the lane is 8 bodies deep -- 8 units to seal it, not 5")
	check(float(l.lane_h) > BattlerRules.new().aggro_px,
		"and deeper than one unit's aggro, or a centre unit would see the whole lane")
	check(l.lane_centre_y() > l.lane_rect.position.y
			and l.lane_centre_y() < l.lane_rect.end.y,
		"the spawn seed depth sits inside the lane")

	# The lane must not stray over the playfield, which is what the old grid did at any
	# window under about 1240px wide -- it was centred by halving slack that had gone
	# negative. A full-width lane cannot repeat that, so this just pins it.
	var field_rect := Rect2(l.field_origin, l.field_size)
	var strays: Array = []
	for vw in [1280, 1100, 900, 700, 640]:
		var lw := ViewLayout.new(b, Vector2(float(vw), 720.0), 30)
		if lw.lane_rect.position.x < lw.right_pane.position.x:
			strays.append("%d" % vw)
		if Rect2(lw.field_origin, lw.field_size).has_point(lw.lane_rect.position):
			strays.append("%d over the field" % vw)
	check(strays.is_empty(), "the lane never starts left of its own pane%s" % [
		"" if strays.is_empty() else " (%s)" % str(strays)])

	# The debug readout draws over the right pane, which is the battler's lane, so it has to
	# open hidden or the game starts with debug text across the minigame. Pinned because the
	# default is one word and the symptom is cosmetic enough to ship by accident.
	var ov: Node2D = load("res://view/debug_overlay.gd").new()
	check(not ov.visible_overlay, "the debug overlay starts hidden, off the lane")
	# free(), not queue_free(): it was never added to the tree, and leaving it would show up
	# as "Pages in use exist at exit" long after this group has passed.
	ov.free()

	# A window with no room must leave a zero-width pane, not a negative one -- every
	# consumer's `size.x <= 0.0` guard depends on it.
	var narrow := ViewLayout.new(b, Vector2(640, 720), 30)
	eq(narrow.right_pane.size.x, 0.0, "a window with no room leaves a zero-width pane")
	eq(narrow.lane_rect.size.x, 0.0, "   and a zero-width lane")

	group("RuleSet: soft drop factor per preset")
	eq(RuleSet.guideline().sdf, 20.0, "guideline soft drop is 20x gravity, as specified")
	eq(RuleSet.competitive().sdf, 60.0, "competitive is 60x -- one cell per tick at level 1")
	# The two presets must actually differ here, or the competitive override is dead code.
	check(RuleSet.competitive().sdf > RuleSet.guideline().sdf,
		"competitive soft drop is the faster of the two")
	# One cell per tick at level 1 is the whole point of 60; check the arithmetic, since
	# sdf multiplies gravity and the relationship is easy to break.
	var rc := RuleSet.competitive()
	var cells_per_tick := rc.gravity_cells_per_tick(1, 60.0) * rc.sdf
	check(absf(cells_per_tick - 1.0) < 0.0001,
		"at level 1 soft drop moves exactly one cell per tick")

	group("AttackTable: legacy b2b_squares is floor(sqrt(b2b))")
	var at := AttackTable.new()
	at.b2b_scaling = AttackTable.B2bScaling.SQUARES
	# No chain pays nothing: -1 is "no b2b", 0 is the first difficult clear, which starts
	# the chain rather than being rewarded by it.
	eq(at.b2b_bonus_for(-1), 0, "no chain pays nothing")
	eq(at.b2b_bonus_for(0), 0, "the first difficult clear starts the chain, no bonus")
	for pair in [[1, 1], [2, 1], [3, 1], [4, 2], [8, 2], [9, 3], [15, 3], [16, 4],
			[24, 4], [25, 5], [27, 5], [35, 5], [36, 6], [48, 6], [49, 7]]:
		eq(at.b2b_bonus_for(int(pair[0])), int(pair[1]),
			"b2b %d pays +%d" % [int(pair[0]), int(pair[1])])
	# Each step up costs quadratically more than the last, which is the point of sqrt.
	var steps: Array = []
	var last := 0
	for n in range(1, 50):
		var v := at.b2b_bonus_for(n)
		if v != last:
			steps.append(n)
			last = v
	eq(steps, [1, 4, 9, 16, 25, 36, 49], "the bonus steps at the perfect squares")

	# And it reaches attack_for: a quad on a 27-chain is 4 + 5.
	var qr := ClearResult.new()
	qr.lines = 4
	qr.piece_id = &"I"
	qr.b2b = 27
	eq(at.attack_for(qr), 9, "a quad on a 27-chain sends 4 + 5 = 9")
	qr.b2b = 0
	eq(at.attack_for(qr), 4, "the same quad starting a chain sends just 4")
	# A single is not difficult, so it never collects a chain bonus even if one is passed.
	var sr := ClearResult.new()
	sr.lines = 1
	sr.piece_id = &"I"
	sr.b2b = 27
	eq(at.attack_for(sr), 0, "a single collects no chain bonus -- it is not a difficult clear")

	group("AttackTable: b2b_fibonacci (default) and b2b_prime")
	var fib := AttackTable.new()
	check(fib.b2b_scaling == AttackTable.B2bScaling.FIBONACCI, "fibonacci is the default")
	for pair in [[0, 0], [1, 1], [3, 1], [4, 2], [8, 2], [9, 3], [24, 3], [25, 4], [63, 4],
			[64, 5], [168, 5], [169, 6], [440, 6], [441, 7], [10000, 7]]:
		eq(fib.b2b_bonus_for(int(pair[0])), int(pair[1]), "fibonacci: b2b %d pays +%d" % [int(pair[0]), int(pair[1])])
	var pr := AttackTable.new()
	pr.b2b_scaling = AttackTable.B2bScaling.PRIME
	for pair in [[1, 0], [3, 0], [4, 1], [9, 2], [25, 3], [48, 3], [49, 4], [121, 5], [169, 6],
			[289, 7], [10000, 7]]:
		eq(pr.b2b_bonus_for(int(pair[0])), int(pair[1]), "prime: b2b %d pays +%d" % [int(pair[0]), int(pair[1])])
	qr.b2b = 70
	eq(fib.attack_for(qr), 9, "fibonacci: a quad on a 70-chain sends 4 + 5")

	# docs/replay_format.md claims five fields are enough to view a record. Prove it, by
	# writing a minimal file and loading it -- a contract doc nobody tests is a guess.
	group("ReplayLog: the documented minimal record")
	var blank_row := "."
	blank_row = blank_row.repeat(10)
	var rows: Array = []
	for i in 40:
		rows.append(blank_row)
	# A flat T on the floor: three cells on row 39, the nub on row 38.
	var lines_out: Array = []
	for placement in 2:
		var rec := {
			"replay": "contract_probe",
			"n": placement,
			"board": rows,
			"piece": "t",
			"place": {"type": "t", "cells": [[3, 39], [4, 39], [5, 39], [4, 38]]},
		}
		lines_out.append(JSON.stringify(rec))
	var probe := "user://_contract_probe.jsonl"
	var fh := FileAccess.open(probe, FileAccess.WRITE)
	check(fh != null, "the probe file can be written")
	if fh != null:
		fh.store_string("
".join(lines_out) + "
")
		fh.close()
		var rl := ReplayLog.new(PieceSet.tetrominoes())
		check(rl.load_file(probe), "a file of minimal records loads: %s" % rl.error)
		eq(rl.run_count(), 1, "   and holds one run")
		eq(rl.names[0], "contract_probe", "   named by the replay field")
		var recs := rl.steps_for("contract_probe")
		eq(recs.size(), 2, "   with both records")
		if recs.size() == 2:
			var gp := make_game(Board.new(10, 20, 20))
			gp.start()
			rl.apply(recs[0], gp)
			check(gp.active != null, "   the placed piece is recovered from absolute cells")
			if gp.active != null:
				eq(gp.active.def.id, &"T", "      as a T")
				var got := {}
				for c in gp.active.board_cells():
					got[c] = true
				var want_cells := [Vector2i(3, 39), Vector2i(4, 39), Vector2i(5, 39), Vector2i(4, 38)]
				var all_there := true
				for c in want_cells:
					if not got.has(c):
						all_there = false
				check(all_there, "      at exactly the cells given")
			eq(gp.pieces_placed, 0, "   and n becomes the placement number")
		# Garbage must be uppercase G; lowercase is not a piece letter and reads as empty.
		var grows: Array = rows.duplicate()
		grows[39] = "GGGGGGGGG."
		var g1 := {"replay": "g_probe", "n": 0, "board": grows, "piece": "t",
			"place": {"type": "t", "cells": [[3, 38], [4, 38], [5, 38], [4, 37]]}}
		var fh2 := FileAccess.open(probe, FileAccess.WRITE)
		fh2.store_string(JSON.stringify(g1) + "
")
		fh2.close()
		var rl2 := ReplayLog.new(PieceSet.tetrominoes())
		check(rl2.load_file(probe), "a record with garbage loads")
		var gp2 := make_game(Board.new(10, 20, 20))
		gp2.start()
		rl2.apply(rl2.steps_for("g_probe")[0], gp2)
		eq(gp2.board.get_cell(0, 39), Board.GARBAGE, "uppercase G reads as garbage")
		eq(gp2.board.get_cell(9, 39), 0, "   and the hole stays empty")
		DirAccess.remove_absolute(ProjectSettings.globalize_path(probe))

# ------------------------------------------------------------------------------
# InputMap
# ------------------------------------------------------------------------------

## Keys bound to an action in the InputMap, as physical keycodes.
func _mapped_keys(action: String) -> Array:
	var out: Array = []
	if not InputMap.has_action(action):
		return out
	for e in InputMap.action_get_events(action):
		if e is InputEventKey:
			out.append(e.physical_keycode)
	return out
