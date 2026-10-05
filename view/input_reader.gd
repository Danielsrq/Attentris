class_name InputReader
extends RefCounted

## Reads the keyboard once per physics tick and produces an InputState.
##
## Edge detection is computed per TICK, not per frame. That is the whole reason this
## class exists, and it survived the move to the InputMap: `poll` snapshots
## `Input.is_action_pressed` -- the level query -- into a double buffer and derives
## rising edges itself. `Input.is_action_just_pressed` is deliberately NOT used, because
## it is frame-scoped: a physics frame can run twice or not at all for a given rendered
## frame, so a press read that way can be double-counted or missed entirely. Every
## binding is still a physical keycode, so the sim only ever sees an InputState and
## nothing about replays or headless tests depends on this class.
##
## The move also made this class testable for the first time.
## `Input.is_physical_key_pressed` cannot be faked, so no test could prove `pressed` was
## an edge rather than a level; `Input.action_press`/`action_release` write action state
## directly and work headless, so test_input_map now asserts it.
##
## `is_action_pressed` is left on its default INEXACT match, which is required rather
## than incidental: `hold` sits on KEY_SHIFT in two profiles, so Shift+A has to keep
## moving left. Passing exact_match would make every binding played with a modifier
## held silently stop matching.
##
## Tetris actions come from a ControlProfiles profile; the meta keys come from META below
## and are fixed everywhere, so a profile can never shadow restart or the debug toggles.

const META := {
	"restart": [KEY_R],
	"toggle_debug": [KEY_F1],
	"toggle_handling": [KEY_F2],
	"toggle_gravity": [KEY_F3],
	"toggle_grid": [KEY_F4],
	"cycle_profile": [KEY_F5],
	"toggle_hints": [KEY_F6],
	"toggle_ghost": [KEY_F7],
	"toggle_autoplay": [KEY_F9],
	"pps_down": [KEY_F10],
	"pps_up": [KEY_F11],
	"toggle_pilot": [KEY_F12],
	# B, not another function key: F12 is the last one, and B is free in every play profile
	# (they use WASD, ZXC, the numpad and the arrows).
	"cycle_bot": [KEY_B],
}

var profile_name: String = "wasd_numpad"
var profile: Dictionary = {}

## Every action name this reader polls. A set, not a key table: the keys now live in the
## InputMap, and this is only the list of names to ask about.
var _all: Dictionary = {}
var _cur: Dictionary = {}
var _prev: Dictionary = {}

func _init(starting_profile: String = "wasd_numpad") -> void:
	ControlProfiles.bind_table(META)
	set_profile(starting_profile)

func set_profile(new_name: String) -> void:
	profile_name = new_name
	profile = ControlProfiles.get_profile(new_name)
	ControlProfiles.apply(new_name)
	# Only actions the InputMap actually declares are polled. Everything is declared in
	# project.godot through the editor (see docs/input_map.md), so a name that is not
	# there yet becomes a dead key rather than an error on every tick; `missing()` is how
	# you find out which.
	_all = {}
	for table in [profile, META]:
		for action in table:
			if InputMap.has_action(action):
				_all[action] = true
	for action in ControlProfiles.EXTERNAL:
		if InputMap.has_action(action):
			_all[action] = true
	# Drop edge history so switching mid-run cannot fire a phantom press.
	_cur = {}
	_prev = {}

## Actions expected but not declared in project.godot, for the debug overlay and tests.
func missing_actions() -> Array:
	return ControlProfiles.missing(META)

## The same, for actions that are declared but not needed yet.
func missing_optional_actions() -> Array:
	return ControlProfiles.missing_optional(META)

func next_profile() -> void:
	var all_names := ControlProfiles.names()
	var i := all_names.find(profile_name)
	set_profile(all_names[(i + 1) % all_names.size()])

## Snapshot the keyboard. Call once at the top of each physics tick.
func poll() -> void:
	_prev = _cur
	_cur = {}
	for action in _all:
		# The LEVEL query. See the class doc for why is_action_just_pressed is not used.
		_cur[action] = Input.is_action_pressed(action)

func held(action: String) -> bool:
	return _cur.get(action, false)

func pressed(action: String) -> bool:
	return _cur.get(action, false) and not _prev.get(action, false)

func to_input_state(out: InputState) -> InputState:
	out.left = held("left")
	out.right = held("right")
	out.soft_drop = held("soft_drop")
	out.hard_drop = pressed("hard_drop")
	out.rotate_cw = pressed("rotate_cw")
	out.rotate_ccw = pressed("rotate_ccw")
	out.rotate_180 = pressed("rotate_180")
	out.hold = pressed("hold")
	return out

## Actions currently down, so the overlay can show a binding registering. If a key
## never lights up here, the binding is wrong rather than the sim.
func held_actions() -> Array:
	var out: Array = []
	for action in ControlProfiles.ACTIONS:
		if held(action):
			out.append(action)
	return out

## Binding list read back from the InputMap rather than hardcoded text, so it shows what
## is actually live. Returns [label, keys] pairs; an empty label means "draw the value
## full width".
func binding_text() -> Array:
	var out: Array = []
	for action in ControlProfiles.ACTIONS:
		out.append([ControlProfiles.LABELS[action], ControlProfiles.keys_text(action)])
	out.append(["restart", ControlProfiles.keys_text("restart")])
	out.append(["switch board", ControlProfiles.keys_text("switch_board")])
	out.append(["", "F1 blank pane  F2 handling  F3 gravity  F4 grid  F5 profile"])
	out.append(["", "F6 hints  F7 spin assist"])
	out.append(["", "F9 autoplay  F10/F11 pps"])
	return out
