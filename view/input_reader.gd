class_name InputReader
extends RefCounted

## Reads the keyboard once per physics tick and produces an InputState.
##
## Deliberately bypasses InputMap: edge detection is computed per tick rather than
## per frame, and the sim still only ever sees an InputState -- so nothing about
## replays or headless tests depends on this class existing.
##
## Game actions come from a ControlProfiles profile. Meta keys are fixed so a
## profile can never shadow the restart or debug toggles.

const META := {
	"restart": [KEY_R],
	"toggle_debug": [KEY_F1],
	"toggle_handling": [KEY_F2],
	"toggle_gravity": [KEY_F3],
	"toggle_grid": [KEY_F4],
	"cycle_profile": [KEY_F5],
	"toggle_hints": [KEY_F6],
	"toggle_ghost": [KEY_F7],
	# TAB, not F8: F8 is Godot's "Stop Running Project" and the editor keeps it even while
	# the game has focus, so binding it here just quit the game. TAB also says "tab".
	"cycle_pane": [KEY_TAB],
	"toggle_autoplay": [KEY_F9],
	"pps_down": [KEY_F10],
	"pps_up": [KEY_F11],
}

var profile_name: String = "wasd_numpad"
var profile: Dictionary = {}

var _all: Dictionary = {}
var _cur: Dictionary = {}
var _prev: Dictionary = {}

func _init(starting_profile: String = "wasd_numpad") -> void:
	set_profile(starting_profile)

func set_profile(new_name: String) -> void:
	profile_name = new_name
	profile = ControlProfiles.get_profile(new_name)
	_all = {}
	for action in profile:
		_all[action] = profile[action]
	for action in META:
		_all[action] = META[action]
	# Drop edge history so switching mid-run cannot fire a phantom press.
	_cur = {}
	_prev = {}

func next_profile() -> void:
	var all_names := ControlProfiles.names()
	var i := all_names.find(profile_name)
	set_profile(all_names[(i + 1) % all_names.size()])

## Snapshot the keyboard. Call once at the top of each physics tick.
func poll() -> void:
	_prev = _cur
	_cur = {}
	for action in _all:
		var down := false
		for key in _all[action]:
			# Physical keycodes: position on the board, not the printed label. Keeps
			# WASD under the same fingers on non-QWERTY layouts and makes the numpad
			# work regardless of NumLock.
			if Input.is_physical_key_pressed(key):
				down = true
				break
		_cur[action] = down

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

## Binding list derived from the live profile rather than hardcoded text.
## Returns [label, keys] pairs; an empty label means "draw the value full width".
func binding_text() -> Array:
	var out: Array = []
	for action in ControlProfiles.ACTIONS:
		out.append([ControlProfiles.LABELS[action], ControlProfiles.keys_text(profile, action)])
	out.append(["restart", "R"])
	out.append(["", "F1 blank pane  F2 handling  F3 gravity  F4 grid  F5 profile"])
	out.append(["", "F6 hints  F7 spin assist  TAB pane tab"])
	out.append(["", "F9 autoplay  F10/F11 pps"])
	return out
