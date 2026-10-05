class_name ControlProfiles
extends RefCounted

## Named key bindings, and the one place they are pushed into Godot's InputMap.
##
## A profile covers the eight Tetris actions only. The meta keys (restart, debug
## toggles, profile cycling) live in InputReader and do not vary by profile.
##
## Bindings are physical keycodes: they follow the key's position on the board, so
## WASD stays where your fingers are on AZERTY or Dvorak, and the numpad keys are
## recognised whether or not NumLock is on. `bind_table` therefore sets
## `physical_keycode` and leaves `keycode` at 0 -- the same shape as the hand-authored
## `switch_board` action in project.godot. Setting `keycode` instead would silently undo
## both of those properties.
##
## These tables stay the source of truth rather than the InputMap: the project builds
## every resource in code and ships no .tres, runtime profile cycling needs to rewrite
## the bindings anyway, and a project.godot diff per keybind change is worse than this.
## The InputMap is a cache of what is written here, rebuilt by `apply`.
##
## Action names are unprefixed: the Tetris and meta actions were already distinct and are
## referenced by name from main.gd and the tests.

const ACTIONS := [
	"left", "right", "soft_drop", "hard_drop",
	"rotate_ccw", "rotate_cw", "rotate_180", "hold",
]

## The action the user bound by hand in project.godot. Listed so `apply` knows to leave
## it alone -- it is the one action this file does not own.
const EXTERNAL := ["switch_board"]

## Declared but not REQUIRED, so `missing()` does not report them as a gap.
##
## The distinction is whether leaving one out stops you playing or just costs that one key.
## These only cost their key: F2 handling, F3 gravity, F4 grid, F5 profile, F6 hints,
## F7 spin assist, F9 autoplay and F10/F11 its rate.
##
## Deliberately NOT here, because without them something is actually unusable: the eight
## Tetris actions, `restart` (the game can be lost), and `toggle_debug` (how you see
## anything when it misbehaves).
const OPTIONAL := [
	"toggle_handling", "toggle_gravity", "toggle_grid", "cycle_profile",
	"toggle_hints", "toggle_ghost", "toggle_autoplay", "pps_down", "pps_up",
]

## Human-readable labels for the overlay, in display order.
const LABELS := {
	"left": "move left",
	"right": "move right",
	"soft_drop": "soft drop",
	"hard_drop": "hard drop",
	"rotate_ccw": "rotate ccw",
	"rotate_cw": "rotate cw",
	"rotate_180": "rotate 180",
	"hold": "hold",
}

static func names() -> Array:
	return ["wasd_numpad", "competitive", "guideline"]

static func get_profile(profile_name: String) -> Dictionary:
	match profile_name:
		"competitive": return competitive()
		"guideline": return guideline()
		_: return wasd_numpad()

## Left hand on WASD, right hand on the numpad.
## Numpad 8 sits between the two rotation keys, so it falls naturally to 180.
## Numpad 0 is the thumb key, so it takes hold.
static func wasd_numpad() -> Dictionary:
	return {
		"left": [KEY_A],
		"right": [KEY_D],
		"soft_drop": [KEY_W],
		"hard_drop": [KEY_S],
		"rotate_ccw": [KEY_KP_7],
		"rotate_cw": [KEY_KP_9],
		"rotate_180": [KEY_KP_8],
		"hold": [KEY_KP_0, KEY_SHIFT],
	}

## What modern stacking games (TETR.IO, Jstris, Puyo Puyo Tetris) default to.
static func competitive() -> Dictionary:
	return {
		"left": [KEY_LEFT],
		"right": [KEY_RIGHT],
		"soft_drop": [KEY_DOWN],
		"hard_drop": [KEY_SPACE],
		"rotate_ccw": [KEY_Z],
		"rotate_cw": [KEY_X, KEY_UP],
		"rotate_180": [KEY_A],
		"hold": [KEY_C, KEY_SHIFT],
	}

## Official Tetris keyboard layout. Up is hard drop rather than rotate, which is the
## single biggest difference from the competitive layout, and there is no 180.
static func guideline() -> Dictionary:
	return {
		"left": [KEY_LEFT],
		"right": [KEY_RIGHT],
		"soft_drop": [KEY_DOWN],
		"hard_drop": [KEY_UP, KEY_SPACE],
		"rotate_ccw": [KEY_Z, KEY_CTRL],
		"rotate_cw": [KEY_X],
		"rotate_180": [],
		"hold": [KEY_SHIFT, KEY_C],
	}

# ------------------------------------------------------------------------------
# InputMap
# ------------------------------------------------------------------------------

## Rebinds the keys of actions that ALREADY EXIST, replacing whatever they were bound to.
##
## Deliberately never calls `InputMap.add_action`: every action is declared in
## project.godot through the editor's Input Map, because actions created at runtime are
## not reliable. See docs/input_map.md for the table to enter, and `missing()` for what is
## not there yet. An action absent from the map is skipped here and never polled, so a
## half-entered map degrades to dead keys rather than an error every tick.
##
## Rebinding an existing action is a different operation from creating one, and it is the
## one thing that cannot be done from the editor: F5 profile cycling has to rewrite these
## at runtime, so only one of the three profiles could ever be the one in the file.
static func bind_table(table: Dictionary) -> void:
	for action in table:
		if not InputMap.has_action(action):
			continue
		InputMap.action_erase_events(action)
		for key in table[action]:
			var e := InputEventKey.new()
			# Physical, not `keycode`: position on the board rather than printed label.
			# InputEventKey.action_match prefers keycode when it is non-zero, so setting
			# both would quietly reintroduce the layout dependence this avoids.
			e.physical_keycode = key
			InputMap.action_add_event(action, e)

## Binds the named profile. Call on each profile switch.
##
## `switch_board` is untouched -- it is authored in project.godot and owned by the user.
static func apply(profile_name: String) -> void:
	bind_table(get_profile(profile_name))

## Actions this file expects but project.godot has not declared, so a half-entered Input
## Map reports itself as one list instead of as mystery dead keys. Anything in OPTIONAL is
## excluded -- see `missing_optional` for those.
static func missing(meta: Dictionary) -> Array:
	var out: Array = []
	for a in _expected(meta):
		if a in OPTIONAL:
			continue
		if not InputMap.has_action(a):
			out.append(a)
	return out

## The not-needed-yet actions that are also absent. Worth reporting, never worth failing.
static func missing_optional(meta: Dictionary) -> Array:
	var out: Array = []
	for a in OPTIONAL:
		if not InputMap.has_action(a):
			out.append(a)
	return out

## Every action name the game expects, in a stable order.
static func _expected(meta: Dictionary) -> Array:
	var out: Array = []
	for a in ACTIONS:
		out.append(a)
	for a in meta:
		out.append(a)
	for a in EXTERNAL:
		out.append(a)
	return out

## "A, D" etc. for the overlay, read back from the InputMap so it shows what is actually
## live rather than what a table says should be.
static func keys_text(action: String) -> String:
	if not InputMap.has_action(action):
		return "-"
	var parts: Array = []
	for e in InputMap.action_get_events(action):
		if e is InputEventKey:
			parts.append(OS.get_keycode_string(e.physical_keycode))
		elif e is InputEventJoypadButton:
			parts.append("pad %d" % e.button_index)
	if parts.is_empty():
		return "-"
	return ", ".join(parts)
