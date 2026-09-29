class_name ControlProfiles
extends RefCounted

## Named key bindings. A profile covers the eight game actions only -- the meta keys
## (restart, debug toggles, profile cycling) live in InputReader and are the same
## everywhere, so they can never be shadowed by a profile.
##
## Bindings are physical keycodes: they follow the key's position on the board, so
## WASD stays where your fingers are on AZERTY or Dvorak, and the numpad keys are
## recognised whether or not NumLock is on.

const ACTIONS := [
	"left", "right", "soft_drop", "hard_drop",
	"rotate_ccw", "rotate_cw", "rotate_180", "hold",
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

## "A, D" etc. for the overlay.
static func keys_text(profile: Dictionary, action: String) -> String:
	var keys: Array = profile.get(action, [])
	if keys.is_empty():
		return "-"
	var parts: Array = []
	for k in keys:
		parts.append(OS.get_keycode_string(k))
	return ", ".join(parts)
