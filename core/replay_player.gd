class_name ReplayPlayer
extends RefCounted

## Drives a Game from a parsed TTRReplay and reports where the reconstruction stops
## matching the recorded outcome.
##
## This is a DIAGNOSTIC first and a reconstructor second. The interesting output is
## not "it worked" but the first point of divergence, because every mismatch names a
## concrete difference between our engine and TETR.IO's:
##
##   piece count diverges early  -> the randomiser is wrong (bagtype is not 7-bag)
##   a piece locks by lock_delay -> our timing differs; the human hard-dropped all of them
##   inputs dropped in a frame   -> sub-frame resolution we do not model
##   line count off at the end   -> kicks, spin rules, or garbage handling
##
## Anything that fails validation gets discarded rather than trusted. A usable
## training corpus needs a verified subset, not every replay.

var replay: TTRReplay
var game: Game

## --- report ---
var frames_run: int = 0
var pieces_placed: int = 0
var lines_cleared: int = 0
var locks_by_delay: int = 0          ## should be 0 for an all-hard-drop replay
var dropped_inputs: int = 0          ## edge presses lost to one-per-frame ticking
var topped_out_at: int = -1
var notes: Array = []

func _init(r: TTRReplay, g: Game) -> void:
	replay = r
	game = g

## Runs the whole replay, or until `max_frames`. Returns true if it completed
## without topping out -- which is necessary for a match, never sufficient.
func run(max_frames: int = 0) -> bool:
	var limit: int = replay.frames if max_frames <= 0 else mini(max_frames, replay.frames)
	var held := {}
	var idx := 0
	var input := InputState.new()

	for frame in range(limit + 1):
		input.clear()
		var edges_this_frame := {}
		# Consume every event stamped for this frame, in sub-frame order.
		while idx < replay.inputs.size() and int(replay.inputs[idx]["frame"]) == frame:
			var e: Dictionary = replay.inputs[idx]
			var key: String = e["key"]
			if e["down"]:
				held[key] = true
				# Our tick is whole-frame, so a second press of the same edge action
				# inside one frame has nowhere to go.
				if edges_this_frame.has(key):
					dropped_inputs += 1
				edges_this_frame[key] = true
			else:
				held[key] = false
			idx += 1

		input.left = bool(held.get("left", false))
		input.right = bool(held.get("right", false))
		input.soft_drop = bool(held.get("soft_drop", false))
		input.hard_drop = edges_this_frame.has("hard_drop")
		input.rotate_cw = edges_this_frame.has("rotate_cw")
		input.rotate_ccw = edges_this_frame.has("rotate_ccw")
		input.rotate_180 = edges_this_frame.has("rotate_180")
		input.hold = edges_this_frame.has("hold")

		for ev in game.tick(input):
			if ev.type == GameEvent.Type.PIECE_LOCKED:
				pieces_placed += 1
				if str(ev.data.get("cause", "")) == "lock_delay":
					locks_by_delay += 1
			elif ev.type == GameEvent.Type.TOP_OUT:
				topped_out_at = frame
		frames_run = frame
		if topped_out_at >= 0:
			break

	lines_cleared = game.lines
	_diagnose()
	return topped_out_at < 0

func _diagnose() -> void:
	notes.clear()
	if replay.bagtype != "7-bag":
		notes.append("bagtype is '%s', not 7-bag -- our piece sequence cannot match" % replay.bagtype)
	if topped_out_at >= 0:
		notes.append("topped out at frame %d of %d (%.1f%% through)" % [
			topped_out_at, replay.frames, 100.0 * float(topped_out_at) / maxf(1.0, float(replay.frames))])
	if locks_by_delay > 0:
		notes.append("%d pieces locked by lock delay, but the replay hard-dropped every piece" % locks_by_delay)
	if dropped_inputs > 0:
		notes.append("%d presses lost to sub-frame resolution" % dropped_inputs)
	if pieces_placed != replay.stat_pieces:
		notes.append("pieces %d vs recorded %d" % [pieces_placed, replay.stat_pieces])
	if lines_cleared != replay.stat_lines:
		notes.append("lines %d vs recorded %d" % [lines_cleared, replay.stat_lines])

## True only when the reconstruction reproduced the recorded outcome exactly.
func verified() -> bool:
	return topped_out_at < 0 \
		and pieces_placed == replay.stat_pieces \
		and lines_cleared == replay.stat_lines
