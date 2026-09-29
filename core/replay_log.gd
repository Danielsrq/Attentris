class_name ReplayLog
extends RefCounted

## Reader for the harvested TETR.IO Zenith dataset (data/zenith_clean.jsonl).
##
## One JSON object per line: the board as it stood when a piece spawned, the piece,
## the next queue, hold, and where that piece finished. Because every sample carries
## its own board, VIEWING a replay needs no simulation at all -- step n renders
## board(n) with the piece drawn at its landing cells, and step n+1's board is already
## the post-clear result. Nothing can drift, because nothing is being derived.
##
## Why the dataset stops partway through a run: Zenith omits most of the garbage a
## player receives (the tower's environmental garbage is generated client-side and is
## absent from the file), so a full reconstruction desyncs the moment garbage lands.
## Samples cover the clean climb before that, and carry a `tier`:
##   "strict"   -- certainly garbage-free: cut at the first garbage the sim inserts
##   "extended" -- garbage-free on the evidence of the board itself
##
## The extended cut ends a run when sealed holes reach 8 or the stack passes 17. Holes
## are the load-bearing half: deliberate upstacking is tall and CLEAN, while a
## reconstruction that has lost sync is hole-rich, because the recorded inputs are
## aimed at rows that do not exist on our board. They must be SEALED rather than merely
## covered -- an overhang is what an all-spin player builds on purpose.
##
## Neither threshold is calibrated, because nothing in the replay says when the
## unrecorded garbage landed: the garbage that IS recorded never exceeds these players'
## attack output at any moment, so it never forces a landing. Every sample therefore
## carries its own `sealed` and `height`, so the set can be re-cut without re-harvesting.
##
## Loading is deliberately lazy. The file holds ~12k samples across ~50 runs, and a
## viewer shows one run at a time, so lines are kept as TEXT and only the selected
## run's lines are handed to the JSON parser.

## The full reconstruction, to the topout. Training subsets are cut from it with
## tools/cut.mjs; the viewer loads everything so the divergence can be watched.
const DEFAULT_PATH := "res://data/zenith_full.jsonl"

## Dataset piece letters -> our PieceSet ids.
const LETTERS := {
	"i": &"I", "o": &"O", "t": &"T", "s": &"S", "z": &"Z", "j": &"J", "l": &"L",
}
## The dataset marks garbage cells 'G'.
const GARBAGE_CHAR := "G"
## Bot replays mark wound-line cells 'W' (all-spin repeat penalty).
const WOUND_CHAR := "W"

var piece_set: PieceSet
var error: String = ""

## Run names in the order they first appear.
var names: Array[String] = []
## name -> {user, mods, samples, strict, extended}
var meta: Dictionary = {}

var _lines: Array[String] = []
## name -> Array[int] of line numbers. Deliberately an Array and not a
## PackedInt32Array: packed arrays are value types, so `_index[name].append(i)`
## can mutate a temporary copy instead of the stored one.
var _index: Dictionary = {}

func _init(ps: PieceSet) -> void:
	piece_set = ps

## Reads the file and indexes it by run. Returns false and sets `error` on failure.
func load_file(path: String = DEFAULT_PATH) -> bool:
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		error = "cannot open %s (error %d)" % [path, FileAccess.get_open_error()]
		return false
	names.clear()
	meta.clear()
	_index.clear()
	_lines.clear()
	# One read and a split, not get_line() per line: FileAccess.get_line() walks the
	# file a byte at a time, which costs minutes on a file this size rather than
	# milliseconds.
	var text := f.get_as_text()
	f.close()
	for line in text.split("\n", false):
		if line.length() < 2:
			continue
		# The run name is pulled out by slicing rather than parsing: at ~12k lines,
		# parsing every record just to group them is the whole load time.
		var name := _field(line, "replay")
		if name == "":
			continue
		var i := _lines.size()
		_lines.append(line)
		if not _index.has(name):
			_index[name] = ([] as Array[int])
			names.append(name)
			meta[name] = {
				"user": _field(line, "user"), "mods": _field(line, "mods"),
				"samples": 0, "strict": 0, "extended": 0,
			}
		_index[name].append(i)
		var m: Dictionary = meta[name]
		m["samples"] = int(m["samples"]) + 1
		var tier := _field(line, "tier")
		m[tier] = int(m.get(tier, 0)) + 1
	if names.is_empty():
		error = "%s held no usable samples" % path
		return false
	return true

## Cheap extraction of a short string field, avoiding a full parse per line.
func _field(line: String, key: String) -> String:
	var needle := '"%s":"' % key
	var at := line.find(needle)
	if at < 0:
		return ""
	var from := at + needle.length()
	var to := line.find('"', from)
	return line.substr(from, to - from) if to > from else ""

func run_count() -> int:
	return names.size()

## Total samples across every run in the file.
func total_samples() -> int:
	return _lines.size()

func sample_count(name: String) -> int:
	return _index[name].size() if _index.has(name) else 0

## Parses one run's samples. Called once per run selection, not per frame.
func steps_for(name: String) -> Array:
	var out: Array = []
	if not _index.has(name):
		return out
	for i in _index[name]:
		var parsed = JSON.parse_string(_lines[i])
		if typeof(parsed) == TYPE_DICTIONARY:
			out.append(parsed)
	return out

## Writes one sample into a Game so the ordinary BoardView and HUD can draw it.
## The board, queue and hold come straight from the record; the piece is placed at
## its recorded landing cells.
func apply(step: Dictionary, game: Game) -> void:
	var board := game.board
	board.clear()
	var rows: Array = step.get("board", [])
	var blank := ".".repeat(board.width)
	for y in mini(rows.size(), board.total_height):
		var row: String = rows[y]
		if row == blank:
			continue
		for x in mini(row.length(), board.width):
			var ch := row[x]
			if ch == ".":
				continue
			if ch == WOUND_CHAR:
				board.set_cell(x, y, Board.WOUND)
			else:
				board.set_cell(x, y, Board.GARBAGE if ch == GARBAGE_CHAR else _value(ch))
	# Wound lines, oldest first, as the bottom block. Their parked contents are not in
	# the record (the viewer never heals them itself), so they park as empty rows.
	for wd in step.get("wounds", []):
		var parked := PackedByteArray()
		parked.resize(board.width)
		var parked_owner := PackedInt32Array()
		parked_owner.resize(board.width)
		board.wounds.append({"left": int(wd.get("left", 0)), "cells": parked, "owner": parked_owner})

	# LOCK-time queue and hold, not the sample's spawn-time ones.
	#
	# A sample's `queue`/`hold` are the state when the piece SPAWNED, which is what a
	# learner needs: board + piece + queue + hold -> action. But the board here is
	# drawn with the piece at its LANDING position, and a hold swap happens between
	# those two moments -- the held piece has moved into the hold slot and, if that
	# slot was empty, the swap consumed a queue entry. Drawing the placed piece beside
	# the spawn-time queue puts two different instants on screen at once, which shows
	# up as the piece on the board also sitting in NEXT.
	game.next_queue.clear()
	# The dataset records six upcoming pieces; the NEXT panel has room for five.
	var shown: int = game.rules.next_count if game.rules != null else 5
	for letter in step.get("lock_queue", step.get("queue", [])):
		if game.next_queue.size() >= shown:
			break
		var d := _def(letter)
		if d != null:
			game.next_queue.append(d)
	game.hold_def = _def(step.get("lock_hold", step.get("hold", null)))
	# Dimmed when hold has already been spent on this piece.
	game.hold_used = bool(step.get("lock_hold_locked", false))

	game.active = piece_for(step)
	game.pieces_placed = int(step.get("n", 0))
	game.topped_out = false

## The placed piece as an ActivePiece, or null if the cells cannot be matched.
func piece_for(step: Dictionary) -> ActivePiece:
	var place: Dictionary = step.get("place", {})
	var d := _def(place.get("type", null))
	if d == null:
		return null
	var abs_cells: Array = []
	for c in place.get("cells", []):
		abs_cells.append(Vector2i(int(c[0]), int(c[1])))
	if abs_cells.is_empty():
		return null
	return _match_rotation(d, abs_cells)

## Recovers (rot, pos) from absolute cells.
##
## The dataset's own (x, y, r) is pivot-relative and specific to the exporting
## engine's rotation tables, so translating it would mean committing to that
## convention. The absolute cells are convention-free, so instead we find which of
## our four rotations reproduces them and where. A rotation matches when its offsets,
## shifted by the difference of the two bounding-box corners, give the same cell set.
func _match_rotation(d: PieceDef, abs_cells: Array) -> ActivePiece:
	var want := {}
	var want_min := Vector2i(1 << 30, 1 << 30)
	for c in abs_cells:
		want[c] = true
		want_min.x = mini(want_min.x, c.x)
		want_min.y = mini(want_min.y, c.y)
	for rot in d.rotations.size():
		var offs: Array = d.rotations[rot]
		if offs.size() != abs_cells.size():
			continue
		var off_min := Vector2i(1 << 30, 1 << 30)
		for o in offs:
			off_min.x = mini(off_min.x, o.x)
			off_min.y = mini(off_min.y, o.y)
		var pos: Vector2i = want_min - off_min
		var hit := true
		for o in offs:
			if not want.has(pos + o):
				hit = false
				break
		if hit:
			var p := ActivePiece.new()
			p.def = d
			p.rot = rot
			p.pos = pos
			return p
	return null

func _def(letter) -> PieceDef:
	if typeof(letter) != TYPE_STRING or letter == "":
		return null
	var id = LETTERS.get(String(letter).to_lower(), null)
	return piece_set.get_def(id) if id != null else null

func _value(letter: String) -> int:
	var d := _def(letter)
	return d.index + 1 if d != null else 0
