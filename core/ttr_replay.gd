class_name TTRReplay
extends RefCounted

## Parser for TETR.IO .ttr / .ttrm replay files (plain JSON, no compression).
##
## A replay stores INPUTS, not board states: key events with frame and sub-frame
## timestamps, plus the settings needed to reproduce the game deterministically.
## Reconstructing `(board, placement)` pairs therefore means replaying those inputs
## through an engine that matches theirs -- which is what ReplayPlayer attempts and,
## more importantly, reports on when it fails.
##
## Obtained without any account: TETR.IO's public Tetra Channel API supplies the
## replay id, and a community backend forwards the file itself.

var ok: bool = false
var error: String = ""

var gamemode: String = ""
var version: int = 0
var frames: int = 0

## --- reproduction-critical settings ---
var seed_value: int = 0
## "zenith", "7-bag", etc. NOT always a plain 7-bag, which is the first thing that
## will diverge if it is assumed.
var bagtype: String = ""
var allow180: bool = true
## e.g. "all-mini+" -- a spin-scoring variant, not necessarily what SpinDetect does.
var spinbonuses: String = ""
var has_garbage: bool = false
## Guideline opening rule: the first piece is never S, Z or O.
var no_szo: bool = false

## Handling, in FRAMES as TETR.IO records them (das 5.7 means 5.7 frames, not ms).
var das_frames: float = 10.0
var arr_frames: float = 0.0
var sdf: float = 41.0
var dcd_frames: float = 0.0

## Key events, sorted by (frame, subframe). Each: {frame, subframe, down, key}.
var inputs: Array = []
## In-game events -- garbage arrival and mode plumbing. Not yet interpreted.
var iges: Array = []

## Recorded outcome, the target any reconstruction has to match.
var stat_lines: int = 0
var stat_score: int = 0
var stat_pieces: int = 0
var stat_holds: int = 0
var stat_inputs: int = 0
var stat_topbtb: int = 0

const KEY_MAP := {
	"moveLeft": "left",
	"moveRight": "right",
	"softDrop": "soft_drop",
	"hardDrop": "hard_drop",
	"rotateCW": "rotate_cw",
	"rotateCCW": "rotate_ccw",
	"rotate180": "rotate_180",
	"hold": "hold",
}

static func load_file(path: String) -> TTRReplay:
	var r := TTRReplay.new()
	var f := FileAccess.open(path, FileAccess.READ)
	if f == null:
		r.error = "cannot open %s (%d)" % [path, FileAccess.get_open_error()]
		return r
	var text := f.get_as_text()
	f.close()
	var parsed = JSON.parse_string(text)
	if typeof(parsed) != TYPE_DICTIONARY:
		r.error = "not valid JSON"
		return r
	r._ingest(parsed)
	return r

func _ingest(j: Dictionary) -> void:
	gamemode = str(j.get("gamemode", ""))
	version = int(j.get("version", 0))
	var rep: Dictionary = j.get("replay", {})
	frames = int(rep.get("frames", 0))

	var o: Dictionary = rep.get("options", {})
	seed_value = int(o.get("seed", 0))
	# 40L and Blitz omit bagtype entirely and use the default; Zenith names its own.
	bagtype = str(o.get("bagtype", "7-bag"))
	no_szo = bool(o.get("no_szo", false))
	allow180 = bool(o.get("allow180", true))
	spinbonuses = str(o.get("spinbonuses", ""))
	has_garbage = bool(o.get("hasgarbage", false))
	var h: Dictionary = o.get("handling", {})
	das_frames = float(h.get("das", 10.0))
	arr_frames = float(h.get("arr", 0.0))
	sdf = float(h.get("sdf", 41.0))
	dcd_frames = float(h.get("dcd", 0.0))

	for e in rep.get("events", []):
		var t := str(e.get("type", ""))
		match t:
			"keydown", "keyup":
				var d: Dictionary = e.get("data", {})
				var key := str(d.get("key", ""))
				if not KEY_MAP.has(key):
					continue
				inputs.append({
					"frame": int(e.get("frame", 0)),
					"subframe": float(d.get("subframe", 0.0)),
					"down": t == "keydown",
					"key": KEY_MAP[key],
				})
			"ige":
				iges.append(e)
	inputs.sort_custom(func(a, b):
		if a["frame"] != b["frame"]:
			return a["frame"] < b["frame"]
		return a["subframe"] < b["subframe"])

	var st: Dictionary = rep.get("results", {}).get("stats", {})
	stat_lines = int(st.get("lines", 0))
	stat_score = int(st.get("score", 0))
	stat_pieces = int(st.get("piecesplaced", 0))
	stat_holds = int(st.get("holds", 0))
	stat_inputs = int(st.get("inputs", 0))
	stat_topbtb = int(st.get("topbtb", 0))
	ok = true

## Handling translated into our RuleSet. TETR.IO stores these in frames; we store
## milliseconds. sdf 41 is TETR.IO's "infinite" sentinel.
func to_rules() -> RuleSet:
	var r := RuleSet.competitive()
	r.das_ms = das_frames * 1000.0 / 60.0
	r.arr_ms = arr_frames * 1000.0 / 60.0
	r.dcd_ms = dcd_frames * 1000.0 / 60.0
	r.sdf = 1000.0 if sdf >= 41.0 else sdf
	r.allow_180 = allow180
	r.all_spin = spinbonuses.begins_with("all")
	var excl: Array[String] = []
	if no_szo:
		excl = ["S", "Z", "O"]
	r.first_piece_excludes = excl
	return r

## Counts by key, for checking the parse against the recorded totals before trusting
## anything downstream.
func keydown_histogram() -> Dictionary:
	var out := {}
	for e in inputs:
		if e["down"]:
			out[e["key"]] = int(out.get(e["key"], 0)) + 1
	return out
