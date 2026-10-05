class_name SpriteSheet
extends RefCounted

## Builds a SpriteFrames from an LPC Universal spritesheet, in code.
##
## Built rather than authored because the project ships no .tres and defines every resource
## in code -- see ControlProfiles. The alternative, hand-keyed Animation resources in a
## .tscn, would mean authoring four facings per clip per character and would make the
## animation system the single most resource-file-shaped thing in a codebase that has none.
##
## LPC layout: frames are square, laid out in a grid, and the ROWS are facings in the order
## up, left, down, right. Columns are the animation's frames.
##
## `cols` and `rows` are always DERIVED from the texture size, never declared. One rule then
## covers three irregularities in the same art set:
##   hurt.png / climb.png   832x64  -> rows = 1, so every facing resolves to row 0
##   thrust_oversize.png   1536x768 -> 8 columns at 192px
##   slash_oversize.png    1152x768 -> 6 columns at 192px   <-- a DIFFERENT width
## Those last two were assumed to match and do not. Declaring a shared column count would
## have been wrong for one of the two characters, silently.

## Facings, in LPC row order.
const UP := 0
const LEFT := 1
const DOWN := 2
const RIGHT := 3

## Built SpriteFrames, keyed by unit id. A SpriteFrames is a Resource and each
## AnimatedSprite2D keeps its own frame counter, so one build is safely shared by every
## unit of a kind -- where building per unit would mean ~24 AtlasTexture allocations on a
## spawn that can fire several times a second.
static var _cache: Dictionary = {}

## Columns and rows a sheet divides into. The whole reason nothing declares them.
static func grid(size: Vector2i, frame_px: int) -> Vector2i:
	if frame_px <= 0:
		return Vector2i.ZERO
	return Vector2i(size.x / frame_px, size.y / frame_px)

## The source rect of one frame.
static func region(frame_px: int, col: int, row: int) -> Rect2i:
	return Rect2i(col * frame_px, row * frame_px, frame_px, frame_px)

## Which row holds a facing. A single-row sheet has no facings, so everything reads row 0 --
## expressed as the general case rather than as a special case for hurt.png.
static func facing_row(facing: int, rows: int) -> int:
	if rows <= 1:
		return 0
	return clampi(facing, 0, rows - 1)

## Builds every clip a UnitDef declares, for every facing it needs.
##
## Clip names are "<clip>_<facing>", e.g. "walk_3", so the unit can ask for a clip and a
## direction independently. Single-row sheets get the same frames under every facing name,
## which costs a few AtlasTextures and saves every caller a special case.
static func build(def: UnitDef, frame_seconds: float) -> SpriteFrames:
	if _cache.has(def.id):
		return _cache[def.id]
	var sf := SpriteFrames.new()
	sf.remove_animation(&"default")
	for clip in def.clips:
		var spec: Dictionary = def.clips[clip]
		var tex: Texture2D = spec["sheet"]
		var frame_px: int = spec["frame_px"]
		var count: int = spec["frames"]
		var g := grid(Vector2i(tex.get_size()), frame_px)
		for facing in [LEFT, RIGHT, UP, DOWN]:
			var name := StringName("%s_%d" % [clip, facing])
			sf.add_animation(name)
			sf.set_animation_loop(name, bool(spec["loop"]))
			sf.set_animation_speed(name, 1.0 / maxf(0.01, frame_seconds))
			var row := facing_row(facing, g.y)
			for col in mini(count, g.x):
				var at := AtlasTexture.new()
				at.atlas = tex
				at.region = Rect2(region(frame_px, col, row))
				sf.add_frame(name, at)
	_cache[def.id] = sf
	return sf

## Retimes every clip without rebuilding a single AtlasTexture, so frame_seconds can be a
## live inspector knob rather than something that costs a rebuild.
static func apply_speed(frames: SpriteFrames, frame_seconds: float) -> void:
	if frames == null:
		return
	var fps := 1.0 / maxf(0.01, frame_seconds)
	for name in frames.get_animation_names():
		frames.set_animation_speed(name, fps)

## Drops the shared cache. Only for tests, which build defs repeatedly.
static func clear_cache() -> void:
	_cache.clear()
