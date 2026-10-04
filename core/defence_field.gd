class_name DefenceField
extends RefCounted

## The placement grid. Occupancy only -- enemies and shots move continuously and are not
## in here at all, which is the whole point of "the grid is for placement".
##
## Stores the tower id per cell, 0 for empty, in a flat array indexed y * cols + x, the
## same shape Board uses. Towers are the only thing that occupies a cell, and a tower's
## cells are recorded for every cell of its footprint, so a lookup is O(1) and removal
## does not have to search.

var cols: int
var rows: int
var cells: PackedInt32Array

func _init(c: int, r: int) -> void:
	cols = c
	rows = r
	cells = PackedInt32Array()
	cells.resize(cols * rows)
	clear()

func clear() -> void:
	cells.fill(0)

func in_bounds(x: int, y: int) -> bool:
	return x >= 0 and x < cols and y >= 0 and y < rows

func at(x: int, y: int) -> int:
	if not in_bounds(x, y):
		return -1          # out of bounds reads as occupied-by-nothing-placeable
	return cells[y * cols + x]

func is_free(x: int, y: int) -> bool:
	return in_bounds(x, y) and cells[y * cols + x] == 0

## Can this footprint sit here? Every cell must be inside the grid and empty.
##
## Takes the def and rotation rather than a cell list so the caller cannot accidentally
## test a footprint from one rotation against the extent of another.
func can_place(def: TowerDef, rot: int, origin: Vector2i) -> bool:
	for c in def.cells(rot):
		var p: Vector2i = origin + c
		if not is_free(p.x, p.y):
			return false
	return true

func place(tower: DefenceTower) -> void:
	for p in tower.cells():
		if in_bounds(p.x, p.y):
			cells[p.y * cols + p.x] = tower.id

func remove(tower: DefenceTower) -> void:
	for p in tower.cells():
		if in_bounds(p.x, p.y) and cells[p.y * cols + p.x] == tower.id:
			cells[p.y * cols + p.x] = 0

## Pixel bounds of one cell, relative to the field's top-left.
func cell_box(x: int, y: int, cell_px: int) -> Rect2:
	return Rect2(Vector2(float(x * cell_px), float(y * cell_px)),
		Vector2(float(cell_px), float(cell_px)))

## Pixel size of the whole field.
func pixel_size(cell_px: int) -> Vector2:
	return Vector2(float(cols * cell_px), float(rows * cell_px))

## The cell a pixel position falls in. May be out of bounds; callers check.
func cell_at(p: Vector2, cell_px: int) -> Vector2i:
	return Vector2i(floori(p.x / float(cell_px)), floori(p.y / float(cell_px)))

## Clamps an origin so a footprint of this extent stays fully inside the grid. What the
## build cursor uses, so it can never be walked somewhere a tower could not go.
func clamp_origin(origin: Vector2i, extent: Vector2i) -> Vector2i:
	return Vector2i(
		clampi(origin.x, 0, maxi(0, cols - extent.x)),
		clampi(origin.y, 0, maxi(0, rows - extent.y)))

func occupied_count() -> int:
	var n := 0
	for i in cells.size():
		if cells[i] != 0:
			n += 1
	return n
