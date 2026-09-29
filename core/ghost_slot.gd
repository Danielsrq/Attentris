class_name GhostSlot
extends RefCounted

## A teaching aid: a phantom column just right of the field that builds the overhang a
## spin into the LAST column needs.
##
## A spin at column width-1 is normally impossible, and not for a scoring reason. The
## pocket a T rotates into spans three columns, so against the right wall the piece
## would occupy the column past the edge and can_place rejects it outright. The same
## is true of the J and Z spins that fit the same pocket. Supplying that column as
## phantom space is the whole mechanism.
##
## What it builds, with the well at column w-1 and the phantom column at w:
##
##     hang      . .  #     covered on the phantom side, so the pocket cannot be
##     pocket   . .  .     entered by dropping -- only by rotating in
##     nub      X .  #     the right wall of the notch
##
## ARMED only when the stack immediately left of the well is FLAT across
## `flat_span` columns. That is deliberate: the assist should appear because the player
## built a real setup, not as a permanent free spin. Flat also means the pocket sits at
## a height the player can reach and read.
##
## The pattern is rebuilt from the surface every time the board changes, so after a
## double clears two rows it simply re-forms two rows lower.

## Set false to disable without tearing the ghost column out of the board.
var enabled: bool = true

var armed: bool = false
## Row of the notch the piece's nub drops into, or -1 when not armed.
var nub_row: int = -1

## Rebuilds the phantom column for the current board. Cheap: three height probes
## and three writes.
func update(board: Board) -> void:
	armed = false
	nub_row = -1
	if board.ghost_width <= 0:
		return
	_clear(board)
	if not enabled:
		return

	var well := board.width - 1               # the open column the nub drops into
	var edge := well - 1                      # one lower than the shelf: the step
	var shelf := edge - 1                     # the pocket row's left wall
	if shelf < 0:
		return

	# The shape is local to three columns: nothing further left has to be flat, because
	# nothing further left touches the pocket.
	#
	# The surface must step down at `edge`. That single cell is what
	# makes this a double rather than a single: the shelf stays filled at the pocket
	# row, so that row completes along with the nub row, and clearing both leaves the
	# stack flat again. A shelf that ran flat all the way to the well would leave the
	# pocket row short and score a T-spin single.
	#
	# Heights are computed here for the few columns that matter rather than taken as
	# an argument, so the assist needs nothing from the solver's surface scan.
	var h_edge := _height(board, edge)
	if h_edge <= 0:
		return
	# The well has to be lower still, or there is no notch for the nub.
	if _height(board, well) >= h_edge:
		return

	# The notch row is the top filled row of the STEP, so the nub rests beside it.
	var nub := board.total_height - h_edge
	var pocket := nub - 1
	var hang := nub - 2
	if hang < 1:
		return

	# The shelf only has to be filled at the POCKET row, so that row can complete. What
	# it does at the hang row is irrelevant: the piece spans edge/well/phantom and
	# never touches the shelf, and the 3-corner test reads the phantom column and the
	# step, not the shelf. So this arms over a wall to the left just as happily as over
	# a one-cell step.
	if board.get_cell(shelf, pocket) == 0:
		return

	# The well itself must be open for the three rows the piece needs.
	for y in [nub, pocket, hang]:
		if board.get_cell(well, y) != 0:
			return

	# The pocket, with the shelf, the step and the well (columns w-3 .. w):
	#
	#     hang      - - - H      overhang: blocks a straight drop into the pocket
	#     pocket   X T T T      the only OPEN cell in the whole phantom column
	#     nub      X X T H      support under the overhang; X is real board
	#
	# Everything else in the column stays WALL, so the piece can turn in here and
	# nowhere else. The two H cells are also what satisfies the 3-corner T-spin test:
	# with the shelf at X they give three filled corners around the T's centre.
	_write(board, hang, Board.GHOST_SHOWN)
	_write(board, pocket, Board.GHOST_OPEN)
	_write(board, nub, Board.GHOST_SHOWN)
	armed = true
	nub_row = nub

## Filled cells above the floor in column x, as a height.
func _height(board: Board, x: int) -> int:
	for y in board.total_height:
		if board.get_cell(x, y) != 0:
			return board.total_height - y
	return 0

## Back to solid. WALL is the default state, so a cleared column is simply the field
## boundary again -- nothing drawn, nothing enterable.
func _clear(board: Board) -> void:
	for i in board.ghost.size():
		board.ghost[i] = Board.GHOST_WALL

func _write(board: Board, y: int, v: int) -> void:
	if y < 0 or y >= board.total_height:
		return
	board.ghost[y * board.ghost_width] = v
