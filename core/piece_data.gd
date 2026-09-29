class_name PieceData
extends RefCounted

## Piece specs as ASCII art inside an NxN bounding box. This is the only place
## shapes and colours are declared; PieceSet compiles these into PieceDefs.
##
## Adding trominoes or pentominoes later means adding a function here plus a kick
## class in KickTables -- no changes to the board, randomizer, renderer or rules.

## Tetris Guideline colours and spawn orientations.
## I and O get their own boxes (4 and 2) so the box-rotation primitive lands them
## correctly; the other five are the standard 3x3 SRS boxes.
static func tetrominoes() -> Dictionary:
	return {
		&"I": {"box": 4, "rows": ["....", "XXXX", "....", "...."], "color": Color8(0, 240, 240), "kick": &"i"},
		&"O": {"box": 2, "rows": ["XX", "XX"], "color": Color8(240, 240, 0), "kick": &"o"},
		&"T": {"box": 3, "rows": [".X.", "XXX", "..."], "color": Color8(160, 0, 240), "kick": &"jlstz"},
		&"S": {"box": 3, "rows": [".XX", "XX.", "..."], "color": Color8(0, 240, 0), "kick": &"jlstz"},
		&"Z": {"box": 3, "rows": ["XX.", ".XX", "..."], "color": Color8(240, 0, 0), "kick": &"jlstz"},
		&"J": {"box": 3, "rows": ["X..", "XXX", "..."], "color": Color8(0, 0, 240), "kick": &"jlstz"},
		&"L": {"box": 3, "rows": ["..X", "XXX", "..."], "color": Color8(240, 160, 0), "kick": &"jlstz"},
	}
