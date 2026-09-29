class_name ScoreTable
extends Resource

## Pure scoring data. Changing the scoring system means editing this resource; to
## replace the *formula* (e.g. a TETR.IO-style attack table for the resource
## economy) subclass Scorer instead.
##
## Dictionary keys are line counts.

@export var line_scores: Dictionary = {1: 100, 2: 300, 3: 500, 4: 800}
@export var t_spin_scores: Dictionary = {0: 400, 1: 800, 2: 1200, 3: 1600}
@export var t_spin_mini_scores: Dictionary = {0: 100, 1: 200, 2: 400}
## All-spin values for S/Z/L/J/I. Not a guideline concept, so these are a starting point.
@export var other_spin_scores: Dictionary = {0: 100, 1: 200, 2: 400, 3: 600, 4: 800}
## With all-spin on, a non-T piece is frequently immobile after an incidental
## rotation. Paying for those would let a player farm points by wiggling pieces, so
## by default non-T spins only score when they clear a line. T-spins are unaffected:
## a T-Spin Zero is worth 400 either way, per the guideline.
@export var score_non_t_zero_line_spins: bool = false
@export var b2b_multiplier: float = 1.5
@export var combo_per_line: int = 50
@export var perfect_clear_scores: Dictionary = {1: 800, 2: 1200, 3: 1800, 4: 2000}
@export var b2b_tetris_perfect_clear: int = 3200
@export var soft_drop_per_cell: int = 1
@export var hard_drop_per_cell: int = 2
@export var scale_with_level: bool = true
