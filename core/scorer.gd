class_name Scorer
extends RefCounted

## Turns a ClearResult into points. Subclass and override score_clear() to replace
## the formula wholesale without touching Game.

var table: ScoreTable

func _init(t: ScoreTable = null) -> void:
	table = t if t != null else ScoreTable.new()

func score_clear(r: ClearResult) -> int:
	var base := base_value(r)
	if r.is_difficult() and r.b2b_active():
		base = int(round(float(base) * table.b2b_multiplier))
	if r.perfect_clear:
		if r.lines == 4 and r.b2b_active():
			base += int(table.b2b_tetris_perfect_clear)
		else:
			base += int(table.perfect_clear_scores.get(r.lines, 0))
	var total := base
	if table.scale_with_level:
		total *= maxi(1, r.level)
	if r.combo >= 1:
		total += table.combo_per_line * r.combo * maxi(1, r.level)
	return total

func base_value(r: ClearResult) -> int:
	if r.spin != SpinDetect.Kind.NONE and r.piece_id == &"T":
		if r.spin == SpinDetect.Kind.FULL:
			return int(table.t_spin_scores.get(r.lines, 0))
		return int(table.t_spin_mini_scores.get(r.lines, 0))
	if r.spin != SpinDetect.Kind.NONE:
		if r.lines == 0 and not table.score_non_t_zero_line_spins:
			return 0
		return int(table.other_spin_scores.get(r.lines, 0))
	return int(table.line_scores.get(r.lines, 0))
