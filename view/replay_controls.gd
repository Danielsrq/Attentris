@tool
extends CanvasLayer

## On-screen transport for the replay viewer: play/pause, a pacing toggle, a speed
## slider and a scrubber.
##
## The speed slider drives whichever pacing is active, re-ranging itself when the mode
## changes -- pieces per second and a real-time multiplier want completely different
## scales (0.25..20 against 0.05..8), and two permanently half-disabled sliders would
## be worse than one that follows the mode.
##
## Built in code rather than authored in the scene so the layout stays in one place
## next to the logic that keeps it in sync.

const BAR_H := 46.0
const PAD := 10.0
const BG := Color(0.11, 0.11, 0.14, 0.96)
const EDGE := Color(1, 1, 1, 0.16)
const TEXT := Color(0.88, 0.90, 0.96)
const DIM := Color(0.55, 0.58, 0.66)

var viewer: Node

var _bar: Panel
var _play: Button
var _mode: Button
var _speed: HSlider
var _speed_label: Label
var _scrub: Control
var _pos_label: Label
var _runs: OptionButton
var _syncing: bool = false

func setup(v: Node) -> void:
	viewer = v
	if _bar == null:
		_build()
	refresh()

func _build() -> void:
	_bar = Panel.new()
	_bar.name = "Bar"
	var style := StyleBoxFlat.new()
	style.bg_color = BG
	style.border_color = EDGE
	style.border_width_top = 1
	_bar.add_theme_stylebox_override("panel", style)
	# Anchored across the bottom so the bar follows the window rather than assuming
	# the 1280x720 design size.
	_bar.anchor_left = 0.0
	_bar.anchor_right = 1.0
	_bar.anchor_top = 1.0
	_bar.anchor_bottom = 1.0
	_bar.offset_top = -BAR_H
	add_child(_bar)

	var row := HBoxContainer.new()
	row.anchor_right = 1.0
	row.anchor_bottom = 1.0
	row.offset_left = PAD
	row.offset_right = -PAD
	row.offset_top = PAD * 0.5
	row.offset_bottom = -PAD * 0.5
	row.add_theme_constant_override("separation", 10)
	_bar.add_child(row)

	_play = _button("Play", func(): viewer.set_playing(not viewer.playing))
	row.add_child(_play)

	_mode = _button("pps", func(): viewer.toggle_pace())
	_mode.custom_minimum_size = Vector2(96, 0)
	row.add_child(_mode)

	_speed = _slider(190.0, func(v): if not _syncing: viewer.set_speed_value(v))
	# Exponential travel. Both scales run over roughly two orders of magnitude, and
	# the readable end is the bottom of each: linearly, everything between 1 and 4 pps
	# would sit in the first fifth of the track.
	_speed.exp_edit = true
	row.add_child(_speed)

	_speed_label = _label("", TEXT, 116.0)
	row.add_child(_speed_label)

	row.add_child(_sep())

	# A coloured timeline rather than a plain slider: the bands show where the
	# incoming-garbage queue was empty, filling, and overdue, so the desync estimate
	# can be checked by eye.
	_scrub = load("res://view/replay_timeline.gd").new()
	_scrub.scrubbed.connect(func(i): viewer.seek_to(i))
	row.add_child(_scrub)

	_pos_label = _label("", TEXT, 104.0)
	row.add_child(_pos_label)

	row.add_child(_sep())

	row.add_child(_button("<", func(): viewer.step_run(-1)))
	# A dropdown rather than a label: there are 50 runs, and stepping through them with
	# the arrows to find one by id is unusable.
	_runs = OptionButton.new()
	_runs.custom_minimum_size = Vector2(250, 0)
	_runs.focus_mode = Control.FOCUS_NONE
	_runs.fit_to_longest_item = false
	_runs.item_selected.connect(func(i): if not _syncing: viewer.open_run_index(i))
	row.add_child(_runs)
	row.add_child(_button(">", func(): viewer.step_run(1)))

## Pulls every widget back into line with the viewer's state. Called whenever that
## state changes, including from the keyboard, so the bar never disagrees with the keys.
func refresh() -> void:
	if viewer == null or _bar == null:
		return
	_syncing = true                       # setting .value would echo back as an edit
	_play.text = "Pause" if viewer.playing else "Play"
	_mode.text = "pps" if viewer.is_pps() else "real time"

	var r: Vector3 = viewer.speed_range()
	_speed.min_value = r.x
	_speed.max_value = r.y
	# r.z is the KEYBOARD nudge, which wants to be coarse and predictable. Dragging
	# wants the opposite, and on an exponential track a coarse step would quantise the
	# low end into a handful of positions.
	_speed.step = 0.01
	_speed.set_value_no_signal(viewer.speed_value())
	_speed_label.text = viewer.speed_text()

	var total: int = viewer.steps.size()
	var d: int = viewer.desync_index()
	_scrub.set_marks(viewer.timeline_marks(), d, viewer.garbage_marks())
	_scrub.set_index(viewer.index)
	_pos_label.text = "%d / %d" % [viewer.index + 1, total]
	if d >= 0:
		_pos_label.text += "
breaks at %d" % (d + 1)

	# Populated once the dataset has loaded, which happens after setup().
	if _runs.item_count != viewer.run_count():
		_runs.clear()
		for label in viewer.run_labels():
			_runs.add_item(label)
	if _runs.item_count > 0 and _runs.selected != viewer.run_index:
		_runs.select(viewer.run_index)

	var theirs: float = viewer.their_pps()
	if theirs > 0.0:
		_speed_label.text += "   (them %.1f)" % theirs
	_syncing = false

func _button(text: String, on_press: Callable) -> Button:
	var b := Button.new()
	b.text = text
	# Without this the buttons take focus and eat the arrow keys the viewer uses for
	# stepping.
	b.focus_mode = Control.FOCUS_NONE
	b.pressed.connect(on_press)
	return b

func _slider(width: float, on_change: Callable) -> HSlider:
	var s := HSlider.new()
	s.custom_minimum_size = Vector2(width, 0)
	s.size_flags_vertical = Control.SIZE_SHRINK_CENTER
	s.focus_mode = Control.FOCUS_NONE
	s.value_changed.connect(on_change)
	return s

func _label(text: String, col: Color, width: float) -> Label:
	var l := Label.new()
	l.text = text
	l.custom_minimum_size = Vector2(width, 0)
	l.vertical_alignment = VERTICAL_ALIGNMENT_CENTER
	l.add_theme_color_override("font_color", col)
	l.add_theme_font_size_override("font_size", 12)
	return l

func _sep() -> Control:
	var c := Control.new()
	c.custom_minimum_size = Vector2(6, 0)
	return c
