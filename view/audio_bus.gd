class_name AudioBus
extends Node

## Turns game events into sound. Reads the same event stream the HUD does and never
## touches the sim, so it cannot desync anything and can be deleted without
## consequence.
##
## TO SWAP A SOUND: select the Audio node in main.tscn and drag a different file
## onto the slot in the "Sound files" group. No code edit. Clearing a slot silences
## that cue. The defaults below are only defaults.
##
## Cue mapping:
##   click            a rotation that landed in a spin position -- "if this clears,
##                    it counts". Follows the active RuleSet, so with all-spin off it
##                    only fires for T-spins.
##   piece_move       any side-to-side movement (coalesced: one per tick, so an
##                    ARR 0 wall slide clicks once, not ten times)
##   piece_drop       every lock
##   line_clear       single, double, triple
##   line_clear_plus  quad, or any spin that cleared a line
##   combo_break      a combo chain ending

const CUE_CLICK := "click"
const CUE_MOVE := "piece_move"
const CUE_DROP := "piece_drop"
const CUE_CLEAR := "line_clear"
const CUE_CLEAR_PLUS := "line_clear_plus"
const CUE_COMBO_BREAK := "combo_break"

## Simultaneous voices. Round-robin, oldest stolen first. A quad landing next to a
## move and a drop needs three at once; 12 is comfortable headroom.
const VOICES := 12

@export_group("Gain (dB)")
## Applied on top of every cue.
@export_range(-32.0, 12.0, 0.5) var master_db: float = 0.0
## Per-cue trim. These attach to the ROLE, not the file, so swapping a sound keeps
## whatever balance you dialled in for that cue.
@export_range(-32.0, 12.0, 0.5) var gain_click: float = 0.0
@export_range(-32.0, 12.0, 0.5) var gain_move: float = 0.0
@export_range(-32.0, 12.0, 0.5) var gain_drop: float = 0.0
@export_range(-32.0, 12.0, 0.5) var gain_clear: float = 0.0
@export_range(-32.0, 12.0, 0.5) var gain_clear_plus: float = 0.0
@export_range(-32.0, 12.0, 0.5) var gain_combo_break: float = -3.0

@export_group("Mix")
@export var muted: bool = false
## Route to a custom audio bus once you add one; "Master" needs no setup.
@export var output_bus: StringName = &"Master"

@export_group("Behaviour")
## Play the drop sound on a placement that also cleared lines. Off makes clears
## cleaner; on keeps the placement tactile.
@export var drop_sound_on_clear: bool = true
## Fire combo_break when a combo chain ends. You did not spec this one, so it is
## easy to switch off.
@export var play_combo_break: bool = true

@export_group("Sound files")
## Rotation that landed in a spin position.
@export var stream_click: AudioStream = preload("res://audio/spin_click.ogg")
## Side-to-side movement.
@export var stream_move: AudioStream = preload("res://audio/piece_move.ogg")
## Piece locked.
@export var stream_drop: AudioStream = preload("res://audio/piece_drop.ogg")
## Single, double, triple.
@export var stream_clear: AudioStream = preload("res://audio/line_clear.ogg")
## Quad, or any spin that cleared.
@export var stream_clear_plus: AudioStream = preload("res://audio/line_clear_plus.ogg")
## Combo chain ending.
@export var stream_combo_break: AudioStream = preload("res://audio/combo_break.ogg")

var _players: Array[AudioStreamPlayer] = []
var _next_voice: int = 0

func _ready() -> void:
	for i in VOICES:
		var p := AudioStreamPlayer.new()
		p.bus = output_bus
		add_child(p)
		_players.append(p)

## The only entry point. Wire it to the same event list the views receive.
func handle_event(e: GameEvent) -> void:
	match e.type:
		GameEvent.Type.PIECE_MOVED:
			play(CUE_MOVE)
		GameEvent.Type.PIECE_ROTATED:
			# Not every rotation -- only one that lands in a spin position.
			if e.data["spin"] != SpinDetect.Kind.NONE:
				play(CUE_CLICK)
		GameEvent.Type.PIECE_LOCKED:
			_on_locked(e)
		_:
			pass

func _on_locked(e: GameEvent) -> void:
	var r: ClearResult = e.data["result"]
	var cleared := r.lines > 0
	if not cleared or drop_sound_on_clear:
		play(CUE_DROP)
	if cleared:
		var special := r.lines >= 4 or r.spin != SpinDetect.Kind.NONE
		play(CUE_CLEAR_PLUS if special else CUE_CLEAR)
	if play_combo_break and e.data.get("combo_broken", false):
		play(CUE_COMBO_BREAK)

## Which file a cue plays. Add a case here only when adding a new cue; swapping an
## existing sound is an inspector change.
func stream_for(cue: String) -> AudioStream:
	match cue:
		CUE_CLICK: return stream_click
		CUE_MOVE: return stream_move
		CUE_DROP: return stream_drop
		CUE_CLEAR: return stream_clear
		CUE_CLEAR_PLUS: return stream_clear_plus
		CUE_COMBO_BREAK: return stream_combo_break
		_: return null

func play(cue: String, pitch: float = 1.0) -> void:
	if muted or _players.is_empty():
		return
	# An empty slot is a deliberate silence, not a mistake -- clearing a stream in the
	# inspector is the intended way to mute one cue.
	var stream := stream_for(cue)
	if stream == null:
		return
	var p := _players[_next_voice]
	_next_voice = (_next_voice + 1) % _players.size()
	p.stream = stream
	p.volume_db = master_db + gain_for(cue)
	p.pitch_scale = pitch
	p.play()

## Per-cue trim in dB. Add a case here only when adding a new cue.
func gain_for(cue: String) -> float:
	match cue:
		CUE_CLICK: return gain_click
		CUE_MOVE: return gain_move
		CUE_DROP: return gain_drop
		CUE_CLEAR: return gain_clear
		CUE_CLEAR_PLUS: return gain_clear_plus
		CUE_COMBO_BREAK: return gain_combo_break
		_: return 0.0

## Runtime trim, for a settings menu later.
func set_cue_gain(cue: String, db: float) -> void:
	match cue:
		CUE_CLICK: gain_click = db
		CUE_MOVE: gain_move = db
		CUE_DROP: gain_drop = db
		CUE_CLEAR: gain_clear = db
		CUE_CLEAR_PLUS: gain_clear_plus = db
		CUE_COMBO_BREAK: gain_combo_break = db
