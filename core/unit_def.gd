class_name UnitDef
extends RefCounted

## One kind of fighter: which sheets it animates from, and its numbers.
##
## Built by static factory like every other definition here (RuleSet.normal,
## PieceSet.tetrominoes), and the sheets are `preload`ed rather than named by an exported
## path -- a mistyped path is a runtime nil with no autocomplete and no parse-time check,
## while a bad preload fails when the file is parsed and stays visible to the exporter.
##
## FRAME COUNTS ARE DECLARED, not detected. Detecting would mean decoding a PNG on the CPU
## at startup to recover a constant of the asset. They are instead measured once and pinned
## by a test that reads the source PNGs and checks the first empty frame lands exactly where
## the table says -- so a re-export from the LPC generator cannot silently start playing
## blank frames. LPC sheets are 13 columns wide but almost no animation uses 13; walk uses
## 9, and playing 13 shows four blank frames as a stutter at the end of every cycle.

const TEAL := 0
const RED := 1

const TEAL_DIR := "res://assets/character_sprites/lpc_male_pumpkin_teal_staff/"
const RED_DIR := "res://assets/character_sprites/lpc_female_human_red_rapier/"

const TEAL_WALK := preload(TEAL_DIR + "standard/walk.png")
const TEAL_ATTACK := preload(TEAL_DIR + "custom/thrust_oversize.png")
const TEAL_DEATH := preload(TEAL_DIR + "standard/hurt.png")
const RED_WALK := preload(RED_DIR + "standard/walk.png")
const RED_ATTACK := preload(RED_DIR + "custom/slash_oversize.png")
const RED_DEATH := preload(RED_DIR + "standard/hurt.png")

var id: StringName = &""
var team: int = TEAL
## Which way it marches, and therefore which LPC row it faces along.
var facing: int = SpriteSheet.RIGHT
## -1 or +1 along x.
var heading: int = 1

var hp: int = 30
var damage: int = 10
## Seconds a given ATTACKER must wait before it can hurt the same target again. Per
## (attacker, target) pair, not per target -- see HitLedger.
var iframe_sec: float = 0.3
## Pixels of reach in front, and how far ahead it looks for something to fight. Sight is
## the longer of the two so a unit stops just outside its own reach rather than inside it.
var reach_px: float = 34.0
var sight_px: float = 44.0

## clip -> {sheet, frame_px, frames, loop, contact}
##
## `contact` is the frame on which the hurtbox opens, and it must never be 0: calling play()
## on a clip already sitting at frame 0 emits no frame_changed, so a contact frame of 0
## silently misses its first swing.
var clips: Dictionary = {}

## The attack clip is the OVERSIZE sheet, at 192px.
##
## Measured, because it is not obvious: the 192 frame is the 64 frame padded by exactly 64px
## on every side -- same body, same height -- so scaling it down to 64 would shrink the
## character to a third rather than fitting it. Used at native size it costs nothing and
## draws more of the weapon: teal's staff is 40px of art here against 27px in the standard
## sheet.
##
## Their overhang past the nominal 64px frame also differs, and only one of them needs room:
##   thrust_oversize (teal)  0px every side -- safe anywhere in the lane
##   slash_oversize  (red)   49px left and right, 17px down
## So red's swing can reach ~11px outside a pane with 38px of slack, at the very edge. That
## is cosmetic; clip the lane if it shows.
static func teal() -> UnitDef:
	var d := UnitDef.new()
	d.id = &"teal"
	d.team = TEAL
	d.facing = SpriteSheet.RIGHT
	d.heading = 1
	d.clips = {
		&"walk": {"sheet": TEAL_WALK, "frame_px": 64, "frames": 9, "loop": true, "contact": -1},
		&"attack": {"sheet": TEAL_ATTACK, "frame_px": 192, "frames": 8, "loop": true, "contact": 5},
		&"death": {"sheet": TEAL_DEATH, "frame_px": 64, "frames": 6, "loop": false, "contact": -1},
	}
	return d

static func red() -> UnitDef:
	var d := UnitDef.new()
	d.id = &"red"
	d.team = RED
	d.facing = SpriteSheet.LEFT
	d.heading = -1
	d.clips = {
		&"walk": {"sheet": RED_WALK, "frame_px": 64, "frames": 9, "loop": true, "contact": -1},
		&"attack": {"sheet": RED_ATTACK, "frame_px": 192, "frames": 6, "loop": true, "contact": 3},
		&"death": {"sheet": RED_DEATH, "frame_px": 64, "frames": 6, "loop": false, "contact": -1},
	}
	return d

static func of(team_id: int) -> UnitDef:
	return red() if team_id == RED else teal()

## Physics layer bit this team's HURTBOX sits on. The hurtbox is the only thing that ever
## occupies a layer, which is what lets one shape disable make a corpse untargetable and
## non-blocking at once.
func own_layer() -> int:
	return 1 if team == TEAL else 2

func enemy_layer() -> int:
	return 2 if team == TEAL else 1

func contact_frame(clip: StringName) -> int:
	return int(clips.get(clip, {}).get("contact", -1))

func frame_count(clip: StringName) -> int:
	return int(clips.get(clip, {}).get("frames", 0))
