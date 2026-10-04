class_name DefenceEvent
extends RefCounted

## What DefenceSim.tick() returns, mirroring GameEvent: the sim reports what happened and
## the view renders it. Entity lifecycle travels as events so the view can create and
## release scenes; positions do NOT -- those are read straight off the sim each frame.
## Eventing a position would mean one event per shot per tick.

enum Type {
	TOWER_BUILT,     ## data: id, tower
	TOWER_REMOVED,   ## data: id
	SHOT_FIRED,      ## data: id, shot, tower_id, cause ("cooldown" or "line_clear")
	SHOT_BOUNCED,    ## data: id, surface ("wall"/"tower"/"enemy"), pos
	SHOT_EXPIRED,    ## data: id, reason ("bounces" or "life")
	ENEMY_SPAWNED,   ## data: id, enemy
	ENEMY_HIT,       ## data: id, damage, hp_left, pos
	ENEMY_KILLED,    ## data: id, pos
	ENEMY_BREACHED,  ## data: id, pos -- reached the left wall; feeds the garbage buffer
	GARBAGE_FLUSHED, ## data: rows, holes -- hand these to Game.receive_garbage
}

var type: int
var data: Dictionary

func _init(t: int, d: Dictionary = {}) -> void:
	type = t
	data = d

func type_name() -> String:
	return Type.keys()[type]

func _to_string() -> String:
	return "%s %s" % [type_name(), data]
