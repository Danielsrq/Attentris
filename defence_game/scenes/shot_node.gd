extends Node2D

## The visible half of a DefenceShot. Pooled by the view, so it must survive being hidden
## and reused -- nothing here may accumulate state across lives.
##
## The texture is set on the `Body` Sprite2D in shot.tscn. Size it to
## DefenceRules.projectile_radius * 2.

## On demand, not @onready -- see the note in tower_node.gd.
var _body: Sprite2D

func _resolve() -> void:
	if _body == null:
		_body = get_node_or_null("Body")

## Nothing to do but resolve: the texture is on the Body Sprite2D in shot.tscn. Kept as a
## hook because the pool calls it on every reuse, which is where per-life state would
## belong if any is ever added.
func setup() -> void:
	_resolve()

func apply(shot: DefenceShot) -> void:
	_resolve()
	position = shot.pos
