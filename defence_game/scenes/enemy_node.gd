extends Node2D

## The visible half of a DefenceEnemy. Position comes from the sim every frame; the only
## thing this decides is how to show damage, and even that is a tint rather than state.
##
## The texture lives on the `Body` Sprite2D in enemy.tscn, NOT behind an export here. An
## export would mean the sprite you can see in the editor sits empty and only fills in at
## runtime, which is the opposite of what an art-swappable scene should feel like: open the
## scene, click Body, change the texture.
##
## Size it to DefenceRules.enemy_half_extents * 2 so the art matches the box it actually
## collides with -- art larger than the box makes shots look like they pass through.

## Played when the enemy reaches the wall, before the node is released. Stubbed: add an
## animation of this name and it is used.
@export var breach_anim: StringName = &"breach"

## Looked up on demand, not with @onready: a pooled node is set up right after add_child,
## and if the parent is not in the tree yet then _ready has not run.
var _body: Sprite2D
var _anim: AnimationPlayer

var _max_hp: int = 1

func _resolve() -> void:
	if _body != null:
		return
	_body = get_node_or_null("Body")
	_anim = get_node_or_null("Anim")

func setup(enemy: DefenceEnemy) -> void:
	_resolve()
	_max_hp = maxi(1, enemy.hp)
	apply(enemy)

func apply(enemy: DefenceEnemy) -> void:
	_resolve()
	position = enemy.pos
	if _body == null:
		return
	# Darkens as it is worn down, so three hits reads as three hits without a health bar.
	var t := float(enemy.hp) / float(_max_hp)
	_body.modulate = Color(1.0, 0.45 + 0.55 * t, 0.45 + 0.55 * t)

## Plays the breach stub if one exists. Returns the seconds to wait before releasing the
## node, so the view does not have to know whether an animation was supplied.
func play_breach() -> float:
	_resolve()
	if _anim != null and _anim.has_animation(breach_anim):
		_anim.play(breach_anim)
		return _anim.get_animation(breach_anim).length
	return 0.0
