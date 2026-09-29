class_name PaneTab
extends RefCounted

## Which page the right-hand pane shows.
##
## The pane has one job now and another later: today it is the only place the economy is
## visible, and the mechanics readout has been sharing it. Those two audiences are
## different -- GAMEPLAY is what a player reads, DEBUG is what the sim's author reads --
## so they are tabs rather than a pile, and the pane is the minigame's future home.

enum { GAMEPLAY, DEBUG, OFF }

const NAMES := {GAMEPLAY: "GAMEPLAY", DEBUG: "DEBUG", OFF: "OFF"}
## OFF is reachable by cycling, but only after the two real pages, so TAB mostly just
## flips between them.
const ORDER := [GAMEPLAY, DEBUG, OFF]

## Height the tab strip occupies at the top of the pane. Both pages start below it, so
## the strip is never drawn over -- the debug page indents by this too.
const STRIP_H := 32.0

static func name_of(tab: int) -> String:
	return NAMES.get(tab, "?")

static func next(tab: int) -> int:
	var i := ORDER.find(tab)
	return ORDER[(i + 1) % ORDER.size()] if i >= 0 else GAMEPLAY

## The tabs that are actual pages, for drawing the strip. OFF is a state, not a tab.
static func pages() -> Array:
	return [GAMEPLAY, DEBUG]
