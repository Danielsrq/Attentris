#!/bin/sh
# Persona / belief experiment under STANDARD rules, 20 games x 500 pieces, same seeds.
B=/c/Users/Admin/Documents/tetris/.claude/worktrees/normal-rules/bot/target/release/bot.exe
W=C:/Users/Admin/Documents/tetris/.claude/worktrees/normal-rules/bot/weights
O=/c/Users/Admin/Documents/tetris/.claude/worktrees/normal-rules/bot/results/belief_normal
mkdir -p "$O"
one() {
    name=$1
    shift
    [ -f "$O/$name.txt" ] && { cat "$O/$name.txt"; return; }
    "$B" play --mode normal --games 20 --pieces 500 --threads ${THREADS:-10} --label "$name" "$@" 2>/dev/null \
        | grep -E "^$name |clears per 100|phantom" > "$O/$name.part" && mv "$O/$name.part" "$O/$name.txt"
    cat "$O/$name.txt"
}
one A_std_weights_knows_rules      --weights "$W/standard.txt"
one B_allspin_weights_knows_rules  --weights "$W/allspin_slots.txt"
one C_allspin_weights_believes_as  --weights "$W/allspin_slots.txt" --believe allspin
one D_std_weights_believes_as      --weights "$W/standard.txt" --believe allspin
