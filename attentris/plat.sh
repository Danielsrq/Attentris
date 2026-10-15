# Windows (Git Bash) vs Linux, sourced by the run scripts right after DIR=... (ATT = this directory):
#   EXE    suffix of native programs: "$ROOT/bot/target/release/bot$EXE"
#   VPY    python inside a venv: "$DIR/.venv/$VPY"
#   CPUPY  CPU python for play / versus: the TetrisBattle venv when it exists (Windows setup),
#          else the AtTenTris venv (a Linux training machine has no TetrisBattle copy)
#   wpath  a path as native programs want it: C:/... under Git Bash, unchanged on Linux
if command -v cygpath >/dev/null 2>&1; then
    EXE=.exe
    VPY=Scripts/python.exe
    wpath() { cygpath -m "$1"; }
else
    EXE=
    VPY=bin/python
    wpath() { printf '%s\n' "$1"; }
fi
CPUPY="$ATT/../external/tetrisbattle/.venv/$VPY"
[ -x "$CPUPY" ] || CPUPY="$ATT/.venv/$VPY"
