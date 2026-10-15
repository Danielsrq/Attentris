# Attentris repo: instructions for Claude Code

Start here: read `attentris/NOTES.md`. It has the project, architecture, data, model lineage, benchmark protocol, latest results, working rules and the next plan. Then:
- `SETUP.md`: machine setup (Windows / Ubuntu), GitHub, data sync through Google Drive (`attentris/data_bundle.py`);
- `attentris/DESIGN.md`: tokenizer and model spec.

Rules the user has set (details and reasons in NOTES.md, "Working rules"):
- Never add Claude co-author / attribution trailers to commits. The user pushes.
- Runs are resumable and show progress. Say how many games and how long before large runs. Never restart a killed job without asking.
- Pin compute to P-cores (`pcores.pin()` in Python; the Rust bot pins itself). GPU only for training, one run at a time.
- Play tests: 64 games per setting; depth 1, depth 1 with garbage (80/20, 50/50), depth 2 beam 4. No depth 3+ unless asked.
- New run scripts source `attentris/plat.sh` (`ATT="$DIR"; . "$ATT/plat.sh"`) and use `$EXE`, `$VPY`, `$CPUPY`, `wpath`, so they work on Windows (Git Bash) and Linux.
- Models and datasets are not in git. After producing new ones, remind the user to run `python attentris/data_bundle.py pack --delta --out <bundle>` and upload the delta zip.
