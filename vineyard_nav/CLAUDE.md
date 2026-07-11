# Claude Code Working Rules for vineyard_nav

## Project context
MSc dissertation project. Three-arm controlled comparison for vineyard in-row navigation:
- Phase A: U-Net binary (SMP + ImageNet ResNet-34)
- Phase B: YOLOv11-seg binary
- Phase C: YOLOv11-seg multiclass + downstream config sweep

Read these before making design decisions:
- docs/STATUS.md — current progress and immediate next task
- docs/PROJECT_PLAN.md — full scope and three-arm design
- docs/DECISIONS.md — every locked decision with rationale
- docs/PHASE_A_SPEC.md — Phase A implementation contract
- docs/PHASE_B_SPEC.md — Phase B implementation contract
- docs/PHASE_C_SPEC.md — Phase C implementation contract

## Working rules (LOCKED — do not deviate without explicit approval)

1. Do not open or re-litigate decisions marked LOCKED in docs/DECISIONS.md. If you believe a locked decision should change, raise it as a discussion point, do not act on it.
2. Every claim in code comments and docstrings must be defensible against a marker's questioning. No hand-waving.
3. No hallucinated APIs, library functions, or citations. Verify before writing.
4. Follow the phase-spec implementation order exactly. Do not skip ahead.
5. Do not evaluate on the test set until a phase's training is complete and best checkpoint is locked. Test set is evaluated ONCE per phase.
6. Do not commit to a directional finding (e.g. "multiclass improves...") before Phase C results are in. All framing is neutral until Results.
7. Reproducibility is non-negotiable: seed 42 everywhere, git commit hash saved with every checkpoint, versions pinned in requirements.txt.
8. Working rule addition to CLAUDE.md: When creating or overwriting text files (requirements.txt, YAML configs, .py files), use dedicated file-writing tools rather than shell heredocs. If a heredoc is used, always immediately verify the file's contents with cat <file> or head -n 3 <file> before proceeding, and check that the first line is not cat > ... and the last line is not a bare EOF.

## Coding conventions

- Python 3.10, PyTorch 2.11 with AMP for U-Net training
- Devcontainer has system + venv Python overlay — use `pip install --upgrade <pkg>` when adding new packages
- All experiments go under results/runs/<experiment_name>_<timestamp>/
- Config files in configs/ as YAML, one per experiment
- Never hardcode paths; take them from config
- Prefer explicit over clever

## Workflow

1. State what you're about to do before doing it (one sentence)
2. Do the smallest useful chunk of work
3. Show the result and confirm before continuing
4. Do not batch large multi-file changes without approval

## Environment quirks

- L-CAS ROS2 Humble devcontainer
- GPU: RTX 5050 Laptop, 8GB VRAM, Blackwell sm_120 — requires PyTorch 2.11+
- Working directory: /workspaces/dissertation/vineyard_nav/
- Dataset: /workspaces/dissertation/SemanticBLT.v1-2024-june.coco-segmentation/
- Ignore harmless warnings: conan PyYAML, Axes3D import, grpcio-tools protobuf

## Git operations

Do not run any git operations. Edosa handles all commits and pushes manually.

Never run: `git add`, `git commit`, `git push`, `git checkout`, `git reset`, `git rebase`, `git merge`, `git stash`, `git branch`, `git tag`, or any other command that modifies the working tree or repository state.

You may run read-only git commands when they help diagnose something: `git status`, `git log`, `git diff`, `git show`, `git branch --show-current`, `git remote -v`.

When you complete a substantial unit of work — a working module, a passing test suite, a locked decision — mention it in your response so Edosa can decide whether to commit. Do not propose a commit message unless asked.

**No attribution trailers.** Never add Co-Authored-By trailers to commit messages. Never add Co-Authored-By text or Claude-attribution wording to any file, comment, docstring, or documentation. This is not a project requirement. Only add it if the user explicitly requests it for a specific commit.