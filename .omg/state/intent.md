# Intent: Optimize and Fix according to optimize.md and suggest.md

## Primary Objective
Follow all instructions and fix all identified issues in `optimize.md` and `suggest.md`.

## Prioritized Goals (from suggest.md)
1. **P0: Critical Fixes**
    - Security: Secure API authentication in `src/plasmid_priority/api/app.py`.
    - Cleanup: Remove shadow `_impl.py` files.
    - Robustness: Remove generic `except Exception:` and fix silent mathematical failures.
2. **P1: Architectural Changes**
    - Orchestration: Replace `run_workflow.py` with DVC-only management.
    - Refactoring: Split "God-Files" (e.g., `module_a_impl.py`).
    - Protocol: Split `ScientificProtocol` into `Execution` and `Evaluation` protocols.
3. **P2: Quality & Maintainability**
    - Testing: Realistic coverage and unit tests.
    - Configuration: Centralize hardcoded paths.
    - Typing: Enforce type safety and clean up `mypy` overrides.

## Workspace State
- Branch: `codex/perfection-pass`
- Dirty: `scripts/15_normalize_and_score.py`
- Untracked: `optimize.md`, `suggest.md`

## Next Steps
1. Assemble team.
2. Create detailed taskboard.
3. Execute P0 fixes.
