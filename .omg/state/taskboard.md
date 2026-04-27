# OmG Taskboard - Remediation Phase

## Active Lane: Project Perfection (Remediation)

### Phase 0: Critical Stability & Security (P0)
- [x] `rem-p0-security`: Fix API Security & Endpoints (hmac, fail-closed, real resolvers). (Done)
- [x] `rem-p0-shadow-impls`: Eradicate all `_impl.py` shadow implementations. (Done)
- [x] `rem-p0-silent-failures`: Expose silent failures (remove `except Exception:`, fix LinAlgError masking). (Done)

### Phase 1: Architecture & Orchestration (P1)
- [ ] `rem-p1-shatter-god-files`: Shatter `module_a.py` and other God-files into SRP submodules.
- [ ] `rem-p1-dvc-sync`: Delete `run_workflow.py`, migrate fully to `dvc.yaml`.
- [ ] `rem-p1-protocol-hash`: Split `ScientificProtocol` into Execution and Evaluation protocols.

### Phase 2: Quality & Maintainability (P2)
- [ ] `rem-p2-test-coverage`: Remove coverage omissions and add unit tests.
- [ ] `rem-p2-centralize-paths`: Consolidate magic strings into `config.py`.
- [ ] `rem-p2-type-safety`: Enforce strict `mypy` checks and remove type-system bypasses.

## Dependencies
- `rem-p0-silent-failures` depends on `rem-p0-shadow-impls`. (Met)
- `rem-p1-shatter-god-files` depends on `rem-p0-shadow-impls`. (Met)
- `rem-p1-protocol-hash` depends on `rem-p1-dvc-sync`.
- `rem-p2-test-coverage` depends on `rem-p1-shatter-god-files`.
- `rem-p2-type-safety` depends on `rem-p1-shatter-god-files`.
