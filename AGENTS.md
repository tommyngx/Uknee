# Uknee agent guide

## Scope routing

- Treat `landmark/` and `segment/` as the active runtimes.
- Treat `landmark0/` as archived legacy code. Do not read, search, import, edit,
  or run its tests unless the user explicitly requests `landmark0`, legacy
  compatibility, migration, or golden-parity debugging.
- Read only `landmark/architecture.md` for landmark or detection work.
- Read only `segment/architecture.md` for segmentation work.
- For shared CLI or packaging work, start with `uknee_cli.py`, `pyproject.toml`,
  and the two architecture files only when both runtimes are affected.

## Context budget

1. Identify the owning subsystem from the request and changed paths.
2. Read its architecture file, then inspect exact symbols with `rg`.
3. Do not inventory broad trees such as `segment/models/`, `Ref/`, `info/`, or
   `landmark0/_vendor/`; query a model or file by name when needed.
4. Do not load historical plans (`next_clean.md`, upgrade reports) unless the
   task is explicitly historical or architectural.
5. Preserve unrelated user changes and keep fixes inside the owning subsystem.

## Verification

- Landmark: run the narrow test module first; use
  `python -m unittest discover -s landmark/tests -v` for cross-cutting changes.
- Segment: run the matching module under `tests/code/`; use
  `python -m unittest discover -s tests/code -p 'test_segment*.py' -v` for
  cross-cutting segment changes.
- Do not run `landmark0/tests` by default.
- Training, GPU, DDP, and large-model checks require a focused smoke test; do
  not start a full training run unless requested.

