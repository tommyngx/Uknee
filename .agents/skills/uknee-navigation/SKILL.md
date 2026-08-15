---
name: uknee-navigation
description: Route Uknee repository work to the active landmark or segment subsystem with minimal context. Use for implementation, debugging, review, testing, architecture, training, data, model, export, or deployment tasks touching `landmark/`, `segment/`, shared Uknee CLI/package files, or the archived `landmark0/` runtime.
---

# Uknee navigation

Route first, then load only the architecture and symbols needed for the task.

## Route the task

1. Use `landmark/architecture.md` for knee detection, landmark pose, MESKO
   keypoints, YOLO/OA26, pose metrics, or landmark export work.
2. Use `segment/architecture.md` for mask datasets, segmentation models,
   segmentation metrics, training, ONNX, or deployment work.
3. For `uknee_cli.py`, packaging, or a contract shared by both runtimes, read
   both architecture files, but inspect only the affected entry points.
4. Treat `landmark0/` as legacy. Do not search, read, import, edit, or test it
   unless the user explicitly requests legacy work, migration, compatibility,
   or golden-parity debugging.

## Minimize discovery

- Read the selected architecture file before source code.
- Use `rg` for exact symbols and filenames; avoid full directory dumps.
- Do not inventory `segment/models/`, `Ref/`, `info/`, or
  `landmark0/_vendor/`.
- Load a model implementation only after resolving its exact registry or YAML
  entry.
- Ignore historical planning and upgrade documents unless history is relevant.

## Verify proportionally

- Run the narrowest owning test first.
- Use the subsystem-wide command listed in its architecture file only for
  cross-cutting changes.
- Do not run full training, multi-GPU, or legacy suites unless requested or
  necessary to validate the specific risk.
