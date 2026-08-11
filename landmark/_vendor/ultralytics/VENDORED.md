# Vendored Ultralytics

This directory is a source snapshot of the local `Ref/ultralytics` package,
Ultralytics version **8.4.87**, vendored on 2026-08-10 for reproducible Uknee
pose training. Runtime bytecode caches, sample assets, HUB, Solutions,
tracking and SAM were excluded. Public registration is narrowed to detect/pose,
while source retained for shared indirect imports remains internal and is not
exposed by the Uknee API.

The SHA-256 manifest digest of the source snapshot before Uknee-specific
changes was:

```text
9489747e8f01cfd3ac47fa7fc2d3cb49e63c49cbea3b213faa9c62c96a2e8b9a
```

Ultralytics files retain their original AGPL-3.0 headers. The repository root
contains the full AGPL-3.0 license. This snapshot is intentionally loaded ahead
of any site-installed `ultralytics` package.

## Uknee modifications

- MESKO4GF2 model YAMLs are exposed through `landmark/cfg/models` with four
  region classes and 51 padded keypoints per instance.
- OA26 auxiliary heatmap target construction is class-aware and maps all four
  region rows to the canonical 129-landmark order.
- OA26 neighbour and curvature losses respect the six independent anatomical
  paths, including the three separate tibial paths.
- Explicit model-YAML scales are preserved when a filename has no n/s/m/l/x
  suffix; standard scaled filenames still take precedence.
- ONNX export recognizes the Uknee fixed-output wrapper and assigns stable
  `detections`, `num_detections`, and `canonical_keypoints` output names with
  dynamic batch axes.
- Pose training supports flat checkpoints, MRE-based `best.pt`, medical metrics
  and consolidated reporting through the landmark integration layer.
- Public model registration is intentionally limited to YOLO detect/pose;
  cloud HUB sessions, tracking, Solutions and unrelated task frontends are not
  part of the standalone Uknee runtime.

Reference: https://www.ultralytics.com/license
