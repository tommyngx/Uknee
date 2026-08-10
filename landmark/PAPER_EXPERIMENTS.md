# Paper experiment protocol

Use one immutable dataset split, `seed=2006`, identical `imgsz`, epochs, batch
and hardware for all ablations. Archive the run's `args.yaml`,
`dataset_resolved.yaml`, `results.csv`, `best.pt`, `best_mre.pt`, exported
state-dict and environment metadata.

## Primary ablation

```bash
for model in yolo26-pose yolo26-pose-v1 yolo26-pose-v9; do
  python -m landmark.train \
    --model landmark/cfg/models/${model}.yaml \
    --data landmark/cfg/datasets/mesko4gf2.yaml \
    --epochs 100 --imgsz 640 --batch 16 --device 0 \
    --seed 2006 --project landmark/runs/paper --name ${model}
done
```

Report box and pose precision/recall/mAP50/mAP50-95 together with MRE,
median/p95, PCK2/4/8, failure rate, four per-region MRE values, order accuracy,
throughput and peak VRAM. YOLO26-Pose is the stable reference. Do not label V1
or V9 as the recommended model until repeated experiments show no regression
in pose mAP or MRE.

## Required pre-paper checks

1. Source/vendor golden parity for YOLO26 forward, FP32 loss, fixed-RNG data,
   optimizer groups, warmup/LR, EMA and resume.
2. V1/V9 target and gradient tests covering all four classes, padded masks and
   all six paths.
3. Full 442-label audit, inverse LetterBox and case leakage checks.
4. Overfit 8–16 images without augmentation; loss and MRE must fall without
   NaN/Inf.
5. Two-process DDP smoke: synchronized weights and rank-zero-only artifacts.
6. ONNX/TorchScript parity for all five models; V9 canonical output must use
   refined keypoints.

Performance acceptance for the unchanged YOLO26 reference is at most `1e-6`
absolute FP32 forward/loss drift and at most 3% throughput or peak-VRAM drift
on the same GPU and batch.
