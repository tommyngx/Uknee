# Architecture

## Contract

Input is `image[B, 1, H, W]`. The RWKV adapter repeats grayscale images to the
checkpoint's three input channels. Inference returns:

- `segmentation_logits[B, C_mask, H, W]`
- `coarse_landmarks[B, 129, 2]`
- `final_landmarks[B, 129, 2]`
- `landmark_confidence[B, 129]`
- optionally `local_heatmaps[B, 129, P, P]`

Coordinates are always normalised `(x, y)`.

## Adaptive RWKV query flow

```text
image
  └─ RWKVUNetBackboneAdapter
       ├─ unchanged segmentation logits
       ├─ P2 projected feature (1/4)
       ├─ P3 projected feature (1/8)
       └─ P4 projected feature (1/16)
            └─ CoarseReferenceHead
                 └─ 129 image-conditioned references
                      └─ MultiScaleLocalFeatureSampler
                           ├─ local P2 patches
                           └─ multi-scale local tokens
                                └─ landmark + bone + coordinate embeddings
                                     └─ 2-layer query self-attention
                                          └─ FiLM local heatmap refinement
                                               └─ final points + confidence
```

This is an adaptive, identity-preserving landmark query model. It is not a full
DETR detector: there are no anonymous object slots, object classes, boxes,
no-object class, encoder-decoder cross attention, or Hungarian matching.

## Backbone integration

`RWKVUNetBackboneAdapter` executes the existing named stages without editing
`models/RWKV/RWKV_UNet/RWKV_UNetV3.py` and without forward hooks. It exposes
decoder features at 1/4 and 1/8 resolution and the encoder bottleneck at 1/16.
Frozen backbone parameters do not receive gradients, while 1×1 feature
projections remain trainable.

## Extending models

Add one file under `landmark/models/`, then register a builder in
`landmark/models/registry.py`. Every model must at minimum return
`coarse_landmarks`, `final_landmarks`, and `landmark_confidence`.
