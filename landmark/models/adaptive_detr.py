"""Compatibility name for the adaptive RWKV landmark-query architecture.

The model borrows identity-preserving queries and self-attention from DETR, but
intentionally has no object slots, Hungarian matching, or no-object class.
"""

from .adaptive_rwkv import RWKVUNetLandmarkModel

AdaptiveRWKVLandmarkDETR = RWKVUNetLandmarkModel

__all__ = ["AdaptiveRWKVLandmarkDETR"]
