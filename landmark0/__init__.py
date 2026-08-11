"""Standalone knee pose training built on a pinned Ultralytics backend."""

from ._vendor.bootstrap import load_vendored_ultralytics

# This must happen before importing modules that use absolute ``ultralytics``
# imports. It also makes accidental use of a pip installation fail loudly.
load_vendored_ultralytics()

from .utils.api import KneePose

__all__ = ["KneePose", "load_vendored_ultralytics"]
