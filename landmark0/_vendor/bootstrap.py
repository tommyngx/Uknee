"""Load the pinned Ultralytics snapshot bundled with ``landmark0``.

Ultralytics uses absolute ``ultralytics.*`` imports internally.  The vendored
directory is therefore added to the front of ``sys.path`` before any landmark0
runtime module imports it.  A package imported from pip or ``Ref`` is rejected
instead of being used silently.
"""

from __future__ import annotations

import importlib
import os
import sys
import tempfile
from pathlib import Path


VENDOR_ROOT = Path(__file__).resolve().parent
ULTRALYTICS_ROOT = (VENDOR_ROOT / "ultralytics").resolve()
EXPECTED_VERSION = "8.4.87"


def load_vendored_ultralytics():
    """Return the pinned vendored package and reject a conflicting import."""
    cache_root = Path(tempfile.gettempdir()) / "uknee-runtime-cache"
    cache_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_root / "matplotlib"))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root))
    vendor_path = str(VENDOR_ROOT)
    if not sys.path or sys.path[0] != vendor_path:
        if vendor_path in sys.path:
            sys.path.remove(vendor_path)
        sys.path.insert(0, vendor_path)

    loaded = sys.modules.get("ultralytics")
    if loaded is not None:
        loaded_file = Path(getattr(loaded, "__file__", "")).resolve()
        if ULTRALYTICS_ROOT not in loaded_file.parents:
            raise RuntimeError(
                "A non-vendored 'ultralytics' package was imported before landmark0: "
                f"{loaded_file}. Import landmark0 before ultralytics, or remove the "
                "conflicting package from the process."
            )
    else:
        loaded = importlib.import_module("ultralytics")

    loaded_file = Path(getattr(loaded, "__file__", "")).resolve()
    if ULTRALYTICS_ROOT not in loaded_file.parents:
        raise RuntimeError(f"landmark0 resolved the wrong Ultralytics backend: {loaded_file}")
    if getattr(loaded, "__version__", None) != EXPECTED_VERSION:
        raise RuntimeError(
            "Unexpected vendored Ultralytics version: "
            f"{getattr(loaded, '__version__', None)!r}; expected {EXPECTED_VERSION}."
        )
    return loaded
