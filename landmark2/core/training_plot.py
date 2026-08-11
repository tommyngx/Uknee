"""Compatibility callback for OA26 trainers.

The consolidated landmark2 reporter owns all plotting, so the old per-variant
plot callback is intentionally empty and removed by ``FlatPoseTrainerMixin``.
"""


def plot_v9_performance_on_epoch_end(_trainer) -> None:
    return None


__all__ = ["plot_v9_performance_on_epoch_end"]
