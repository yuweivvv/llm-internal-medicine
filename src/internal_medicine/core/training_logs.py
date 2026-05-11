"""
Framework-agnostic training metrics store.

All values stored as Python floats. Framework-specific tensor conversion
is the caller's responsibility.
"""

import logging
from collections import defaultdict
from collections.abc import Callable

logger = logging.getLogger(__name__)

__all__ = ("SmoothedValue", "TrainingLogs", "training_logs")


class SmoothedValue:
    """Track a series of scalar values and provide smoothed access."""

    def __init__(self, skip_zero=False, mode="mean"):
        self.total = 0.0
        self.count = 0
        self._skip_zero = skip_zero
        self.mode = mode
        if self.mode == "max":
            self.max_value = float("-inf")
        if self.mode == "min":
            self.min_value = float("inf")

    def update(self, value: float):
        if self._skip_zero and value == 0:
            return
        self.count += 1
        self.total += value
        if self.mode == "max":
            self.max_value = max(self.max_value, value)
        if self.mode == "min":
            self.min_value = min(self.min_value, value)

    @property
    def global_avg(self):
        return self.total / max(self.count, 1e-6)

    @property
    def log(self):
        if self.mode == "max":
            return self.max_value
        if self.mode == "min":
            return self.min_value
        return self.global_avg

    def reset(self):
        self.total = 0.0
        self.count = 0
        if self.mode == "max":
            self.max_value = float("-inf")
        if self.mode == "min":
            self.min_value = float("inf")


class TrainingLogs:
    """Singleton metric store. All monitors write here."""

    _instance = None

    def __new__(cls, *args, **kw):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __init__(self, gather_fn: Callable | None = None):
        if not hasattr(self, "meters"):
            self.meters = {}
        if gather_fn is not None:
            self._gather_fn = gather_fn

    def set_gather_fn(self, fn: Callable):
        """Set the distributed gather function (backend provides this)."""
        self._gather_fn = fn

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = v

    def __setitem__(self, k, v):
        val = float(v) if isinstance(v, int | float) else float(v.item() if hasattr(v, "item") else v)
        if k not in self.meters:
            if "/max" in k or k.endswith("_max"):
                mode = "max"
            elif "/min" in k or k.endswith("_min"):
                mode = "min"
            else:
                mode = "mean"
            self.meters[k] = SmoothedValue(mode=mode)
        self.meters[k].update(val)

    def __getitem__(self, v):
        return self.meters[v]

    def __getattr__(self, attr):
        if attr in ("meters", "_gather_fn"):
            raise AttributeError(attr)
        if attr in self.meters:
            return self.meters[attr]
        raise AttributeError(f"'{type(self).__name__}' has no attribute '{attr}'")

    def dict(self):
        return {k: v.log for k, v in self.meters.items()}

    def get_latest(self, prefix=None):
        result = {}
        for k, v in self.meters.items():
            if prefix is None or k.startswith(prefix):
                result[k] = v.log
        return result

    def print_metrics(self, metrics=None, prefix=None, format_fn=None):
        if format_fn is None:
            format_fn = logger.info
        if metrics is None:
            metrics = self.get_latest(prefix)
        elif prefix:
            metrics = {k: v for k, v in metrics.items() if k.startswith(prefix)}
        if not metrics:
            return
        grouped = defaultdict(dict)
        for k, v in metrics.items():
            if "/" in k:
                parts = k.split("/")
                category = parts[0]
                metric_name = "/".join(parts[1:])
                grouped[category][metric_name] = v
            else:
                grouped["other"][k] = v
        for category, items in sorted(grouped.items()):
            format_fn(f"[{category}]")
            for name, value in sorted(items.items()):
                format_fn(f"  {name}: {value:.4f}")

    def reset(self):
        self.meters.clear()

    def gather_and_aggregate(self):
        """Gather metrics from all ranks and aggregate by naming convention."""
        all_metrics = self.get_latest()
        if not all_metrics:
            return {}

        gather_fn = getattr(self, "_gather_fn", None)
        if gather_fn is None:
            return all_metrics

        info_list = gather_fn(all_metrics)
        if info_list is None:
            return all_metrics

        aggregated = {}
        all_keys = {key for item in info_list for key in item}
        for k in all_keys:
            values = [v[k] for v in info_list if k in v]
            if not values:
                continue
            if "_max" in k or k.endswith("/max"):
                aggregated[k] = max(values)
            elif "_min" in k or k.endswith("/min"):
                aggregated[k] = min(values)
            else:
                aggregated[k] = sum(values) / len(values)
        return aggregated


training_logs = TrainingLogs()
