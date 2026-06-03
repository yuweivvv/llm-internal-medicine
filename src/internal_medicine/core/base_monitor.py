"""Base class for all monitors (probes)."""

from abc import ABC, abstractmethod

from .training_logs import training_logs


class Probe(ABC):
    """Base class shared by all backend-specific monitors.

    Subclasses set METRIC_PREFIX plus aggregation overrides, then call
    _record_metrics() from their hooks. Global aggregation and
    flushing are handled here automatically.
    """

    METRIC_PREFIX: str = ""
    MAX_AGGREGATED: set[str] = set()
    MIN_AGGREGATED: set[str] = set()

    def __init__(self, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False):
        self.log_per_layer = log_per_layer
        self.log_global = log_global
        self.monitor_interval = monitor_interval
        self.verbose = verbose
        self.hooks = []
        self.step_count = 0
        self.pp_rank = 0
        self._global_accum: dict[str, float] = {}
        self._global_count: int = 0

    @abstractmethod
    def register_hooks(self, model) -> None: ...

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def step(self):
        self.step_count += 1
        if self.log_global and self._global_accum:
            self._flush_global_metrics()

    def _should_monitor(self) -> bool:
        return self.step_count % self.monitor_interval == 0

    def _record_metrics(self, layer_idx: int, metrics: dict[str, float]):
        """Log per-layer metrics and accumulate for global aggregation.

        Increments _global_count once (one layer observation).
        Skips entirely when metrics is empty.
        """
        if not metrics:
            return
        self._log_per_layer_metrics(layer_idx, metrics)
        if self.log_global:
            self._accumulate_global(metrics)
            self._count_global_observation()

    def _log_per_layer_metrics(self, layer_idx: int, metrics: dict[str, float]):
        """Write per-layer metric keys without touching global aggregation."""
        if self.log_per_layer and metrics:
            training_logs.update(**{f"{self.METRIC_PREFIX}/layer_{layer_idx}/{k}": v for k, v in metrics.items()})

    def _accumulate_global(self, metrics: dict[str, float]):
        """Accumulate into global buffer WITHOUT incrementing count.

        Use when a layer emits metrics from multiple hooks (e.g., MoE
        router + expert). Pair with a single _count_global_observation()
        call when the complete layer observation is finished.
        """
        for name, val in metrics.items():
            if self._is_max_aggregated(name):
                self._global_accum[name] = max(self._global_accum.get(name, float("-inf")), val)
            elif name in self.MIN_AGGREGATED:
                self._global_accum[name] = min(self._global_accum.get(name, float("inf")), val)
            else:
                self._global_accum[name] = self._global_accum.get(name, 0.0) + val

    def _is_max_aggregated(self, name: str) -> bool:
        return (
            name in self.MAX_AGGREGATED
            or name == "massive_act_channel_count"
            or name.startswith("channel_count_gt_")
        )

    def _count_global_observation(self):
        """Count one complete layer observation for global averages."""
        if self.log_global:
            self._global_count += 1

    def _flush_global_metrics(self):
        """Aggregate accumulated metrics into global keys and write to training_logs."""
        if self._global_count == 0:
            self._global_accum.clear()
            return
        log_dict = {}
        for name, val in self._global_accum.items():
            if self._is_max_aggregated(name) or name in self.MIN_AGGREGATED:
                log_dict[f"{self.METRIC_PREFIX}/global_{name}"] = val
            else:
                log_dict[f"{self.METRIC_PREFIX}/global_{name}"] = val / self._global_count
        training_logs.update(**log_dict)
        self._global_accum = {}
        self._global_count = 0

    def _resolve_layer_idx(self, layer, local_idx: int, num_local_layers: int, layer_offset: int = 0) -> int:
        """Resolve a stable global layer id when model layers expose one."""
        for attr in ("layer_idx", "layer_index", "idx"):
            value = getattr(layer, attr, None)
            if isinstance(value, int):
                return value
        layer_number = getattr(layer, "layer_number", None)
        if isinstance(layer_number, int):
            return layer_number - 1 if layer_number > 0 else layer_number
        return self.pp_rank * num_local_layers + layer_offset + local_idx


BaseMonitor = Probe
