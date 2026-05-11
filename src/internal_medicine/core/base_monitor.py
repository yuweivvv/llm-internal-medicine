"""Abstract base class for all monitors."""

from abc import ABC, abstractmethod


class BaseMonitor(ABC):
    """Base class shared by all backend-specific monitors."""

    def __init__(self, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False):
        self.log_per_layer = log_per_layer
        self.log_global = log_global
        self.monitor_interval = monitor_interval
        self.verbose = verbose
        self.hooks = []
        self.step_count = 0
        self.pp_rank = 0

    @abstractmethod
    def register_hooks(self, model) -> None: ...

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def step(self):
        self.step_count += 1

    def _should_monitor(self) -> bool:
        return self.step_count % self.monitor_interval == 0
