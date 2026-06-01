"""Megatron-specific Probe base class."""

import torch

from ...core.base_monitor import Probe


class TorchProbe(Probe):
    """Probe subclass with recompute (gradient checkpointing) guard.

    During recompute, forward hooks fire again in the backward pass with
    grad disabled. Accessing parameters at that point can crash when ZeRO
    has freed their storage.
    """

    def _should_monitor(self) -> bool:
        if not torch.is_grad_enabled():
            return False
        return super()._should_monitor()
