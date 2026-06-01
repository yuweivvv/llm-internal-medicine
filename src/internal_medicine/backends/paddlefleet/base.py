"""PaddleFleet-specific Probe base class."""

import paddle

from ...core.base_monitor import Probe


class PaddleProbe(Probe):
    """Probe subclass with recompute (gradient checkpointing) guard.

    During recompute, forward hooks fire again in the backward pass with
    grad disabled. Accessing parameters at that point can crash when ZeRO
    has freed their storage (holder->size():0).
    """

    def _should_monitor(self) -> bool:
        if not paddle.is_grad_enabled():
            return False
        return super()._should_monitor()
