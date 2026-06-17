"""PaddleFleet backend for internal_medicine."""

import logging

from .base import PaddleProbe
from .gather import install_gather_fn
from .massive_activation_monitor import PaddleMassiveActivationMonitor, setup_massive_activation_monitor
from .moe_monitor import PaddleMoEMonitor, setup_moe_monitor
from .qk_monitor import PaddleQKStatsMonitor, setup_qk_monitor

logger = logging.getLogger(__name__)

_MONITOR_MAP = {
    "qk_stats": setup_qk_monitor,
    "moe_health": setup_moe_monitor,
    "massive_act": setup_massive_activation_monitor,
}


def setup_monitors(model, monitors=None, monitor_dict=None, monitor_interval=1, verbose=False, **kwargs):
    """Setup all requested monitors on a PaddleFleet model."""
    install_gather_fn()

    if monitors is None:
        monitors = ["all"]
    if "all" in monitors:
        monitors = list(_MONITOR_MAP.keys())
    if monitor_dict is None:
        monitor_dict = {}

    model_monitor_dict = getattr(model, "_internal_medicine_paddlefleet_monitors", None)
    if model_monitor_dict is None:
        model_monitor_dict = {}
        setattr(model, "_internal_medicine_paddlefleet_monitors", model_monitor_dict)

    for name in monitors:
        if name not in _MONITOR_MAP:
            logger.warning(f"[InternalMedicine/paddlefleet] Unknown monitor: {name}, skipping")
            continue
        if name in model_monitor_dict:
            monitor_dict[name] = model_monitor_dict[name]
            logger.info(f"[InternalMedicine/paddlefleet] Monitor already enabled: {name}, skipping duplicate setup")
            continue
        try:
            setup_dict = {}
            _MONITOR_MAP[name](
                model,
                monitor_dict=setup_dict,
                monitor_interval=monitor_interval,
                verbose=verbose,
                **kwargs.get(name, {}),
            )
            monitor = setup_dict.get(name)
            if monitor is not None:
                model_monitor_dict[name] = monitor
                monitor_dict[name] = monitor
            logger.info(f"[InternalMedicine/paddlefleet] Enabled monitor: {name}")
        except Exception as e:
            logger.error(f"[InternalMedicine/paddlefleet] Failed to setup {name}: {e}")

    return model


__all__ = [
    "setup_monitors",
    "PaddleProbe",
    "PaddleQKStatsMonitor",
    "setup_qk_monitor",
    "PaddleMoEMonitor",
    "setup_moe_monitor",
    "PaddleMassiveActivationMonitor",
    "setup_massive_activation_monitor",
]
