"""PaddleFleet distributed gather function for training_logs."""

import logging

logger = logging.getLogger(__name__)


def _paddle_gather(local_metrics: dict) -> list:
    """Gather metrics from all ranks using paddle.distributed."""
    import paddle.distributed as dist

    if not dist.is_initialized():
        return None
    info_list = []
    dist.all_gather_object(info_list, local_metrics)
    return info_list


def install_gather_fn():
    """Install the paddle-based gather function into the global training_logs."""
    from ...core.training_logs import training_logs

    try:
        import paddle.distributed as dist

        if dist.is_initialized():
            training_logs.set_gather_fn(_paddle_gather)
    except ImportError:
        pass
