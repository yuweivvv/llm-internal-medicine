"""
QK Stats Monitor for PaddleFleet.

Uses paddle hooks on core_attention to capture Q, K tensors and compute
attention statistics. Pure paddle implementation (no Triton).
"""

import logging
import math

import paddle
import paddle.nn as nn

from ...core.base_monitor import BaseMonitor
from ...core.training_logs import training_logs

logger = logging.getLogger(__name__)


def compute_qk_stats_paddle(q, k, causal=True):
    """
    Reference paddle implementation of QK statistics.

    Args:
        q: [B, S, H, D] or [S, B, H, D]
        k: same shape as q

    Returns:
        dict with max_global, mean_global, entropy_global, sink_global,
        entropy_per_head [B, H]
    """
    if q.ndim == 4 and q.shape[0] < q.shape[1]:
        q = q.transpose([1, 0, 2, 3])
        k = k.transpose([1, 0, 2, 3])

    # Ensure [B, H, S, D]
    if q.shape[2] > q.shape[1]:
        # [B, S, H, D] -> [B, H, S, D]
        q = q.transpose([0, 2, 1, 3])
        k = k.transpose([0, 2, 1, 3])

    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / math.sqrt(head_dim)

    logits = paddle.matmul(q, k, transpose_y=True) * scale

    if causal:
        mask = paddle.triu(paddle.ones([seq_len, seq_len], dtype="bool"), diagonal=1)
        logits = paddle.where(~mask, logits, paddle.to_tensor(float("-inf")))

    valid_mask = logits > -1e9
    max_per_head = logits.max(axis=-1).max(axis=-1)

    logits_zeroed = paddle.where(valid_mask, logits, paddle.zeros_like(logits))
    sum_logits = logits_zeroed.sum(axis=(-2, -1))
    count = valid_mask.astype("float32").sum(axis=(-2, -1))
    mean_per_head = sum_logits / count.clip(min=1)

    probs = paddle.nn.functional.softmax(logits, axis=-1)
    log_probs = paddle.nn.functional.log_softmax(logits, axis=-1)
    entropy_map = -(probs * log_probs)
    entropy_map = paddle.where(valid_mask, entropy_map, paddle.zeros_like(entropy_map))
    row_entropy = entropy_map.sum(axis=-1)
    avg_entropy = row_entropy.mean(axis=-1)  # [B, H]

    sink_probs = probs[:, :, :, 0]  # [B, H, S]
    avg_sink = sink_probs.mean(axis=-1)  # [B, H]

    return {
        "max_per_head": max_per_head,
        "mean_per_head": mean_per_head,
        "entropy_per_head": avg_entropy,
        "sink_per_head": avg_sink,
        "max_global": float(max_per_head.max()),
        "mean_global": float(mean_per_head.mean()),
        "entropy_global": float(avg_entropy.mean()),
        "sink_global": float(avg_sink.mean()),
    }


class PaddleQKStatsMonitor(BaseMonitor):
    def __init__(self, causal=True, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False):
        super().__init__(
            log_per_layer=log_per_layer, log_global=log_global, monitor_interval=monitor_interval, verbose=verbose
        )
        self.causal = causal
        self.compute_count = 0
        self.tp_size = 1
        self.tp_rank = 0
        self.tp_group = None

    def register_hooks(self, model: nn.Layer):
        try:
            from paddlefleet.process_groups_config import ProcessGroupCollection
            from paddlefleet.utils import get_pg_rank, get_pg_size

            pg = ProcessGroupCollection.use_mpu_process_groups(required_pgs=["tp"])
            self.tp_size = get_pg_size(pg.tp)
            self.tp_rank = get_pg_rank(pg.tp)
            self.tp_group = pg.tp
        except Exception:
            pass

        try:
            from paddlefleet.parallel_state import get_pipeline_model_parallel_rank

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

        attention_layers = self._find_attention_layers(model)
        if not attention_layers:
            logger.warning("[PaddleQKMonitor] No attention layers found!")
            return

        if self.verbose:
            logger.info(f"[PaddleQKMonitor] Found {len(attention_layers)} attention layers. TP={self.tp_size}")

        for layer_idx, attn_module in attention_layers:
            if hasattr(attn_module, "core_attention"):
                hook = attn_module.core_attention.register_forward_pre_hook(self._make_compute_hook(layer_idx))
                self.hooks.append(hook)

        logger.info(f"[PaddleQKMonitor] Registered {len(self.hooks)} hooks.")

    def _find_attention_layers(self, model: nn.Layer) -> list[tuple[int, nn.Layer]]:
        attention_layers = []
        layers = self._get_decoder_layers(model)
        if layers is None:
            return []
        for local_idx, layer in enumerate(layers):
            global_idx = self.pp_rank * len(layers) + local_idx
            attn = getattr(layer, "self_attn", None) or getattr(layer, "self_attention", None)
            if attn is not None:
                attention_layers.append((global_idx, attn))
        return attention_layers

    def _get_decoder_layers(self, model):
        """Walk PaddleFleet model hierarchy to find decoder layers."""
        # PipelineLayer wraps layers in _layers_desc, actual layers in _sublayers
        # For non-PP: model.decoder.layers or direct model.layers
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            return model.decoder.layers
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            return model.encoder.layers
        if hasattr(model, "layers"):
            return model.layers
        # PaddleFleet PipelineLayer: _sublayers ordered dict of TransformerLayers
        transformer_layers = []
        for _name, sublayer in model.named_sublayers():
            if sublayer.__class__.__name__ == "TransformerLayer":
                transformer_layers.append(sublayer)
        return transformer_layers if transformer_layers else None

    def _make_compute_hook(self, layer_idx: int):
        def hook_fn(layer, inputs):
            if not layer.training:
                return
            if not self._should_monitor():
                return
            try:
                query, key = inputs[0], inputs[1]
                with paddle.no_grad():
                    # Handle GQA
                    if query.shape[2] != key.shape[2]:
                        heads_per_group = query.shape[2] // key.shape[2]
                        key = key.repeat_interleave(heads_per_group, axis=2)
                    stats = compute_qk_stats_paddle(query, key, causal=self.causal)

                all_heads = stats["entropy_per_head"]
                metrics = {
                    "max": stats["max_global"],
                    "mean": stats["mean_global"],
                    "entropy_avg": stats["entropy_global"],
                    "sink": stats["sink_global"],
                    "entropy_min": float(all_heads.min()),
                    "entropy_max": float(all_heads.max()),
                    "entropy_std": float(all_heads.std()),
                }

                log_dict = {}
                if self.log_per_layer:
                    for name, val in metrics.items():
                        log_dict[f"qk_stats/layer_{layer_idx}/{name}"] = val
                if self.log_global:
                    for name, val in metrics.items():
                        log_dict[f"qk_stats/global_{name}"] = val
                training_logs.update(**log_dict)
                self.compute_count += 1
            except Exception as e:
                logger.error(f"[PaddleQKMonitor] Error layer {layer_idx}: {e}")

        return hook_fn


def setup_qk_monitor(
    model,
    causal=True,
    verbose=False,
    log_per_layer=True,
    log_global=True,
    monitor_interval=1,
    monitor_dict=None,
    return_monitor=False,
):
    monitor = PaddleQKStatsMonitor(
        causal=causal,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
    )
    models = [model] if not isinstance(model, list) else model
    for m in models:
        monitor.register_hooks(m)
    logger.info(f"[PaddleQKMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers.")
    if monitor_dict is not None:
        monitor_dict["qk_stats"] = monitor
    if return_monitor:
        return model, monitor
    return model
