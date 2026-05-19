"""
Massive Activation Monitor for PaddleFleet.

Monitors massive activations in post-residual hidden states — extreme outlier
values that appear in a few channels and persist across intermediate layers
via the residual connection.

Based on findings from:
    Sun, S., Canziani, A., LeCun, Y., & Zhu, J. (2026).
    "The Spike, the Sparse and the Sink: Anatomy of Massive Activations
    and Attention Sinks." arXiv:2603.05498.

This monitor hooks into pre-norm points (residual stream BEFORE RMSNorm)
to capture raw activation magnitudes, and post-norm points to detect the
sparsification that enables attention sinks.

Metrics produced:
    massive_act/layer_{i}/channel_max
    massive_act/layer_{i}/channel_max_ratio
    massive_act/layer_{i}/massive_act_channel_count
    massive_act/layer_{i}/topk_channel_norm
    massive_act/layer_{i}/post_norm_sparsity
    massive_act/layer_{i}/post_norm_cosine
    massive_act/global_*
"""

import logging

import paddle
import paddle.nn as nn

from ...core.base_monitor import Probe
from .massive_activation_metrics import (
    compute_post_norm_cosine_stability,
    compute_post_norm_sparsity,
    compute_pre_norm_metrics,
)

logger = logging.getLogger(__name__)


class PaddleMassiveActivationMonitor(Probe):
    """Monitor massive activations in the residual stream.

    Hooks into Transformer layers via forward pre-hooks to capture the
    hidden_states (residual stream) before normalization.
    """

    METRIC_PREFIX = "massive_act"
    MAX_AGGREGATED = {"channel_max", "channel_max_ratio", "topk_channel_norm"}

    def __init__(
        self,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
        spike_threshold_multiplier: float = 100.0,
        topk_channels: int = 3,
        sparsity_epsilon: float = 0.01,
        cosine_sample_pairs: int = 256,
        sample_layers: list[int] | None = None,
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
        )
        self.spike_threshold_multiplier = spike_threshold_multiplier
        self.topk_channels = topk_channels
        self.sparsity_epsilon = sparsity_epsilon
        self.cosine_sample_pairs = cosine_sample_pairs
        self.sample_layers = set(sample_layers) if sample_layers else None

    def register_hooks(self, model: nn.Layer):
        try:
            from paddlefleet.parallel_state import get_pipeline_model_parallel_rank

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

        layers = self._get_decoder_layers(model)
        if not layers:
            logger.warning("[MassiveActMonitor] No transformer layers found!")
            return

        registered = 0
        for local_idx, layer in enumerate(layers):
            global_idx = self.pp_rank * len(layers) + local_idx

            if self.sample_layers and global_idx not in self.sample_layers:
                continue

            hook = layer.register_forward_pre_hook(self._make_residual_hook(global_idx))
            self.hooks.append(hook)
            registered += 1

        logger.info(f"[MassiveActMonitor] Registered {registered} hooks.")

    def _get_decoder_layers(self, model):
        """Find transformer decoder layers in PaddleFleet model hierarchy."""
        if hasattr(model, "_layers") and hasattr(model._layers, "run_function"):
            model = model._layers
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "run_function"):
            return [
                layer
                for layer in model.run_function
                if hasattr(layer, "self_attn") or hasattr(layer, "self_attention") or hasattr(layer, "input_layernorm")
            ]
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            return model.decoder.layers
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            return model.encoder.layers
        if hasattr(model, "layers"):
            return model.layers
        transformer_layers = []
        for _name, sublayer in model.named_sublayers():
            if hasattr(sublayer, "self_attn") or hasattr(sublayer, "self_attention"):
                transformer_layers.append(sublayer)
        return transformer_layers if transformer_layers else None

    def _make_residual_hook(self, layer_idx: int):
        def hook_fn(module, inputs):
            if not module.training:
                return
            if not self._should_monitor():
                return
            try:
                hidden_states = self._extract_hidden_states(inputs)
                if hidden_states is None:
                    return

                with paddle.no_grad():
                    self._compute_and_log(layer_idx, hidden_states.detach(), module)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MassiveActMonitor] Error at layer {layer_idx}: {e}")

        return hook_fn

    @staticmethod
    def _extract_hidden_states(inputs):
        """Extract hidden_states from pre-hook inputs.

        Handles two PaddleFleet conventions:
        - Dict-based: forward(self, dict_args) where dict_args["hidden_states"] is the tensor
        - Positional: forward(self, hidden_states, ...) where inputs[0] is the tensor
        """
        if not inputs:
            return None
        first = inputs[0]
        if isinstance(first, dict):
            return first.get("hidden_states")
        if isinstance(first, paddle.Tensor):
            return first
        return None

    def _compute_and_log(self, layer_idx: int, hidden_states: paddle.Tensor, module: nn.Layer):
        metrics = compute_pre_norm_metrics(
            hidden_states,
            threshold_multiplier=self.spike_threshold_multiplier,
            k=self.topk_channels,
        )

        norm_layer = getattr(module, "input_layernorm", None)
        if norm_layer is not None:
            try:
                normalized = norm_layer(hidden_states)
                metrics["post_norm_sparsity"] = compute_post_norm_sparsity(normalized, epsilon=self.sparsity_epsilon)
                metrics["post_norm_cosine"] = compute_post_norm_cosine_stability(
                    normalized, num_sample_pairs=self.cosine_sample_pairs
                )
            except Exception:
                pass

        self._record_metrics(layer_idx, metrics)


def setup_massive_activation_monitor(
    model,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    verbose: bool = False,
    spike_threshold_multiplier: float = 100.0,
    topk_channels: int = 3,
    sparsity_epsilon: float = 0.01,
    cosine_sample_pairs: int = 256,
    sample_layers: list[int] | None = None,
    monitor_dict: dict | None = None,
):
    monitor = PaddleMassiveActivationMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        spike_threshold_multiplier=spike_threshold_multiplier,
        topk_channels=topk_channels,
        sparsity_epsilon=sparsity_epsilon,
        cosine_sample_pairs=cosine_sample_pairs,
        sample_layers=sample_layers,
    )
    monitor.register_hooks(model)
    logger.info(f"[MassiveActMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers.")
    if monitor_dict is not None:
        monitor_dict["massive_act"] = monitor
    return model
