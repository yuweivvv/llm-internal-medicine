"""
MoE Monitor for PaddleFleet.

Monitors MoE router health and expert weight norms using paddle hooks.

PaddleFleet MoE structure:
- MoELayer.gate (TopKRouter) — returns 8-tuple:
    (capacity, topk_weights, topk_indices, gates_masked, mask, priorities, aux_loss, z_loss)
- MoELayer.experts (nn.LayerList) or .grouped_gemm_experts (GroupedMLPExpert)
- MoELayer.shared_experts (StandardMLPSharedExpert or None)
"""

import logging

import paddle
import paddle.nn as nn

from ...core.base_monitor import BaseMonitor
from ...core.training_logs import training_logs

logger = logging.getLogger(__name__)


def _compute_router_entropy(probs):
    """Router entropy from probability distribution. probs: [tokens, experts]."""
    probs = probs.astype("float32").clip(min=1e-10)
    entropy = -(probs * probs.log()).sum(axis=-1)
    return float(entropy.mean())


def _compute_expert_norms_paddle(weight_list):
    """Compute L2 norms for a list of paddle weight tensors."""
    if not weight_list:
        return {"expert_norm_mean": 0.0, "expert_norm_std": 0.0, "expert_norm_min": 0.0, "expert_norm_max": 0.0}
    norms = paddle.stack([w.astype("float32").norm() for w in weight_list])
    return {
        "expert_norm_mean": float(norms.mean()),
        "expert_norm_std": float(norms.std()) if norms.numel() > 1 else 0.0,
        "expert_norm_min": float(norms.min()),
        "expert_norm_max": float(norms.max()),
    }


class PaddleMoEMonitor(BaseMonitor):
    def __init__(self, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False):
        super().__init__(
            log_per_layer=log_per_layer, log_global=log_global, monitor_interval=monitor_interval, verbose=verbose
        )
        self.pp_rank = 0

    def register_hooks(self, model: nn.Layer):
        try:
            from paddlefleet.parallel_state import get_pipeline_model_parallel_rank

            self.pp_rank = get_pipeline_model_parallel_rank()
        except Exception:
            pass

        moe_layers = self._find_moe_layers(model)
        if not moe_layers:
            logger.warning("[PaddleMoEMonitor] No MoE layers found!")
            return
        if self.verbose:
            logger.info(f"[PaddleMoEMonitor] Found {len(moe_layers)} MoE layers.")

        for layer_idx, moe_layer in moe_layers:
            # Hook on gate (TopKRouter)
            if hasattr(moe_layer, "gate"):
                hook = moe_layer.gate.register_forward_post_hook(self._make_gate_hook(layer_idx, moe_layer))
                self.hooks.append(hook)
            # Hook on MoELayer for expert norms
            hook = moe_layer.register_forward_post_hook(self._make_moe_layer_hook(layer_idx, moe_layer))
            self.hooks.append(hook)

        logger.info(f"[PaddleMoEMonitor] Registered {len(self.hooks)} hooks on {len(moe_layers)} layers.")

    def _find_moe_layers(self, model: nn.Layer) -> list[tuple[int, nn.Layer]]:
        moe_layers = []
        layers = self._get_decoder_layers(model)
        if layers is None:
            # Fallback: search all sublayers
            for _name, sublayer in model.named_sublayers():
                if sublayer.__class__.__name__ == "MoELayer":
                    moe_layers.append((len(moe_layers), sublayer))
            return moe_layers

        for local_idx, layer in enumerate(layers):
            global_idx = self.pp_rank * len(layers) + local_idx
            moe_module = None
            # PaddleFleet: TransformerLayer.mlp can be MoELayer
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                moe_module = layer.mlp
            elif hasattr(layer, "moe"):
                moe_module = layer.moe
            elif hasattr(layer, "gate"):
                moe_module = layer
            if moe_module is not None:
                moe_layers.append((global_idx, moe_module))
        return moe_layers

    def _get_decoder_layers(self, model):
        if hasattr(model, "_layers") and hasattr(model._layers, "run_function"):
            model = model._layers
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "run_function"):
            return [
                layer
                for layer in model.run_function
                if hasattr(layer, "self_attn")
                or hasattr(layer, "self_attention")
                or (hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"))
                or hasattr(layer, "moe")
            ]
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            return model.decoder.layers
        if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            return model.encoder.layers
        if hasattr(model, "layers"):
            return model.layers
        transformer_layers = []
        for _name, sublayer in model.named_sublayers():
            if (
                hasattr(sublayer, "self_attn")
                or hasattr(sublayer, "self_attention")
                or (hasattr(sublayer, "mlp") and hasattr(sublayer.mlp, "gate"))
                or hasattr(sublayer, "moe")
            ):
                transformer_layers.append(sublayer)
        return transformer_layers if transformer_layers else None

    def _make_gate_hook(self, layer_idx: int, moe_layer: nn.Layer):
        def hook_fn(layer, inputs, outputs):
            if not layer.training:
                return
            if not self._should_monitor():
                return
            try:
                with paddle.no_grad():
                    self._compute_gate_metrics(layer_idx, layer, outputs, moe_layer)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[PaddleMoEMonitor] Gate hook error layer {layer_idx}: {e}")

        return hook_fn

    def _make_moe_layer_hook(self, layer_idx: int, moe_layer: nn.Layer):
        def hook_fn(layer, inputs, outputs):
            if not layer.training:
                return
            if not self._should_monitor():
                return
            try:
                with paddle.no_grad():
                    self._compute_expert_metrics(layer_idx, layer)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[PaddleMoEMonitor] MoE layer hook error layer {layer_idx}: {e}")

        return hook_fn

    def _compute_gate_metrics(self, layer_idx, gate, outputs, moe_layer):
        """
        PaddleFleet TopKRouter.forward returns:
        (capacity, topk_weights, topk_indices, gates_masked, mask, priorities, aux_loss, z_loss)
        """
        metrics = {}

        # outputs is the 8-tuple from TopKRouter
        if isinstance(outputs, tuple) and len(outputs) >= 5:
            _, topk_weights, _, gates_masked, _ = outputs[:5]

            # Router entropy from gates_masked (non-zero entries form the prob distribution)
            if gates_masked is not None:
                # Normalize to probability distribution for entropy
                row_sum = gates_masked.sum(axis=-1, keepdim=True).clip(min=1e-10)
                probs = gates_masked / row_sum
                metrics["router_entropy"] = _compute_router_entropy(probs)

            # TopK score sum
            if topk_weights is not None:
                score_sum = topk_weights.sum(axis=-1) if topk_weights.ndim > 1 else topk_weights
                metrics["score_sum_mean"] = float(score_sum.mean())
                metrics["score_sum_min"] = float(score_sum.min())
                metrics["score_sum_max"] = float(score_sum.max())

        # Expert bias (e_score_correction_bias for noaux_tc)
        if hasattr(gate, "e_score_correction_bias"):
            bias = gate.e_score_correction_bias
            metrics["expert_bias_mean"] = float(bias.mean())
            metrics["expert_bias_std"] = float(bias.std())

        log_dict = {}
        if self.log_per_layer:
            for name, val in metrics.items():
                log_dict[f"moe_health/layer_{layer_idx}/{name}"] = val
        if self.log_global:
            for name, val in metrics.items():
                log_dict[f"moe_health/global_{name}"] = val
        if log_dict:
            training_logs.update(**log_dict)

    def _compute_expert_metrics(self, layer_idx, moe_layer):
        metrics = {}
        routed_norm_mean = None

        # Expert norms — handle both grouped_gemm and standard experts
        if hasattr(moe_layer, "grouped_gemm_experts") and moe_layer.grouped_gemm_experts is not None:
            ggm = moe_layer.grouped_gemm_experts
            expert_weights = []
            if hasattr(ggm, "weight1") and hasattr(ggm, "weight2"):
                # GroupedMLPExpert: weight1 [E, H, F], weight2 [E, F, H]
                w1 = ggm.weight1
                w2 = ggm.weight2
                num_experts = w1.shape[0]
                for i in range(num_experts):
                    combined = paddle.concat([w1[i].flatten(), w2[i].flatten()])
                    expert_weights.append(combined)
            if expert_weights:
                norm_stats = _compute_expert_norms_paddle(expert_weights)
                metrics.update(norm_stats)
                routed_norm_mean = norm_stats["expert_norm_mean"]

        elif hasattr(moe_layer, "experts") and moe_layer.experts is not None:
            expert_weights = []
            for expert in moe_layer.experts:
                if expert is not None:
                    weights = [p.flatten() for p in expert.parameters()]
                    if weights:
                        expert_weights.append(paddle.concat(weights))
            if expert_weights:
                norm_stats = _compute_expert_norms_paddle(expert_weights)
                metrics.update(norm_stats)
                routed_norm_mean = norm_stats["expert_norm_mean"]

        # Shared expert norms
        if hasattr(moe_layer, "shared_experts") and moe_layer.shared_experts is not None:
            shared_weights = [p.flatten() for p in moe_layer.shared_experts.parameters()]
            if shared_weights:
                all_params = paddle.concat(shared_weights)
                shared_norm = float(all_params.astype("float32").norm())
                metrics["shared_expert_norm"] = shared_norm
                if routed_norm_mean is not None and routed_norm_mean > 1e-8:
                    metrics["shared_routed_ratio"] = shared_norm / routed_norm_mean

        log_dict = {}
        if self.log_per_layer:
            for name, val in metrics.items():
                log_dict[f"moe_health/layer_{layer_idx}/{name}"] = val
        if self.log_global:
            for name, val in metrics.items():
                log_dict[f"moe_health/global_{name}"] = val
        if log_dict:
            training_logs.update(**log_dict)


def setup_moe_monitor(
    model,
    log_per_layer=True,
    log_global=True,
    monitor_interval=1,
    verbose=False,
    monitor_dict=None,
):
    monitor = PaddleMoEMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
    )
    monitor.register_hooks(model)
    logger.info(f"[PaddleMoEMonitor] Setup complete. Monitoring {len(monitor.hooks)} hooks.")
    if monitor_dict is not None:
        monitor_dict["moe_health"] = monitor
    return model
