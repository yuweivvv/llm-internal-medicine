"""
MoE Monitor for PaddleFleet.

Monitors MoE router health and expert weight norms using paddle hooks.

PaddleFleet MoE structure:
- MoELayer.gate — varies by model:
    StandardMoEGate: (capacity, top_gate, top_idx, gates_masked, mask, token_priority, l_aux, l_zloss)
    Other gates may return different formats.
- MoELayer.experts (nn.LayerList) or .grouped_gemm_experts (GroupedMLPExpert)
- MoELayer.shared_experts (StandardMLPSharedExpert or None)

TODO: Gate output format is model-specific. When switching to a different model
family, verify that outputs[2] (top_idx) is still at index 2 in the gate's
return tuple. Currently adapted for StandardMoEGate (PaddleFormers).
"""

import logging

import paddle
import paddle.nn as nn

from .base import PaddleProbe

logger = logging.getLogger(__name__)


def _compute_router_entropy(probs):
    """Router entropy from probability distribution. probs: [tokens, experts]."""
    probs = probs.astype("float32").clip(min=1e-10)
    probs = probs / probs.sum(axis=-1, keepdim=True)
    entropy = -(probs * probs.log()).sum(axis=-1)
    return entropy.mean()


def _compute_bias_affinity_jaccard(top_idx_with_bias, gates_no_bias, k, n_group=1, topk_group=1):
    """Compute mean Jaccard similarity between routing with and without correction_bias.

    Paddle port of megatron/moe_metrics.compute_bias_affinity_jaccard, extended with
    group-limited topk support for PaddleFleet's TopKRouter.

    Args:
        top_idx_with_bias: [tokens, k] — actual routing indices (with bias)
        gates_no_bias: [tokens, experts] — original gate scores (no bias)
        k: num_experts_per_tok
        n_group / topk_group: group-limited topk params

    Returns:
        mean Jaccard similarity (1 = identical routing, 0 = completely different)
    """
    num_tokens, num_experts = gates_no_bias.shape

    if n_group > 1 and topk_group >= 1:
        group_size = num_experts // n_group
        gates_reshaped = gates_no_bias.reshape([num_tokens, n_group, group_size])
        group_max = gates_reshaped.max(axis=-1)
        _, top_groups = paddle.topk(group_max, topk_group, axis=-1)
        group_mask = paddle.zeros([num_tokens, n_group], dtype="int32")
        group_mask = group_mask.put_along_axis(top_groups, paddle.to_tensor(1, dtype="int32"), axis=1)
        group_mask = group_mask.unsqueeze(-1).expand([-1, -1, group_size]).reshape([num_tokens, num_experts])
        masked_gates = gates_no_bias.clone()
        masked_gates = paddle.where(group_mask > 0, masked_gates, paddle.full_like(masked_gates, float("-inf")))
        _, top_idx_no_bias = paddle.topk(masked_gates, k, axis=-1)
    else:
        _, top_idx_no_bias = paddle.topk(gates_no_bias, k, axis=-1)

    set_with = paddle.zeros([num_tokens, num_experts], dtype="int32")
    set_without = paddle.zeros([num_tokens, num_experts], dtype="int32")
    set_with = set_with.put_along_axis(top_idx_with_bias, paddle.to_tensor(1, dtype="int32"), axis=1)
    set_without = set_without.put_along_axis(top_idx_no_bias, paddle.to_tensor(1, dtype="int32"), axis=1)

    intersection = (set_with & set_without).astype("float32").sum()
    union = (set_with | set_without).astype("float32").sum()
    return intersection / union.clip(min=1.0)


def _compute_expert_norms_paddle(weight_list):
    """Compute L2 norms for a list of paddle weight tensors. Returns GPU tensors."""
    if not weight_list:
        return {}
    norms = paddle.stack([w.astype("float32").norm() for w in weight_list])
    return {
        "expert_norm_mean": norms.mean(),
        "expert_norm_std": norms.std() if norms.numel() > 1 else paddle.zeros(()),
        "expert_norm_min": norms.min(),
        "expert_norm_max": norms.max(),
    }


class PaddleMoEMonitor(PaddleProbe):
    METRIC_PREFIX = "moe_health"
    MAX_AGGREGATED = {"score_sum_max", "expert_norm_max", "expert_bias_max"}
    MIN_AGGREGATED = {"score_sum_min", "expert_norm_min", "expert_bias_min"}

    def __init__(self, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False):
        super().__init__(
            log_per_layer=log_per_layer, log_global=log_global, monitor_interval=monitor_interval, verbose=verbose
        )
        self._patched_gates = []

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

        # Declare metric schema
        for layer_idx, moe_layer in moe_layers:
            gate_metrics = ["router_entropy", "score_sum_mean", "score_sum_min", "score_sum_max"]
            if hasattr(moe_layer, "gate") and hasattr(moe_layer.gate, "e_score_correction_bias"):
                gate_metrics += [
                    "bias_affinity_jaccard",
                    "expert_bias_mean",
                    "expert_bias_std",
                    "expert_bias_max",
                    "expert_bias_min",
                ]
            expert_metrics = [
                "expert_norm_mean",
                "expert_norm_std",
                "expert_norm_min",
                "expert_norm_max",
                "shared_expert_norm",
                "shared_routed_ratio",
            ]
            for m in gate_metrics + expert_metrics:
                self.declare_layer_metric(layer_idx, m)

        self.allocate_buffers()

        for layer_idx, moe_layer in moe_layers:
            if hasattr(moe_layer, "gate"):
                self._patch_gate_cache(moe_layer.gate)
                hook = moe_layer.gate.register_forward_post_hook(self._make_gate_hook(layer_idx, moe_layer))
                self.hooks.append(hook)
            hook = moe_layer.register_forward_post_hook(self._make_moe_layer_hook(layer_idx, moe_layer))
            self.hooks.append(hook)

        logger.info(f"[PaddleMoEMonitor] Registered {len(self.hooks)} hooks on {len(moe_layers)} layers.")

    def _patch_gate_cache(self, gate):
        """Monkey-patch gate.gate_score_func to cache pre-bias gates."""
        if not hasattr(gate, "gate_score_func"):
            if self.verbose:
                logger.warning("[PaddleMoEMonitor] Gate has no gate_score_func; router metrics may be unavailable")
            return
        if hasattr(gate, "_im_patched"):
            if self.verbose:
                logger.warning("[PaddleMoEMonitor] Gate is already patched; skipping duplicate patch")
            return
        original_fn = gate.gate_score_func
        monitor = self

        def cached_gate_score_func(logits):
            result = original_fn(logits)
            if monitor._should_monitor():
                gate._cached_gates = result.detach()
            else:
                gate._cached_gates = None
            return result

        gate._im_original_gate_score_func = original_fn
        gate.gate_score_func = cached_gate_score_func

        # Also patch _hash_routing for hash-routed layers (DeepSeek V4+).
        # Hash layers return early from forward() without calling gate_score_func,
        # so we intercept _hash_routing to capture the scores computed there.
        if hasattr(gate, "_hash_routing"):
            original_hash_routing = gate._hash_routing

            def cached_hash_routing(logits, flat_ids):
                result = original_hash_routing(logits, flat_ids)
                # _hash_routing computes scores internally (softmax/sigmoid on logits).
                # Recompute here to cache the full [N, num_experts] score distribution.
                import paddle.nn.functional as F

                logits_fp32 = logits.cast("float32")
                scoring_func = getattr(gate, "scoring_func", "softmax")
                if scoring_func == "softmax":
                    scores = F.softmax(logits_fp32, axis=-1)
                elif scoring_func == "sigmoid":
                    scores = F.sigmoid(logits_fp32)
                else:
                    scores = F.softmax(logits_fp32, axis=-1)
                gate._cached_gates = scores.detach()
                return result

            gate._hash_routing = cached_hash_routing

        gate._im_patched = True
        self._patched_gates.append(gate)

    def _find_moe_layers(self, model: nn.Layer) -> list[tuple[int, nn.Layer]]:
        moe_layers = []
        layers = self._get_decoder_layers(model)
        if layers is None:
            for _name, sublayer in model.named_sublayers():
                if sublayer.__class__.__name__ == "MoELayer":
                    moe_layers.append((len(moe_layers), sublayer))
            return moe_layers

        for local_idx, layer in enumerate(layers):
            global_idx = self._resolve_layer_idx(layer, local_idx, len(layers))
            moe_module = None
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
                if hasattr(layer, "_cached_gates"):
                    layer._cached_gates = None
                return
            if not self._should_monitor():
                if hasattr(layer, "_cached_gates"):
                    layer._cached_gates = None
                return
            try:
                with paddle.no_grad():
                    self._compute_gate_metrics(layer_idx, layer, outputs, moe_layer)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[PaddleMoEMonitor] Gate hook error layer {layer_idx}: {e}")
            finally:
                if hasattr(layer, "_cached_gates"):
                    layer._cached_gates = None

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
        """Compute router metrics from gate forward output."""
        cached_gates = getattr(gate, "_cached_gates", None)
        k = getattr(gate, "num_experts_per_tok", None)

        if cached_gates is None:
            if self.verbose:
                logger.warning(f"[PaddleMoEMonitor] layer {layer_idx}: _cached_gates is None, gate patch may not work")
            return

        self.record_layer_metric(layer_idx, "router_entropy", _compute_router_entropy(cached_gates))
        if k is not None:
            topk_vals, _ = paddle.topk(cached_gates, k, axis=-1)
            score_sum = topk_vals.sum(axis=-1)
            self.record_layer_metric(layer_idx, "score_sum_mean", score_sum.mean())
            self.record_layer_metric(layer_idx, "score_sum_min", score_sum.min())
            self.record_layer_metric(layer_idx, "score_sum_max", score_sum.max())

        if hasattr(gate, "e_score_correction_bias"):
            top_idx_with_bias = None
            if isinstance(outputs, tuple) and len(outputs) >= 3:
                top_idx_with_bias = outputs[2]
            if top_idx_with_bias is not None and k is not None:
                n_group = getattr(gate, "n_group", 1)
                topk_group = getattr(gate, "topk_group", 1)
                self.record_layer_metric(
                    layer_idx,
                    "bias_affinity_jaccard",
                    _compute_bias_affinity_jaccard(top_idx_with_bias, cached_gates, k, n_group, topk_group),
                )
            bias = gate.e_score_correction_bias
            self.record_layer_metric(layer_idx, "expert_bias_mean", bias.mean())
            self.record_layer_metric(layer_idx, "expert_bias_std", bias.std())
            self.record_layer_metric(layer_idx, "expert_bias_max", bias.max())
            self.record_layer_metric(layer_idx, "expert_bias_min", bias.min())

    def _compute_expert_metrics(self, layer_idx, moe_layer):
        routed_norm_mean = None

        if hasattr(moe_layer, "grouped_gemm_experts") and moe_layer.grouped_gemm_experts is not None:
            ggm = moe_layer.grouped_gemm_experts
            expert_weights = []
            if hasattr(ggm, "weight1") and hasattr(ggm, "weight2"):
                w1 = ggm.weight1
                w2 = ggm.weight2
                num_experts = w1.shape[0]
                for i in range(num_experts):
                    combined = paddle.concat([w1[i].flatten(), w2[i].flatten()])
                    expert_weights.append(combined)
            if expert_weights:
                norm_stats = _compute_expert_norms_paddle(expert_weights)
                for name, val in norm_stats.items():
                    self.record_layer_metric(layer_idx, name, val)
                routed_norm_mean = norm_stats.get("expert_norm_mean")

        elif hasattr(moe_layer, "experts") and moe_layer.experts is not None:
            expert_weights = []
            for expert in moe_layer.experts:
                if expert is not None:
                    weights = [p.flatten() for p in expert.parameters()]
                    if weights:
                        expert_weights.append(paddle.concat(weights))
            if expert_weights:
                norm_stats = _compute_expert_norms_paddle(expert_weights)
                for name, val in norm_stats.items():
                    self.record_layer_metric(layer_idx, name, val)
                routed_norm_mean = norm_stats.get("expert_norm_mean")

        if hasattr(moe_layer, "shared_experts") and moe_layer.shared_experts is not None:
            shared_weights = [p.flatten() for p in moe_layer.shared_experts.parameters()]
            if shared_weights:
                all_params = paddle.concat(shared_weights)
                shared_norm = all_params.astype("float32").norm()
                self.record_layer_metric(layer_idx, "shared_expert_norm", shared_norm)
                if routed_norm_mean is not None:
                    # clip 防止除零（对齐 megatron compute_shared_routed_ratio），保持 GPU 张量无 D2H
                    self.record_layer_metric(
                        layer_idx, "shared_routed_ratio", shared_norm / routed_norm_mean.clip(min=1e-8)
                    )

    def remove_hooks(self):
        super().remove_hooks()
        for gate in self._patched_gates:
            original_fn = getattr(gate, "_im_original_gate_score_func", None)
            if original_fn is not None:
                gate.gate_score_func = original_fn
            for attr in ("_im_original_gate_score_func", "_im_patched", "_cached_gates"):
                if hasattr(gate, attr):
                    delattr(gate, attr)
        self._patched_gates = []

    def step(self):
        super().step()


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
