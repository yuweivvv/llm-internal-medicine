"""
MoE Specialist Monitor for Megatron-Bridge.
Migrated from src/internal_medicine/moe_specialist/.
"""

import logging
import weakref
from typing import Any

import torch
import torch.nn as nn

from ...core.training_logs import training_logs
from .base import TorchProbe
from .moe_metrics import (
    compute_bias_affinity_jaccard,
    compute_expert_norms,
    compute_router_entropy,
    compute_shared_expert_norm,
    compute_shared_routed_ratio,
    compute_topk_score_sum,
)

logger = logging.getLogger(__name__)


_ROUTER_METRICS = (
    "router_entropy",
    "score_sum_mean",
    "score_sum_min",
    "score_sum_max",
    "expert_bias_mean",
    "expert_bias_std",
    "bias_affinity_jaccard",
)

_EXPERT_METRICS = (
    "expert_norm_mean",
    "expert_norm_std",
    "expert_norm_min",
    "expert_norm_max",
    "shared_expert_norm",
    "shared_routed_ratio",
)


class MoESpecialistMonitor(TorchProbe):
    METRIC_PREFIX = "moe_health"
    MAX_AGGREGATED = {"score_sum_max", "expert_norm_max"}
    MIN_AGGREGATED = {"score_sum_min", "expert_norm_min"}

    def __init__(
        self, log_per_layer=True, log_global=True, monitor_interval=1, verbose=False, hook_timing_enabled=False
    ):
        super().__init__(
            log_per_layer=log_per_layer,
            log_global=log_global,
            monitor_interval=monitor_interval,
            verbose=verbose,
            hook_timing_enabled=hook_timing_enabled,
        )
        self._monitored_moe_layers: list[tuple[int, weakref.ref]] = []
        self._patched_routers: list[weakref.ref] = []

    def register_hooks(self, model: nn.Module):
        self._init_parallel_state()
        targets = self._prepare_layers(model)
        if not targets:
            return
        self.allocate_buffers(next(model.parameters()).device)
        self._attach_hooks(targets)

    def _init_parallel_state(self):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
        except ImportError:
            pass

    def _prepare_layers(self, model: nn.Module) -> list[tuple[int, nn.Module]]:
        moe_layers = self._find_moe_layers(model)
        if len(moe_layers) == 0:
            logger.warning("[MoEMonitor] No MoE layers found!")
            return []
        if self.verbose:
            logger.info(f"[MoEMonitor] Found {len(moe_layers)} MoE layers.")

        for layer_idx, _ in moe_layers:
            for name in (*_ROUTER_METRICS, *_EXPERT_METRICS):
                self.declare_layer_metric(layer_idx, name)
        return moe_layers

    def _attach_hooks(self, targets: list[tuple[int, nn.Module]]):
        for layer_idx, moe_layer in targets:
            self._monitored_moe_layers.append((layer_idx, weakref.ref(moe_layer)))
            if hasattr(moe_layer, "router"):
                self._patch_router_cache(moe_layer.router)
                hook = moe_layer.router.register_forward_hook(
                    self.timed_hook("router", self._make_router_hook(layer_idx, moe_layer))
                )
                self.hooks.append(hook)

        logger.info(f"[MoEMonitor] Registered {len(self.hooks)} hooks on {len(targets)} layers.")

    def _patch_router_cache(self, router):
        if not hasattr(router, "_apply_aux_loss"):
            if self.verbose:
                logger.warning("[MoEMonitor] Router has no _apply_aux_loss; router metrics may be unavailable")
            return
        if getattr(router, "_im_patched", False):
            if self.verbose:
                logger.warning("[MoEMonitor] Router is already patched; skipping duplicate patch")
            return
        original_apply = router._apply_aux_loss
        monitor = self

        def patched_apply(probs, scores_for_aux_loss, routing_map, *args, **kwargs):
            if monitor._should_monitor():
                router._cached_scores_for_aux_loss = scores_for_aux_loss.detach()
                router._cached_routing_map_for_aux_loss = routing_map.detach()
            else:
                router._cached_scores_for_aux_loss = None
                router._cached_routing_map_for_aux_loss = None
            return original_apply(probs, scores_for_aux_loss, routing_map, *args, **kwargs)

        router._im_original_apply_aux_loss = original_apply
        router._apply_aux_loss = patched_apply
        router._im_patched = True
        self._patched_routers.append(weakref.ref(router))

    def _find_moe_layers(self, model: nn.Module) -> list[tuple[int, nn.Module]]:
        moe_layers = []
        if hasattr(model, "module"):
            model = model.module
        layers = None
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            layers = model.decoder.layers
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            layers = model.encoder.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        elif hasattr(model, "language_model"):
            lm = model.language_model
            if hasattr(lm, "decoder") and hasattr(lm.decoder, "layers"):
                layers = lm.decoder.layers
            elif hasattr(lm, "encoder") and hasattr(lm.encoder, "layers"):
                layers = lm.encoder.layers
        if layers is None:
            for _, module in model.named_modules():
                if module.__class__.__name__ in ("MoELayer", "BaseMoELayer"):
                    moe_layers.append((len(moe_layers), module))
            return moe_layers
        for local_idx, layer in enumerate(layers):
            global_idx = self._resolve_layer_idx(layer, local_idx, len(layers))
            moe_module = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                moe_module = layer.mlp
            elif hasattr(layer, "moe"):
                moe_module = layer.moe
            elif hasattr(layer, "router"):
                moe_module = layer
            if moe_module is not None:
                moe_layers.append((global_idx, moe_module))
        return moe_layers

    def _make_router_hook(self, layer_idx: int, moe_layer: nn.Module):
        def hook_fn(module, _inputs, outputs):
            if not self._should_monitor():
                for attr in ("_cached_scores_for_aux_loss", "_cached_routing_map_for_aux_loss"):
                    if hasattr(module, attr):
                        setattr(module, attr, None)
                return
            try:
                with torch.no_grad():
                    self._compute_router_metrics(layer_idx, module, outputs, moe_layer)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MoEMonitor] Router hook error layer {layer_idx}: {e}")
            finally:
                for attr in ("_cached_scores_for_aux_loss", "_cached_routing_map_for_aux_loss"):
                    if hasattr(module, attr):
                        setattr(module, attr, None)

        return hook_fn

    def _compute_expert_metrics_for_all_layers(self):
        """Compute expert/shared norms once per step.

        Expert weights are constant within a step (only optimizer.step() updates
        them). Running this per microbatch on the forward stream queues kernels
        that compete with EP a2a — see monitor-hook-perf-rules skill.
        """
        for layer_idx, moe_ref in self._monitored_moe_layers:
            moe_layer = moe_ref()
            if moe_layer is None:
                continue
            try:
                with torch.no_grad():
                    self._compute_expert_metrics(layer_idx, moe_layer)
            except Exception as e:
                if self.verbose:
                    logger.error(f"[MoEMonitor] Step expert-metric error layer {layer_idx}: {e}")

    def step(self):
        # Bypass TorchProbe._should_monitor's torch.is_grad_enabled() guard:
        # that guard exists to skip recompute-pass forward hooks, but step()
        # runs outside any forward and a caller-side no_grad wrapper must not
        # silently disable expert-weight metrics.
        if self.step_count % self.monitor_interval == 0:
            self._compute_expert_metrics_for_all_layers()
        super().step()

    def _compute_router_metrics(self, layer_idx, router, outputs, _moe_layer):
        topk = getattr(router, "topk", None)
        scores_for_aux_loss = getattr(router, "_cached_scores_for_aux_loss", None)

        if scores_for_aux_loss is not None:
            self.record_layer_metric(layer_idx, "router_entropy", compute_router_entropy(scores_for_aux_loss))
            if topk is not None:
                stats = compute_topk_score_sum(scores_for_aux_loss, topk)
                self.record_layer_metric(layer_idx, "score_sum_mean", stats["score_sum_mean"])
                self.record_layer_metric(layer_idx, "score_sum_min", stats["score_sum_min"])
                self.record_layer_metric(layer_idx, "score_sum_max", stats["score_sum_max"])

        if hasattr(router, "expert_bias") and router.expert_bias is not None:
            self.record_layer_metric(layer_idx, "expert_bias_mean", router.expert_bias.mean())
            self.record_layer_metric(layer_idx, "expert_bias_std", router.expert_bias.std())

        routing_map_for_aux_loss = getattr(router, "_cached_routing_map_for_aux_loss", None)
        if routing_map_for_aux_loss is not None and isinstance(outputs, tuple) and len(outputs) >= 2:
            routing_after = outputs[1]
            num_experts = getattr(router, "num_experts", None) or getattr(router, "num_moe_experts", None)
            jaccard = compute_bias_affinity_jaccard(routing_map_for_aux_loss, routing_after, num_experts=num_experts)
            self.record_layer_metric(layer_idx, "bias_affinity_jaccard", jaccard)

    def _compute_expert_metrics(self, layer_idx, moe_layer):
        # expert_norm_mean aggregates only this rank's local experts; the
        # flush-time global is correct across EP only when each rank holds
        # the same number of local experts (the typical EP layout).
        routed_norm_mean = None

        if hasattr(moe_layer, "experts") and moe_layer.experts is not None:
            experts = moe_layer.experts
            expert_weights = []
            if hasattr(experts, "weight1") and hasattr(experts, "weight2"):
                num_experts = experts.num_local_experts
                hidden_size = experts.config.hidden_size
                w1 = experts.weight1.data.view(num_experts, hidden_size, -1)
                w2 = experts.weight2.data.view(num_experts, -1, hidden_size)
                for i in range(num_experts):
                    combined = torch.cat([w1[i].flatten(), w2[i].flatten()])
                    expert_weights.append(combined)
            elif hasattr(experts, "linear_fc1"):
                num_experts = experts.num_local_experts
                for i in range(num_experts):
                    w1 = getattr(experts.linear_fc1, f"weight{i}", None)
                    w2 = getattr(experts.linear_fc2, f"weight{i}", None)
                    if w1 is not None and w2 is not None:
                        combined = torch.cat([w1.data.flatten(), w2.data.flatten()])
                        expert_weights.append(combined)
            elif hasattr(experts, "local_experts"):
                for expert in experts.local_experts:
                    weights = [p.data.flatten() for p in expert.parameters()]
                    if weights:
                        expert_weights.append(torch.cat(weights))

            if expert_weights:
                stats = compute_expert_norms(expert_weights)
                self.record_layer_metric(layer_idx, "expert_norm_mean", stats["expert_norm_mean"])
                self.record_layer_metric(layer_idx, "expert_norm_std", stats["expert_norm_std"])
                self.record_layer_metric(layer_idx, "expert_norm_min", stats["expert_norm_min"])
                self.record_layer_metric(layer_idx, "expert_norm_max", stats["expert_norm_max"])
                routed_norm_mean = stats["expert_norm_mean"]

        if hasattr(moe_layer, "shared_experts") and moe_layer.shared_experts is not None:
            shared_weights = [p.data for p in moe_layer.shared_experts.parameters()]
            if shared_weights:
                shared_norm = compute_shared_expert_norm(shared_weights)
                self.record_layer_metric(layer_idx, "shared_expert_norm", shared_norm)
                if routed_norm_mean is not None:
                    ratio = compute_shared_routed_ratio(shared_norm, routed_norm_mean)
                    self.record_layer_metric(layer_idx, "shared_routed_ratio", ratio)

    def remove_hooks(self):
        super().remove_hooks()
        for router_ref in self._patched_routers:
            router = router_ref()
            if router is None:
                continue
            original_apply = getattr(router, "_im_original_apply_aux_loss", None)
            if original_apply is not None:
                router._apply_aux_loss = original_apply
            for attr in (
                "_im_original_apply_aux_loss",
                "_im_patched",
                "_cached_scores_for_aux_loss",
                "_cached_routing_map_for_aux_loss",
            ):
                if hasattr(router, attr):
                    delattr(router, attr)
        self._monitored_moe_layers = []
        self._patched_routers = []

    def get_health_summary(self) -> dict[str, Any]:
        metrics = training_logs.get_latest(prefix="moe_health")
        summary = {"num_layers_monitored": len(self._monitored_moe_layers), "total_steps": self.step_count}
        for key, val in metrics.items():
            if "bias_affinity_jaccard" in key:
                summary["router_conflict"] = "SEVERE" if val < 0.3 else "WARNING" if val < 0.7 else "OK"
            if "shared_routed_ratio" in key:
                summary["shared_expert"] = "MONOPOLY" if val > 3.0 else "INEFFECTIVE" if val < 0.3 else "OK"
        return summary


def setup_moe_monitor(
    model,
    log_per_layer=True,
    log_global=True,
    monitor_interval=1,
    verbose=False,
    hook_timing_enabled=False,
    monitor_dict=None,
):
    monitor = MoESpecialistMonitor(
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
        hook_timing_enabled=hook_timing_enabled,
    )
    models = [model] if not isinstance(model, list) else model
    monitor._init_parallel_state()
    chunk_targets = []
    for m in models:
        chunk_targets.append((m, monitor._prepare_layers(m)))
    if any(targets for _, targets in chunk_targets):
        device = next((p.device for m in models for p in m.parameters()), None)
        assert device is not None, "no parameters across model chunks; cannot pick a device"
        monitor.allocate_buffers(device)
        for _, targets in chunk_targets:
            monitor._attach_hooks(targets)
    logger.info(f"[MoEMonitor] Setup complete. Monitoring {len(monitor.hooks)} hooks.")
    if monitor_dict is not None:
        monitor_dict["moe_health"] = monitor
    return model
