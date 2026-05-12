"""
QK Stats Monitor for Megatron-Bridge + Transformer Engine.
Migrated from src/internal_medicine/qk_logits/.
"""

import logging

import torch
import torch.distributed as dist
import torch.nn as nn

from ...core.base_monitor import BaseMonitor
from ...core.training_logs import training_logs
from .triton_kernels import compute_qk_stats

logger = logging.getLogger(__name__)


class QKStatsMonitor(BaseMonitor):
    def __init__(
        self,
        causal: bool = True,
        use_triton: bool = True,
        log_per_layer: bool = True,
        log_global: bool = True,
        monitor_interval: int = 1,
        verbose: bool = False,
    ):
        super().__init__(
            log_per_layer=log_per_layer, log_global=log_global, monitor_interval=monitor_interval, verbose=verbose
        )
        self.causal = causal
        self.use_triton = use_triton
        self.compute_count = 0
        self.tp_size = 1
        self.tp_rank = 0
        self.tp_group = None

    def register_hooks(self, model: nn.Module):
        try:
            from megatron.core import parallel_state

            if parallel_state.model_parallel_is_initialized():
                self.tp_size = parallel_state.get_tensor_model_parallel_world_size()
                self.tp_rank = parallel_state.get_tensor_model_parallel_rank()
                self.pp_rank = parallel_state.get_pipeline_model_parallel_rank()
                self.tp_group = parallel_state.get_tensor_model_parallel_group()
        except ImportError:
            pass

        attention_layers = self._find_attention_layers(model)
        if len(attention_layers) == 0:
            logger.warning("[QKMonitor] No attention layers found!")
            return

        if self.verbose:
            logger.info(f"[QKMonitor] Found {len(attention_layers)} attention layers. TP={self.tp_size}")

        for layer_idx, attention_module in attention_layers:
            if hasattr(attention_module, "core_attention"):
                hook = attention_module.core_attention.register_forward_pre_hook(self._make_compute_hook(layer_idx))
                self.hooks.append(hook)

        logger.info(f"[QKMonitor] Registered {len(self.hooks)} hooks.")

    def _find_attention_layers(self, model: nn.Module) -> list[tuple[int, nn.Module]]:
        attention_layers = []
        if hasattr(model, "module"):
            model = model.module
        layers = None
        if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
            layers = model.decoder.layers
        elif hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
            layers = model.encoder.layers
        elif hasattr(model, "layers"):
            layers = model.layers
        if layers is None:
            return []
        for local_idx, layer in enumerate(layers):
            global_idx = self.pp_rank * len(layers) + local_idx
            attn = None
            if hasattr(layer, "self_attention"):
                attn = layer.self_attention
            elif hasattr(layer, "attention"):
                attn = layer.attention
            if attn:
                attention_layers.append((global_idx, attn))
        return attention_layers

    def _aggregate_tp_stats(self, stats: dict) -> dict:
        data = torch.tensor(
            [stats["max_global"], stats["mean_global"], stats["entropy_global"], stats["sink_global"]],
            device="cuda",
            dtype=torch.float32,
        )
        max_val = data[0:1].clone()
        if self.tp_group is not None:
            dist.all_reduce(max_val, op=dist.ReduceOp.MAX, group=self.tp_group)
        sum_vals = data[1:4].clone()
        if self.tp_group is not None:
            dist.all_reduce(sum_vals, op=dist.ReduceOp.SUM, group=self.tp_group)
        avg_vals = sum_vals / self.tp_size

        local_head_entropy = stats["entropy_per_head"].mean(dim=0)
        full_head_entropy = local_head_entropy
        if self.tp_size > 1 and self.tp_group is not None:
            gathered_list = [torch.zeros_like(local_head_entropy) for _ in range(self.tp_size)]
            dist.all_gather(gathered_list, local_head_entropy, group=self.tp_group)
            full_head_entropy = torch.cat(gathered_list)

        return {
            "max_global": max_val.item(),
            "mean_global": avg_vals[0].item(),
            "entropy_global": avg_vals[1].item(),
            "sink_global": avg_vals[2].item(),
            "entropy_per_head_tensor": full_head_entropy,
        }

    def _make_compute_hook(self, layer_idx: int):
        def hook_fn(module, args):
            if not torch.is_grad_enabled():
                return
            if not self._should_monitor():
                return
            try:
                query, key = args[0], args[1]
                with torch.no_grad():
                    stats = compute_qk_stats(query, key, causal=self.causal, use_triton=self.use_triton)
                if self.tp_size > 1 and dist.is_initialized():
                    stats = self._aggregate_tp_stats(stats)
                else:
                    stats["entropy_per_head_tensor"] = stats["entropy_per_head"].mean(dim=0)

                all_heads = stats["entropy_per_head_tensor"]
                metrics = {
                    "max": stats["max_global"],
                    "mean": stats["mean_global"],
                    "entropy_avg": stats["entropy_global"],
                    "sink": stats["sink_global"],
                    "entropy_min": all_heads.min().item(),
                    "entropy_max": all_heads.max().item(),
                    "entropy_std": all_heads.std().item(),
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
                logger.error(f"[QKMonitor] Error layer {layer_idx}: {e}")

        return hook_fn


def setup_qk_monitor(
    model: nn.Module,
    causal: bool = True,
    use_triton: bool = True,
    verbose: bool = False,
    log_per_layer: bool = True,
    log_global: bool = True,
    monitor_interval: int = 1,
    monitor_dict: dict | None = None,
    return_monitor: bool = False,
) -> nn.Module:
    monitor = QKStatsMonitor(
        causal=causal,
        use_triton=use_triton,
        log_per_layer=log_per_layer,
        log_global=log_global,
        monitor_interval=monitor_interval,
        verbose=verbose,
    )
    monitor.register_hooks(model)
    logger.info(f"[QKMonitor] Setup complete. Monitoring {len(monitor.hooks)} layers.")
    if monitor_dict is not None:
        monitor_dict["qk_stats"] = monitor
    if return_monitor:
        return model, monitor
    return model
