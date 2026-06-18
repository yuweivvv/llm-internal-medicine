"""Layer discovery helpers for PaddleFleet monitors."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class MonitorLayer:
    """A transformer layer together with its metric layer id."""

    idx: int
    layer: object
    is_mtp: bool = False


def get_decoder_layers(model) -> list[object] | None:
    """Find PaddleFleet decoder layers, including MTP wrapper layers."""
    if hasattr(model, "_layers") and hasattr(model._layers, "run_function"):
        model = model._layers
    if hasattr(model, "module"):
        model = model.module
    if hasattr(model, "run_function"):
        return list(model.run_function)
    if hasattr(model, "decoder") and hasattr(model.decoder, "layers"):
        return list(model.decoder.layers)
    if hasattr(model, "encoder") and hasattr(model.encoder, "layers"):
        return list(model.encoder.layers)
    if hasattr(model, "layers"):
        return list(model.layers)
    return None


def is_mtp_wrapper(layer) -> bool:
    """Return True for wrapper layers that contain a real MTP transformer layer."""
    inner = getattr(layer, "transformer_layer", None)
    return inner is not None and inner is not layer


def unwrap_mtp_layer(layer):
    """Return the transformer layer to hook for a possible MTP wrapper."""
    return getattr(layer, "transformer_layer", None) if is_mtp_wrapper(layer) else layer


def resolve_layer_idx(layer, local_idx: int, num_local_layers: int, pp_rank: int = 0, layer_offset: int = 0) -> int:
    """Resolve a PaddleFleet metric layer id without converting 0-based ids."""
    for attr in ("layer_idx", "layer_index", "idx", "layer_number"):
        value = getattr(layer, attr, None)
        if isinstance(value, int):
            return value
    return pp_rank * num_local_layers + layer_offset + local_idx


def iter_monitor_layers(
    layers: Iterable[object],
    matches: Callable[[object], bool],
    *,
    pp_rank: int = 0,
    layer_offset: int = 0,
) -> list[MonitorLayer]:
    """Return main + MTP layers that satisfy ``matches``.

    PaddleFleet MTP layers are wrappers whose real transformer layer lives at
    ``wrapper.transformer_layer``. Pipeline ``run_function`` also contains
    embedding, norm, empty, and LM head entries, so MTP metric ids must be
    assigned after matched main transformer layers rather than physical entries.
    """
    layers = list(layers)
    main_layers = [layer for layer in layers if not is_mtp_wrapper(layer)]
    mtp_wrappers = [layer for layer in layers if is_mtp_wrapper(layer)]
    matched_main_layers = [layer for layer in main_layers if matches(layer)]
    num_main_layers = len(matched_main_layers)
    monitor_layers: list[MonitorLayer] = []

    for local_idx, layer in enumerate(matched_main_layers):
        idx = resolve_layer_idx(layer, local_idx, num_main_layers, pp_rank=pp_rank, layer_offset=layer_offset)
        monitor_layers.append(MonitorLayer(idx=idx, layer=layer, is_mtp=False))

    next_mtp_idx = max((item.idx for item in monitor_layers), default=layer_offset - 1) + 1
    for mtp_idx, wrapper in enumerate(mtp_wrappers):
        layer = unwrap_mtp_layer(wrapper)
        if not matches(layer):
            continue
        idx = next_mtp_idx + mtp_idx
        monitor_layers.append(MonitorLayer(idx=idx, layer=layer, is_mtp=True))

    return monitor_layers
