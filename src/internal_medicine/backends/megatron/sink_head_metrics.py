"""
Enhanced QK Stats: Sink Head Classification Metrics.

Extends the base QK Stats Monitor with per-head sink classification,
based on findings from Sun et al. (2026) arXiv:2603.05498:

- Sink heads: heads where token-0 attention > threshold (typically >0.3)
  These heads act as "learned gates" that modulate attention output.
- Non-sink heads: heads with normal attention distribution.

Additional metrics:
    qk_stats/.../sink_head_ratio   — fraction of heads classified as sink heads
    qk_stats/.../sink_head_max     — strongest sink head's token-0 weight
    qk_stats/.../sink_nonsink_gap  — mean sink weight gap between sink/non-sink heads

Reference:
    Sun, S., Canziani, A., LeCun, Y., & Zhu, J. (2026).
    The Spike, the Sparse and the Sink. arXiv:2603.05498.
"""

import torch


def compute_sink_head_classification(
    sink_per_head: torch.Tensor,
    threshold: float = 0.3,
) -> dict[str, torch.Tensor]:
    """Classify attention heads as sink vs non-sink.

    A "sink head" is one where the average attention weight on token-0
    exceeds the threshold. These heads act as implicit gates — they dump
    excess attention mass onto a fixed position rather than distributing
    it semantically (Section 4.3 of Sun et al. 2026).

    Args:
        sink_per_head: [num_heads] mean attention weight on token-0 per head.
            (After TP gather, this should be the full head set.)
        threshold: attention weight above which a head is classified as "sink".
            Default 0.3 based on empirical observation that sink heads
            typically allocate >30% attention to token-0.

    Returns:
        Dict with:
            sink_head_ratio: fraction of heads classified as sink (0.0 to 1.0)
            sink_head_max: maximum sink weight across all heads
            sink_nonsink_gap: mean(sink_heads) - mean(nonsink_heads)
                Measures the logit gap proxy. Higher gap = more extreme sinks.
    """
    if sink_per_head.numel() == 0:
        zeros = torch.tensor(0.0, device=sink_per_head.device)
        return {
            "sink_head_ratio": zeros,
            "sink_head_max": zeros,
            "sink_nonsink_gap": zeros,
        }

    is_sink = sink_per_head > threshold
    num_heads = sink_per_head.numel()
    sink_count = is_sink.sum().float()

    sink_head_ratio = sink_count / num_heads
    sink_head_max = sink_per_head.max()

    # Compute gap between sink and non-sink heads
    if sink_count > 0 and sink_count < num_heads:
        sink_mean = sink_per_head[is_sink].mean()
        nonsink_mean = sink_per_head[~is_sink].mean()
        gap = sink_mean - nonsink_mean
    elif sink_count == num_heads:
        # All heads are sinks — gap is vs 0
        gap = sink_per_head.mean()
    else:
        # No sinks — gap is 0
        gap = torch.tensor(0.0, device=sink_per_head.device)

    return {
        "sink_head_ratio": sink_head_ratio,
        "sink_head_max": sink_head_max,
        "sink_nonsink_gap": gap,
    }
