"""
MoE Health Metrics Computation Functions (Simplified).

Core metrics:
1. Router Entropy - 路由熵
2. Router Score Sum - TopK选择的score求和
3. Bias-Affinity Correlation - 使用Jaccard相似度
4. Expert Norms - 专家权重L2范数
5. Shared/Routed Ratio - Shared Expert与Routed Expert的比例

注意: 所有指标只计算本地值，EP 聚合在 training_logs 层面统一处理
(通过 gather_object_list 收集所有卡的结果，然后按 mean/max/min 聚合)
"""

import torch


def compute_router_entropy(router_probs: torch.Tensor) -> torch.Tensor:
    """
    计算 Router Entropy (路由熵).

    H = -sum(p * log(p))

    Args:
        router_probs: Softmax probabilities [tokens, num_experts] (scores_for_aux_loss)

    Returns:
        entropy_mean: 本地 batch 的平均熵 (Scalar Tensor)
    """
    probs = router_probs.float().clamp(min=1e-10)
    entropy = -(probs * probs.log()).sum(dim=-1)  # [tokens]
    return entropy.mean()


def compute_topk_score_sum(scores: torch.Tensor, topk: int) -> dict[str, torch.Tensor]:
    """
    计算 Router TopK Score Sum.

    Args:
        scores: Router scores [tokens, num_experts]，softmax/sigmoid 后的分数
        topk: TopK 值

    Returns:
        Dict with score_sum statistics
    """
    topk_scores, _ = scores.float().topk(topk, dim=-1)  # [tokens, topk]
    score_sum = topk_scores.sum(dim=-1)  # [tokens]
    return {
        "score_sum_mean": score_sum.mean(),
        "score_sum_min": score_sum.min(),
        "score_sum_max": score_sum.max(),
    }


def compute_bias_affinity_jaccard(
    routing_map_before_bias: torch.Tensor,
    routing_map_after_bias: torch.Tensor,
) -> torch.Tensor:
    """
    计算 Bias-Affinity Jaccard 相似度.

    Jaccard(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        routing_map_before_bias: Bias前的routing map [tokens, topk] 或 [tokens, num_experts]
        routing_map_after_bias: Bias后的routing map

    Returns:
        jaccard: Jaccard相似度 (0-1)
    """
    if routing_map_before_bias.dim() == 2:
        is_onehot = routing_map_before_bias.unique().numel() <= 2

        if not is_onehot:
            # expert indices -> one-hot
            num_tokens, topk = routing_map_before_bias.shape
            num_experts = int(max(routing_map_before_bias.max().item(), routing_map_after_bias.max().item())) + 1

            before_onehot = torch.zeros(
                num_tokens, num_experts, device=routing_map_before_bias.device, dtype=torch.bool
            )
            after_onehot = torch.zeros(num_tokens, num_experts, device=routing_map_after_bias.device, dtype=torch.bool)

            for k in range(topk):
                before_onehot.scatter_(1, routing_map_before_bias[:, k : k + 1].long(), True)
                after_onehot.scatter_(1, routing_map_after_bias[:, k : k + 1].long(), True)

            routing_map_before_bias = before_onehot
            routing_map_after_bias = after_onehot

    before = routing_map_before_bias.bool()
    after = routing_map_after_bias.bool()

    intersection = (before & after).float().sum()
    union = (before | after).float().sum()
    return intersection / union.clamp(min=1e-8)


def compute_expert_norms(expert_weights: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    计算 Expert Norms (专家权重L2范数).

    Args:
        expert_weights: 每个本地专家的权重张量列表

    Returns:
        Dict with norm statistics (本地专家)
    """
    if not expert_weights:
        return {
            "expert_norm_mean": torch.tensor(0.0),
            "expert_norm_std": torch.tensor(0.0),
            "expert_norm_min": torch.tensor(0.0),
            "expert_norm_max": torch.tensor(0.0),
        }

    norms = torch.stack([w.float().norm() for w in expert_weights])
    return {
        "expert_norm_mean": norms.mean(),
        "expert_norm_std": norms.std() if norms.numel() > 1 else torch.tensor(0.0, device=norms.device),
        "expert_norm_min": norms.min(),
        "expert_norm_max": norms.max(),
    }


def compute_shared_expert_norm(shared_expert_weights: list[torch.Tensor]) -> torch.Tensor:
    """
    计算 SharedExpert 的 L2 Norm.

    Args:
        shared_expert_weights: Shared Expert 的权重张量列表

    Returns:
        shared_norm: L2 Norm
    """
    if not shared_expert_weights:
        return torch.tensor(0.0)

    all_params = torch.cat([w.flatten() for w in shared_expert_weights])
    return all_params.float().norm()


def compute_shared_routed_ratio(
    shared_norm: torch.Tensor,
    routed_norm_mean: torch.Tensor,
) -> torch.Tensor:
    """
    计算 Shared/Routed Ratio.

    Args:
        shared_norm: SharedExpert 的 L2 Norm
        routed_norm_mean: Routed Experts 的平均 L2 Norm

    Returns:
        ratio: Shared/Routed Ratio
    """
    return shared_norm / routed_norm_mean.clamp(min=1e-8)
