"""
Triton kernels for efficient QK attention statistics computation.
Includes: Max Logits, Mean Logits, Attention Entropy, and Attention Sink Weights.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def qk_stats_kernel(
    # Pointers
    Q_ptr,
    K_ptr,
    max_logits_ptr,
    mean_logits_ptr,
    entropy_ptr,
    sink_ptr,
    count_ptr,  # Valid token count per head
    # Shapes
    batch,
    num_heads,
    seq_len,
    head_dim,
    # Strides
    stride_q_batch,
    stride_q_head,
    stride_q_seq,
    stride_q_dim,
    stride_k_batch,
    stride_k_head,
    stride_k_seq,
    stride_k_dim,
    stride_out_batch,
    stride_out_head,
    # Configs
    scale: tl.constexpr,
    apply_causal_mask: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """
    Fused kernel for computing Attention statistics using Online Softmax logic.
    Computes per-head:
    1. Max Logit (Global max in the attention matrix)
    2. Mean Logit (Average of unnormalized logits)
    3. Entropy (Shannon entropy of the Softmax distribution)
    4. Sink Weight (Attention weight assigned to the first token)
    """
    # Grid: (batch * num_heads)
    pid = tl.program_id(0)
    batch_idx = pid // num_heads
    head_idx = pid % num_heads

    # Base pointers
    q_base = Q_ptr + batch_idx * stride_q_batch + head_idx * stride_q_head
    k_base = K_ptr + batch_idx * stride_k_batch + head_idx * stride_k_head

    # --- Global Accumulators for the Head ---
    # These accumulate statistics across all queries (rows) in the head
    head_max_logit = -1e10
    head_sum_logit = 0.0
    head_valid_count = 0.0

    head_sum_entropy = 0.0
    head_sum_sink = 0.0
    head_valid_rows = 0.0

    # Loop over Query blocks (Rows)
    for m_start in range(0, seq_len, BLOCK_M):
        m_offsets = m_start + tl.arange(0, BLOCK_M)
        m_mask = m_offsets < seq_len

        # --- Row-wise Accumulators (for Online Softmax) ---
        # m_i: current max logit for each row
        # d_i: current denominator (sum exp) for each row
        # h_i: current unnormalized entropy term sum((x-m)*exp(x-m))
        # s_i: current unnormalized sink weight (exp(x_0 - m))

        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
        d_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        h_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        s_i = tl.zeros([BLOCK_M], dtype=tl.float32)

        # Track raw logits stats for this block
        row_max_logit_raw = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e10
        row_sum_logit_raw = tl.zeros([BLOCK_M], dtype=tl.float32)
        row_count_raw = tl.zeros([BLOCK_M], dtype=tl.float32)

        # Loop over Key blocks (Columns)
        for n_start in range(0, seq_len, BLOCK_N):
            n_offsets = n_start + tl.arange(0, BLOCK_N)
            n_mask = n_offsets < seq_len

            # 1. Compute QK^T for this block [BLOCK_M, BLOCK_N]
            acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

            for k_start in range(0, head_dim, BLOCK_K):
                k_offsets = k_start + tl.arange(0, BLOCK_K)
                k_mask = k_offsets < head_dim

                # Load Q: [BLOCK_M, BLOCK_K]
                q_ptr = q_base + m_offsets[:, None] * stride_q_seq + k_offsets[None, :] * stride_q_dim
                q = tl.load(q_ptr, mask=m_mask[:, None] & k_mask[None, :], other=0.0)

                # Load K: [BLOCK_N, BLOCK_K]
                k_ptr = k_base + n_offsets[:, None] * stride_k_seq + k_offsets[None, :] * stride_k_dim
                k = tl.load(k_ptr, mask=n_mask[:, None] & k_mask[None, :], other=0.0)  # Transpose handled in dot

                acc += tl.dot(q, tl.trans(k))

            # 2. Scale and Mask
            logits = acc * scale

            mask_val = n_mask[None, :] & m_mask[:, None]
            if apply_causal_mask:
                causal = m_offsets[:, None] >= n_offsets[None, :]
                mask_val = mask_val & causal

            # Apply mask (set to -inf)
            logits = tl.where(mask_val, logits, -1e10)

            # --- Update Raw Stats (Max/Mean Logits) ---
            block_max = tl.max(logits, 1)
            row_max_logit_raw = tl.maximum(row_max_logit_raw, block_max)

            # For sum/mean, we only count valid masked values
            # Using 0.0 for invalid positions to not affect sum
            logits_zeroed = tl.where(mask_val, logits, 0.0)
            row_sum_logit_raw += tl.sum(logits_zeroed, 1)
            row_count_raw += tl.sum(mask_val.to(tl.float32), 1)

            # --- Online Softmax Update ---
            # Current block max for stability
            m_curr = block_max  # [BLOCK_M]

            # Compute new global max for these rows
            m_new = tl.maximum(m_i, m_curr)

            # Scaling factors
            alpha_prev = tl.exp(m_i - m_new)
            alpha_curr = tl.exp(logits - m_new[:, None])
            # Apply mask to exponentials (0.0 for invalid)
            alpha_curr = tl.where(mask_val, alpha_curr, 0.0)

            # Sum of exp for current block
            d_curr = tl.sum(alpha_curr, 1)

            # Update Denominator (L)
            d_new = d_i * alpha_prev + d_curr

            # Update Entropy Term H: sum((x - m_new) * exp(x - m_new))
            # H_new = H_prev * alpha_prev + (m_prev - m_new) * D_prev * alpha_prev + sum((x - m_new) * exp)

            # Term from previous blocks adjusted for new max
            h_term_prev = h_i * alpha_prev + (m_i - m_new) * d_i * alpha_prev

            # Term from current block
            # (x - m_new) * exp(x - m_new)
            logits_diff = logits - m_new[:, None]
            h_term_curr = tl.sum(tl.where(mask_val, logits_diff * alpha_curr, 0.0), 1)

            h_new = h_term_prev + h_term_curr

            # Update Sink (Attention to token 0)
            # Check if column 0 is in this block
            if n_start == 0:
                # Column 0 is at local index 0
                is_first = n_offsets == 0  # [BLOCK_N]
                # Extract sink logits: exp(x_0 - m_new)
                # alpha_curr is already exp(x - m_new) and masked
                # We need column 0.
                sink_contrib = tl.sum(tl.where(is_first[None, :], alpha_curr, 0.0), 1)
                s_new = s_i * alpha_prev + sink_contrib
            else:
                s_new = s_i * alpha_prev

            # Commit updates
            m_i = m_new
            d_i = d_new
            h_i = h_new
            s_i = s_new

        # --- Finalize Block Rows ---
        # 1. Entropy = Log(L) - (1/L) * Sum((x-m)*exp(x-m))
        # H = log(d_i) - h_i / d_i
        log_d = tl.log(d_i)
        row_entropy = log_d - (h_i / d_i)

        # 2. Sink Weight = s_i / d_i
        row_sink = s_i / d_i

        # Mask out rows that were purely padding (count == 0)
        row_has_data = row_count_raw > 0

        # Accumulate to Head Globals
        head_max_logit = tl.maximum(head_max_logit, tl.max(row_max_logit_raw))
        head_sum_logit += tl.sum(tl.where(row_has_data, row_sum_logit_raw, 0.0))
        head_valid_count += tl.sum(row_count_raw)

        head_sum_entropy += tl.sum(tl.where(row_has_data & m_mask, row_entropy, 0.0))
        head_sum_sink += tl.sum(tl.where(row_has_data & m_mask, row_sink, 0.0))
        head_valid_rows += tl.sum((row_has_data & m_mask).to(tl.float32))

    # --- Write Output ---
    out_offset = batch_idx * stride_out_batch + head_idx * stride_out_head

    # Avoid division by zero
    safe_count = tl.maximum(head_valid_count, 1.0)
    safe_rows = tl.maximum(head_valid_rows, 1.0)

    tl.store(max_logits_ptr + out_offset, head_max_logit)
    tl.store(mean_logits_ptr + out_offset, head_sum_logit / safe_count)
    tl.store(entropy_ptr + out_offset, head_sum_entropy / safe_rows)
    tl.store(sink_ptr + out_offset, head_sum_sink / safe_rows)
    tl.store(count_ptr + out_offset, head_valid_count)


def compute_qk_stats_triton(q: torch.Tensor, k: torch.Tensor, causal: bool = True) -> dict:
    """
    Compute QK statistics using optimized Triton kernel.
    Input: [B, H, S, D] (already permuted by compute_qk_stats).
    Returns: Max Logits, Mean Logits, Entropy, Sink Weights.
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / (head_dim**0.5)

    # Output tensors [batch, num_heads]
    max_logits = torch.empty((batch, num_heads), device=q.device, dtype=torch.float32)
    mean_logits = torch.empty((batch, num_heads), device=q.device, dtype=torch.float32)
    entropy = torch.empty((batch, num_heads), device=q.device, dtype=torch.float32)
    sink = torch.empty((batch, num_heads), device=q.device, dtype=torch.float32)
    count = torch.empty((batch, num_heads), device=q.device, dtype=torch.float32)

    grid = (batch * num_heads,)

    # Tuning block sizes for performance
    BLOCK_M = 64
    BLOCK_N = 64
    BLOCK_K = 64
    if head_dim > 64:
        BLOCK_K = 128

    qk_stats_kernel[grid](
        q,
        k,
        max_logits,
        mean_logits,
        entropy,
        sink,
        count,
        batch,
        num_heads,
        seq_len,
        head_dim,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        max_logits.stride(0),
        max_logits.stride(1),
        scale=scale,
        apply_causal_mask=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return {
        "max_per_head": max_logits,
        "mean_per_head": mean_logits,
        "entropy_per_head": entropy,
        "sink_per_head": sink,
        "max_global": max_logits.max().item(),
        "mean_global": mean_logits.mean().item(),
        "entropy_global": entropy.mean().item(),
        "sink_global": sink.mean().item(),
    }


def compute_qk_stats_pytorch(q: torch.Tensor, k: torch.Tensor, causal: bool = True) -> dict:
    """
    Reference PyTorch implementation including Entropy and Sink.
    Input: [B, H, S, D] (already permuted by compute_qk_stats).
    """
    batch, num_heads, seq_len, head_dim = q.shape
    scale = 1.0 / (head_dim**0.5)

    # 1. Logits
    # [B, H, S, D] @ [B, H, D, S] -> [B, H, S, S]
    logits = torch.matmul(q, k.transpose(-2, -1)) * scale

    if causal:
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool), diagonal=1)
        logits.masked_fill_(mask, float("-inf"))

    # 2. Stats: Max & Mean (Raw Logits)
    # Filter out -inf for mean calculation
    valid_mask = logits > -1e9
    max_per_head = logits.max(dim=-1)[0].max(dim=-1)[0]

    logits_zeroed = torch.where(valid_mask, logits, torch.tensor(0.0, device=q.device))
    sum_logits = logits_zeroed.sum(dim=(-2, -1))
    count = valid_mask.sum(dim=(-2, -1))
    mean_per_head = sum_logits / count.clamp(min=1)

    # 3. Softmax & Entropy & Sink
    # Softmax along last dim (keys)
    probs = torch.softmax(logits, dim=-1)  # [B, H, S, S]

    # Entropy: -sum(p * log(p))
    # Note: masked positions have p=0, 0*log0=0 (handled by where)
    log_probs = torch.log_softmax(logits, dim=-1)
    entropy_map = -(probs * log_probs)
    # Mask out invalid (causal) entries which might be NaN
    entropy_map = torch.where(valid_mask, entropy_map, torch.tensor(0.0, device=q.device))
    row_entropy = entropy_map.sum(dim=-1)  # [B, H, S]
    # Average entropy over valid queries (rows)
    # For causal, first row has 1 valid, second 2...
    avg_entropy = row_entropy.mean(dim=-1)  # [B, H]

    # Sink: Probability of token 0
    sink_probs = probs[..., 0]  # [B, H, S]
    avg_sink = sink_probs.mean(dim=-1)  # [B, H]

    return {
        "max_per_head": max_per_head,
        "mean_per_head": mean_per_head,
        "entropy_per_head": avg_entropy,
        "sink_per_head": avg_sink,
        "max_global": max_per_head.max().item(),
        "mean_global": mean_per_head.mean().item(),
        "entropy_global": avg_entropy.mean().item(),
        "sink_global": avg_sink.mean().item(),
    }


def compute_qk_stats(q: torch.Tensor, k: torch.Tensor, causal: bool = True, use_triton: bool = True) -> dict:
    """
    Unified entry point. Input layout: [S, B, H, D] (Megatron core_attention convention).
    Permutes to [B, H, S, D] before dispatching to backend.
    """
    seq_len, batch, num_q_heads, head_dim = q.shape
    _, _, num_k_heads, _ = k.shape

    if num_q_heads != num_k_heads:
        heads_per_group = num_q_heads // num_k_heads
        k = k.repeat_interleave(heads_per_group, dim=2)

    # Permute [S, B, H, D] -> [B, H, S, D] for both backends
    q = q.permute(1, 2, 0, 3).contiguous()
    k = k.permute(1, 2, 0, 3).contiguous()

    if use_triton and torch.cuda.is_available():
        try:
            return compute_qk_stats_triton(q, k, causal)
        except Exception as e:
            print(f"Triton kernel error: {e}. Falling back to PyTorch.")

    return compute_qk_stats_pytorch(q, k, causal)
