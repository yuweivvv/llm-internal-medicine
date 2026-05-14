# QK Stats Monitor

QK 注意力统计监控模块，监控 7 个核心指标。

通过 Triton 优化的 Online Softmax 内核，在单次前向传播中高效计算注意力矩阵的统计特征，覆盖数值稳定性、注意力集中度和 attention sink 现象。

---

## 监控指标

### 1. Max Logit (注意力 Logit 最大值)

**数学公式:**
```
max = max(Q × K^T / √d)
```

对注意力矩阵中所有有效位置（考虑因果 mask 和 padding）取全局最大值，然后在所有 head 之间取最大，再在 batch 之间取最大。

**TP 聚合:** `all_reduce(MAX)` 跨 TP ranks

**诊断意义:**
- 过大的值预示数值不稳定，可能导致 softmax 溢出
- 训练过程中应保持稳定，突增需要警惕

---

### 2. Mean Logit (注意力 Logit 均值)

**数学公式:**
```
mean = sum(valid_logits) / count(valid_positions)
```

对注意力矩阵中所有有效位置（排除因果 mask 的 `-inf` 位置）求均值，然后在所有 head 和 batch 之间取均值。

**TP 聚合:** `all_reduce(SUM) / tp_size` 跨 TP ranks

**诊断意义:**
- 反映注意力 logit 的基准量级
- 值过大或过小可能表示 Q/K 投影权重的量级异常

---

### 3. Entropy Avg (注意力熵均值)

**数学公式:**

标准 Shannon 熵：
```
H = -Σ(p_i × log(p_i))
```

其中 `p = softmax(Q × K^T / √d)` 为注意力概率分布。

**Triton 实现 (Online Softmax):**
```
entropy = log(L) - (1/L) × Σ((x_i - m) × exp(x_i - m))
```

其中 `m = max(logits)` 为行最大值，`L = Σ(exp(x_i - m))` 为 softmax 分母。该公式与标准 Shannon 熵数学等价，但通过 Online Softmax 可以单次遍历完成计算。

对每个 query 行计算熵后取均值，再在所有 head 和 batch 之间取均值。

**TP 聚合:** `all_reduce(SUM) / tp_size` 跨 TP ranks

**诊断意义:**
- **低熵**: 注意力集中在少数 token 上，可能存在注意力坍缩
- **高熵**: 注意力分布均匀，模型对当前位置的关注不够聚焦
- 正常训练过程中应随层深度变化呈现合理梯度

---

### 4. Sink (Attention Sink 权重)

**数学公式:**
```
sink = mean(softmax(logits)[..., 0])
```

即 attention 概率分布中分配给第一个 token (position 0) 的权重，对所有 query 行取均值。

**Triton 实现:**
```
sink_per_row = exp(x_0 - m) / L
```

其中 `x_0` 为 position 0 对应的 logit 值，通过 Online Softmax 状态变量 `s_i` 追踪。

**TP 聚合:** `all_reduce(SUM) / tp_size` 跨 TP ranks

**诊断意义:**
- **高 sink 值**: 出现 "attention sink" 现象 — 大量注意力被分配到第一个 token，而非语义相关的位置
- 是 LLM 训练中常见的现象，但过度 sink 可能影响模型质量

---

### 5. Entropy Min (最小熵 Head)

**数学公式:**
```
entropy_min = min(entropy_per_head)
```

从所有 head（跨 TP ranks gather 后）中选取熵最小的 head。

**诊断意义:**
- 识别最"尖锐"的 head，即注意力最集中的 head
- 极低值可能表示某个 head 发生了注意力坍缩

---

### 6. Entropy Max (最大熵 Head)

**数学公式:**
```
entropy_max = max(entropy_per_head)
```

从所有 head 中选取熵最大的 head。

**诊断意义:**
- 识别最"分散"的 head，即注意力最均匀的 head
- 极高值可能表示某个 head 未学到有效的注意力模式

---

### 7. Entropy Std (熵标准差)

**数学公式:**
```
entropy_std = std(entropy_per_head)
```

所有 head 的熵的标准差。

**诊断意义:**
- **高标准差**: 各 head 行为分化严重，部分 head 非常集中而部分非常分散
- **低标准差**: 各 head 行为一致
- 适度的标准差是健康的，说明不同 head 承担了不同的注意力模式

---


---

### 8. Sink Head Ratio (Sink Head 占比)

**数学公式:**
```
sink_head_ratio = count(sink_per_head > threshold) / num_heads
```

被分类为 "sink head" 的 head 占总 head 数的比例。默认阈值 0.3。

**理论基础:**
Sun et al. (2026, arXiv:2603.05498) 证明 attention sinks 是 per-head 现象：
某些 head 将 >30% 注意力分配给 token-0，作为 "learned gate" 来关闭不需要的信息通道。

**诊断意义:**
- **0.0 ~ 0.3**: 正常，少数 head 有 sink 行为
- **0.3 ~ 0.6**: 显著 sink 行为，模型在积极使用 implicit gating
- **> 0.6**: 大部分 head 都是 sink head，可能影响长上下文能力

---

### 9. Sink Head Max (最强 Sink Head 权重)

**数学公式:**
```
sink_head_max = max(sink_per_head)
```

所有 head 中对 token-0 分配的最大平均注意力权重。

**诊断意义:**
- 识别最极端的 sink head
- 值接近 1.0 说明该 head 几乎完全 "dormant"（所有注意力给 token-0）
- 可用于定位 pruning 候选 head

---

### 10. Sink Non-Sink Gap (Sink/非Sink Gap)

**数学公式:**
```
sink_nonsink_gap = mean(sink_heads_weight) - mean(nonsink_heads_weight)
```

Sink heads 和非 sink heads 在 token-0 注意力权重上的差异。

**理论基础:**
这是 "logit gap" 的代理指标。Sun et al. 发现 sink head 中 k^(s) 和 q^(n) 的子空间
比非 sink head 更接近（Figure 6），产生持续的 logit gap。

**诊断意义:**
- 高 gap = sink/非sink 分化明显，模型在清晰地区分两类 head
- 低 gap = 所有 head 行为相似，没有强 sink 分化
- Gap 上升趋势 = 模型在训练中逐步强化 sink 策略


## Triton 内核说明

### Online Softmax 算法

QK 统计的核心是 `qk_stats_kernel`，采用 Online Softmax 算法在单次遍历中同时计算 max logit、mean logit、entropy 和 sink weight，避免了完整注意力矩阵的存储。

每个 Triton program 处理一个 `(batch, head)` 对。通过以下状态变量实现增量更新：

| 状态变量 | 含义 |
|----------|------|
| `m_i` | 当前行的最大 logit |
| `d_i` | 当前行的 softmax 分母 `Σexp(x-m)` |
| `h_i` | 当前行的熵分子项 `Σ(x-m)·exp(x-m)` |
| `s_i` | 当前行的 sink 分子项 `exp(x₀-m)` |

当遇到新的 Key block 时，按以下公式更新 (设新最大值 `m_new = max(m_i, block_max)`)：
```
α_prev = exp(m_i - m_new)
α_curr = exp(logits - m_new)

d_new = d_i × α_prev + Σ(α_curr)
h_new = h_i × α_prev + (m_i - m_new) × d_i × α_prev + Σ((logits - m_new) × α_curr)
s_new = s_i × α_prev + α_curr[:, 0]    (仅当 block 包含 position 0)
```

最终：
```
entropy = log(d_i) - h_i / d_i
sink    = s_i / d_i
```

### GQA 处理

当 `num_q_heads ≠ num_k_heads` 时（Grouped Query Attention），K 通过 `repeat_interleave` 扩展以匹配 Q 的 head 数量。

### 因果 Mask

默认启用因果 mask (`causal=True`)，适用于 GPT 类自回归模型。注意力矩阵中 `query_pos < key_pos` 的位置被设为 `-1e10`。

---

## 使用方式

### 基本用法

```python
from internal_medicine.qk_logits import setup_qk_monitor

# 设置监控
model = setup_qk_monitor(
    model,
    causal=True,          # 因果 mask (GPT 类模型)
    use_triton=True,      # 使用 Triton 加速
    log_per_layer=True,   # 记录每层指标
    log_global=True,      # 记录全局聚合指标
    monitor_interval=1,   # 每步监控
    verbose=False,
)

# 获取 monitor 实例
model, monitor = setup_qk_monitor(model, return_monitor=True)

# 训练循环中
for step in range(num_steps):
    loss = model(inputs)
    loss.backward()
    optimizer.step()
    monitor.step()  # 必须在每步结束时调用
```

### 配置选项

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `causal` | `True` | 是否应用因果 mask |
| `use_triton` | `True` | 使用 Triton 内核加速 (CUDA 不可用时自动回退到 PyTorch) |
| `log_per_layer` | `True` | 记录每层指标 (`qk_stats/layer_{idx}/{metric}`) |
| `log_global` | `True` | 记录全局指标 (`qk_stats/global_{metric}`) |
| `monitor_interval` | `1` | 监控间隔 (步数) |
| `verbose` | `False` | 打印调试信息 |

### 日志格式

**Per-layer 指标:**
```
qk_stats/layer_0/max
qk_stats/layer_0/mean
qk_stats/layer_0/entropy_avg
qk_stats/layer_0/sink
qk_stats/layer_0/entropy_min
qk_stats/layer_0/entropy_max
qk_stats/layer_0/entropy_std
...
```

**Global 指标:**
```
qk_stats/global_max
qk_stats/global_mean
qk_stats/global_entropy_avg
qk_stats/global_sink
qk_stats/global_entropy_min
qk_stats/global_entropy_max
qk_stats/global_entropy_std
```

---

## Hook 挂载点

| Hook 位置 | 类型 | 捕获内容 | 产出指标 |
|-----------|------|----------|----------|
| `attention.core_attention` | forward **pre**-hook | `args[0]` = Query, `args[1]` = Key | 全部 7 个指标 |

注意: 使用 `forward_pre_hook` 而非 `forward_hook`，直接拦截送入 `core_attention` 的 Q/K 张量。

---

## TP 聚合机制

| 指标 | TP 聚合方式 |
|------|------------|
| `max` | `all_reduce(MAX)` |
| `mean`, `entropy_avg`, `sink` | `all_reduce(SUM) / tp_size` |
| `entropy_min/max/std` | `all_gather` 所有 rank 的 per-head entropy → 拼接后计算 |
