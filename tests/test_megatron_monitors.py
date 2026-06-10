import importlib
import sys
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

importlib.import_module("_backend_env").skip_unless_backend("megatron")

try:
    torch = importlib.import_module("torch")
    nn = importlib.import_module("torch.nn")
    F = importlib.import_module("torch.nn.functional")
except Exception as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"torch backend unavailable: {exc}") from exc

MassiveActivationMonitor = importlib.import_module(
    "internal_medicine.backends.megatron.massive_activation_monitor"
).MassiveActivationMonitor
MoESpecialistMonitor = importlib.import_module("internal_medicine.backends.megatron.moe_monitor").MoESpecialistMonitor
moe_monitor_module = importlib.import_module("internal_medicine.backends.megatron.moe_monitor")
PLEHealthMonitor = importlib.import_module("internal_medicine.backends.megatron.ple_monitor").PLEHealthMonitor
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs
compute_sink_head_classification = importlib.import_module(
    "internal_medicine.backends.megatron.sink_head_metrics"
).compute_sink_head_classification


class FakePLESublayer:
    act_fn = F.gelu


class FakeMoELayer(nn.Module):
    def __init__(self, experts):
        super().__init__()
        self.experts = experts
        self.shared_experts = None


class MegatronMoEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_router_metrics_flush_from_gpu_buffer(self):
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        for name in moe_monitor_module._ROUTER_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        router = SimpleNamespace(
            topk=2,
            _cached_scores_for_aux_loss=torch.tensor(
                [
                    [0.7, 0.2, 0.1],
                    [0.1, 0.6, 0.3],
                ],
                dtype=torch.float32,
            ),
        )
        monitor._compute_router_metrics(0, router, None, None)

        monitor.step()
        latest = training_logs.get_latest(prefix="moe_health")
        self.assertIn("moe_health/layer_0/router_entropy", latest)
        self.assertIn("moe_health/layer_0/score_sum_mean", latest)
        self.assertIn("moe_health/global_router_entropy", latest)
        self.assertIn("moe_health/global_score_sum_max", latest)

    def test_step_computes_expert_metrics_even_under_no_grad(self):
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
        for name in moe_monitor_module._EXPERT_METRICS:
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(torch.device("cpu"))

        hidden_size = 4
        ffn_hidden = 8
        num_experts = 2
        experts = SimpleNamespace(
            num_local_experts=num_experts,
            config=SimpleNamespace(hidden_size=hidden_size),
            weight1=torch.nn.Parameter(torch.ones(num_experts * hidden_size, ffn_hidden)),
            weight2=torch.nn.Parameter(torch.ones(num_experts * ffn_hidden, hidden_size)),
        )
        moe_layer = FakeMoELayer(experts)
        monitor._monitored_moe_layers = [(0, weakref.ref(moe_layer))]

        with torch.no_grad():
            monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertIn("moe_health/layer_0/expert_norm_mean", latest)
        self.assertIn("moe_health/global_expert_norm_mean", latest)


class MegatronPLEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_global_hooks_are_disabled_when_log_global_is_false(self):
        monitor = PLEHealthMonitor(log_global=False)
        monitor._num_layers = 2
        monitor._hidden_size = 6
        monitor._hidden_size_ple = 3

        monitor._make_token_ple_hook()(None, None, torch.randn(2, 4, 6))
        monitor._make_proj_ple_hook()(None, None, torch.randn(2, 4, 6))

        self.assertIsNone(monitor._token_ple_buf)
        self.assertIsNone(monitor._proj_ple_buf)
        self.assertEqual(training_logs.get_latest(prefix="ple_health"), {})

    def test_layer_hook_records_residual_and_gate_metrics_as_one_observation(self):
        monitor = PLEHealthMonitor(log_per_layer=True, log_global=True, gate_sparsity_threshold=0.01)
        hidden_states = torch.ones(2, 3, 4)
        for name in ("residual_ratio", "gate_activation_mean", "gate_sparsity"):
            monitor.declare_layer_metric(5, name)
        monitor.allocate_buffers(hidden_states.device)

        monitor._gate_out_buf[5] = torch.ones(2, 3, 4)
        output = hidden_states * 1.5

        hook = monitor._make_ple_layer_hook(5, FakePLESublayer())
        hook(None, (hidden_states,), output)
        monitor.step()

        latest = training_logs.get_latest(prefix="ple_health")
        self.assertIn("ple_health/layer_5/residual_ratio", latest)
        self.assertIn("ple_health/layer_5/gate_activation_mean", latest)
        self.assertIn("ple_health/global_residual_ratio", latest)
        self.assertEqual(monitor._gate_out_buf, {})


class MegatronMassiveActivationMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_compute_and_log_records_pre_norm_metrics(self):
        monitor = MassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            cosine_sample_pairs=4,
            absolute_thresholds=(2.0, 3.0),
        )
        hidden_states = torch.tensor(
            [
                [[1.0, -2.0, 0.5, 4.0]],
                [[3.0, 1.0, -0.5, 2.0]],
            ]
        )
        for name in monitor._layer_metric_names():
            monitor.declare_layer_metric(0, name)
        monitor.allocate_buffers(hidden_states.device)

        monitor._compute_residual_metrics(0, hidden_states)
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        for key in (
            "channel_max",
            "channel_median",
            "channel_p95",
            "channel_p99",
            "channel_max_ratio",
            "massive_act_channel_count",
            "channel_count_gt_2",
            "channel_count_gt_3",
            "topk_channel_norm",
            "activation_rms",
        ):
            self.assertIn(f"massive_act/layer_0/{key}", latest)
            self.assertIn(f"massive_act/global_{key}", latest)
        self.assertEqual(latest["massive_act/layer_0/channel_count_gt_2"], 2.0)
        self.assertEqual(latest["massive_act/layer_0/channel_count_gt_3"], 1.0)


class SinkHeadClassificationTest(unittest.TestCase):
    """The gap computation is branchless to avoid a GPU->CPU sync on the hot
    path (Python comparisons on a tensor sink_count would .item()). These cases
    pin the branchless result against a readable branched reference.
    """

    THRESHOLD = 0.3

    def _reference_gap(self, sink_per_head):
        is_sink = sink_per_head > self.THRESHOLD
        num_heads = sink_per_head.numel()
        sink_count = int(is_sink.sum())
        if 0 < sink_count < num_heads:
            return (sink_per_head[is_sink].mean() - sink_per_head[~is_sink].mean()).item()
        if sink_count == num_heads:
            return sink_per_head.mean().item()
        return 0.0

    def _assert_gap(self, sink_per_head):
        result = compute_sink_head_classification(sink_per_head, threshold=self.THRESHOLD)
        self.assertAlmostEqual(result["sink_nonsink_gap"].item(), self._reference_gap(sink_per_head), places=5)

    def test_mixed_sink_and_nonsink(self):
        self._assert_gap(torch.tensor([0.5, 0.1, 0.8, 0.05]))

    def test_all_heads_are_sinks(self):
        self._assert_gap(torch.tensor([0.5, 0.6, 0.9]))

    def test_no_sinks(self):
        self._assert_gap(torch.tensor([0.1, 0.2, 0.05]))

    def test_single_head(self):
        self._assert_gap(torch.tensor([0.9]))
        self._assert_gap(torch.tensor([0.1]))

    def test_empty_input_is_zero(self):
        result = compute_sink_head_classification(torch.tensor([]), threshold=self.THRESHOLD)
        self.assertEqual(result["sink_nonsink_gap"].item(), 0.0)
        self.assertEqual(result["sink_head_ratio"].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
