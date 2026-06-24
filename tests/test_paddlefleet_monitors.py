import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

importlib.import_module("_backend_env").skip_unless_backend("paddlefleet")

try:
    paddle = importlib.import_module("paddle")
    nn = importlib.import_module("paddle.nn")
except Exception as exc:  # pragma: no cover - depends on optional backend install
    raise unittest.SkipTest(f"paddle backend unavailable: {exc}") from exc

PaddleMassiveActivationMonitor = importlib.import_module(
    "internal_medicine.backends.paddlefleet.massive_activation_monitor"
).PaddleMassiveActivationMonitor
PaddleMoEMonitor = importlib.import_module("internal_medicine.backends.paddlefleet.moe_monitor").PaddleMoEMonitor
PaddleQKStatsMonitor = importlib.import_module("internal_medicine.backends.paddlefleet.qk_monitor").PaddleQKStatsMonitor
paddlefleet_backend = importlib.import_module("internal_medicine.backends.paddlefleet")
layer_discovery = importlib.import_module("internal_medicine.backends.paddlefleet.layer_discovery")
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs


class BrokenPaddleMoELayer:
    training = True

    @property
    def grouped_gemm_experts(self):
        raise RuntimeError("grouped expert read failed")


class DummyHook:
    def __init__(self):
        self.removed = False

    def remove(self):
        self.removed = True


class DummyMonitor:
    def __init__(self):
        self.hooks = [DummyHook()]
        self.removed = False

    def remove_hooks(self):
        self.removed = True
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


class PaddleFleetSetupTest(unittest.TestCase):
    def setUp(self):
        self.original_monitor_map = dict(paddlefleet_backend._MONITOR_MAP)

    def tearDown(self):
        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP.update(self.original_monitor_map)

    def test_setup_monitors_reuses_existing_monitor_for_same_config(self):
        created = []

        def setup_dummy(_model, monitor_dict=None, **_kwargs):
            monitor = DummyMonitor()
            created.append(monitor)
            monitor_dict["dummy"] = monitor

        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP["dummy"] = setup_dummy
        model = SimpleNamespace()
        first = {}
        second = {}

        paddlefleet_backend.setup_monitors(model, monitors=["dummy"], monitor_dict=first, monitor_interval=2)
        paddlefleet_backend.setup_monitors(model, monitors=["dummy"], monitor_dict=second, monitor_interval=2)

        self.assertEqual(len(created), 1)
        self.assertIs(second["dummy"], first["dummy"])
        self.assertFalse(first["dummy"].removed)

    def test_setup_monitors_replaces_existing_monitor_when_config_changes(self):
        created = []

        def setup_dummy(_model, monitor_dict=None, **_kwargs):
            monitor = DummyMonitor()
            created.append(monitor)
            monitor_dict["dummy"] = monitor

        paddlefleet_backend._MONITOR_MAP.clear()
        paddlefleet_backend._MONITOR_MAP["dummy"] = setup_dummy
        model = SimpleNamespace()
        first = {}
        second = {}

        paddlefleet_backend.setup_monitors(model, monitors=["dummy"], monitor_dict=first, monitor_interval=1)
        paddlefleet_backend.setup_monitors(model, monitors=["dummy"], monitor_dict=second, monitor_interval=2)

        self.assertEqual(len(created), 2)
        self.assertTrue(first["dummy"].removed)
        self.assertIs(second["dummy"], created[1])


class PaddleLayerDiscoveryTest(unittest.TestCase):
    def test_get_decoder_layers_flattens_virtual_pipeline_chunks(self):
        layer0 = SimpleNamespace(layer_idx=0)
        layer1 = SimpleNamespace(layer_idx=1)
        layer2 = SimpleNamespace(layer_idx=2)
        model = SimpleNamespace(
            _model_chunks=[
                SimpleNamespace(run_function=[layer0, layer1]),
                SimpleNamespace(run_function=[layer2]),
            ]
        )

        self.assertEqual(layer_discovery.get_decoder_layers(model), [layer0, layer1, layer2])

    def test_get_decoder_layers_checks_wrapped_module_after_empty_layers_wrapper(self):
        layer = SimpleNamespace(layer_idx=0)
        model = SimpleNamespace(_layers=SimpleNamespace(), module=SimpleNamespace(run_function=[layer]))

        self.assertEqual(layer_discovery.get_decoder_layers(model), [layer])


class PaddleMoEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_gpu_buffer_gate_metrics_recorded(self):
        """Gate hook records metrics via GPU-buffer API without D2H sync."""
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True)
        # Manually declare and allocate for layer 0
        for m in ["router_entropy", "score_sum_mean", "score_sum_min", "score_sum_max"]:
            monitor.declare_layer_metric(0, m)
        monitor.allocate_buffers()

        # Simulate recording
        monitor.record_layer_metric(0, "router_entropy", paddle.to_tensor(2.0))
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(latest["moe_health/layer_0/router_entropy"], 2.0, places=4)
        self.assertAlmostEqual(latest["moe_health/global_router_entropy"], 2.0, places=4)

    def test_gpu_buffer_multi_layer_global_aggregation(self):
        """Global metrics are derived from layer accumulators at flush time."""
        monitor = PaddleMoEMonitor(log_per_layer=False, log_global=True)
        for layer_idx in (0, 1):
            for m in ["router_entropy", "score_sum_max"]:
                monitor.declare_layer_metric(layer_idx, m)
        monitor.allocate_buffers()

        monitor.record_layer_metric(0, "router_entropy", paddle.to_tensor(2.0))
        monitor.record_layer_metric(1, "router_entropy", paddle.to_tensor(4.0))
        monitor.record_layer_metric(0, "score_sum_max", paddle.to_tensor(0.8))
        monitor.record_layer_metric(1, "score_sum_max", paddle.to_tensor(0.9))
        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertAlmostEqual(latest["moe_health/global_router_entropy"], 3.0, places=4)
        self.assertAlmostEqual(latest["moe_health/global_score_sum_max"], 0.9, places=4)

    def test_expert_hook_exception_does_not_crash(self):
        """Expert hook exceptions are caught without crashing the step."""
        monitor = PaddleMoEMonitor(log_per_layer=False, log_global=True, verbose=False)
        for m in ["expert_norm_mean", "expert_norm_std", "expert_norm_min", "expert_norm_max"]:
            monitor.declare_layer_metric(2, m)
        monitor.allocate_buffers()

        hook = monitor._make_moe_layer_hook(2, BrokenPaddleMoELayer())
        hook(BrokenPaddleMoELayer(), (), None)

        # Should not crash; step should still work
        monitor.step()

    def test_hash_routing_cache_supports_sqrtsoftplus(self):
        monitor = PaddleMoEMonitor(log_per_layer=False, log_global=True, verbose=False)
        logits = paddle.to_tensor([[0.0, 1.0]], dtype="float32")

        def original_hash_routing(logits, flat_ids):
            return "ok"

        gate = SimpleNamespace(
            gate_score_func=lambda logits: logits,
            _hash_routing=original_hash_routing,
            scoring_func="sqrtsoftplus",
        )
        monitor._patch_gate_cache(gate)

        self.assertEqual(gate._hash_routing(logits, paddle.to_tensor([0], dtype="int64")), "ok")
        expected = paddle.sqrt(paddle.nn.functional.softplus(logits) + 1e-20)
        self.assertTrue(bool(paddle.allclose(gate._cached_gates, expected)))

        monitor.remove_hooks()
        self.assertIs(gate._hash_routing, original_hash_routing)

    def test_hash_routing_cache_respects_monitor_interval(self):
        monitor = PaddleMoEMonitor(log_per_layer=False, log_global=True, monitor_interval=2, verbose=False)
        logits = paddle.to_tensor([[0.0, 1.0]], dtype="float32")
        gate = SimpleNamespace(
            gate_score_func=lambda logits: logits,
            _hash_routing=lambda logits, flat_ids: "ok",
            scoring_func="sigmoid",
        )
        monitor._patch_gate_cache(gate)
        monitor.step_count = 1

        self.assertEqual(gate._hash_routing(logits, paddle.to_tensor([0], dtype="int64")), "ok")
        self.assertIsNone(gate._cached_gates)


class PaddleMassiveActivationMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_extract_hidden_states_supports_dict_and_positional_inputs(self):
        monitor = PaddleMassiveActivationMonitor()
        hidden_states = paddle.randn([2, 3, 4], dtype="float32")

        self.assertIs(monitor._extract_hidden_states(({"hidden_states": hidden_states},)), hidden_states)
        self.assertIs(monitor._extract_hidden_states((hidden_states,)), hidden_states)
        self.assertIsNone(monitor._extract_hidden_states(()))
        self.assertIsNone(monitor._extract_hidden_states(({"other": hidden_states},)))

    def test_compute_and_log_records_pre_norm_metrics(self):
        monitor = PaddleMassiveActivationMonitor(
            log_per_layer=True,
            log_global=True,
            cosine_sample_pairs=4,
            absolute_thresholds=(2.0, 3.0),
        )
        hidden_states = paddle.to_tensor(
            [
                [[1.0, -2.0, 0.5, 4.0]],
                [[3.0, 1.0, -0.5, 2.0]],
            ],
            dtype="float32",
        )

        # Must declare + allocate before _compute_and_log
        metric_names = [
            "channel_max",
            "channel_median",
            "channel_p95",
            "channel_p99",
            "channel_max_ratio",
            "massive_act_channel_count",
            "topk_channel_norm",
            "activation_rms",
            "post_norm_sparsity",
            "post_norm_cosine",
            "channel_count_gt_2",
            "channel_count_gt_3",
        ]
        for m in metric_names:
            monitor.declare_layer_metric(0, m)
        monitor.allocate_buffers()

        monitor._compute_and_log(0, hidden_states, nn.Layer())
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

    def test_count_metrics_are_max_aggregated_across_layers(self):
        monitor = PaddleMassiveActivationMonitor(
            log_per_layer=False,
            log_global=True,
            absolute_thresholds=(2.0, 3.0),
        )

        # Declare and allocate for 2 layers
        for layer_idx in (0, 1):
            for m in ["massive_act_channel_count", "channel_count_gt_2", "channel_count_gt_3"]:
                monitor.declare_layer_metric(layer_idx, m)
        monitor.allocate_buffers()

        monitor.record_layer_metric(0, "massive_act_channel_count", paddle.to_tensor(0.0))
        monitor.record_layer_metric(0, "channel_count_gt_2", paddle.to_tensor(1.0))
        monitor.record_layer_metric(0, "channel_count_gt_3", paddle.to_tensor(0.0))
        monitor.record_layer_metric(1, "massive_act_channel_count", paddle.to_tensor(2.0))
        monitor.record_layer_metric(1, "channel_count_gt_2", paddle.to_tensor(3.0))
        monitor.record_layer_metric(1, "channel_count_gt_3", paddle.to_tensor(1.0))
        monitor.step()

        latest = training_logs.get_latest(prefix="massive_act")
        self.assertEqual(latest["massive_act/global_massive_act_channel_count"], 2.0)
        self.assertEqual(latest["massive_act/global_channel_count_gt_2"], 3.0)
        self.assertEqual(latest["massive_act/global_channel_count_gt_3"], 1.0)


class PaddleQKMonitorTest(unittest.TestCase):
    def test_resolve_layer_idx_uses_shared_base_logic(self):
        monitor = PaddleQKStatsMonitor()

        self.assertEqual(monitor._resolve_layer_idx(SimpleNamespace(layer_idx=8), 0, 4), 8)
        self.assertEqual(monitor._resolve_layer_idx(SimpleNamespace(layer_number=2), 0, 4), 1)

        monitor.pp_rank = 1
        self.assertEqual(monitor._resolve_layer_idx(SimpleNamespace(), 2, 4), 6)


if __name__ == "__main__":
    unittest.main()
