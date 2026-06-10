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
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs


class BrokenPaddleMoELayer:
    training = True

    @property
    def grouped_gemm_experts(self):
        raise RuntimeError("grouped expert read failed")


class PaddleMoEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_router_only_global_observation_is_counted_once(self):
        monitor = PaddleMoEMonitor(log_per_layer=True, log_global=True)
        monitor._log_per_layer_metrics(0, {"router_entropy": 2.0})
        monitor._accumulate_global({"router_entropy": 2.0})
        monitor._pending_router_global_layers.add(0)

        monitor._finalize_layer_observation(0)
        self.assertEqual(monitor._global_count, 1)
        self.assertEqual(monitor._pending_router_global_layers, set())

        monitor.step()
        latest = training_logs.get_latest(prefix="moe_health")
        self.assertEqual(latest["moe_health/layer_0/router_entropy"], 2.0)
        self.assertEqual(latest["moe_health/global_router_entropy"], 2.0)

    def test_step_finalizes_pending_router_metrics_before_flush(self):
        monitor = PaddleMoEMonitor(log_per_layer=False, log_global=True)
        monitor._accumulate_global({"router_entropy": 4.0})
        monitor._pending_router_global_layers.add(3)

        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertEqual(latest["moe_health/global_router_entropy"], 4.0)
        self.assertEqual(monitor._global_count, 0)
        self.assertEqual(monitor._pending_router_global_layers, set())

    def test_expert_hook_exception_finalizes_pending_router_metrics(self):
        monitor = PaddleMoEMonitor(log_per_layer=False, log_global=True, verbose=False)
        monitor._accumulate_global({"router_entropy": 6.0})
        monitor._pending_router_global_layers.add(2)

        hook = monitor._make_moe_layer_hook(2, BrokenPaddleMoELayer())
        hook(BrokenPaddleMoELayer(), (), None)

        self.assertEqual(monitor._global_count, 1)
        self.assertEqual(monitor._pending_router_global_layers, set())
        monitor.step()
        latest = training_logs.get_latest(prefix="moe_health")
        self.assertEqual(latest["moe_health/global_router_entropy"], 6.0)


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

        monitor._record_metrics(
            0,
            {
                "massive_act_channel_count": 0.0,
                "channel_count_gt_2": 1.0,
                "channel_count_gt_3": 0.0,
            },
        )
        monitor._record_metrics(
            1,
            {
                "massive_act_channel_count": 2.0,
                "channel_count_gt_2": 3.0,
                "channel_count_gt_3": 1.0,
            },
        )
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
