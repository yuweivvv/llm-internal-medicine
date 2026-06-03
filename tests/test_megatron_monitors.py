import importlib
import sys
import unittest
from pathlib import Path

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
PLEHealthMonitor = importlib.import_module("internal_medicine.backends.megatron.ple_monitor").PLEHealthMonitor
training_logs = importlib.import_module("internal_medicine.core.training_logs").training_logs


class BrokenExperts:
    @property
    def weight1(self):
        raise RuntimeError("expert weight read failed")


class BrokenMoELayer:
    experts = BrokenExperts()
    shared_experts = None


class FakePLESublayer:
    act_fn = F.gelu


class MegatronMoEMonitorTest(unittest.TestCase):
    def setUp(self):
        training_logs.reset()

    def tearDown(self):
        training_logs.reset()

    def test_router_only_global_observation_is_counted_once(self):
        monitor = MoESpecialistMonitor(log_per_layer=True, log_global=True)
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

    def test_expert_only_and_router_plus_expert_count_one_observation_per_layer(self):
        monitor = MoESpecialistMonitor(log_per_layer=False, log_global=True)

        monitor._finalize_layer_observation(0, has_expert_metrics=True)
        self.assertEqual(monitor._global_count, 1)

        monitor._accumulate_global({"router_entropy": 2.0})
        monitor._pending_router_global_layers.add(1)
        monitor._accumulate_global({"expert_norm_mean": 4.0})
        monitor._finalize_layer_observation(1, has_expert_metrics=True)
        self.assertEqual(monitor._global_count, 2)

        monitor.step()
        latest = training_logs.get_latest(prefix="moe_health")
        self.assertEqual(latest["moe_health/global_router_entropy"], 1.0)
        self.assertEqual(latest["moe_health/global_expert_norm_mean"], 2.0)

    def test_step_finalizes_pending_router_metrics_before_flush(self):
        monitor = MoESpecialistMonitor(log_per_layer=False, log_global=True)
        monitor._accumulate_global({"router_entropy": 4.0})
        monitor._pending_router_global_layers.add(3)

        monitor.step()

        latest = training_logs.get_latest(prefix="moe_health")
        self.assertEqual(latest["moe_health/global_router_entropy"], 4.0)
        self.assertEqual(monitor._global_count, 0)
        self.assertEqual(monitor._pending_router_global_layers, set())

    def test_expert_hook_exception_finalizes_pending_router_metrics(self):
        monitor = MoESpecialistMonitor(log_per_layer=False, log_global=True, verbose=False)
        monitor._accumulate_global({"router_entropy": 6.0})
        monitor._pending_router_global_layers.add(2)

        hook = monitor._make_moe_layer_hook(2, BrokenMoELayer())
        hook(BrokenMoELayer(), (), None)

        self.assertEqual(monitor._global_count, 1)
        self.assertEqual(monitor._pending_router_global_layers, set())
        monitor.step()
        latest = training_logs.get_latest(prefix="moe_health")
        self.assertEqual(latest["moe_health/global_router_entropy"], 6.0)


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
        monitor._gate_out_buf[5] = torch.ones(2, 3, 4)
        hidden_states = torch.ones(2, 3, 4)
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

        monitor._compute_and_log(0, hidden_states, nn.Module())
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


if __name__ == "__main__":
    unittest.main()
