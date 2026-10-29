"""Unit Test Suite for Benchmark Simulation Engine.
"""

import os
import unittest
import numpy as np
import torch

from engine.client import EdgeClassifier
from engine.storage import EdgeStore, TelemetryEngine, initialize_storage_pipeline
from simulate import evaluate_global_model, run_regime


class TestSimulationEngine(unittest.TestCase):
    """Verifies the benchmark orchestrator and global evaluation logic."""

    @classmethod
    def setUpClass(cls):
        cls.data_root = "data"
        if not os.path.exists(os.path.join(cls.data_root, "test.parquet")):
            initialize_storage_pipeline(data_root=cls.data_root, num_clients=20, alpha=0.5)
        cls.store = EdgeStore(data_root=cls.data_root)
        cls.x_test, cls.y_test = cls.store.get_test_data()
        cls.telemetry = TelemetryEngine(db_path=os.path.join(cls.data_root, "telemetry.duckdb"))

    @classmethod
    def tearDownClass(cls):
        cls.telemetry.close()
        cls.store.close()

    def test_01_global_model_evaluation(self):
        """Verify global evaluation accurately computes accuracy and loss on test shard."""
        model = EdgeClassifier()
        acc, loss = evaluate_global_model(model, self.x_test, self.y_test, batch_size=512)

        self.assertGreaterEqual(acc, 0.0)
        self.assertLessEqual(acc, 100.0)
        self.assertGreater(loss, 0.0)

    def test_02_regime_execution_smoke_test(self):
        """Verify run_regime executes 1 round across 4 clients and updates telemetry."""
        model = EdgeClassifier()
        initial_weights = model.get_flat_parameters()

        exp_id = "test_smoke_regime"
        result = run_regime(
            regime_name="Smoke_Test_MultiKrum",
            aggregation_method="Multi-Krum",
            byzantine_ratio=0.25,  # 1 out of 4 clients
            quantize=True,
            num_rounds=1,
            num_clients=4,
            batch_size=256,
            lr=0.05,
            store=self.store,
            telemetry=self.telemetry,
            experiment_id=exp_id,
            x_test=self.x_test,
            y_test=self.y_test,
            initial_weights=initial_weights,
        )

        self.assertEqual(result["regime_name"], "Smoke_Test_MultiKrum")
        self.assertTrue(result["quantized"])
        self.assertGreater(result["final_accuracy"], 0.0)
        self.assertEqual(result["mitigation_rate"], 100.0)
        # INT8 payload per client should be ~99.4 KB
        self.assertLess(result["payload_per_client_kb"], 105.0)

    def test_03_regime_gaussian_attack_execution(self):
        """Verify run_regime executes cleanly with Gaussian noise attack vector."""
        model = EdgeClassifier()
        initial_weights = model.get_flat_parameters()

        exp_id = "test_smoke_gaussian"
        result = run_regime(
            regime_name="Smoke_Test_Gaussian_MultiKrum",
            aggregation_method="Multi-Krum",
            byzantine_ratio=0.25,
            quantize=False,
            num_rounds=1,
            num_clients=4,
            batch_size=256,
            lr=0.05,
            store=self.store,
            telemetry=self.telemetry,
            experiment_id=exp_id,
            x_test=self.x_test,
            y_test=self.y_test,
            initial_weights=initial_weights,
            attack_type="gaussian",
        )

        self.assertEqual(result["regime_name"], "Smoke_Test_Gaussian_MultiKrum")
        self.assertFalse(result["quantized"])
        self.assertGreater(result["final_accuracy"], 0.0)

    def test_04_regime_coordinate_median_execution(self):
        """Verify run_regime executes cleanly with Coordinate-wise Median defense."""
        model = EdgeClassifier()
        initial_weights = model.get_flat_parameters()

        exp_id = "test_smoke_median"
        result = run_regime(
            regime_name="Smoke_Test_Coordinate_Median",
            aggregation_method="Coordinate-Median",
            byzantine_ratio=0.25,
            quantize=False,
            num_rounds=1,
            num_clients=4,
            batch_size=256,
            lr=0.05,
            store=self.store,
            telemetry=self.telemetry,
            experiment_id=exp_id,
            x_test=self.x_test,
            y_test=self.y_test,
            initial_weights=initial_weights,
            attack_type="sign_flip",
        )

        self.assertEqual(result["regime_name"], "Smoke_Test_Coordinate_Median")
        self.assertGreater(result["final_accuracy"], 0.0)
        self.assertEqual(result["mitigation_rate"], 100.0)

    def test_05_seed_determinism_exact_match(self):
        """Verify two independent runs with identical config and seed give exact matching accuracy."""
        seed = 77
        num_rounds = 2
        num_clients = 4
        byzantine_ratio = 0.25

        # Run 1
        torch.manual_seed(seed)
        np.random.seed(seed)
        model_1 = EdgeClassifier()
        init_weights_1 = model_1.get_flat_parameters()

        res_1 = run_regime(
            regime_name="Determinism_Run_1",
            aggregation_method="Multi-Krum",
            byzantine_ratio=byzantine_ratio,
            quantize=True,
            num_rounds=num_rounds,
            num_clients=num_clients,
            batch_size=256,
            lr=0.05,
            store=self.store,
            telemetry=self.telemetry,
            experiment_id="test_det_1",
            x_test=self.x_test,
            y_test=self.y_test,
            initial_weights=init_weights_1,
            attack_type="gaussian",
            seed=seed,
        )

        # Run 2
        torch.manual_seed(seed)
        np.random.seed(seed)
        model_2 = EdgeClassifier()
        init_weights_2 = model_2.get_flat_parameters()

        res_2 = run_regime(
            regime_name="Determinism_Run_2",
            aggregation_method="Multi-Krum",
            byzantine_ratio=byzantine_ratio,
            quantize=True,
            num_rounds=num_rounds,
            num_clients=num_clients,
            batch_size=256,
            lr=0.05,
            store=self.store,
            telemetry=self.telemetry,
            experiment_id="test_det_2",
            x_test=self.x_test,
            y_test=self.y_test,
            initial_weights=init_weights_2,
            attack_type="gaussian",
            seed=seed,
        )

        # Exact match assertion on CPU
        self.assertEqual(res_1["final_accuracy"], res_2["final_accuracy"])
        self.assertAlmostEqual(res_1["final_loss"], res_2["final_loss"], places=7)


if __name__ == "__main__":
    unittest.main()
