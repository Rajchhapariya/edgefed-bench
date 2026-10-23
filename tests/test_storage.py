"""Unit test suite for storage, Parquet sharding, and DuckDB telemetry."""

import os
import unittest
from pathlib import Path
import numpy as np

from engine.storage import DataPartitioner, EdgeStore, TelemetryEngine, initialize_storage_pipeline


class TestStoragePipeline(unittest.TestCase):
    """Tests verifying data partitioning, Parquet I/O, and DuckDB telemetry."""

    @classmethod
    def setUpClass(cls):
        cls.data_root = "data"
        cls.clients_dir = Path(cls.data_root) / "clients"
        cls.test_parquet = Path(cls.data_root) / "test.parquet"
        cls.telemetry_db = Path(cls.data_root) / "telemetry.duckdb"
        if not cls.test_parquet.exists() or not cls.clients_dir.exists():
            initialize_storage_pipeline(data_root=cls.data_root, num_clients=20, alpha=0.5)
        cls.store = EdgeStore(data_root=cls.data_root)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_01_parquet_shards_exist(self):
        """Verify all 20 client Parquet shards and the global test shard exist on disk."""
        self.assertTrue(self.test_parquet.exists(), "Global test.parquet does not exist.")
        for cid in range(20):
            shard_path = self.clients_dir / f"client_{cid:02d}.parquet"
            self.assertTrue(shard_path.exists(), f"Missing shard: {shard_path}")
            # Ensure file is not empty
            self.assertGreater(shard_path.stat().st_size, 100_000, f"Shard {shard_path} is suspiciously small.")

    def test_02_sample_conservation(self):
        """Verify total training samples across all 20 clients equals 60,000 and test equals 10,000."""
        total_train_samples = 0
        for cid in range(20):
            features, labels = self.store.get_client_data(cid)
            self.assertEqual(len(features), len(labels))
            total_train_samples += len(labels)

        self.assertEqual(total_train_samples, 60000, f"Expected 60,000 total train samples, got {total_train_samples}")

        test_features, test_labels = self.store.get_test_data()
        self.assertEqual(len(test_labels), 10000, f"Expected 10,000 test samples, got {len(test_labels)}")
        self.assertEqual(len(test_features), 10000)

    def test_03_feature_dimensionality_and_bounds(self):
        """Verify features are 784-dimensional float arrays normalized in [0.0, 1.0]."""
        features, labels = self.store.get_client_data(0)
        self.assertEqual(features.shape[1], 784, "Features are not 784-dimensional.")
        self.assertEqual(features.dtype, np.float32, "Feature dtype is not float32.")
        self.assertGreaterEqual(features.min(), 0.0, "Feature value below 0.0 found.")
        self.assertLessEqual(features.max(), 1.0, "Feature value above 1.0 found.")
        self.assertTrue(np.all(labels >= 0) and np.all(labels < 10), "Invalid class label found.")

    def test_04_dirichlet_non_iid_heterogeneity(self):
        """Statistically verify that class distributions across clients exhibit significant variance (non-IID)."""
        client_class_matrix = np.zeros((20, 10), dtype=np.int64)
        for cid in range(20):
            dist = self.store.get_client_class_distribution(cid)
            for label, count in dist.items():
                client_class_matrix[cid, label] = count

        # If data were IID, each client would have approximately 300 samples per class (std dev ~ 15)
        # Under Dirichlet alpha=0.5, standard deviation across clients for any class should be high (> 100)
        class_std_devs = np.std(client_class_matrix, axis=0)
        mean_std_dev = float(np.mean(class_std_devs))
        self.assertGreater(mean_std_dev, 100.0, f"Standard deviation {mean_std_dev} too low for non-IID Dirichlet skew.")

    def test_05_telemetry_engine_crud(self):
        """Verify TelemetryEngine creates tables, inserts records, and performs SQL aggregations."""
        telemetry = TelemetryEngine(db_path=str(self.telemetry_db))
        test_exp_id = "test_crud_exp"

        telemetry.log_round_telemetry(
            experiment_id=test_exp_id,
            regime_name="Unit_Test_Regime",
            round_num=1,
            client_id=5,
            is_byzantine=False,
            gradient_norm=1.234,
            payload_bytes=1024,
            quantized=True,
            local_loss=0.88,
        )

        telemetry.log_global_eval(
            experiment_id=test_exp_id,
            regime_name="Unit_Test_Regime",
            round_num=1,
            global_accuracy=81.5,
            global_loss=0.62,
            aggregation_method="Multi-Krum",
            byzantine_fraction=0.2,
            quantized=True,
            round_duration_sec=0.45,
        )

        summary = telemetry.get_regime_summary(test_exp_id)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["regime_name"], "Unit_Test_Regime")
        self.assertAlmostEqual(summary[0]["peak_accuracy"], 81.5)
        self.assertAlmostEqual(summary[0]["best_loss"], 0.62)
        telemetry.close()

    def test_06_idempotent_pipeline_skips_rebuild(self):
        """Verify initialize_storage_pipeline recognizes existing shards and skips rebuild."""
        paths, summary = initialize_storage_pipeline(
            data_root=self.data_root,
            num_clients=20,
            alpha=0.5,
            force_rebuild=False,
        )
        self.assertTrue(summary.get("skipped", False))
        self.assertEqual(len(paths), 20)


if __name__ == "__main__":
    unittest.main()
