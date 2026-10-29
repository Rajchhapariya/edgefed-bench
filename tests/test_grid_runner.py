"""Unit Test Suite for Config-Driven Grid Runner and DuckDB Persistence.
"""

import os
import tempfile
import unittest
from pathlib import Path

import duckdb
import torch

from engine.storage import EdgeStore, initialize_storage_pipeline
from experiments.run_grid import (
    compute_config_hash,
    execute_single_run,
    generate_grid_configurations,
    init_database,
)


class TestGridRunner(unittest.TestCase):
    """Verifies experiment grid generation, hashing determinism, and DuckDB storage."""

    @classmethod
    def setUpClass(cls):
        cls.config_path = Path("experiments/config.yaml")
        cls.data_root = "data"
        if not os.path.exists(os.path.join(cls.data_root, "test.parquet")):
            initialize_storage_pipeline(data_root=cls.data_root, num_clients=20, alpha=0.5, seed=0)
        cls.store = EdgeStore(data_root=cls.data_root, seed=0)
        cls.x_test_raw, cls.y_test_raw = cls.store.get_test_data()
        cls.x_test_t = torch.from_numpy(cls.x_test_raw).float()
        cls.y_test_t = torch.from_numpy(cls.y_test_raw).long()

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_01_grid_generation_counts(self):
        """Verify grid configurations generate exactly 280 main grid runs and 40 wrong-f runs."""
        grid_runs, wrong_f_runs = generate_grid_configurations(self.config_path)
        self.assertEqual(len(grid_runs), 280)
        self.assertEqual(len(wrong_f_runs), 40)

        # Count clean vs attacked
        clean_runs = [r for r in grid_runs if r["attack"] == "none"]
        attacked_runs = [r for r in grid_runs if r["attack"] != "none"]
        self.assertEqual(len(clean_runs), 40)
        self.assertEqual(len(attacked_runs), 240)

    def test_02_config_hash_determinism(self):
        """Verify compute_config_hash yields exact same hash across independent calls."""
        cfg = {
            "experiment_type": "grid",
            "aggregator": "Multi-Krum",
            "attack": "sign-flip",
            "attacker_share": 0.20,
            "actual_f": 4,
            "assumed_f": 4,
            "wire_precision": "INT8",
            "seed": 2,
            "num_rounds": 50,
            "num_clients": 20,
            "dirichlet_alpha": 0.5,
        }
        hash1 = compute_config_hash(cfg)
        hash2 = compute_config_hash(cfg)
        self.assertEqual(hash1, hash2)
        self.assertEqual(len(hash1), 16)

    def test_03_execute_single_run_and_duckdb_persistence(self):
        """Smoke test executing a fast 2-round run and verifying DuckDB schema insertion."""
        test_cfg = {
            "experiment_type": "grid",
            "aggregator": "Trimmed mean",
            "attack": "sign-flip",
            "attacker_share": 0.20,
            "actual_f": 4,
            "assumed_f": 4,
            "wire_precision": "INT8",
            "seed": 0,
            "num_rounds": 2,
            "num_clients": 20,
            "dirichlet_alpha": 0.5,
            "batch_size": 128,
            "lr": 0.05,
            "momentum": 0.9,
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "test_results.duckdb"
            con = init_database(db_path)

            summary, round_records = execute_single_run(
                test_cfg,
                store=self.store,
                x_test_tensor=self.x_test_t,
                y_test_tensor=self.y_test_t,
            )

            self.assertGreater(summary["final_accuracy"], 0.0)
            self.assertEqual(len(round_records), 2)

            chash = compute_config_hash(test_cfg)
            con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                chash,
                test_cfg["experiment_type"],
                test_cfg["aggregator"],
                test_cfg["attack"],
                test_cfg["attacker_share"],
                test_cfg["actual_f"],
                test_cfg["assumed_f"],
                test_cfg["wire_precision"],
                test_cfg["seed"],
                test_cfg["num_rounds"],
                test_cfg["num_clients"],
                summary["final_accuracy"],
                summary["best_accuracy"],
                summary["final_loss"],
                summary["acc_round_10"],
                summary["acc_round_25"],
                summary["acc_round_50"],
                summary["rounds_to_70"],
                summary["total_uplink_bytes"],
                summary["bytes_per_client_round"],
                summary["wall_clock_sec"],
                summary["attacker_rejection_rate"],
                summary["honest_false_rejection_rate"],
                summary["completed_at"],
            ])

            row_count = con.execute("SELECT count(*) FROM runs").fetchone()[0]
            self.assertEqual(row_count, 1)

            rounds_count = con.execute("SELECT count(*) FROM round_metrics").fetchone()[0]
            self.assertEqual(rounds_count, 0)
            con.close()


if __name__ == "__main__":
    unittest.main()

