"""Regression tests for the fixes made after the audit (AUDIT_REPORT.md).

- Config hashes of the stored grid stay the same, but a change of lr, batch size
  or momentum gives a new hash.
- Client shards built with one alpha are rebuilt when another alpha is requested.
- EdgeClient uses the momentum it is given.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from engine.storage import PARTITION_META_FILENAME, DataPartitioner, EdgeStore, initialize_storage_pipeline
from experiments.run_grid import compute_config_hash, generate_grid_configurations


def _fake_fashion_mnist(self):
    """Small synthetic stand-in for Fashion-MNIST so the test needs no download."""
    rng = np.random.default_rng(0)
    x_train = rng.random((400, 784), dtype=np.float32)
    y_train = np.repeat(np.arange(10), 40).astype(np.int64)
    x_test = rng.random((50, 784), dtype=np.float32)
    y_test = np.repeat(np.arange(10), 5).astype(np.int64)
    return x_train, y_train, x_test, y_test


class TestConfigHashHyperparameters(unittest.TestCase):
    def test_01_stored_hash_ignores_legacy_hyperparameters(self):
        """Adding lr, batch size and momentum at their stored values keeps the hash."""
        grid, _ = generate_grid_configurations(Path("experiments/config.yaml"))
        cfg = dict(grid[0])
        without = {k: v for k, v in cfg.items() if k not in ("lr", "batch_size", "momentum")}
        self.assertEqual(compute_config_hash(cfg), compute_config_hash(without))

    def test_02_changed_hyperparameters_change_hash(self):
        """A run with a different lr, batch size or momentum is not mistaken for a stored run."""
        grid, _ = generate_grid_configurations(Path("experiments/config.yaml"))
        base = dict(grid[0])
        base_hash = compute_config_hash(base)
        for key, value in (("lr", 0.01), ("batch_size", 64), ("momentum", 0.0)):
            changed = dict(base, **{key: value})
            self.assertNotEqual(compute_config_hash(changed), base_hash, key)


class TestPartitionRebuild(unittest.TestCase):
    def test_03_alpha_change_triggers_rebuild(self):
        """Shards built with alpha 0.5 are rebuilt, not reused, when alpha 0.1 is requested."""
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            DataPartitioner, "load_fashion_mnist", _fake_fashion_mnist
        ):
            initialize_storage_pipeline(data_root=tmp, num_clients=4, alpha=0.5, seed=7)
            meta_path = Path(tmp) / "clients_seed_7" / PARTITION_META_FILENAME
            with open(meta_path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["alpha"], 0.5)

            _, summary = initialize_storage_pipeline(data_root=tmp, num_clients=4, alpha=0.5, seed=7)
            self.assertTrue(summary.get("skipped"))

            _, summary = initialize_storage_pipeline(data_root=tmp, num_clients=4, alpha=0.1, seed=7)
            self.assertFalse(summary.get("skipped", False))
            with open(meta_path, encoding="utf-8") as fh:
                self.assertEqual(json.load(fh)["alpha"], 0.1)

    def test_04_client_count_change_removes_stale_shards(self):
        """Going from 6 to 4 clients leaves exactly 4 shards."""
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            DataPartitioner, "load_fashion_mnist", _fake_fashion_mnist
        ):
            initialize_storage_pipeline(data_root=tmp, num_clients=6, alpha=0.5, seed=7)
            initialize_storage_pipeline(data_root=tmp, num_clients=4, alpha=0.5, seed=7)
            shards = sorted((Path(tmp) / "clients_seed_7").glob("client_*.parquet"))
            self.assertEqual(len(shards), 4)


class TestClientMomentum(unittest.TestCase):
    def test_05_client_uses_given_momentum(self):
        """Momentum 0 and momentum 0.9 give different updates from the same start."""
        from engine.client import EdgeClassifier, EdgeClient

        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            DataPartitioner, "load_fashion_mnist", _fake_fashion_mnist
        ):
            initialize_storage_pipeline(data_root=tmp, num_clients=2, alpha=0.5, seed=3)
            with EdgeStore(data_root=tmp, seed=3) as store:
                start = EdgeClassifier().get_flat_parameters()
                updates = []
                for momentum in (0.0, 0.9):
                    client = EdgeClient(client_id=0, store=store, batch_size=16, momentum=momentum)
                    updates.append(client.train_round(start, round_num=1, seed=3).update)
                self.assertFalse(np.allclose(updates[0], updates[1]))


class TestInt8AttackerWireFormat(unittest.TestCase):
    def test_06_attackers_send_valid_int8(self):
        """With wire_int8, every attacker update is an INT8 value times one scale."""
        from engine.adversary import ByzantineAdversary
        from engine.quantization import Int8Quantizer

        rng = np.random.default_rng(1)
        honest = [Int8Quantizer.dequantize(*Int8Quantizer.quantize(rng.normal(size=500).astype(np.float32)))
                  for _ in range(5)]
        for attack in ("sign_flip", "gaussian"):
            out, _ = ByzantineAdversary.corrupt_client_updates(
                honest, byzantine_client_ids=[1, 3], attack_type=attack, seed=7, wire_int8=True
            )
            for cid in (1, 3):
                q, scale = Int8Quantizer.quantize(out[cid])
                np.testing.assert_array_equal(Int8Quantizer.dequantize(q, scale), out[cid])
            np.testing.assert_array_equal(out[0], honest[0])

    def test_07_empty_client_raises(self):
        """A Dirichlet split that leaves a client empty is rejected."""
        labels = np.repeat(np.arange(2), 3)
        with self.assertRaises(ValueError):
            DataPartitioner.__new__(DataPartitioner).partition_dirichlet(
                labels=labels, num_clients=50, alpha=0.01, seed=0
            )


if __name__ == "__main__":
    unittest.main()
