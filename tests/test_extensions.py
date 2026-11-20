"""Tests for the features added for the extended study: wire formats, ALIE and IPM,
the CNN, the validation split, mechanism metrics, result merging and statistics."""

import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch

from engine import compression
from engine.adversary import ByzantineAdversary
from engine.client import EdgeClient, build_model
from engine.quantization import Int8Quantizer
from engine.storage import DataPartitioner, EdgeStore, initialize_storage_pipeline
from experiments.run_grid import (
    compute_config_hash,
    execute_single_run,
    init_database,
    insert_run,
    mechanism_metrics,
)
from experiments.merge_results import merge
from experiments import stats


def _fake_fashion_mnist(self):
    rng = np.random.default_rng(0)
    x_train = rng.random((400, 784), dtype=np.float32)
    y_train = np.repeat(np.arange(10), 40).astype(np.int64)
    x_test = rng.random((50, 784), dtype=np.float32)
    y_test = np.repeat(np.arange(10), 5).astype(np.int64)
    return x_train, y_train, x_test, y_test


class TestCompression(unittest.TestCase):
    def test_01_int8_matches_quantizer(self):
        x = np.random.default_rng(0).normal(size=1000).astype(np.float32)
        rec, nbytes = compression.encode(x, "INT8")
        np.testing.assert_array_equal(rec, Int8Quantizer.dequantize(*Int8Quantizer.quantize(x)))
        self.assertEqual(nbytes, 1000 + 4)

    def test_02_int4_levels_and_bound(self):
        x = np.random.default_rng(1).normal(size=1001).astype(np.float32)
        rec, nbytes = compression.encode(x, "INT4")
        scale = np.max(np.abs(x)) / 7
        self.assertLessEqual(len(np.unique(np.round(rec / scale))), 15)
        self.assertLessEqual(np.max(np.abs(rec - x)), scale / 2 + 1e-6)
        self.assertEqual(nbytes, math.ceil(1001 / 2) + 4)

    def test_03_sign_values_and_size(self):
        x = np.array([0.5, -2.0, 0.0, 1.5], dtype=np.float32)
        rec, nbytes = compression.encode(x, "Sign")
        np.testing.assert_allclose(rec, np.array([1, -1, 1, 1]) * 1.0)
        self.assertEqual(nbytes, 1 + 4)

    def test_04_payload_ratios(self):
        d = 101_770
        self.assertEqual(compression.payload_bytes("Float32", d), 407_080)
        self.assertEqual(compression.payload_bytes("INT8", d), 101_774)
        self.assertEqual(compression.payload_bytes("INT4", d), 50_889)
        self.assertEqual(compression.payload_bytes("Sign", d), 12_726)


class TestStrongAttacks(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(2)
        self.updates = [rng.normal(loc=1.0, size=300).astype(np.float32) for _ in range(20)]

    def test_05_alie_z_values(self):
        # s = floor(N/2 + 1) - f; z = Phi^-1((N - s) / N)
        self.assertAlmostEqual(ByzantineAdversary.alie_z(20, 4), 0.3853, places=3)
        self.assertAlmostEqual(ByzantineAdversary.alie_z(20, 6), 0.6745, places=3)

    def test_06_alie_inside_honest_spread(self):
        out, _ = ByzantineAdversary.corrupt_client_updates(self.updates, [0, 1, 2, 3], attack_type="alie")
        stacked = np.stack(self.updates)
        mu, sd = stacked.mean(0), stacked.std(0)
        np.testing.assert_allclose(out[0], mu - ByzantineAdversary.alie_z(20, 4) * sd, rtol=1e-5)
        np.testing.assert_array_equal(out[0], out[3])
        np.testing.assert_array_equal(out[5], self.updates[5])

    def test_07_ipm_points_against_mean(self):
        out, _ = ByzantineAdversary.corrupt_client_updates(self.updates, [7], attack_type="ipm")
        mu = np.stack(self.updates).mean(0)
        np.testing.assert_allclose(out[7], -0.1 * mu, rtol=1e-5)
        self.assertLess(float(np.dot(out[7], mu)), 0.0)

    def test_08_attackers_encoded_in_every_wire_format(self):
        for prec in ("INT8", "INT4", "Sign"):
            out, _ = ByzantineAdversary.corrupt_client_updates(
                self.updates, [1], attack_type="alie", wire_precision=prec)
            again, _ = compression.encode(out[1], prec)
            np.testing.assert_array_equal(again, out[1])


class TestModelsAndData(unittest.TestCase):
    def test_09_cnn_shapes_and_flat_params(self):
        model = build_model("cnn")
        self.assertEqual(model(torch.zeros(3, 784)).shape, (3, 10))
        flat = model.get_flat_parameters()
        model.set_flat_parameters(flat + 1.0)
        np.testing.assert_allclose(model.get_flat_parameters(), flat + 1.0, rtol=1e-6)
        with self.assertRaises(ValueError):
            build_model("resnet")

    def test_10_validation_split_is_disjoint(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            DataPartitioner, "load_fashion_mnist", _fake_fashion_mnist
        ):
            initialize_storage_pipeline(data_root=tmp, num_clients=4, alpha=0.5, seed=1, validation_size=40)
            with EdgeStore(data_root=tmp, seed=1, num_clients=4, validation_size=40) as store:
                xv, yv = store.get_validation_data()
                self.assertEqual(len(yv), 40)
                val_ids = set(store.con.execute(
                    f"SELECT sample_id FROM '{(store.clients_dir / 'val.parquet').as_posix()}'").fetchnumpy()["sample_id"])
                client_ids = set()
                for c in range(4):
                    path = (store.clients_dir / f"client_{c:02d}.parquet").as_posix()
                    client_ids |= set(store.con.execute(f"SELECT sample_id FROM '{path}'").fetchnumpy()["sample_id"])
                self.assertFalse(val_ids & client_ids)
                self.assertEqual(len(val_ids) + len(client_ids), 400)


class TestMechanismAndMerge(unittest.TestCase):
    def test_11_mechanism_metrics(self):
        v = [np.zeros(3), np.array([3.0, 4.0, 0.0]), np.array([0.0, 0.0, 10.0])]
        m = mechanism_metrics(v, byzantine_ids=[2], codec_errors=[0.5, 1.5, None])
        self.assertAlmostEqual(m["honest_pair_dist"], 5.0)
        self.assertAlmostEqual(m["honest_attacker_dist"], (10.0 + math.sqrt(9 + 16 + 100)) / 2)
        self.assertAlmostEqual(m["honest_codec_error"], 1.0)
        self.assertAlmostEqual(m["attacker_update_norm"], 10.0)

    def test_12_run_insert_and_merge(self):
        cfg = dict(experiment_type="strong_attacks", aggregator="Trimmed mean", attack="ALIE",
                   attacker_share=0.2, actual_f=4, assumed_f=4, wire_precision="INT4", seed=0,
                   num_rounds=1, num_clients=4, dirichlet_alpha=0.5, batch_size=32, lr=0.05,
                   momentum=0.9, model="mlp")
        cfg["actual_f"] = cfg["assumed_f"] = 1
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(
            DataPartitioner, "load_fashion_mnist", _fake_fashion_mnist
        ):
            initialize_storage_pipeline(data_root=tmp, num_clients=4, alpha=0.5, seed=0)
            with EdgeStore(data_root=tmp, seed=0, num_clients=4) as store:
                x, y = store.get_test_data()
                summary, records = execute_single_run(
                    cfg, store, torch.from_numpy(x).float(), torch.from_numpy(y).long())
            self.assertEqual(summary["bytes_per_client_round"], compression.payload_bytes("INT4", 101_770))
            shard = Path(tmp) / "shard.duckdb"
            con = init_database(shard)
            insert_run(con, compute_config_hash(cfg), cfg, summary, records)
            con.close()
            target = Path(tmp) / "main.duckdb"
            self.assertEqual(merge(target, [str(shard)]), 1)
            self.assertEqual(merge(target, [str(shard)]), 0)
            import duckdb
            with duckdb.connect(str(target), read_only=True) as c:
                self.assertEqual(c.execute("SELECT count(*) FROM round_mechanism").fetchone()[0], 1)
                self.assertEqual(c.execute("SELECT wire_precision FROM runs").fetchone()[0], "INT4")


class TestStats(unittest.TestCase):
    def test_13_t_quantiles(self):
        # Reference values from standard t tables.
        self.assertAlmostEqual(stats.t_ppf(0.975, 4), 2.776, places=3)
        self.assertAlmostEqual(stats.t_ppf(0.975, 9), 2.262, places=3)
        self.assertAlmostEqual(stats.t_ppf(0.975, 1000), 1.962, places=2)

    def test_14_paired_test(self):
        a = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        res = stats.paired_comparison(a + 0.5, a)
        self.assertAlmostEqual(res["mean_diff"], 0.5)
        self.assertEqual(res["ci_low"], 0.5)
        self.assertAlmostEqual(res["p_perm"], 2 / 32)
        res0 = stats.paired_comparison(a, a)
        self.assertEqual(res0["p_perm"], 1.0)

    def test_15_holm(self):
        adj = stats.holm([0.01, 0.04, 0.03])
        np.testing.assert_allclose(adj, [0.03, 0.06, 0.06])


class TestSecondDataset(unittest.TestCase):
    def test_16_cifar_cnn_shape(self):
        model = build_model("cnn_cifar")
        self.assertEqual(sum(p.numel() for p in model.parameters()), 30138)
        out = model(torch.zeros(4, 3072))
        self.assertEqual(tuple(out.shape), (4, 10))

    def test_17_uint8_features_roundtrip(self):
        from engine.storage import _feature_array, _feature_matrix, dataset_root
        x = np.random.default_rng(0).integers(0, 256, size=(5, 12), dtype=np.uint8)
        arr = _feature_array(x)
        back = _feature_matrix(arr.to_numpy(zero_copy_only=False))
        np.testing.assert_allclose(back, x.astype(np.float32) / 255.0)
        self.assertEqual(dataset_root("data"), Path("data"))
        self.assertEqual(dataset_root("data", "cifar10"), Path("data") / "cifar10")
        with self.assertRaises(ValueError):
            dataset_root("data", "imagenet")

    def test_18_new_experiments_and_hashes(self):
        from experiments.run_grid import all_configurations
        cfgs = all_configurations(Path("experiments/config.yaml"))
        self.assertEqual(len(cfgs["cifar10"]), 72)
        self.assertEqual(len(cfgs["tuned"]), 120)
        self.assertTrue(all(r["dataset"] == "cifar10" and r["model"] == "cnn_cifar" for r in cfgs["cifar10"]))
        self.assertTrue(all(r["batch_size"] == 64 for r in cfgs["tuned"]))
        # The dataset key changes the hash only for non-default datasets.
        base = dict(cfgs["grid"][0])
        self.assertEqual(compute_config_hash(base), compute_config_hash(dict(base, dataset="fashion_mnist")))
        self.assertNotEqual(compute_config_hash(base), compute_config_hash(dict(base, dataset="cifar10")))
        hashes = [compute_config_hash(r) for runs in cfgs.values() for r in runs]
        self.assertEqual(len(hashes), len(set(hashes)))


if __name__ == "__main__":
    unittest.main()
