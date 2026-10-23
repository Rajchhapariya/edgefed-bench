"""Formal Unit Test Suite for Byzantine Adversaries and Robust Aggregation.
"""

import unittest
import numpy as np

from engine.adversary import ByzantineAdversary
from engine.aggregator import RobustAggregator
from engine.quantization import Int8Quantizer


class TestAdversaryAndAggregator(unittest.TestCase):
    """Rigorous verification of Byzantine attacks and defensive aggregators."""

    def setUp(self):
        self.rng = np.random.default_rng(42)
        self.dim = 101770
        self.num_clients = 20
        self.num_byzantine = 4  # 20% of 20 nodes

        # Synthesize honest cluster with realistic distributed variance
        base_signal = self.rng.normal(loc=0.05, scale=0.01, size=self.dim).astype(np.float32)
        self.honest_updates = [
            base_signal + self.rng.normal(loc=0.0, scale=0.005, size=self.dim).astype(np.float32)
            for _ in range(self.num_clients)
        ]

    def test_01_byzantine_attacks(self):
        """Verify mathematical properties of sign-flip and Gaussian attacks."""
        honest_vec = np.ones(100, dtype=np.float32)

        # Sign flip
        flipped = ByzantineAdversary.sign_flip(honest_vec, scaling_factor=2.5)
        self.assertTrue(np.allclose(flipped, -2.5 * honest_vec))
        dot_prod = float(np.dot(honest_vec, flipped))
        self.assertLess(dot_prod, 0.0, "Sign flip must produce negative inner product.")

        # Gaussian attack
        gaussian_vec = ByzantineAdversary.gaussian_noise(honest_vec, mean=0.0, std=5.0, seed=123)
        self.assertEqual(gaussian_vec.shape, honest_vec.shape)
        self.assertAlmostEqual(float(np.mean(gaussian_vec)), 0.0, delta=1.0)
        self.assertAlmostEqual(float(np.std(gaussian_vec)), 5.0, delta=1.0)

    def test_02_fed_avg_poisoning_vulnerability(self):
        """Verify that standard FedAvg fails when 20% of nodes execute a sign-flip attack."""
        byz_ids = [0, 1, 2, 3]
        poisoned_updates, _ = ByzantineAdversary.corrupt_client_updates(
            self.honest_updates,
            byzantine_client_ids=byz_ids,
            attack_type="sign_flip",
            scaling_factor=5.0,  # 4 attackers at -5.0 strictly invert 16 honest nodes at +1.0
        )

        clean_avg = RobustAggregator.fed_avg(self.honest_updates)
        poisoned_avg = RobustAggregator.fed_avg(poisoned_updates)

        # Honest mean is positive; poisoned mean is driven negative
        self.assertGreater(float(np.mean(clean_avg)), 0.0)
        self.assertLess(float(np.mean(poisoned_avg)), 0.0, "FedAvg was not successfully inverted by attack.")

    def test_03_coordinate_median_resilience(self):
        """Verify that Coordinate-wise Median rejects the 20% sign-flip attackers."""
        byz_ids = [0, 1, 2, 3]
        poisoned_updates, _ = ByzantineAdversary.corrupt_client_updates(
            self.honest_updates,
            byzantine_client_ids=byz_ids,
            attack_type="sign_flip",
            scaling_factor=4.0,
        )

        median_agg = RobustAggregator.coordinate_median(poisoned_updates)
        # Median is dominated by the 16 honest nodes, remaining positive
        self.assertGreater(float(np.mean(median_agg)), 0.04)

    def test_04_multi_krum_outlier_rejection(self):
        """Verify Multi-Krum detects and rejects all 4 Byzantine nodes in Float32."""
        byz_ids = [0, 1, 2, 3]
        poisoned_updates, _ = ByzantineAdversary.corrupt_client_updates(
            self.honest_updates,
            byzantine_client_ids=byz_ids,
            attack_type="sign_flip",
            scaling_factor=2.5,
        )

        agg_vec, selected, metadata = RobustAggregator.multi_krum(
            poisoned_updates,
            num_byzantine=self.num_byzantine,
            m_candidates=self.num_clients - self.num_byzantine,  # 16
        )

        # Assert all 4 Byzantine nodes were rejected
        for bad_id in byz_ids:
            self.assertIn(bad_id, metadata["rejected_indices"], f"Byzantine node {bad_id} was not rejected.")
            self.assertNotIn(bad_id, selected, f"Byzantine node {bad_id} was selected by Multi-Krum.")

        # Krum scores of attackers should be substantially higher than honest clients
        scores = metadata["krum_scores"]
        honest_max_score = max(scores[i] for i in range(self.num_clients) if i not in byz_ids)
        byzantine_min_score = min(scores[i] for i in byz_ids)
        self.assertGreater(byzantine_min_score, honest_max_score)

    def test_05_quantized_multi_krum_defense_resilience(self):
        """Core Research Question Unit Test:
        Verify Multi-Krum still isolates all Byzantine nodes under INT8 quantization noise.
        """
        byz_ids = [0, 1, 2, 3]
        poisoned_updates, _ = ByzantineAdversary.corrupt_client_updates(
            self.honest_updates,
            byzantine_client_ids=byz_ids,
            attack_type="sign_flip",
            scaling_factor=2.5,
        )

        # Simulate edge nodes quantizing updates to INT8 before sending to server
        quantized_dequantized_updates = []
        for upd in poisoned_updates:
            q_arr, scale = Int8Quantizer.quantize(upd)
            rec = Int8Quantizer.dequantize(q_arr, scale)
            quantized_dequantized_updates.append(rec)

        # Run Multi-Krum on reconstructed INT8 updates
        agg_vec, selected, metadata = RobustAggregator.multi_krum(
            quantized_dequantized_updates,
            num_byzantine=self.num_byzantine,
            m_candidates=self.num_clients - self.num_byzantine,
        )

        # Assert defense resilience under INT8 compression
        for bad_id in byz_ids:
            self.assertIn(bad_id, metadata["rejected_indices"])
            self.assertNotIn(bad_id, selected)

        # Final aggregated vector remains aligned with honest direction
        self.assertGreater(float(np.mean(agg_vec)), 0.04)


if __name__ == "__main__":
    unittest.main()
