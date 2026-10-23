"""Unit test suite for INT8 quantization and edge client optimization."""

import unittest
import numpy as np
import torch

from engine.quantization import Int8Quantizer
from engine.client import EdgeClassifier, EdgeClient
from engine.storage import EdgeStore


class TestQuantizationAndClient(unittest.TestCase):
    """Tests verifying INT8 bounds, quantization noise, and edge training."""

    def test_01_quantize_dequantize_bounds(self):
        """Verify INT8 quantization produces int8 types within [-127, 127] and strictly bounded error."""
        rng = np.random.default_rng(1234)
        for size in [100, 1000, 101770]:
            tensor = rng.normal(loc=0.0, scale=0.5, size=size).astype(np.float32)
            q_arr, scale = Int8Quantizer.quantize(tensor)

            self.assertEqual(q_arr.dtype, np.int8)
            self.assertGreaterEqual(int(q_arr.min()), -127)
            self.assertLessEqual(int(q_arr.max()), 127)
            self.assertGreater(scale, 0.0)

            rec = Int8Quantizer.dequantize(q_arr, scale)
            self.assertEqual(rec.dtype, np.float32)
            self.assertEqual(rec.shape, tensor.shape)

            # Mathematical bound: element-wise error <= scale / 2 + numerical epsilon
            max_abs_err = np.max(np.abs(tensor - rec))
            self.assertLessEqual(max_abs_err, (scale / 2.0) + 1e-6)

        # Verify unsupported bit depths raise NotImplementedError
        with self.assertRaises(NotImplementedError):
            Int8Quantizer.quantize(np.ones(10, dtype=np.float32), num_bits=4)

    def test_02_all_zeros_and_extreme_inputs(self):
        """Verify numerical stability on degenerate arrays (zeros, single-element, extreme magnitudes)."""
        zeros = np.zeros(500, dtype=np.float32)
        q_arr, scale = Int8Quantizer.quantize(zeros)
        self.assertEqual(scale, 1.0)
        self.assertTrue(np.all(q_arr == 0))
        rec = Int8Quantizer.dequantize(q_arr, scale)
        self.assertTrue(np.all(rec == 0.0))

        # Extreme values
        extreme = np.array([-1e6, 0.0, 1e6], dtype=np.float32)
        q_ext, scale_ext = Int8Quantizer.quantize(extreme)
        self.assertEqual(q_ext[0], -127)
        self.assertEqual(q_ext[2], 127)
        rec_ext = Int8Quantizer.dequantize(q_ext, scale_ext)
        self.assertAlmostEqual(rec_ext[0], -1e6, delta=1e4)

    def test_03_payload_bandwidth_savings(self):
        """Verify that INT8 quantization cuts network payload by ~75% on 101,770 model parameters."""
        param_count = 101770
        dummy_tensor = np.ones(param_count, dtype=np.float32)

        unquantized_bytes = Int8Quantizer.measure_unquantized_payload(dummy_tensor)
        self.assertEqual(unquantized_bytes, 101770 * 4)  # 407,080 bytes

        q_arr, scale = Int8Quantizer.quantize(dummy_tensor)
        quantized_bytes = Int8Quantizer.measure_payload(q_arr, scale)
        self.assertEqual(quantized_bytes, 101770 * 1 + 4)  # 101,774 bytes

        reduction_ratio = (unquantized_bytes - quantized_bytes) / unquantized_bytes
        self.assertGreater(reduction_ratio, 0.7499)
        self.assertLess(reduction_ratio, 0.7501)

    def test_04_quantization_noise_metrics(self):
        """Verify SQNR, MSE, and residual norm metrics are calculated correctly."""
        orig = np.linspace(-2.0, 2.0, 1000, dtype=np.float32)
        q_arr, scale = Int8Quantizer.quantize(orig)
        rec = Int8Quantizer.dequantize(q_arr, scale)

        metrics = Int8Quantizer.compute_noise_metrics(orig, rec)
        self.assertIn("mse", metrics)
        self.assertIn("l2_error", metrics)
        self.assertIn("max_error", metrics)
        self.assertIn("sqnr_db", metrics)

        self.assertGreater(metrics["sqnr_db"], 30.0, "SQNR should be above 30 dB for 8-bit quantization.")
        self.assertLess(metrics["mse"], 0.001)

    def test_05_edge_classifier_bidirectional_parameter_sync(self):
        """Verify flat vector synchronization into PyTorch model layers."""
        model = EdgeClassifier()
        flat_initial = model.get_flat_parameters()
        self.assertEqual(len(flat_initial), 101770)

        modified = flat_initial + 0.12345
        model.set_flat_parameters(modified)
        flat_after = model.get_flat_parameters()

        self.assertTrue(np.allclose(modified, flat_after, atol=1e-6))

    def test_06_edge_client_training_unquantized_and_quantized(self):
        """Verify client training rounds produce valid parameter deltas and payload telemetry."""
        store = EdgeStore(data_root="data")
        client = EdgeClient(client_id=0, store=store, lr=0.01, batch_size=128, local_epochs=1)
        global_params = client.model.get_flat_parameters()

        # 1. Unquantized training round
        update_fp32 = client.train_round(global_params, quantize=False)
        self.assertFalse(update_fp32.quantized)
        self.assertEqual(len(update_fp32.update), 101770)
        self.assertEqual(update_fp32.payload_bytes, 407080)
        self.assertGreater(update_fp32.gradient_norm, 0.0)
        self.assertGreater(update_fp32.local_loss, 0.0)
        self.assertIsNone(update_fp32.noise_metrics)

        # 2. Quantized training round
        update_int8 = client.train_round(global_params, quantize=True)
        self.assertTrue(update_int8.quantized)
        self.assertEqual(len(update_int8.update), 101770)
        self.assertEqual(update_int8.payload_bytes, 101774)
        self.assertGreater(update_int8.gradient_norm, 0.0)
        self.assertIsNotNone(update_int8.noise_metrics)
        self.assertGreater(update_int8.noise_metrics["sqnr_db"], 18.0)


if __name__ == "__main__":
    unittest.main()
