"""INT8 Uniform Symmetric Gradient Quantizer and Compression Engine.

Provides mathematically verified quantization and dequantization routines,
network payload byte accounting, and quantization noise metrics.
"""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np


class Int8Quantizer:
    """Uniform Symmetric INT8 Quantizer for gradient and model parameter compression.
    
    Mathematical Formulation:
        Let x in R^D be a continuous 32-bit floating-point gradient vector.
        Let q_max = 2^(b - 1) - 1 = 127 for b = 8 bits.
        
        Dynamic range:
            alpha = max(|x|)
            scale S = alpha / q_max = max(|x|) / 127
            
        Quantization mapping:
            q = clip(round(x / S), -127, 127)  [stored as int8]
            
        Dequantization reconstruction:
            x_hat = q * S  [reconstructed as float32]
            
        Theoretical bound on individual element error:
            |x_i - x_hat_i| <= S / 2 = alpha / 254
    """

    @staticmethod
    def quantize(
        tensor: np.ndarray,
        num_bits: int = 8,
    ) -> Tuple[np.ndarray, float]:
        """Quantizes a float32 array into signed integers and a floating-point scale factor.

        Args:
            tensor: Input array of float32 values.
            num_bits: Bit depth (currently strictly 8 bits, yielding range [-127, 127]).

        Returns:
            Tuple of:
            - q_tensor: Quantized array of signed integers (np.int8).
            - scale: Floating-point scaling factor (float).

        Raises:
            NotImplementedError: If num_bits != 8.
        """
        if num_bits != 8:
            raise NotImplementedError(
                f"Only 8-bit (num_bits=8) quantization is supported; got num_bits={num_bits}. "
                "Sub-byte precision requires explicit custom bit-packing."
            )

        arr = np.asarray(tensor, dtype=np.float32)
        q_max = (1 << (num_bits - 1)) - 1  # 127 for 8 bits

        abs_max = float(np.max(np.abs(arr)))
        if abs_max == 0.0 or np.isnan(abs_max):
            return np.zeros_like(arr, dtype=np.int8), 1.0

        scale = abs_max / q_max
        scaled = arr / scale
        rounded = np.round(scaled)
        clipped = np.clip(rounded, -q_max, q_max).astype(np.int8)

        return clipped, scale

    @staticmethod
    def dequantize(q_tensor: np.ndarray, scale: float) -> np.ndarray:
        """Reconstructs float32 values from quantized int8 values and the scale factor.

        Args:
            q_tensor: Array of signed 8-bit integers.
            scale: Floating-point scale factor.

        Returns:
            Reconstructed float32 numpy array.
        """
        return q_tensor.astype(np.float32) * float(scale)

    @staticmethod
    def measure_payload(q_tensor: np.ndarray, scale: float) -> int:
        """Calculates exact network transmission size in bytes.

        An edge client transmitting an INT8 quantized gradient vector sends:
        1. The quantized payload array: 1 byte per element.
        2. The scaling factor: 4 bytes (Float32).

        Args:
            q_tensor: Quantized int8 array.
            scale: Scale factor.

        Returns:
            Total payload size in bytes.
        """
        # Note: `scale` is retained in the signature for symmetry with Int8Quantizer.quantize().
        # The numeric value of scale does not alter transmission size:
        # Payload = (1 byte per int8 element) + (4 bytes for the float32 scale scalar).
        payload_bytes = q_tensor.nbytes + 4  # 4 bytes for float32 scale
        return payload_bytes

    @staticmethod
    def measure_unquantized_payload(tensor: np.ndarray) -> int:
        """Calculates uncompressed 32-bit floating-point payload in bytes.

        Each element requires 4 bytes.
        """
        return tensor.size * 4

    @staticmethod
    def compute_noise_metrics(
        original: np.ndarray,
        reconstructed: np.ndarray,
    ) -> Dict[str, float]:
        """Calculates statistical metrics characterizing the quantization error.

        Metrics computed:
            - mse: Mean Squared Error between original and reconstructed vectors.
            - l2_error: Euclidean norm of the residual error vector.
            - max_error: Maximum absolute element-wise reconstruction error.
            - sqnr_db: Signal-to-Quantization-Noise Ratio in decibels (dB).

        Args:
            original: Ground-truth float32 array.
            reconstructed: Dequantized float32 array.

        Returns:
            Dictionary containing computed noise metrics.
        """
        orig = original.astype(np.float64)
        rec = reconstructed.astype(np.float64)
        residual = orig - rec

        mse = float(np.mean(residual ** 2))
        l2_error = float(np.linalg.norm(residual))
        max_error = float(np.max(np.abs(residual)))

        signal_power = float(np.sum(orig ** 2))
        noise_power = float(np.sum(residual ** 2))

        if noise_power == 0.0:
            sqnr_db = float("inf")
        else:
            sqnr_db = float(10.0 * np.log10(signal_power / (noise_power + 1e-12)))

        return {
            "mse": mse,
            "l2_error": l2_error,
            "max_error": max_error,
            "sqnr_db": sqnr_db,
        }
