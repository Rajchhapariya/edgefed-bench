"""Wire formats for client updates: Float32, INT8, INT4 and scaled sign.

`encode` returns what the server reconstructs from a payload and the payload size in
bytes. INT8 goes through Int8Quantizer unchanged, so INT8 results match the stored grid
bit for bit.

- INT4: uniform symmetric, one scale per vector, values in [-7, 7]. Two values share a
  byte, so the payload is ceil(D / 2) + 4 bytes.
- Sign: one bit per coordinate plus one float32 scale equal to mean(|x|), as in scaled
  sign compression (Karimireddy et al., 2019). Payload ceil(D / 8) + 4 bytes. A zero
  coordinate is sent as +1, because one bit cannot encode zero.
"""

from __future__ import annotations

import math
from typing import Tuple

import numpy as np

from engine.quantization import Int8Quantizer

PRECISIONS = ("Float32", "INT8", "INT4", "Sign")
SCALE_BYTES = 4


def payload_bytes(precision: str, num_elements: int) -> int:
    """Bytes one client sends for an update with num_elements coordinates."""
    if precision == "Float32":
        return 4 * num_elements
    if precision == "INT8":
        return num_elements + SCALE_BYTES
    if precision == "INT4":
        return math.ceil(num_elements / 2) + SCALE_BYTES
    if precision == "Sign":
        return math.ceil(num_elements / 8) + SCALE_BYTES
    raise ValueError(f"Unknown precision: {precision}")


def _uniform_round_trip(x: np.ndarray, q_max: int) -> np.ndarray:
    abs_max = float(np.max(np.abs(x)))
    if abs_max == 0.0 or np.isnan(abs_max):
        return np.zeros_like(x, dtype=np.float32)
    scale = abs_max / q_max
    q = np.clip(np.round(x / scale), -q_max, q_max)
    return (q.astype(np.float32) * np.float32(scale)).astype(np.float32)


def encode(update: np.ndarray, precision: str) -> Tuple[np.ndarray, int]:
    """Returns (vector the server reconstructs, payload bytes) for one update."""
    x = np.asarray(update, dtype=np.float32)
    if precision == "Float32":
        return x, payload_bytes(precision, x.size)
    if precision == "INT8":
        q, scale = Int8Quantizer.quantize(x)
        return Int8Quantizer.dequantize(q, scale), payload_bytes(precision, x.size)
    if precision == "INT4":
        return _uniform_round_trip(x, q_max=7), payload_bytes(precision, x.size)
    if precision == "Sign":
        scale = np.float32(np.mean(np.abs(x)))
        signs = np.where(x >= 0, np.float32(1.0), np.float32(-1.0))
        return (signs * scale).astype(np.float32), payload_bytes(precision, x.size)
    raise ValueError(f"Unknown precision: {precision}")
