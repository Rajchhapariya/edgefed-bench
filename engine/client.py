"""Edge Client Training and Local Optimization Layer.

Implements edge worker logic:
1. Loads client-specific partitioned data via DuckDB EdgeStore.
2. Performs local Stochastic Gradient Descent (SGD) on an edge classification model.
3. Computes parameter update vectors and optional INT8 symmetric quantization.
4. Accounts for communication payload sizes and gradient telemetry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from engine.quantization import Int8Quantizer
from engine.storage import EdgeStore


class EdgeClassifier(nn.Module):
    """Multi-Layer Perceptron for 784-dimensional edge classification."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))

    def get_flat_parameters(self) -> np.ndarray:
        """Flattens all trainable parameters into a single continuous float32 vector."""
        params = [p.data.detach().cpu().view(-1) for p in self.parameters()]
        return torch.cat(params).numpy().astype(np.float32)

    def set_flat_parameters(self, flat_params: np.ndarray) -> None:
        """Populates model weights and biases from a continuous float32 vector."""
        offset = 0
        flat_tensor = torch.from_numpy(np.asarray(flat_params, dtype=np.float32))
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_tensor[offset : offset + numel].view_as(p))
            offset += numel


@dataclass
class ClientUpdate:
    """Structure encapsulating the output transmitted by an edge client per round."""
    client_id: int
    update: np.ndarray
    gradient_norm: float
    payload_bytes: int
    quantized: bool
    num_samples: int
    local_loss: float
    noise_metrics: Optional[Dict[str, float]] = None


class EdgeClient:
    """Represents an edge device performing local training on its private Parquet partition."""

    def __init__(
        self,
        client_id: int,
        store: EdgeStore,
        lr: float = 0.05,
        batch_size: int = 64,
        local_epochs: int = 1,
        device: str = "cpu",
    ):
        self.client_id = client_id
        self.store = store
        self.lr = lr
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.device = torch.device(device)

        self.model = EdgeClassifier().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Cache local dataset in memory from DuckDB
        features_np, labels_np = self.store.get_client_data(self.client_id)
        self.num_samples = len(labels_np)

        self.dataset = TensorDataset(
            torch.from_numpy(features_np).float(),
            torch.from_numpy(labels_np).long(),
        )

    def train_round(
        self,
        global_params: np.ndarray,
        quantize: bool = False,
    ) -> ClientUpdate:
        """Executes one round of local training starting from global model weights.

        Args:
            global_params: Continuous float32 array of global weights.
            quantize: If True, compresses the parameter update vector to INT8.

        Returns:
            ClientUpdate containing update vector, transmission payload size, and telemetry.
        """
        # Synchronize local model with incoming global parameters
        self.model.set_flat_parameters(global_params)
        self.model.train()

        # Optimizer is intentionally re-instantiated every round.
        # FedAvg (McMahan et al., 2017) assumes stateless local updates:
        # momentum does not carry over across communication rounds.
        # Each round begins from incoming global_params with a fresh optimizer state.
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=0.9)

        # DataLoader is re-created per round to reshuffle the local dataset for each pass.
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True)

        running_loss = 0.0
        batches = 0

        for _ in range(self.local_epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                batches += 1

        avg_loss = running_loss / max(batches, 1)

        # Compute raw parameter delta: Delta w = w_local - w_global
        local_params = self.model.get_flat_parameters()
        raw_update = local_params - global_params
        grad_norm = float(np.linalg.norm(raw_update))

        if quantize:
            q_arr, scale = Int8Quantizer.quantize(raw_update)
            payload_bytes = Int8Quantizer.measure_payload(q_arr, scale)
            reconstructed_update = Int8Quantizer.dequantize(q_arr, scale)
            noise_metrics = Int8Quantizer.compute_noise_metrics(raw_update, reconstructed_update)

            return ClientUpdate(
                client_id=self.client_id,
                update=reconstructed_update,
                gradient_norm=grad_norm,
                payload_bytes=payload_bytes,
                quantized=True,
                num_samples=self.num_samples,
                local_loss=avg_loss,
                noise_metrics=noise_metrics,
            )
        else:
            payload_bytes = Int8Quantizer.measure_unquantized_payload(raw_update)
            return ClientUpdate(
                client_id=self.client_id,
                update=raw_update,
                gradient_norm=grad_norm,
                payload_bytes=payload_bytes,
                quantized=False,
                num_samples=self.num_samples,
                local_loss=avg_loss,
                noise_metrics=None,
            )
