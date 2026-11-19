"""Client model and local training.

A client loads its shard once, trains one local epoch of SGD per round, and
returns its update (optionally INT8-quantized) with the payload size it would send.
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
    """MLP 784-128-10 (101,770 parameters) for Fashion-MNIST."""

    def __init__(self, input_dim: int = 784, hidden_dim: int = 128, num_classes: int = 10):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))

    def get_flat_parameters(self) -> np.ndarray:
        """Returns all trainable parameters as one float32 vector."""
        params = [p.data.detach().cpu().view(-1) for p in self.parameters()]
        return torch.cat(params).numpy().astype(np.float32)

    def set_flat_parameters(self, flat_params: np.ndarray) -> None:
        """Loads all trainable parameters from one float32 vector."""
        offset = 0
        flat_tensor = torch.from_numpy(np.asarray(flat_params, dtype=np.float32))
        for p in self.parameters():
            numel = p.numel()
            p.data.copy_(flat_tensor[offset : offset + numel].view_as(p))
            offset += numel


@dataclass
class ClientUpdate:
    """What a client returns after one round: its update and telemetry."""
    client_id: int
    update: np.ndarray
    gradient_norm: float
    payload_bytes: int
    quantized: bool
    num_samples: int
    local_loss: float
    noise_metrics: Optional[Dict[str, float]] = None


class EdgeClient:
    """One simulated client that trains on its own Parquet shard."""

    def __init__(
        self,
        client_id: int,
        store: EdgeStore,
        lr: float = 0.05,
        batch_size: int = 64,
        local_epochs: int = 1,
        device: str = "cpu",
        momentum: float = 0.9,
    ):
        self.client_id = client_id
        self.store = store
        self.lr = lr
        self.momentum = momentum
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.device = torch.device(device)

        self.model = EdgeClassifier().to(self.device)
        self.criterion = nn.CrossEntropyLoss()

        # Load the shard once; every round reuses it.
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
        round_num: int = 1,
        seed: Optional[int] = None,
    ) -> ClientUpdate:
        """Trains one round from the global weights and returns the update.

        Args:
            global_params: Continuous float32 array of global weights.
            quantize: If True, compresses the parameter update vector to INT8.
            round_num: Current communication round index (for deterministic shuffling).
            seed: Master random seed for deterministic execution.

        Returns:
            ClientUpdate containing update vector, transmission payload size, and telemetry.
        """
        self.model.set_flat_parameters(global_params)
        self.model.train()

        # A fresh optimizer each round keeps local training stateless, as in FedAvg
        # (McMahan et al., 2017). Momentum therefore does not carry across rounds.
        optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)

        # A new loader each round gives a new shuffle. Seeding it from (seed, round,
        # client) makes that order repeatable.
        if seed is not None:
            gen = torch.Generator().manual_seed((seed * 100_000) + (round_num * 1000) + self.client_id)
            loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True, generator=gen)
        else:
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
