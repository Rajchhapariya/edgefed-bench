"""Byzantine attacks used in EdgeFed-Bench.

Sign-flip sends a scaled, negated honest update. Gaussian sends zero-mean noise
that ignores the honest update entirely. Both are easy to spot because they sit far
from honest updates.

ALIE and IPM are built to stay close to honest updates. Both assume an omniscient
attacker that sees every client's true update in the round:
- ALIE (Baruch et al., NeurIPS 2019): every attacker sends mean - z * std, per
  coordinate, with z chosen so the update stays inside the spread of honest ones.
- IPM (Xie et al., UAI 2019): every attacker sends -epsilon * mean, a small update that
  points against the honest direction.
"""

from __future__ import annotations

from statistics import NormalDist
from typing import List, Optional, Tuple

import numpy as np

from engine import compression


class ByzantineAdversary:
    """Turns the updates of selected clients into attacker updates."""

    @staticmethod
    def select_byzantine_clients(num_clients: int, num_byzantine: int, seed: int = 42) -> List[int]:
        """Picks which clients are Byzantine. The same seed always picks the same clients.

        Args:
            num_clients: Total number of clients N.
            num_byzantine: Number of Byzantine attackers f.
            seed: Master random seed.

        Returns:
            Sorted list of selected Byzantine client IDs.
        """
        if num_byzantine <= 0:
            return []
        if num_byzantine >= num_clients:
            return list(range(num_clients))
        rng = np.random.default_rng(seed + 1000)
        chosen = rng.choice(num_clients, size=num_byzantine, replace=False)
        return sorted(int(c) for c in chosen)

    @staticmethod
    def sign_flip(update: np.ndarray, scaling_factor: float = 2.5) -> np.ndarray:
        """Executes a sign-flip attack.

        Mathematical Formulation:
            v_byz = -scaling_factor * v_honest
            
        The attacker pushes the model against the direction honest clients move it.

        Args:
            update: Honest parameter update vector.
            scaling_factor: Magnitude multiplier (default: 2.5).

        Returns:
            Poisoned parameter update vector.
        """
        arr = np.asarray(update, dtype=np.float32)
        return (-scaling_factor * arr).astype(np.float32)

    @staticmethod
    def gaussian_noise(
        shape_reference: np.ndarray,
        mean: float = 0.0,
        std: float = 5.0,
        seed: int | None = None,
    ) -> np.ndarray:
        """Returns a Gaussian noise vector in place of an honest update.

        Mathematical Formulation:
            v_byz ~ Normal(mean, std^2 * I)
            
        The noise does not depend on the honest update.

        Args:
            shape_reference: Baseline array whose .shape determines output dimensions.
                             Array values are not used in the calculation.
            mean: Distribution mean.
            std: Standard deviation of noise injection.
            seed: Optional deterministic seed for reproducibility.

        Returns:
            Noise-injected poisoned update vector.
        """
        rng = np.random.default_rng(seed)
        noise = rng.normal(loc=mean, scale=std, size=shape_reference.shape).astype(np.float32)
        return noise

    @staticmethod
    def alie_z(num_clients: int, num_byzantine: int) -> float:
        """z for ALIE from Baruch et al. (2019): s = floor(N/2 + 1) - f, z = Phi^-1((N - s) / N)."""
        s = (num_clients // 2 + 1) - num_byzantine
        return float(NormalDist().inv_cdf((num_clients - s) / num_clients))

    @staticmethod
    def alie(true_updates: List[np.ndarray], num_byzantine: int) -> np.ndarray:
        """ALIE update: per-coordinate mean - z * std of all clients' true updates."""
        stacked = np.stack([np.asarray(u, dtype=np.float64) for u in true_updates])
        z = ByzantineAdversary.alie_z(len(true_updates), num_byzantine)
        return (stacked.mean(axis=0) - z * stacked.std(axis=0)).astype(np.float32)

    @staticmethod
    def ipm(true_updates: List[np.ndarray], epsilon: float = 0.1) -> np.ndarray:
        """IPM update: -epsilon times the mean of all clients' true updates."""
        stacked = np.stack([np.asarray(u, dtype=np.float64) for u in true_updates])
        return (-epsilon * stacked.mean(axis=0)).astype(np.float32)

    @staticmethod
    def corrupt_client_updates(
        updates: List[np.ndarray],
        byzantine_client_ids: List[int],
        attack_type: str = "sign_flip",
        scaling_factor: float = 2.5,
        noise_std: float = 5.0,
        seed: int = 42,
        wire_int8: bool = False,
        wire_precision: Optional[str] = None,
        ipm_epsilon: float = 0.1,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Replaces the updates of the given Byzantine clients with attacker updates.

        Args:
            updates: List of client update vectors (ordered 0 to N-1).
            byzantine_client_ids: Indices of nodes acting as adversaries.
            attack_type: "sign_flip", "gaussian", "alie" or "ipm".
            scaling_factor: Multiplier for sign flip.
            noise_std: Standard deviation for Gaussian noise.
            seed: Seed for stochastic attacks.
            wire_int8: Same as wire_precision="INT8".
            wire_precision: Wire format of the run. Attacker updates are encoded in it
                too, because the server can only decode payloads in that format.
            ipm_epsilon: Scale of the IPM update.

        Returns:
            Tuple of:
            - corrupted_updates: List of update vectors containing honest and poisoned vectors.
            - byzantine_client_ids: List of confirmed attacker IDs.
        """
        poisoned_list = []
        byz_set = set(byzantine_client_ids)
        if wire_precision is None:
            wire_precision = "INT8" if wire_int8 else "Float32"

        # ALIE and IPM send one shared vector, computed once from every true update.
        shared = None
        if attack_type == "alie":
            shared = ByzantineAdversary.alie(updates, len(byz_set))
        elif attack_type == "ipm":
            shared = ByzantineAdversary.ipm(updates, epsilon=ipm_epsilon)

        for cid, update in enumerate(updates):
            if cid in byz_set:
                if attack_type == "sign_flip":
                    poisoned = ByzantineAdversary.sign_flip(update, scaling_factor=scaling_factor)
                elif attack_type == "gaussian":
                    poisoned = ByzantineAdversary.gaussian_noise(
                        update,
                        std=noise_std,
                        seed=seed + cid,
                    )
                elif shared is not None:
                    poisoned = shared.copy()
                else:
                    raise ValueError(f"Unknown attack type: {attack_type}")
                if wire_precision != "Float32":
                    poisoned, _ = compression.encode(poisoned, wire_precision)
                poisoned_list.append(poisoned)
            else:
                poisoned_list.append(update.copy())

        return poisoned_list, byzantine_client_ids
