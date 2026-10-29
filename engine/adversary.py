"""Byzantine Adversary Engine for EdgeFed-Bench.

Implements realistic adversarial attack vectors in federated learning:
1. Sign-Flip Attack: Inverts the direction of gradient descent to cause divergence.
2. Gaussian Noise Injection: Injects high-variance zero-mean stochastic disturbance.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np


class ByzantineAdversary:
    """Simulates rogue edge nodes executing Byzantine attack vectors."""

    @staticmethod
    def select_byzantine_clients(num_clients: int, num_byzantine: int, seed: int = 42) -> List[int]:
        """Deterministically selects Byzantine client IDs using the seed.

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
            
        Negates the update direction to pull the global model away from optimal minima.

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
        """Executes a Gaussian variance explosion attack.

        Mathematical Formulation:
            v_byz ~ Normal(mean, std^2 * I)
            
        Injects uncorrelated high-magnitude noise to disrupt parameter stability.

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
    def corrupt_client_updates(
        updates: List[np.ndarray],
        byzantine_client_ids: List[int],
        attack_type: str = "sign_flip",
        scaling_factor: float = 2.5,
        noise_std: float = 5.0,
        seed: int = 42,
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Applies Byzantine corruption to designated client update vectors.

        Args:
            updates: List of client update vectors (ordered 0 to N-1).
            byzantine_client_ids: Indices of nodes acting as adversaries.
            attack_type: "sign_flip" or "gaussian".
            scaling_factor: Multiplier for sign flip.
            noise_std: Standard deviation for Gaussian noise.
            seed: Seed for stochastic attacks.

        Returns:
            Tuple of:
            - corrupted_updates: List of update vectors containing honest and poisoned vectors.
            - byzantine_client_ids: List of confirmed attacker IDs.
        """
        poisoned_list = []
        byz_set = set(byzantine_client_ids)

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
                else:
                    raise ValueError(f"Unknown attack type: {attack_type}")
                poisoned_list.append(poisoned)
            else:
                poisoned_list.append(update.copy())

        return poisoned_list, byzantine_client_ids
