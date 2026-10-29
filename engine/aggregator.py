"""Robust Aggregation Engine for EdgeFed-Bench.

Implements baseline and Byzantine-tolerant server aggregation rules:
1. FedAvg: Standard weighted mean (vulnerable to Byzantine manipulation).
2. Coordinate-wise Median: Component-wise median aggregator.
3. Multi-Krum: Byzantine-resilient aggregation via Euclidean distance scoring.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class RobustAggregator:
    """Server-side parameter aggregator comparing baseline and robust strategies."""

    @staticmethod
    def fed_avg(
        updates: List[np.ndarray],
        weights: Optional[List[float]] = None,
    ) -> np.ndarray:
        """Standard Federated Averaging (FedAvg).

        Mathematical Formulation:
            v_agg = sum_{i=1}^N (w_i * v_i) / sum(w_i)
        
        Args:
            updates: List of client parameter update vectors.
            weights: Optional weighting factors (e.g., sample counts).

        Returns:
            Aggregated global update vector.
        """
        if not updates:
            raise ValueError("Cannot aggregate empty list of updates.")

        stacked = np.stack(updates, axis=0)

        if weights is None:
            return np.mean(stacked, axis=0).astype(np.float32)

        w = np.asarray(weights, dtype=np.float64)
        w_norm = w / np.sum(w)
        aggregated = np.sum(stacked * w_norm[:, np.newaxis], axis=0)
        return aggregated.astype(np.float32)

    @staticmethod
    def coordinate_median(updates: List[np.ndarray]) -> np.ndarray:
        """Coordinate-wise Median Aggregator.

        Mathematical Formulation:
            [v_med]_j = median({ [v_1]_j, [v_2]_j, ..., [v_N]_j })

        Args:
            updates: List of client update vectors.

        Returns:
            Aggregated update vector computed component-by-component.
        """
        if not updates:
            raise ValueError("Cannot aggregate empty list of updates.")

        stacked = np.stack(updates, axis=0)
        median_vec = np.median(stacked, axis=0)
        return median_vec.astype(np.float32)

    @staticmethod
    def trimmed_mean(
        updates: List[np.ndarray],
        num_byzantine: Optional[int] = None,
        beta: Optional[float] = None,
    ) -> np.ndarray:
        """Coordinate-wise Trimmed Mean Aggregator.

        Mathematical Formulation:
            For each coordinate j in {1, ..., D}:
            1. Sort the N values: [v_{(1)}]_j <= [v_{(2)}]_j <= ... <= [v_{(N)}]_j
            2. Drop the k smallest and k largest values, where k = ceil(beta * N).
               Default beta = f / N, where f is the assumed number of Byzantine attackers.
            3. Average the remaining N - 2k values:
               [v_trim]_j = (1 / (N - 2k)) * sum_{i=k+1}^{N-k} [v_{(i)}]_j

        Args:
            updates: List of client update vectors of shape (D,).
            num_byzantine: Assumed number of Byzantine attackers (f).
            beta: Trimming fraction. If None, defaults to f / N.

        Returns:
            Aggregated update vector computed component-by-component.

        Raises:
            ValueError: If updates is empty, neither num_byzantine nor beta is provided,
                        or if 2k >= N.
        """
        if not updates:
            raise ValueError("Cannot aggregate empty list of updates.")

        num_clients = len(updates)
        if beta is None:
            if num_byzantine is None:
                raise ValueError("Either num_byzantine or beta must be specified for trimmed_mean.")
            beta = float(num_byzantine) / float(num_clients)

        k = int(np.ceil(beta * num_clients))

        if 2 * k >= num_clients:
            raise ValueError(
                f"Trimmed mean requires 2*k < N to retain at least one candidate update, "
                f"but got 2*{k} = {2 * k} >= {num_clients}."
            )

        stacked = np.stack(updates, axis=0)
        sorted_stacked = np.sort(stacked, axis=0)

        if k == 0:
            trimmed = sorted_stacked
        else:
            trimmed = sorted_stacked[k : num_clients - k, :]

        return np.mean(trimmed, axis=0).astype(np.float32)

    @staticmethod
    def multi_krum(
        updates: List[np.ndarray],
        num_byzantine: int,
        m_candidates: Optional[int] = None,
    ) -> Tuple[np.ndarray, List[int], Dict[str, Any]]:
        """Multi-Krum Byzantine-resilient Aggregator.

        Theoretical Foundation (Blanchard et al., NeurIPS 2017):
            Requires 2f + 2 < N, where:
            - N is total number of clients.
            - f is the number of Byzantine adversaries.
            
            For each client vector v_i:
            1. Computes squared Euclidean distance to all other clients:
               d_ij^2 = ||v_i - v_j||_2^2
            2. Sorts distances in ascending order.
            3. Computes the Krum score:
               s(i) = sum_{k=1}^{N - f - 2} d_{i, (k)}^2
               (sum of squared distances to its N - f - 2 closest neighbors).
            4. In Multi-Krum, selects the m candidate clients with the smallest
               scores s(i) (typically m = N - f) and averages their updates:
               v_agg = (1 / m) * sum_{k in Selected} v_k

        Args:
            updates: List of client update vectors of shape (D,).
            num_byzantine: Count of assumed Byzantine adversaries (f).
            m_candidates: Number of top candidate updates to average (default: N - f).

        Returns:
            Tuple of:
            - aggregated_update: Aggregated vector of shape (D,).
            - selected_indices: Indices of clients chosen by the defense.
            - metadata: Dictionary containing distance matrix, krum scores, and rejected client IDs.
        """
        num_clients = len(updates)
        if num_clients == 0:
            raise ValueError("No updates provided to Multi-Krum.")

        f = num_byzantine
        # Theoretical safety condition
        if 2 * f + 2 >= num_clients:
            # Clamp f to maximum safe bound if needed
            f = max(0, (num_clients - 3) // 2)

        if m_candidates is None:
            m = max(1, num_clients - f)
        else:
            m = max(1, min(m_candidates, num_clients))

        # Number of closest neighbors to sum over
        num_neighbors = num_clients - f - 2
        if num_neighbors <= 0:
            num_neighbors = 1

        # Cast to float64 to prevent numerical cancellation in high dimensions
        stacked_64 = np.stack(updates, axis=0).astype(np.float64)

        # Vectorized Euclidean squared distance matrix:
        # ||u - v||^2 = ||u||^2 + ||v||^2 - 2 * <u, v>
        norms_sq = np.sum(stacked_64 ** 2, axis=1, keepdims=True)
        dist_matrix = np.maximum(norms_sq + norms_sq.T - 2.0 * np.dot(stacked_64, stacked_64.T), 0.0)
        np.fill_diagonal(dist_matrix, 0.0)

        # Calculate Krum score for each client
        scores = np.zeros(num_clients, dtype=np.float64)
        for i in range(num_clients):
            other_distances = np.delete(dist_matrix[i], i)
            sorted_distances = np.sort(other_distances)
            scores[i] = np.sum(sorted_distances[:num_neighbors])

        # Multi-Krum: Select m clients with smallest Krum scores
        sorted_client_indices = np.argsort(scores).tolist()
        selected_indices = sorted_client_indices[:m]
        rejected_indices = sorted_client_indices[m:]

        # Compute average of selected candidate updates
        selected_updates = [updates[idx] for idx in selected_indices]
        aggregated_update = np.mean(np.stack(selected_updates, axis=0), axis=0).astype(np.float32)

        metadata = {
            "num_clients": num_clients,
            "f_tolerated": f,
            "m_selected": m,
            "num_neighbors_summed": num_neighbors,
            "krum_scores": scores.tolist(),
            "selected_indices": selected_indices,
            "rejected_indices": rejected_indices,
        }

        return aggregated_update, selected_indices, metadata
