"""EdgeFed-Bench: Benchmark Orchestrator and Simulation Runner.

Executes a grid of 4 controlled experimental regimes:
1. Baseline FedAvg (Honest clients, Float32).
2. Poisoned FedAvg (20% Byzantine attackers, Float32).
3. Defended Multi-Krum (20% Byzantine attackers, Float32).
4. Quantized Defended Multi-Krum (20% Byzantine attackers, INT8 Quantized).

Logs telemetry to DuckDB and outputs a comparative benchmark summary.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from engine.adversary import ByzantineAdversary
from engine.aggregator import RobustAggregator
from engine.client import EdgeClassifier, EdgeClient
from engine.storage import EdgeStore, TelemetryEngine, initialize_storage_pipeline


def evaluate_global_model(
    model: EdgeClassifier,
    x_test: np.ndarray,
    y_test: np.ndarray,
    batch_size: int = 512,
) -> Tuple[float, float]:
    """Evaluates the global model on the central test Parquet partition."""
    model.eval()
    criterion = nn.CrossEntropyLoss()

    x_tensor = torch.from_numpy(x_test).float()
    y_tensor = torch.from_numpy(y_test).long()

    total_samples = len(y_test)
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            bx = x_tensor[i : i + batch_size]
            by = y_tensor[i : i + batch_size]
            outputs = model(bx)
            loss = criterion(outputs, by)
            total_loss += loss.item() * len(by)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == by).sum().item()

    accuracy = (correct / total_samples) * 100.0
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss


def run_regime(
    regime_name: str,
    aggregation_method: str,
    byzantine_ratio: float,
    quantize: bool,
    num_rounds: int,
    num_clients: int,
    batch_size: int,
    lr: float,
    store: EdgeStore,
    telemetry: TelemetryEngine,
    experiment_id: str,
    x_test: np.ndarray,
    y_test: np.ndarray,
    initial_weights: np.ndarray,
    attack_type: str = "sign_flip",
) -> Dict[str, Any]:
    """Runs a single controlled federated learning regime."""
    print(f"\n========================================================")
    print(f"Starting: {regime_name}")
    print(f"Aggregation: {aggregation_method} | Byzantine: {byzantine_ratio*100:.0f}% | Quantized: {quantize}")
    print(f"========================================================")

    clients = [
        EdgeClient(
            client_id=i,
            store=store,
            lr=lr,
            batch_size=batch_size,
            local_epochs=1,
        )
        for i in range(num_clients)
    ]

    global_model = EdgeClassifier()
    global_model.set_flat_parameters(initial_weights)

    num_byzantine = int(num_clients * byzantine_ratio)
    byzantine_ids = list(range(num_byzantine))  # First f clients designated as Byzantine

    total_payload_bytes = 0
    successful_mitigations = 0
    mitigation_eval_rounds = 0

    history_accuracy = []
    history_loss = []

    for round_num in range(1, num_rounds + 1):
        round_start_time = time.time()
        current_weights = global_model.get_flat_parameters()

        # 1. Edge computation: clients perform local training
        client_updates = []
        for client in clients:
            upd = client.train_round(current_weights, quantize=quantize)
            client_updates.append(upd)
            total_payload_bytes += upd.payload_bytes

            is_byz = client.client_id in byzantine_ids
            telemetry.log_round_telemetry(
                experiment_id=experiment_id,
                regime_name=regime_name,
                round_num=round_num,
                client_id=client.client_id,
                is_byzantine=is_byz,
                gradient_norm=upd.gradient_norm,
                payload_bytes=upd.payload_bytes,
                quantized=upd.quantized,
                local_loss=upd.local_loss,
            )

        # 2. Extract raw update vectors
        raw_vectors = [u.update for u in client_updates]

        # 3. Byzantine attack injection
        if num_byzantine > 0:
            corrupted_vectors, _ = ByzantineAdversary.corrupt_client_updates(
                raw_vectors,
                byzantine_client_ids=byzantine_ids,
                attack_type=attack_type,
                scaling_factor=2.5,
                seed=42 + round_num,
            )
        else:
            corrupted_vectors = raw_vectors

        # 4. Server-side aggregation
        if aggregation_method == "FedAvg":
            aggregated_vector = RobustAggregator.fed_avg(corrupted_vectors)
        elif aggregation_method == "Multi-Krum":
            aggregated_vector, selected_indices, meta = RobustAggregator.multi_krum(
                corrupted_vectors,
                num_byzantine=num_byzantine,
                m_candidates=num_clients - num_byzantine,
            )
            # Track mitigation success: did Multi-Krum reject all Byzantine nodes?
            mitigation_eval_rounds += 1
            attackers_in_selection = set(selected_indices).intersection(set(byzantine_ids))
            if len(attackers_in_selection) == 0:
                successful_mitigations += 1
        elif aggregation_method in ("Coordinate-Median", "Median"):
            aggregated_vector = RobustAggregator.coordinate_median(corrupted_vectors)
            mitigation_eval_rounds += 1
            if float(np.mean(aggregated_vector)) > 0.0:
                successful_mitigations += 1
        else:
            raise ValueError(f"Unknown aggregation method: {aggregation_method}")

        # 5. Global model parameter update: w_{t+1} = w_t + v_agg
        new_weights = current_weights + aggregated_vector
        global_model.set_flat_parameters(new_weights)

        # 6. Global evaluation on test set
        acc, test_loss = evaluate_global_model(global_model, x_test, y_test)
        history_accuracy.append(acc)
        history_loss.append(test_loss)
        duration = time.time() - round_start_time

        telemetry.log_global_eval(
            experiment_id=experiment_id,
            regime_name=regime_name,
            round_num=round_num,
            global_accuracy=acc,
            global_loss=test_loss,
            aggregation_method=aggregation_method,
            byzantine_fraction=byzantine_ratio,
            quantized=quantize,
            round_duration_sec=duration,
        )

        print(
            f"Round {round_num:02d}/{num_rounds:02d} | "
            f"Test Acc: {acc:6.2f}% | Test Loss: {test_loss:6.4f} | "
            f"Duration: {duration:.2f}s"
        )

    mitigation_rate = (
        (successful_mitigations / max(mitigation_eval_rounds, 1)) * 100.0
        if num_byzantine > 0
        else 100.0
    )

    return {
        "regime_name": regime_name,
        "aggregation_method": aggregation_method,
        "quantized": quantize,
        "byzantine_ratio": byzantine_ratio,
        "peak_accuracy": max(history_accuracy),
        "final_accuracy": history_accuracy[-1],
        "final_loss": history_loss[-1],
        "total_payload_mb": total_payload_bytes / (1024 * 1024),
        "payload_per_client_kb": (total_payload_bytes / (num_clients * num_rounds)) / 1024,
        "mitigation_rate": mitigation_rate,
    }


def main():
    parser = argparse.ArgumentParser(description="EdgeFed-Bench Benchmark Runner")
    parser.add_argument("--rounds", type=int, default=5, help="Number of federated communication rounds (default: 5)")
    parser.add_argument("--clients", type=int, default=20, help="Total edge clients (default: 20)")
    parser.add_argument("--alpha", type=float, default=0.5, help="Dirichlet concentration parameter (default: 0.5)")
    parser.add_argument("--batch_size", type=int, default=128, help="Client batch size (default: 128)")
    parser.add_argument("--lr", type=float, default=0.05, help="Client learning rate (default: 0.05)")
    parser.add_argument("--data_root", type=str, default="data", help="Data directory (default: data)")
    parser.add_argument(
        "--attack_type",
        type=str,
        default="sign_flip",
        choices=["sign_flip", "gaussian"],
        help="Byzantine attack strategy: 'sign_flip' or 'gaussian' (default: sign_flip)",
    )
    parser.add_argument(
        "--force_rebuild",
        action="store_true",
        help="Force rebuild of Parquet shards and Dirichlet partitions (default: False)",
    )
    args = parser.parse_args()

    print("==================================================================")
    print("EdgeFed-Bench: Byzantine-Robust Federated Learning Benchmark")
    print("==================================================================")
    print(f"Configuration: {args.clients} clients, alpha={args.alpha}, rounds={args.rounds}, attack={args.attack_type}")

    client_shards_dir = os.path.join(args.data_root, "clients")
    if (
        args.force_rebuild
        or not os.path.exists(client_shards_dir)
        or len(os.listdir(client_shards_dir)) < args.clients
    ):
        print("[!] Initializing storage pipeline...")
        initialize_storage_pipeline(
            data_root=args.data_root,
            num_clients=args.clients,
            alpha=args.alpha,
            force_rebuild=args.force_rebuild,
        )

    store = EdgeStore(data_root=args.data_root)
    x_test, y_test = store.get_test_data()

    exp_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    telemetry = TelemetryEngine(db_path=os.path.join(args.data_root, "telemetry.duckdb"))

    # Initial model weights for fair cross-regime evaluation
    torch.manual_seed(42)
    initial_model = EdgeClassifier()
    initial_weights = initial_model.get_flat_parameters()

    regimes_to_run = [
        {
            "name": "1. Baseline FedAvg (Honest Float32)",
            "aggregation": "FedAvg",
            "byzantine_ratio": 0.0,
            "quantize": False,
        },
        {
            "name": "2. Poisoned FedAvg (20% Byz, Float32)",
            "aggregation": "FedAvg",
            "byzantine_ratio": 0.20,
            "quantize": False,
        },
        {
            "name": "3. Defended Multi-Krum (20% Byz, Float32)",
            "aggregation": "Multi-Krum",
            "byzantine_ratio": 0.20,
            "quantize": False,
        },
        {
            "name": "4. Quantized Defended Multi-Krum (20% Byz, INT8)",
            "aggregation": "Multi-Krum",
            "byzantine_ratio": 0.20,
            "quantize": True,
        },
    ]

    benchmark_results = []
    for reg in regimes_to_run:
        res = run_regime(
            regime_name=reg["name"],
            aggregation_method=reg["aggregation"],
            byzantine_ratio=reg["byzantine_ratio"],
            quantize=reg["quantize"],
            num_rounds=args.rounds,
            num_clients=args.clients,
            batch_size=args.batch_size,
            lr=args.lr,
            store=store,
            telemetry=telemetry,
            experiment_id=exp_id,
            x_test=x_test,
            y_test=y_test,
            initial_weights=initial_weights,
            attack_type=args.attack_type,
        )
        benchmark_results.append(res)

    telemetry.close()
    store.close()

    # Print summary table
    print("\n" + "=" * 105)
    print("FINAL BENCHMARK COMPARISON TABLE")
    print("=" * 105)
    header = (
        f"{'Regime':<46} | {'Peak Acc':<9} | {'Final Acc':<9} | "
        f"{'KB/Client':<10} | {'Total MB':<9} | {'Mitigation':<10}"
    )
    print(header)
    print("-" * 105)

    for r in benchmark_results:
        row = (
            f"{r['regime_name']:<46} | "
            f"{r['peak_accuracy']:>8.2f}% | "
            f"{r['final_accuracy']:>8.2f}% | "
            f"{r['payload_per_client_kb']:>9.1f} | "
            f"{r['total_payload_mb']:>8.2f} | "
            f"{r['mitigation_rate']:>9.1f}%"
        )
        print(row)
    print("=" * 105)


if __name__ == "__main__":
    main()
