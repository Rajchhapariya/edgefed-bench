"""Config-Driven Experiment Grid Runner for EdgeFed-Bench v2.

Executes the full factorial experimental grid:
1. 280-run robustness grid across aggregators, attacks, attacker shares, precisions, and seeds.
2. Wrong-f sensitivity experiment across assumed f in {2, 4, 6, 8}.

Persists results into DuckDB:
- `runs`: Summary metrics, final accuracies, bandwidth, convergence rounds.
- `round_metrics`: Round-by-round trajectory for learning curve visualization.
Supports resume functionality via deterministic config hashes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import duckdb
import numpy as np
import torch
import yaml

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.adversary import ByzantineAdversary
from engine.aggregator import RobustAggregator
from engine.client import EdgeClassifier, EdgeClient
from engine.storage import EdgeStore, initialize_storage_pipeline


def compute_config_hash(run_config: Dict[str, Any]) -> str:
    """Computes a deterministic 16-character hex hash from the configuration dictionary."""
    canonical_keys = [
        "experiment_type",
        "aggregator",
        "attack",
        "attacker_share",
        "actual_f",
        "assumed_f",
        "wire_precision",
        "seed",
        "num_rounds",
        "num_clients",
        "dirichlet_alpha",
    ]
    canonical_dict = {k: run_config.get(k) for k in canonical_keys}
    raw_str = json.dumps(canonical_dict, sort_keys=True)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]


def init_database(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Initializes DuckDB tables with strict schema definitions."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))

    con.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        config_hash VARCHAR PRIMARY KEY,
        experiment_type VARCHAR,
        aggregator VARCHAR,
        attack VARCHAR,
        attacker_share DOUBLE,
        actual_f INTEGER,
        assumed_f INTEGER,
        wire_precision VARCHAR,
        seed INTEGER,
        num_rounds INTEGER,
        num_clients INTEGER,
        final_accuracy DOUBLE,
        best_accuracy DOUBLE,
        final_loss DOUBLE,
        acc_round_10 DOUBLE,
        acc_round_25 DOUBLE,
        acc_round_50 DOUBLE,
        rounds_to_70 INTEGER,
        total_uplink_bytes BIGINT,
        bytes_per_client_round DOUBLE,
        wall_clock_sec DOUBLE,
        attacker_rejection_rate DOUBLE,
        honest_false_rejection_rate DOUBLE,
        completed_at TIMESTAMP
    );
    """)

    con.execute("""
    CREATE TABLE IF NOT EXISTS round_metrics (
        config_hash VARCHAR,
        round_num INTEGER,
        test_accuracy DOUBLE,
        test_loss DOUBLE,
        round_duration_sec DOUBLE,
        round_uplink_bytes BIGINT,
        attacker_rejections INTEGER,
        honest_rejections INTEGER,
        PRIMARY KEY (config_hash, round_num)
    );
    """)

    return con


def evaluate_global_model_tensor(
    model: EdgeClassifier,
    x_test_tensor: torch.Tensor,
    y_test_tensor: torch.Tensor,
    batch_size: int = 512,
) -> Tuple[float, float]:
    """Optimized evaluation function utilizing pre-allocated torch tensors."""
    model.eval()
    criterion = torch.nn.CrossEntropyLoss()
    total_samples = len(y_test_tensor)
    correct = 0
    total_loss = 0.0

    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            bx = x_test_tensor[i : i + batch_size]
            by = y_test_tensor[i : i + batch_size]
            outputs = model(bx)
            loss = criterion(outputs, by)
            total_loss += loss.item() * len(by)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == by).sum().item()

    accuracy = (correct / total_samples) * 100.0
    avg_loss = total_loss / total_samples
    return accuracy, avg_loss


def execute_single_run(
    run_config: Dict[str, Any],
    store: EdgeStore,
    x_test_tensor: torch.Tensor,
    y_test_tensor: torch.Tensor,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Executes a single federated learning run and returns run-level and round-level metrics."""
    aggregator = run_config["aggregator"]
    attack = run_config["attack"]
    actual_f = run_config["actual_f"]
    assumed_f = run_config["assumed_f"]
    precision = run_config["wire_precision"]
    seed = run_config["seed"]
    num_rounds = run_config["num_rounds"]
    num_clients = run_config["num_clients"]
    batch_size = run_config["batch_size"]
    lr = run_config["lr"]
    quantize = (precision == "INT8")

    # Multi-Krum theoretical condition: 2f + 2 < N
    if aggregator == "Multi-Krum" and assumed_f > 0:
        assert 2 * assumed_f + 2 < num_clients, (
            f"Multi-Krum condition 2f + 2 < N violated: 2*{assumed_f} + 2 >= {num_clients}"
        )

    # Deterministic model initialization
    torch.manual_seed(seed)
    np.random.seed(seed)
    global_model = EdgeClassifier()
    current_weights = global_model.get_flat_parameters()

    # Pre-instantiate clients for this run
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

    # Select Byzantine attackers deterministically
    byzantine_ids = (
        ByzantineAdversary.select_byzantine_clients(num_clients, actual_f, seed=seed)
        if actual_f > 0
        else []
    )
    byzantine_set = set(byzantine_ids)
    honest_set = set(range(num_clients)) - byzantine_set

    total_uplink_bytes = 0
    history_accuracy = []
    history_loss = []
    round_records = []
    mk_attacker_rej_rates = []
    mk_honest_rej_rates = []

    start_wall_clock = time.time()

    for round_num in range(1, num_rounds + 1):
        round_start = time.time()
        client_updates = []
        round_bytes = 0

        # Edge client training
        for client in clients:
            upd = client.train_round(
                current_weights,
                quantize=quantize,
                round_num=round_num,
                seed=seed,
            )
            client_updates.append(upd)
            round_bytes += upd.payload_bytes

        total_uplink_bytes += round_bytes
        raw_vectors = [u.update for u in client_updates]

        # Adversarial corruption
        if actual_f > 0:
            attack_type_clean = "sign_flip" if attack == "sign-flip" else "gaussian"
            corrupted_vectors, _ = ByzantineAdversary.corrupt_client_updates(
                raw_vectors,
                byzantine_client_ids=byzantine_ids,
                attack_type=attack_type_clean,
                scaling_factor=2.5,
                seed=(seed * 10_000) + (round_num * 100),
            )
        else:
            corrupted_vectors = raw_vectors

        # Server-side robust aggregation
        round_att_rej: Optional[int] = None
        round_hon_rej: Optional[int] = None

        if aggregator == "FedAvg":
            aggregated_vector = RobustAggregator.fed_avg(corrupted_vectors)
        elif aggregator == "Multi-Krum":
            aggregated_vector, selected_indices, metadata = RobustAggregator.multi_krum(
                corrupted_vectors,
                num_byzantine=assumed_f,
                m_candidates=num_clients - assumed_f,
            )
            rejected_indices = set(metadata["rejected_indices"])
            if actual_f > 0:
                round_att_rej = len(rejected_indices.intersection(byzantine_set))
                mk_attacker_rej_rates.append(round_att_rej / actual_f)
            if len(honest_set) > 0:
                round_hon_rej = len(rejected_indices.intersection(honest_set))
                mk_honest_rej_rates.append(round_hon_rej / len(honest_set))
        elif aggregator == "Coordinate median":
            aggregated_vector = RobustAggregator.coordinate_median(corrupted_vectors)
        elif aggregator == "Trimmed mean":
            aggregated_vector = RobustAggregator.trimmed_mean(
                corrupted_vectors,
                num_byzantine=assumed_f,
            )
        else:
            raise ValueError(f"Unknown aggregator: {aggregator}")

        # Update global parameters
        current_weights = current_weights + aggregated_vector
        global_model.set_flat_parameters(current_weights)

        # Global test set evaluation
        acc, test_loss = evaluate_global_model_tensor(
            global_model,
            x_test_tensor,
            y_test_tensor,
            batch_size=512,
        )
        history_accuracy.append(acc)
        history_loss.append(test_loss)
        round_duration = time.time() - round_start

        round_records.append({
            "round_num": round_num,
            "test_accuracy": acc,
            "test_loss": test_loss,
            "round_duration_sec": round_duration,
            "round_uplink_bytes": round_bytes,
            "attacker_rejections": round_att_rej,
            "honest_rejections": round_hon_rej,
        })

    wall_clock_sec = time.time() - start_wall_clock

    # Calculate convergence and summary metrics
    acc_10 = history_accuracy[9] if len(history_accuracy) >= 10 else None
    acc_25 = history_accuracy[24] if len(history_accuracy) >= 25 else None
    acc_50 = history_accuracy[49] if len(history_accuracy) >= 50 else None

    rounds_to_70 = -1
    for r_idx, acc_val in enumerate(history_accuracy, start=1):
        if acc_val >= 70.0:
            rounds_to_70 = r_idx
            break

    attacker_rejection_rate = (
        float(np.mean(mk_attacker_rej_rates)) if mk_attacker_rej_rates else None
    )
    honest_false_rejection_rate = (
        float(np.mean(mk_honest_rej_rates)) if mk_honest_rej_rates else None
    )

    run_summary = {
        "final_accuracy": history_accuracy[-1],
        "best_accuracy": max(history_accuracy),
        "final_loss": history_loss[-1],
        "acc_round_10": acc_10,
        "acc_round_25": acc_25,
        "acc_round_50": acc_50,
        "rounds_to_70": rounds_to_70,
        "total_uplink_bytes": total_uplink_bytes,
        "bytes_per_client_round": float(total_uplink_bytes) / (num_clients * num_rounds),
        "wall_clock_sec": wall_clock_sec,
        "attacker_rejection_rate": attacker_rejection_rate,
        "honest_false_rejection_rate": honest_false_rejection_rate,
        "completed_at": datetime.now(),
    }

    return run_summary, round_records


def generate_grid_configurations(config_path: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates lists of all parameter combinations for main grid and wrong-f experiment."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    grid_cfg = cfg["grid"]
    grid_runs = []

    # Clean runs: attack = 'none', share = 0.0, actual_f = 0, assumed_f = 0
    # Runs once per aggregator, precision and seed
    for agg in grid_cfg["aggregators"]:
        for prec in grid_cfg["precisions"]:
            for seed in grid_cfg["seeds"]:
                grid_runs.append({
                    "experiment_type": "grid",
                    "aggregator": agg,
                    "attack": "none",
                    "attacker_share": 0.0,
                    "actual_f": 0,
                    "assumed_f": 0,
                    "wire_precision": prec,
                    "seed": seed,
                    "num_rounds": grid_cfg["num_rounds"],
                    "num_clients": grid_cfg["num_clients"],
                    "dirichlet_alpha": grid_cfg["dirichlet_alpha"],
                    "batch_size": grid_cfg["batch_size"],
                    "lr": grid_cfg["lr"],
                    "momentum": grid_cfg["momentum"],
                })

    # Attacked runs: sign-flip and Gaussian across attacker shares (10%, 20%, 30%)
    for agg in grid_cfg["aggregators"]:
        for attack in ["sign-flip", "Gaussian"]:
            for share in grid_cfg["attacker_shares"]:
                actual_f = int(round(grid_cfg["num_clients"] * share))
                assumed_f = actual_f
                for prec in grid_cfg["precisions"]:
                    for seed in grid_cfg["seeds"]:
                        grid_runs.append({
                            "experiment_type": "grid",
                            "aggregator": agg,
                            "attack": attack,
                            "attacker_share": share,
                            "actual_f": actual_f,
                            "assumed_f": assumed_f,
                            "wire_precision": prec,
                            "seed": seed,
                            "num_rounds": grid_cfg["num_rounds"],
                            "num_clients": grid_cfg["num_clients"],
                            "dirichlet_alpha": grid_cfg["dirichlet_alpha"],
                            "batch_size": grid_cfg["batch_size"],
                            "lr": grid_cfg["lr"],
                            "momentum": grid_cfg["momentum"],
                        })

    # Wrong-f experiment configurations (Spec 2.5)
    wf_cfg = cfg["wrong_f_experiment"]
    wrong_f_runs = []
    for agg in wf_cfg["aggregators"]:
        for assumed_f in wf_cfg["assumed_f_values"]:
            for seed in wf_cfg["seeds"]:
                wrong_f_runs.append({
                    "experiment_type": "wrong_f",
                    "aggregator": agg,
                    "attack": wf_cfg["attack"],
                    "attacker_share": float(wf_cfg["actual_f"]) / wf_cfg["num_clients"],
                    "actual_f": wf_cfg["actual_f"],
                    "assumed_f": assumed_f,
                    "wire_precision": wf_cfg["precision"],
                    "seed": seed,
                    "num_rounds": wf_cfg["num_rounds"],
                    "num_clients": wf_cfg["num_clients"],
                    "dirichlet_alpha": wf_cfg["dirichlet_alpha"],
                    "batch_size": wf_cfg["batch_size"],
                    "lr": wf_cfg["lr"],
                    "momentum": wf_cfg["momentum"],
                })

    return grid_runs, wrong_f_runs


def main():
    parser = argparse.ArgumentParser(description="EdgeFed-Bench v2 Grid Runner")
    parser.add_argument(
        "--config",
        type=str,
        default="experiments/config.yaml",
        help="Path to YAML configuration file (default: experiments/config.yaml)",
    )
    parser.add_argument(
        "--db_path",
        type=str,
        default="results/results.duckdb",
        help="Path to DuckDB results database (default: results/results.duckdb)",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "grid", "wrong_f"],
        help="Which experiment subset to run: 'all', 'grid', or 'wrong_f' (default: all)",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Inspect combinations and config hashes without executing simulation.",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    db_path = Path(args.db_path)

    grid_runs, wrong_f_runs = generate_grid_configurations(config_path)

    runs_to_process = []
    if args.experiment in ("all", "grid"):
        runs_to_process.extend(grid_runs)
    if args.experiment in ("all", "wrong_f"):
        runs_to_process.extend(wrong_f_runs)

    print("==================================================================")
    print("EdgeFed-Bench v2: Config-Driven Experiment Grid Runner")
    print("==================================================================")
    print(f"Total planned configurations: {len(runs_to_process)}")
    print(f"  - Main Grid: {len(grid_runs)} runs (40 clean + 240 attacked)")
    print(f"  - Wrong-f Experiment: {len(wrong_f_runs)} runs")
    print(f"Target Database: {db_path}")

    if args.dry_run:
        print("\n[Dry Run] Displaying first 5 configuration hashes:")
        for r in runs_to_process[:5]:
            chash = compute_config_hash(r)
            print(f"  Hash: {chash} | Type: {r['experiment_type']} | Agg: {r['aggregator']} | Atk: {r['attack']} | Share: {r['attacker_share']} | Prec: {r['wire_precision']} | Seed: {r['seed']}")
        print(f"\n[Dry Run] Total {len(runs_to_process)} configs validated.")
        return

    con = init_database(db_path)

    # Fetch existing completed config hashes for resumption
    existing_hashes = set(row[0] for row in con.execute("SELECT config_hash FROM runs").fetchall())
    print(f"Already completed in database: {len(existing_hashes)} runs.")

    # Pre-generate Dirichlet shards for all 5 seeds once to avoid disk churn
    unique_seeds = sorted(list(set(r["seed"] for r in runs_to_process)))
    print(f"Ensuring Dirichlet partitions exist for seeds: {unique_seeds}...")
    for s in unique_seeds:
        initialize_storage_pipeline(
            data_root="data",
            num_clients=20,
            alpha=0.5,
            seed=s,
            force_rebuild=False,
        )

    # Cache stores and test data
    stores_by_seed = {s: EdgeStore(data_root="data", seed=s) for s in unique_seeds}
    x_test_raw, y_test_raw = stores_by_seed[unique_seeds[0]].get_test_data()
    x_test_tensor = torch.from_numpy(x_test_raw).float()
    y_test_tensor = torch.from_numpy(y_test_raw).long()

    completed_count = 0
    skipped_count = 0
    total_runs = len(runs_to_process)

    grid_start_time = time.time()

    for idx, run_cfg in enumerate(runs_to_process, start=1):
        chash = compute_config_hash(run_cfg)
        if chash in existing_hashes:
            skipped_count += 1
            continue

        desc = (
            f"[{idx:03d}/{total_runs:03d}] {run_cfg['experiment_type']} | "
            f"{run_cfg['aggregator']} | {run_cfg['attack']} "
            f"({run_cfg['attacker_share']*100:.0f}%) | "
            f"act_f={run_cfg['actual_f']} ass_f={run_cfg['assumed_f']} | "
            f"{run_cfg['wire_precision']} | Seed {run_cfg['seed']}"
        )
        print(f"Executing {desc} ...", flush=True)

        store = stores_by_seed[run_cfg["seed"]]
        summary, round_records = execute_single_run(
            run_config=run_cfg,
            store=store,
            x_test_tensor=x_test_tensor,
            y_test_tensor=y_test_tensor,
        )

        # Batch insert into DuckDB
        con.execute("""
            INSERT INTO runs VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
            )
        """, [
            chash,
            run_cfg["experiment_type"],
            run_cfg["aggregator"],
            run_cfg["attack"],
            run_cfg["attacker_share"],
            run_cfg["actual_f"],
            run_cfg["assumed_f"],
            run_cfg["wire_precision"],
            run_cfg["seed"],
            run_cfg["num_rounds"],
            run_cfg["num_clients"],
            summary["final_accuracy"],
            summary["best_accuracy"],
            summary["final_loss"],
            summary["acc_round_10"],
            summary["acc_round_25"],
            summary["acc_round_50"],
            summary["rounds_to_70"],
            summary["total_uplink_bytes"],
            summary["bytes_per_client_round"],
            summary["wall_clock_sec"],
            summary["attacker_rejection_rate"],
            summary["honest_false_rejection_rate"],
            summary["completed_at"],
        ])

        round_tuples = [
            (
                chash,
                rr["round_num"],
                rr["test_accuracy"],
                rr["test_loss"],
                rr["round_duration_sec"],
                rr["round_uplink_bytes"],
                rr["attacker_rejections"],
                rr["honest_rejections"],
            )
            for rr in round_records
        ]
        con.executemany("INSERT INTO round_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)", round_tuples)
        existing_hashes.add(chash)
        completed_count += 1

        print(
            f"  -> Done in {summary['wall_clock_sec']:.2f}s | "
            f"Final Acc: {summary['final_accuracy']:.2f}% | "
            f"Best: {summary['best_accuracy']:.2f}% | "
            f"Rounds to 70%: {summary['rounds_to_70']}",
            flush=True,
        )

    for store in stores_by_seed.values():
        store.close()
    con.close()

    total_duration = time.time() - grid_start_time
    print("==================================================================")
    print("Grid Execution Finished.")
    print(f"Total processed: {completed_count} newly executed, {skipped_count} skipped (resumed).")
    print(f"Total elapsed wall-clock time: {total_duration:.2f}s")
    print("==================================================================")


if __name__ == "__main__":
    main()

