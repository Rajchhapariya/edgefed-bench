"""Runs the experiments defined in experiments/config.yaml.

Experiment types (the `experiment_type` column in the results):
- grid: 4 aggregators x 2 attacks x 3 attacker shares x 2 precisions x seeds, plus clean runs.
- wrong_f: the server is told the wrong number of attackers.
- strong_attacks: ALIE and IPM, attacks built to stay close to honest updates.
- alpha_sweep: other Dirichlet alpha values.
- bit_width: INT4 and 1-bit sign updates.
- cnn: a small CNN instead of the MLP.
- long_run: 150 rounds instead of 50.
- tuning: learning rate, momentum and batch size scored on a held-out validation split.

Results go to DuckDB. `runs` holds one summary row per run, `round_metrics` one row per
round, and `round_mechanism` the per-round distances used to explain the results. Each
run is keyed by a hash of its config, so a stopped job resumes without redoing finished
runs. `--shard k/K` runs every K-th pending run, so several processes can work in
parallel; each must write to its own database file (see experiments/merge_results.py).
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

# Lets the script import engine/ when run as experiments/run_grid.py.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.adversary import ByzantineAdversary
from engine.aggregator import RobustAggregator
from engine.client import EdgeClient, build_model
from engine.storage import EdgeStore, initialize_storage_pipeline


# Settings of the committed grid. A key enters the hash only when a run uses a
# different value, so every stored hash stays valid.
LEGACY_DEFAULTS = {
    "lr": 0.05,
    "batch_size": 128,
    "momentum": 0.9,
    "model": "mlp",
    "validation_size": 0,
    "eval_split": "test",
}
# Kept for code that imported the old name.
LEGACY_TRAINING_DEFAULTS = {k: LEGACY_DEFAULTS[k] for k in ("lr", "batch_size", "momentum")}

ATTACK_NAMES = {"sign-flip": "sign_flip", "Gaussian": "gaussian", "ALIE": "alie", "IPM": "ipm"}

EXPERIMENT_TYPES = (
    "grid", "wrong_f", "strong_attacks", "alpha_sweep", "bit_width", "cnn", "long_run", "tuning",
)

RUN_COLUMNS = [
    "config_hash", "experiment_type", "aggregator", "attack", "attacker_share", "actual_f",
    "assumed_f", "wire_precision", "seed", "num_rounds", "num_clients", "final_accuracy",
    "best_accuracy", "final_loss", "acc_round_10", "acc_round_25", "acc_round_50",
    "rounds_to_70", "total_uplink_bytes", "bytes_per_client_round", "wall_clock_sec",
    "attacker_rejection_rate", "honest_false_rejection_rate", "completed_at",
    "dirichlet_alpha", "model", "lr", "momentum", "batch_size", "validation_size",
    "eval_split", "torch_threads",
]


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
    for key, legacy_value in LEGACY_DEFAULTS.items():
        value = run_config.get(key, legacy_value)
        if value != legacy_value:
            canonical_dict[key] = value
    raw_str = json.dumps(canonical_dict, sort_keys=True)
    return hashlib.sha256(raw_str.encode("utf-8")).hexdigest()[:16]


def init_database(db_path: Path) -> duckdb.DuckDBPyConnection:
    """Creates the result tables, and adds columns that older databases lack."""
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
    # Columns added after the first grid. Rows written before them all used these values.
    for column, sql_type, default in (
        ("dirichlet_alpha", "DOUBLE", "0.5"),
        ("model", "VARCHAR", "'mlp'"),
        ("lr", "DOUBLE", "0.05"),
        ("momentum", "DOUBLE", "0.9"),
        ("batch_size", "INTEGER", "128"),
        ("validation_size", "INTEGER", "0"),
        ("eval_split", "VARCHAR", "'test'"),
        ("torch_threads", "INTEGER", "10"),
    ):
        con.execute(f"ALTER TABLE runs ADD COLUMN IF NOT EXISTS {column} {sql_type} DEFAULT {default}")

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

    con.execute("""
    CREATE TABLE IF NOT EXISTS round_mechanism (
        config_hash VARCHAR,
        round_num INTEGER,
        honest_pair_dist DOUBLE,
        honest_attacker_dist DOUBLE,
        honest_codec_error DOUBLE,
        honest_update_norm DOUBLE,
        attacker_update_norm DOUBLE,
        PRIMARY KEY (config_hash, round_num)
    );
    """)

    return con


def evaluate_global_model_tensor(
    model: torch.nn.Module,
    x_test_tensor: torch.Tensor,
    y_test_tensor: torch.Tensor,
    batch_size: int = 512,
) -> Tuple[float, float]:
    """Returns (accuracy %, mean loss) on evaluation tensors built once per process."""
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


def mechanism_metrics(
    vectors: List[np.ndarray],
    byzantine_ids: List[int],
    codec_errors: List[Optional[float]],
) -> Dict[str, Optional[float]]:
    """Distances that show whether compression error could hide attackers.

    honest_pair_dist: mean L2 distance between two honest updates.
    honest_attacker_dist: mean L2 distance between an honest and an attacker update.
    honest_codec_error: mean L2 norm of the compression error of honest updates
        (0 for Float32).
    """
    v = np.stack([np.asarray(x, dtype=np.float64) for x in vectors])
    sq = np.einsum("ij,ij->i", v, v)
    dist = np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * (v @ v.T), 0.0))
    norms = np.sqrt(sq)
    byz = set(byzantine_ids)
    honest = [i for i in range(len(vectors)) if i not in byz]
    attackers = sorted(byz)

    pairs = [dist[i, j] for a, i in enumerate(honest) for j in honest[a + 1:]]
    cross = [dist[i, j] for i in honest for j in attackers]
    errs = [codec_errors[i] or 0.0 for i in honest]
    return {
        "honest_pair_dist": float(np.mean(pairs)) if pairs else None,
        "honest_attacker_dist": float(np.mean(cross)) if cross else None,
        "honest_codec_error": float(np.mean(errs)) if errs else None,
        "honest_update_norm": float(np.mean(norms[honest])) if honest else None,
        "attacker_update_norm": float(np.mean(norms[attackers])) if attackers else None,
    }


def execute_single_run(
    run_config: Dict[str, Any],
    store: EdgeStore,
    x_test_tensor: torch.Tensor,
    y_test_tensor: torch.Tensor,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Executes a single federated learning run and returns run-level and round-level metrics.

    Each round record also carries a "mechanism" entry (None for runs without attackers).
    """
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
    model_name = run_config.get("model", "mlp")

    if aggregator == "Multi-Krum" and assumed_f > 0:
        assert 2 * assumed_f + 2 < num_clients, (
            f"Multi-Krum condition 2f + 2 < N violated: 2*{assumed_f} + 2 >= {num_clients}"
        )

    torch.manual_seed(seed)
    np.random.seed(seed)
    global_model = build_model(model_name)
    current_weights = global_model.get_flat_parameters()

    clients = [
        EdgeClient(
            client_id=i,
            store=store,
            lr=lr,
            batch_size=batch_size,
            local_epochs=1,
            momentum=run_config.get("momentum", 0.9),
            model_name=model_name,
        )
        for i in range(num_clients)
    ]

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

        for client in clients:
            upd = client.train_round(
                current_weights,
                round_num=round_num,
                seed=seed,
                precision=precision,
            )
            client_updates.append(upd)
            round_bytes += upd.payload_bytes

        total_uplink_bytes += round_bytes
        raw_vectors = [u.update for u in client_updates]

        if actual_f > 0:
            corrupted_vectors, _ = ByzantineAdversary.corrupt_client_updates(
                raw_vectors,
                byzantine_client_ids=byzantine_ids,
                attack_type=ATTACK_NAMES[attack],
                scaling_factor=2.5,
                seed=(seed * 10_000) + (round_num * 100),
                wire_precision=precision,
            )
            mechanism = mechanism_metrics(
                corrupted_vectors,
                byzantine_ids,
                [u.noise_metrics["l2_error"] if u.noise_metrics else None for u in client_updates],
            )
        else:
            corrupted_vectors = raw_vectors
            mechanism = None

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

        current_weights = current_weights + aggregated_vector
        global_model.set_flat_parameters(current_weights)

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
            "mechanism": mechanism,
        })

    wall_clock_sec = time.time() - start_wall_clock

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


def _base_run(cfg: Dict[str, Any], **overrides: Any) -> Dict[str, Any]:
    run = {
        "num_rounds": cfg["num_rounds"],
        "num_clients": cfg["num_clients"],
        "dirichlet_alpha": cfg["dirichlet_alpha"],
        "batch_size": cfg["batch_size"],
        "lr": cfg["lr"],
        "momentum": cfg["momentum"],
    }
    run.update(overrides)
    return run


def _scenario_runs(
    experiment_type: str,
    cfg: Dict[str, Any],
    aggregators: List[str],
    scenarios: List[List[Any]],
    precisions: List[str],
    seeds: List[int],
    **overrides: Any,
) -> List[Dict[str, Any]]:
    """One run per aggregator x (attack, share) x precision x seed."""
    runs = []
    for agg in aggregators:
        for attack, share in scenarios:
            f = int(round(cfg["num_clients"] * share))
            for prec in precisions:
                for seed in seeds:
                    runs.append(_base_run(
                        cfg,
                        experiment_type=experiment_type,
                        aggregator=agg,
                        attack=attack,
                        attacker_share=float(share),
                        actual_f=f,
                        assumed_f=f,
                        wire_precision=prec,
                        seed=seed,
                        **overrides,
                    ))
    return runs


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
                grid_runs.append(_base_run(
                    grid_cfg,
                    experiment_type="grid",
                    aggregator=agg,
                    attack="none",
                    attacker_share=0.0,
                    actual_f=0,
                    assumed_f=0,
                    wire_precision=prec,
                    seed=seed,
                ))

    # Attacked runs: sign-flip and Gaussian across attacker shares (10%, 20%, 30%)
    for agg in grid_cfg["aggregators"]:
        for attack in ["sign-flip", "Gaussian"]:
            for share in grid_cfg["attacker_shares"]:
                actual_f = int(round(grid_cfg["num_clients"] * share))
                for prec in grid_cfg["precisions"]:
                    for seed in grid_cfg["seeds"]:
                        grid_runs.append(_base_run(
                            grid_cfg,
                            experiment_type="grid",
                            aggregator=agg,
                            attack=attack,
                            attacker_share=share,
                            actual_f=actual_f,
                            assumed_f=actual_f,
                            wire_precision=prec,
                            seed=seed,
                        ))

    # Wrong-f experiment configurations (Spec 2.5)
    wf_cfg = cfg["wrong_f_experiment"]
    wrong_f_runs = []
    for agg in wf_cfg["aggregators"]:
        for assumed_f in wf_cfg["assumed_f_values"]:
            for seed in wf_cfg["seeds"]:
                wrong_f_runs.append(_base_run(
                    wf_cfg,
                    experiment_type="wrong_f",
                    aggregator=agg,
                    attack=wf_cfg["attack"],
                    attacker_share=float(wf_cfg["actual_f"]) / wf_cfg["num_clients"],
                    actual_f=wf_cfg["actual_f"],
                    assumed_f=assumed_f,
                    wire_precision=wf_cfg["precision"],
                    seed=seed,
                ))

    return grid_runs, wrong_f_runs


def generate_extension_configurations(config_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """Run lists for the experiments added after the main grid, keyed by experiment type."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    ext = cfg.get("extensions")
    if not ext:
        return {}
    base = ext["common"]
    seeds = base["seeds"]
    out: Dict[str, List[Dict[str, Any]]] = {}

    sa = ext["strong_attacks"]
    out["strong_attacks"] = _scenario_runs(
        "strong_attacks", base, sa["aggregators"],
        [[a, s] for a in sa["attacks"] for s in sa["attacker_shares"]],
        sa["precisions"], seeds,
    )

    al = ext["alpha_sweep"]
    out["alpha_sweep"] = []
    for alpha in al["alphas"]:
        out["alpha_sweep"] += _scenario_runs(
            "alpha_sweep", dict(base, dirichlet_alpha=alpha), al["aggregators"],
            al["scenarios"], al["precisions"], seeds,
        )

    bw = ext["bit_width"]
    out["bit_width"] = _scenario_runs(
        "bit_width", base, bw["aggregators"], bw["scenarios"], bw["precisions"], seeds,
    )

    cn = ext["cnn"]
    out["cnn"] = _scenario_runs(
        "cnn", dict(base, num_rounds=cn.get("num_rounds", base["num_rounds"])), cn["aggregators"],
        cn["scenarios"], cn["precisions"], cn.get("seeds", seeds), model="cnn",
    )

    lr_cfg = ext["long_run"]
    out["long_run"] = _scenario_runs(
        "long_run", dict(base, num_rounds=lr_cfg["num_rounds"]), lr_cfg["aggregators"],
        lr_cfg["scenarios"], lr_cfg["precisions"], seeds,
    )

    tu = ext["tuning"]
    out["tuning"] = []
    for lr in tu["lrs"]:
        for momentum in tu["momenta"]:
            for bs in tu["batch_sizes"]:
                out["tuning"] += _scenario_runs(
                    "tuning", dict(base, lr=lr, momentum=momentum, batch_size=bs),
                    [tu["aggregator"]], [["none", 0.0]], [tu["precision"]], tu["seeds"],
                    validation_size=tu["validation_size"], eval_split="val",
                )
    return out


def all_configurations(config_path: Path) -> Dict[str, List[Dict[str, Any]]]:
    grid, wrong_f = generate_grid_configurations(config_path)
    out = {"grid": grid, "wrong_f": wrong_f}
    out.update(generate_extension_configurations(config_path))
    return out


def partition_key(run: Dict[str, Any]) -> Tuple[int, float, int, int]:
    return (run["seed"], float(run["dirichlet_alpha"]), run["num_clients"], run.get("validation_size", 0))


def insert_run(con: duckdb.DuckDBPyConnection, chash: str, run_cfg: Dict[str, Any],
               summary: Dict[str, Any], round_records: List[Dict[str, Any]]) -> None:
    """Writes one finished run to runs, round_metrics and round_mechanism."""
    values = {
        "config_hash": chash,
        **{k: run_cfg[k] for k in ("experiment_type", "aggregator", "attack", "attacker_share",
                                    "actual_f", "assumed_f", "wire_precision", "seed",
                                    "num_rounds", "num_clients")},
        **{k: summary[k] for k in ("final_accuracy", "best_accuracy", "final_loss", "acc_round_10",
                                    "acc_round_25", "acc_round_50", "rounds_to_70",
                                    "total_uplink_bytes", "bytes_per_client_round", "wall_clock_sec",
                                    "attacker_rejection_rate", "honest_false_rejection_rate",
                                    "completed_at")},
        "dirichlet_alpha": run_cfg["dirichlet_alpha"],
        "model": run_cfg.get("model", "mlp"),
        "lr": run_cfg["lr"],
        "momentum": run_cfg["momentum"],
        "batch_size": run_cfg["batch_size"],
        "validation_size": run_cfg.get("validation_size", 0),
        "eval_split": run_cfg.get("eval_split", "test"),
        "torch_threads": torch.get_num_threads(),
    }
    con.execute(
        f"INSERT INTO runs ({', '.join(RUN_COLUMNS)}) VALUES ({', '.join(['?'] * len(RUN_COLUMNS))})",
        [values[c] for c in RUN_COLUMNS],
    )
    con.executemany("INSERT INTO round_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)", [
        (chash, rr["round_num"], rr["test_accuracy"], rr["test_loss"], rr["round_duration_sec"],
         rr["round_uplink_bytes"], rr["attacker_rejections"], rr["honest_rejections"])
        for rr in round_records
    ])
    mech = [
        (chash, rr["round_num"], m["honest_pair_dist"], m["honest_attacker_dist"],
         m["honest_codec_error"], m["honest_update_norm"], m["attacker_update_norm"])
        for rr in round_records if (m := rr.get("mechanism"))
    ]
    if mech:
        con.executemany("INSERT INTO round_mechanism VALUES (?, ?, ?, ?, ?, ?, ?)", mech)


def main():
    parser = argparse.ArgumentParser(description="EdgeFed-Bench experiment runner")
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
        help="Comma-separated experiment types, or 'all' (main grid and wrong-f only) or "
        "'everything'. Types: " + ", ".join(EXPERIMENT_TYPES),
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Inspect combinations and config hashes without executing simulation.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="PyTorch CPU threads. Results repeat bit for bit only at the same thread count. "
        "The thread count of each run is stored in the torch_threads column.",
    )
    parser.add_argument(
        "--shard",
        type=str,
        default="0/1",
        help="k/K: run only every K-th pending run, starting at k. Use a separate --db_path per shard.",
    )
    parser.add_argument(
        "--skip_db",
        type=str,
        default=None,
        help="Optional second database whose finished runs are skipped (for example the main results file).",
    )
    args = parser.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    print(f"PyTorch threads: {torch.get_num_threads()}")

    config_path = Path(args.config)
    db_path = Path(args.db_path)
    shard_k, shard_n = (int(x) for x in args.shard.split("/"))

    configs = all_configurations(config_path)
    if args.experiment == "all":
        selected = ["grid", "wrong_f"]
    elif args.experiment == "everything":
        selected = list(configs)
    else:
        selected = [s.strip() for s in args.experiment.split(",")]
    unknown = [s for s in selected if s not in configs]
    if unknown:
        raise ValueError(f"Unknown experiment types: {unknown}. Known: {sorted(configs)}")

    runs_to_process = [r for s in selected for r in configs[s]]

    print("==================================================================")
    print("EdgeFed-Bench experiment runner")
    print("==================================================================")
    for s in selected:
        print(f"  - {s}: {len(configs[s])} runs")
    print(f"Total planned configurations: {len(runs_to_process)}")
    print(f"Target Database: {db_path}")

    if args.dry_run:
        print("\n[Dry Run] Displaying first 5 configuration hashes:")
        for r in runs_to_process[:5]:
            chash = compute_config_hash(r)
            print(f"  Hash: {chash} | Type: {r['experiment_type']} | Agg: {r['aggregator']} | Atk: {r['attack']} | Share: {r['attacker_share']} | Prec: {r['wire_precision']} | Seed: {r['seed']}")
        print(f"\n[Dry Run] Total {len(runs_to_process)} configs validated.")
        return

    con = init_database(db_path)
    existing_hashes = set(row[0] for row in con.execute("SELECT config_hash FROM runs").fetchall())
    if args.skip_db:
        with duckdb.connect(args.skip_db, read_only=True) as other:
            existing_hashes |= {row[0] for row in other.execute("SELECT config_hash FROM runs").fetchall()}
    print(f"Already completed: {len(existing_hashes)} runs.")

    pending = [r for r in runs_to_process if compute_config_hash(r) not in existing_hashes]
    pending = [r for i, r in enumerate(pending) if i % shard_n == shard_k]
    print(f"This process runs {len(pending)} of them (shard {shard_k}/{shard_n}).")

    stores: Dict[Tuple[int, float, int, int], EdgeStore] = {}
    eval_tensors: Dict[Any, Tuple[torch.Tensor, torch.Tensor]] = {}

    def store_for(run: Dict[str, Any]) -> EdgeStore:
        key = partition_key(run)
        if key not in stores:
            seed, alpha, n_clients, val_size = key
            initialize_storage_pipeline(
                data_root="data", num_clients=n_clients, alpha=alpha, seed=seed,
                validation_size=val_size,
            )
            stores[key] = EdgeStore(data_root="data", seed=seed, alpha=alpha,
                                    num_clients=n_clients, validation_size=val_size)
        return stores[key]

    def eval_for(run: Dict[str, Any], store: EdgeStore) -> Tuple[torch.Tensor, torch.Tensor]:
        key = ("val", partition_key(run)) if run.get("eval_split", "test") == "val" else "test"
        if key not in eval_tensors:
            x, y = store.get_validation_data() if key != "test" else store.get_test_data()
            eval_tensors[key] = (torch.from_numpy(x).float(), torch.from_numpy(y).long())
        return eval_tensors[key]

    completed_count = 0
    grid_start_time = time.time()

    for idx, run_cfg in enumerate(pending, start=1):
        chash = compute_config_hash(run_cfg)
        desc = (
            f"[{idx:03d}/{len(pending):03d}] {run_cfg['experiment_type']} | "
            f"{run_cfg['aggregator']} | {run_cfg['attack']} "
            f"({run_cfg['attacker_share']*100:.0f}%) | "
            f"act_f={run_cfg['actual_f']} ass_f={run_cfg['assumed_f']} | "
            f"{run_cfg['wire_precision']} | {run_cfg.get('model', 'mlp')} | "
            f"alpha={run_cfg['dirichlet_alpha']} | R={run_cfg['num_rounds']} | Seed {run_cfg['seed']}"
        )
        print(f"Executing {desc} ...", flush=True)

        store = store_for(run_cfg)
        x_eval, y_eval = eval_for(run_cfg, store)
        summary, round_records = execute_single_run(
            run_config=run_cfg,
            store=store,
            x_test_tensor=x_eval,
            y_test_tensor=y_eval,
        )
        insert_run(con, chash, run_cfg, summary, round_records)
        completed_count += 1

        print(
            f"  -> Done in {summary['wall_clock_sec']:.2f}s | "
            f"Final Acc: {summary['final_accuracy']:.2f}% | "
            f"Best: {summary['best_accuracy']:.2f}% | "
            f"Rounds to 70%: {summary['rounds_to_70']}",
            flush=True,
        )

    for store in stores.values():
        store.close()
    con.close()

    total_duration = time.time() - grid_start_time
    print("==================================================================")
    print("Run Finished.")
    print(f"Newly executed: {completed_count}")
    print(f"Total elapsed wall-clock time: {total_duration:.2f}s")
    print("SHARD DONE", flush=True)
    print("==================================================================")


if __name__ == "__main__":
    main()
