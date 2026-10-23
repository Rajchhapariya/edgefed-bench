"""Manual smoke verification script for Phase 1: Data Engineering and Storage Layer.
"""

import os
import sys
from pathlib import Path

# Ensure current directory is on PYTHONPATH
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from engine.storage import DataPartitioner, EdgeStore, TelemetryEngine, initialize_storage_pipeline


def main():
    print("=================================================================")
    print("STEP 1: Initializing Dirichlet Partitioning & Parquet Sharding...")
    print("=================================================================")
    data_dir = "data"
    parquet_paths, summary = initialize_storage_pipeline(
        data_root=data_dir,
        num_clients=20,
        alpha=0.5,
        seed=42,
    )
    print(f"Summary: {summary}")
    print(f"Successfully verified {len(parquet_paths)} client Parquet files.")

    print("\n=================================================================")
    print("STEP 2: Verifying EdgeStore DuckDB SQL Queries across Edge Shards")
    print("=================================================================")
    store = EdgeStore(data_root=data_dir)

    for cid in [0, 1, 2]:
        dist = store.get_client_class_distribution(cid)
        features, labels = store.get_client_data(cid)
        print(f"Client {cid:02d}: Total samples = {len(labels)}, Feature shape = {features.shape}")
        print(f"Client {cid:02d} class label distribution: {dist}")

    # Verify global test set
    test_x, test_y = store.get_test_data()
    print(f"\nGlobal Test Set: Total samples = {len(test_y)}, Feature shape = {test_x.shape}")
    store.close()

    print("\n=================================================================")
    print("STEP 3: Testing DuckDB Analytical Telemetry Ingestion & Aggregation")
    print("=================================================================")
    telemetry = TelemetryEngine(db_path=os.path.join(data_dir, "telemetry.duckdb"))
    
    exp_id = "exp_verify_manual"
    telemetry.log_round_telemetry(
        experiment_id=exp_id,
        regime_name="Attacked_FedAvg",
        round_num=1,
        client_id=0,
        is_byzantine=False,
        gradient_norm=0.452,
        payload_bytes=313600,
        quantized=False,
        local_loss=1.82,
    )
    telemetry.log_round_telemetry(
        experiment_id=exp_id,
        regime_name="Attacked_FedAvg",
        round_num=1,
        client_id=1,
        is_byzantine=True,
        gradient_norm=4.891,
        payload_bytes=313600,
        quantized=False,
        local_loss=3.14,
    )
    telemetry.log_global_eval(
        experiment_id=exp_id,
        regime_name="Attacked_FedAvg",
        round_num=1,
        global_accuracy=42.15,
        global_loss=1.95,
        aggregation_method="FedAvg",
        byzantine_fraction=0.2,
        quantized=False,
        round_duration_sec=1.12,
    )

    summary_rows = telemetry.get_regime_summary(exp_id)
    print("Telemetry Summary Query Output from DuckDB:")
    print(summary_rows)
    telemetry.close()

    print("\n[SUCCESS] Storage layer manual smoke verification complete.")


if __name__ == "__main__":
    main()
