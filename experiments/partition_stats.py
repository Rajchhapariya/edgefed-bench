"""Records how skewed each client partition is, in the results database.

For every partition used by the experiments, writes one row per client to the
`partition_stats` table: sample count, number of classes with at least 1% of the
client's samples, and the share of the client's largest class. The report reads label
skew from this table, so no skew figure is typed by hand.

    python experiments/partition_stats.py [results/results.duckdb]
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import duckdb

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.storage import client_shard_dir, dataset_root, initialize_storage_pipeline
from experiments.run_grid import all_configurations, partition_key

MIN_CLASS_SHARE = 0.01


def main(db_path: str = "results/results.duckdb", config: str = "experiments/config.yaml") -> None:
    keys = sorted({partition_key(r) for runs in all_configurations(Path(config)).values() for r in runs})
    con = duckdb.connect(db_path)
    con.execute("""
        CREATE OR REPLACE TABLE partition_stats (
            seed INTEGER, dirichlet_alpha DOUBLE, num_clients INTEGER, validation_size INTEGER,
            client_id INTEGER, num_samples INTEGER, classes_present INTEGER, max_class_share DOUBLE,
            dataset VARCHAR
        )
    """)
    for seed, alpha, n_clients, val, dataset in keys:
        initialize_storage_pipeline(data_root="data", num_clients=n_clients, alpha=alpha, seed=seed,
                                    validation_size=val, dataset=dataset)
        shard_dir = client_shard_dir(dataset_root("data", dataset), seed, alpha, n_clients, val)
        for cid in range(n_clients):
            path = (shard_dir / f"client_{cid:02d}.parquet").as_posix()
            counts = [c for (c,) in con.execute(
                f"SELECT count(*) FROM '{path}' GROUP BY label").fetchall()]
            total = sum(counts)
            con.execute("INSERT INTO partition_stats VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", [
                seed, alpha, n_clients, val, cid, total,
                sum(1 for c in counts if c / total >= MIN_CLASS_SHARE), max(counts) / total, dataset,
            ])
    print(con.execute("""
        SELECT dataset, dirichlet_alpha, validation_size, count(DISTINCT seed) AS seeds,
               round(avg(classes_present), 2), round(avg(max_class_share), 3), min(num_samples)
        FROM partition_stats GROUP BY 1, 2, 3 ORDER BY 1, 2, 3
    """).fetchall())
    con.close()


if __name__ == "__main__":
    main(*(sys.argv[1:2] or []))
