"""Merges shard databases written by `run_grid.py --shard k/K` into the main results file.

Runs already in the target are left alone, so merging twice is safe.

    python experiments/merge_results.py results/results.duckdb shard_0.duckdb shard_1.duckdb ...
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from experiments.run_grid import RUN_COLUMNS, init_database


def merge(target: Path, sources: list) -> int:
    con = init_database(target)
    added = 0
    for i, src in enumerate(sources):
        # init_database also upgrades older shard files to the current columns.
        init_database(Path(src)).close()
        alias = f"s{i}"
        con.execute(f"ATTACH '{Path(src).as_posix()}' AS {alias} (READ_ONLY)")
        new = f"(SELECT config_hash FROM {alias}.runs WHERE config_hash NOT IN (SELECT config_hash FROM runs))"
        cols = ", ".join(RUN_COLUMNS)
        n = con.execute(f"SELECT count(*) FROM {new}").fetchone()[0]
        con.execute("BEGIN")
        con.execute(f"INSERT INTO round_metrics SELECT * FROM {alias}.round_metrics WHERE config_hash IN {new}")
        con.execute(f"INSERT INTO round_mechanism SELECT * FROM {alias}.round_mechanism WHERE config_hash IN {new}")
        con.execute(f"INSERT INTO runs ({cols}) SELECT {cols} FROM {alias}.runs WHERE config_hash IN {new}")
        con.execute("COMMIT")
        con.execute(f"DETACH {alias}")
        print(f"{src}: added {n} runs")
        added += n
    con.execute("CHECKPOINT")
    con.close()
    return added


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    print(f"Added {merge(Path(sys.argv[1]), sys.argv[2:])} runs in total.")
