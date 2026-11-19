"""Data partitioning, Parquet shards and DuckDB access.

Fashion-MNIST is split across clients with a Dirichlet distribution and written to one
Parquet file per client. EdgeStore reads shards through DuckDB. TelemetryEngine logs
per-round telemetry from simulate.py.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import duckdb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from torchvision import datasets, transforms

# File written next to the client shards that records how they were partitioned.
PARTITION_META_FILENAME = "_partition_meta.json"
VALIDATION_FILENAME = "val.parquet"
DEFAULT_SEED = 42

# Offset for the validation-split generator, so the split does not reuse the
# partition's random stream.
_VALIDATION_SEED_OFFSET = 7919


def client_shard_dir(
    data_root: Union[str, Path],
    seed: int = DEFAULT_SEED,
    alpha: float = 0.5,
    num_clients: int = 20,
    validation_size: int = 0,
) -> Path:
    """Directory that holds the shards for one partition.

    The main setting (alpha 0.5, 20 clients, no validation split) keeps the short name
    clients_seed_{seed} used by the stored results. Any other setting gets its own
    directory, so partitions with different parameters never overwrite each other.
    """
    name = f"clients_seed_{seed}"
    if float(alpha) != 0.5 or int(num_clients) != 20:
        name += f"_a{float(alpha):g}_n{int(num_clients)}"
    if int(validation_size) > 0:
        name += f"_val{int(validation_size)}"
    return Path(data_root) / name


class DataPartitioner:
    """Loads Fashion-MNIST, splits it across clients and writes the Parquet shards."""

    def __init__(self, data_root: str = "data"):
        self.data_root = Path(data_root)
        self.cache_dir = self.data_root / "cache"
        self.clients_dir = client_shard_dir(self.data_root)
        self.test_parquet_path = self.data_root / "test.parquet"

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_fashion_mnist(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Loads Fashion-MNIST dataset from cache or downloads it.
        
        Returns:
            Tuple containing:
            - X_train: float32 array of shape (60000, 784) in range [0, 1]
            - y_train: int64 array of shape (60000,)
            - X_test: float32 array of shape (10000, 784) in range [0, 1]
            - y_test: int64 array of shape (10000,)
        """
        transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        train_ds = datasets.FashionMNIST(
            root=str(self.cache_dir),
            train=True,
            download=True,
            transform=transform,
        )
        test_ds = datasets.FashionMNIST(
            root=str(self.cache_dir),
            train=False,
            download=True,
            transform=transform,
        )

        x_train = train_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_train = train_ds.targets.numpy().astype(np.int64)

        x_test = test_ds.data.numpy().reshape(-1, 784).astype(np.float32) / 255.0
        y_test = test_ds.targets.numpy().astype(np.int64)

        return x_train, y_train, x_test, y_test

    def partition_dirichlet(
        self,
        labels: np.ndarray,
        num_clients: int = 20,
        alpha: float = 0.5,
        seed: int = 42,
    ) -> Dict[int, np.ndarray]:
        """Partitions sample indices across clients using a Dirichlet distribution over classes.

        Mathematical Formulation:
            For each class c in {0, ..., K-1}:
                p_c ~ Dir(alpha * 1_N)
            where 1_N is an all-ones vector of length num_clients.
            The indices corresponding to class c are allocated across clients
            proportional to the probability vector p_c.

        Args:
            labels: Array of integer class labels.
            num_clients: Number of simulated edge clients (default: 20).
            alpha: Concentration parameter. Lower values give more label skew; as alpha grows
                   the split approaches IID.
            seed: Random seed; the same seed gives the same split.

        Returns:
            Dictionary mapping client_id (0 to num_clients - 1) to an array of assigned sample indices.
        """
        rng = np.random.default_rng(seed)
        num_classes = len(np.unique(labels))
        client_indices: Dict[int, List[int]] = {i: [] for i in range(num_clients)}

        for c in range(num_classes):
            idx_c = np.where(labels == c)[0]
            rng.shuffle(idx_c)

            proportions = rng.dirichlet(np.repeat(alpha, num_clients))

            splits = (np.cumsum(proportions) * len(idx_c)).astype(int)[:-1]
            client_splits = np.split(idx_c, splits)

            for client_id, split in enumerate(client_splits):
                client_indices[client_id].extend(split.tolist())

        empty = [cid for cid, idx in client_indices.items() if len(idx) == 0]
        if empty:
            raise ValueError(
                f"Dirichlet split left clients {empty} with no samples (alpha={alpha}, seed={seed}). "
                f"Use another seed or a larger alpha."
            )

        final_partitions = {}
        for client_id, indices in client_indices.items():
            arr_indices = np.array(indices, dtype=np.int64)
            rng.shuffle(arr_indices)
            final_partitions[client_id] = arr_indices

        return final_partitions

    def export_to_parquet(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        client_partitions: Dict[int, np.ndarray],
        clients_dir: Optional[Union[str, Path]] = None,
    ) -> List[str]:
        """Serializes client partitions and the global test set into Apache Parquet files.

        Args:
            x_train: Flattened feature matrix for training.
            y_train: Label vector for training.
            x_test: Flattened feature matrix for global evaluation.
            y_test: Label vector for global evaluation.
            client_partitions: Mapping of client_id to training index arrays.
            clients_dir: Optional directory for client Parquet shards (defaults to self.clients_dir).

        Returns:
            List of absolute paths to generated client Parquet files.
        """
        target_dir = Path(clients_dir) if clients_dir is not None else self.clients_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        generated_paths = []

        for client_id, indices in client_partitions.items():
            client_features = x_train[indices]
            client_labels = y_train[indices]
            num_samples = len(indices)

            table = pa.Table.from_arrays(
                [
                    pa.array([client_id] * num_samples, type=pa.int32()),
                    pa.array(indices, type=pa.int64()),
                    pa.array(client_labels, type=pa.int64()),
                    pa.array(client_features.tolist(), type=pa.list_(pa.float32())),
                ],
                names=["client_id", "sample_id", "label", "features"],
            )

            out_path = target_dir / f"client_{client_id:02d}.parquet"
            pq.write_table(table, str(out_path), compression="snappy")
            generated_paths.append(str(out_path.resolve()))

        test_table = pa.Table.from_arrays(
            [
                pa.array(range(len(y_test)), type=pa.int64()),
                pa.array(y_test, type=pa.int64()),
                pa.array(x_test.tolist(), type=pa.list_(pa.float32())),
            ],
            names=["sample_id", "label", "features"],
        )
        pq.write_table(test_table, str(self.test_parquet_path), compression="snappy")

        return generated_paths


class EdgeStore:
    """Reads client shards and the test set through DuckDB."""

    def __init__(
        self,
        data_root: str = "data",
        seed: Optional[int] = None,
        clients_dir: Optional[Union[str, Path]] = None,
        alpha: float = 0.5,
        num_clients: int = 20,
        validation_size: int = 0,
    ):
        self.data_root = Path(data_root)
        if clients_dir is not None:
            self.clients_dir = Path(clients_dir)
        else:
            self.clients_dir = client_shard_dir(
                self.data_root,
                seed=DEFAULT_SEED if seed is None else seed,
                alpha=alpha,
                num_clients=num_clients,
                validation_size=validation_size,
            )
        self.test_parquet_path = self.data_root / "test.parquet"
        self.con = duckdb.connect()

    def close(self) -> None:
        """Closes the in-memory DuckDB connection."""
        self.con.close()

    def __enter__(self) -> "EdgeStore":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def get_client_data(self, client_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Fetches all feature vectors and labels for a specific client using DuckDB.

        Args:
            client_id: Integer identifier of the client (0-19).

        Returns:
            Tuple of (features, labels) as NumPy arrays.
        """
        file_path = (self.clients_dir / f"client_{client_id:02d}.parquet").as_posix()
        query = f"SELECT label, features FROM '{file_path}'"
        result = self.con.execute(query).fetchnumpy()

        labels = result["label"].astype(np.int64)
        features = np.stack(result["features"]).astype(np.float32)

        return features, labels

    def get_client_class_distribution(self, client_id: int) -> Dict[int, int]:
        """Counts samples per class in one client's shard.

        Args:
            client_id: Integer identifier of the client.

        Returns:
            Dictionary mapping class_label to sample count.
        """
        file_path = (self.clients_dir / f"client_{client_id:02d}.parquet").as_posix()
        query = f"""
            SELECT label, COUNT(*) AS count
            FROM '{file_path}'
            GROUP BY label
            ORDER BY label
        """
        result = self.con.execute(query).fetchall()
        return {row[0]: row[1] for row in result}

    def get_validation_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns the held-out validation split stored with these shards."""
        path = self.clients_dir / VALIDATION_FILENAME
        if not path.exists():
            raise FileNotFoundError(
                f"No validation split at {path}. Build the shards with validation_size > 0."
            )
        result = self.con.execute(f"SELECT label, features FROM '{path.as_posix()}'").fetchnumpy()
        return np.stack(result["features"]).astype(np.float32), result["label"].astype(np.int64)

    def get_test_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Fetches global test dataset via DuckDB."""
        file_path = self.test_parquet_path.as_posix()
        query = f"SELECT label, features FROM '{file_path}'"
        result = self.con.execute(query).fetchnumpy()

        labels = result["label"].astype(np.int64)
        features = np.stack(result["features"]).astype(np.float32)

        return features, labels


class TelemetryEngine:
    """Writes simulate.py telemetry to DuckDB."""

    def __init__(self, db_path: str = "data/telemetry.duckdb"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.con = duckdb.connect(str(self.db_path))
        self._init_schema()

    def _init_schema(self) -> None:
        """Initializes tables for edge telemetry and global evaluations."""
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS round_telemetry (
                experiment_id VARCHAR,
                regime_name VARCHAR,
                round_num INTEGER,
                client_id INTEGER,
                is_byzantine BOOLEAN,
                gradient_norm DOUBLE,
                payload_bytes INTEGER,
                quantized BOOLEAN,
                local_loss DOUBLE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self.con.execute("""
            CREATE TABLE IF NOT EXISTS global_eval (
                experiment_id VARCHAR,
                regime_name VARCHAR,
                round_num INTEGER,
                global_accuracy DOUBLE,
                global_loss DOUBLE,
                aggregation_method VARCHAR,
                byzantine_fraction DOUBLE,
                quantized BOOLEAN,
                round_duration_sec DOUBLE,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

    def log_round_telemetry(
        self,
        experiment_id: str,
        regime_name: str,
        round_num: int,
        client_id: int,
        is_byzantine: bool,
        gradient_norm: float,
        payload_bytes: int,
        quantized: bool,
        local_loss: float,
    ) -> None:
        """Appends edge client telemetry entry."""
        self.con.execute(
            """
            INSERT INTO round_telemetry (
                experiment_id, regime_name, round_num, client_id, is_byzantine,
                gradient_norm, payload_bytes, quantized, local_loss
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                experiment_id,
                regime_name,
                round_num,
                client_id,
                is_byzantine,
                gradient_norm,
                payload_bytes,
                quantized,
                local_loss,
            ],
        )

    def log_global_eval(
        self,
        experiment_id: str,
        regime_name: str,
        round_num: int,
        global_accuracy: float,
        global_loss: float,
        aggregation_method: str,
        byzantine_fraction: float,
        quantized: bool,
        round_duration_sec: float,
    ) -> None:
        """Appends global server evaluation record."""
        self.con.execute(
            """
            INSERT INTO global_eval (
                experiment_id, regime_name, round_num, global_accuracy, global_loss,
                aggregation_method, byzantine_fraction, quantized, round_duration_sec
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                experiment_id,
                regime_name,
                round_num,
                global_accuracy,
                global_loss,
                aggregation_method,
                byzantine_fraction,
                quantized,
                round_duration_sec,
            ],
        )

    def get_regime_summary(self, experiment_id: str) -> List[Dict[str, Any]]:
        """Returns peak accuracy, final loss and total payload per regime."""
        query = """
            SELECT 
                regime_name,
                MAX(global_accuracy) AS peak_accuracy,
                AVG(global_accuracy) AS mean_accuracy,
                MIN(global_loss) AS best_loss,
                MAX(round_num) AS total_rounds
            FROM global_eval
            WHERE experiment_id = ?
            GROUP BY regime_name
            ORDER BY regime_name
        """
        rows = self.con.execute(query, [experiment_id]).fetchall()
        columns = ["regime_name", "peak_accuracy", "mean_accuracy", "best_loss", "total_rounds"]
        return [dict(zip(columns, row)) for row in rows]

    def close(self) -> None:
        """Closes DuckDB connection."""
        self.con.close()


def initialize_storage_pipeline(
    data_root: str = "data",
    num_clients: int = 20,
    alpha: float = 0.5,
    seed: int = DEFAULT_SEED,
    force_rebuild: bool = False,
    clients_dir: Optional[Union[str, Path]] = None,
    validation_size: int = 0,
) -> Tuple[List[str], Dict[str, Any]]:
    """Builds the client shards for one seed, or reuses them if they match.

    Args:
        data_root: Root directory for storing Parquet partitions and database.
        num_clients: Number of partitioned client shards.
        alpha: Dirichlet concentration parameter for non-IID label skew.
        seed: Random seed for partition reproducibility.
        force_rebuild: If True, regenerates and overwrites existing Parquet shards.
        clients_dir: Optional custom path for client shards (default: client_shard_dir()).
        validation_size: Training samples held out as a validation set before partitioning.
            They are drawn uniformly at random and stored as val.parquet next to the shards.

    Returns:
        Tuple of (list of parquet file paths, summary dictionary).
    """
    if clients_dir is not None:
        target_clients_dir = Path(clients_dir)
    else:
        target_clients_dir = client_shard_dir(data_root, seed, alpha, num_clients, validation_size)

    target_clients_dir.mkdir(parents=True, exist_ok=True)
    test_parquet_path = Path(data_root) / "test.parquet"
    expected_shards = [target_clients_dir / f"client_{i:02d}.parquet" for i in range(num_clients)]
    shards_exist = all(p.exists() for p in expected_shards) and test_parquet_path.exists()

    # Without this record, shards built with another alpha would be reused silently.
    meta_path = target_clients_dir / PARTITION_META_FILENAME
    requested_meta = {"num_clients": int(num_clients), "alpha": float(alpha), "seed": int(seed)}
    if validation_size:
        requested_meta["validation_size"] = int(validation_size)
    if shards_exist and not force_rebuild:
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as fh:
                stored_meta = json.load(fh)
            if stored_meta != requested_meta:
                print(
                    f"[!] Shards at {target_clients_dir} were built with {stored_meta}, "
                    f"but {requested_meta} was requested. Rebuilding."
                )
                force_rebuild = True
        else:
            print(
                f"[!] Shards at {target_clients_dir} have no {PARTITION_META_FILENAME}; their alpha "
                f"cannot be checked. Pass force_rebuild=True if they were built with other parameters."
            )

    if force_rebuild:
        for old_shard in target_clients_dir.glob("client_*.parquet"):
            old_shard.unlink()

    if shards_exist and not force_rebuild:
        print(
            f"[!] {num_clients} client shards and test.parquet already exist at {target_clients_dir}. "
            f"Skipping rebuild. Pass force_rebuild=True to overwrite."
        )
        return [str(p.resolve()) for p in expected_shards], {
            "num_clients": num_clients,
            "alpha": alpha,
            "seed": seed,
            "parquet_files": num_clients,
            "skipped": True,
        }

    partitioner = DataPartitioner(data_root=data_root)
    x_train, y_train, x_test, y_test = partitioner.load_fashion_mnist()

    train_pool = np.arange(len(y_train))
    if validation_size:
        val_rng = np.random.default_rng(seed + _VALIDATION_SEED_OFFSET)
        val_idx = np.sort(val_rng.choice(len(y_train), size=int(validation_size), replace=False))
        train_pool = np.setdiff1d(train_pool, val_idx)
        val_table = pa.Table.from_arrays(
            [
                pa.array(val_idx, type=pa.int64()),
                pa.array(y_train[val_idx], type=pa.int64()),
                pa.array(x_train[val_idx].tolist(), type=pa.list_(pa.float32())),
            ],
            names=["sample_id", "label", "features"],
        )
        pq.write_table(val_table, str(target_clients_dir / VALIDATION_FILENAME), compression="snappy")

    # Partition positions within the training pool, then map back to dataset indices.
    local = partitioner.partition_dirichlet(
        labels=y_train[train_pool],
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )
    partitions = {cid: train_pool[idx] for cid, idx in local.items()}
    paths = partitioner.export_to_parquet(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        client_partitions=partitions,
        clients_dir=target_clients_dir,
    )
    with open(meta_path, "w", encoding="utf-8") as fh:
        json.dump(requested_meta, fh, indent=2)

    telemetry = TelemetryEngine(db_path=os.path.join(data_root, "telemetry.duckdb"))
    telemetry.close()

    summary = {
        "num_clients": num_clients,
        "alpha": alpha,
        "seed": seed,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "parquet_files": len(paths),
    }
    return paths, summary
