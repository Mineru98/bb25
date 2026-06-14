#!/usr/bin/env python3
"""Write a NumPy default_rng query split for benchmark parity."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


def read_query_ids(path: Path) -> list[str]:
    query_ids: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            query_id = row.get("query_id", row.get("id"))
            if not isinstance(query_id, str):
                raise ValueError(f"{path}:{line_no}: missing string query_id")
            query_ids.append(query_id)
    return query_ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--dataset", default=None)
    parser.add_argument("--train-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    query_ids = read_query_ids(args.queries)
    if len(query_ids) < 2:
        raise ValueError("at least two queries are required for a split")
    if not (0.0 < args.train_ratio < 1.0):
        raise ValueError("--train-ratio must be in (0, 1)")

    shuffled = np.array(query_ids, dtype=object)
    np.random.default_rng(args.seed).shuffle(shuffled)
    n_train = min(len(shuffled) - 1, max(1, int(len(shuffled) * args.train_ratio)))

    payload = {
        "kind": "bb25-numpy-query-split",
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "queries": str(args.queries),
        "seed": args.seed,
        "trainRatio": args.train_ratio,
        "algorithm": "numpy.default_rng(seed).shuffle; floor(n * train_ratio)",
        "queryCount": len(shuffled),
        "trainQueryIds": shuffled[:n_train].tolist(),
        "evalQueryIds": shuffled[n_train:].tolist(),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
