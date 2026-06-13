#!/usr/bin/env python3
"""Evaluate TREC run files with pytrec_eval.

Supports qrels in either TREC/TSV whitespace format or bb25 qrels JSONL.
The output shape mirrors ``bb25 bench --json`` enough for comparison gates:

{
  "cutoffs": [10],
  "results": [
    {"scorer": "bm25", "queries": 100, "metrics": {"ndcg@10": 0.1, ...}}
  ]
}
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qrels", default=None, help="qrels.tsv or qrels.jsonl for a single dataset")
    parser.add_argument(
        "--root",
        default=None,
        help="Root containing <dataset>/qrels.tsv for multi-dataset evaluation",
    )
    parser.add_argument(
        "--datasets",
        default=None,
        help="Comma-separated dataset names for use with --root",
    )
    parser.add_argument(
        "--runs",
        required=True,
        help="TREC run file or directory containing *.trec files",
    )
    parser.add_argument("--cutoffs", default="10", help="Comma-separated cutoffs")
    parser.add_argument("--out", default=None, help="Optional JSON output path")
    return parser.parse_args()


def iter_run_paths(path: Path) -> Iterable[Path]:
    if path.is_dir():
        yield from sorted(path.glob("*.trec"))
    else:
        yield path


def load_qrels(path: Path) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}

    def add(qid: str, docid: str, rel: int) -> None:
        qrels.setdefault(qid, {})[docid] = rel

    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                row = json.loads(line)
                add(str(row["query_id"]), str(row["doc_id"]), int(row.get("relevance", 1)))
        return qrels

    first_data_line = True
    with path.open("r", encoding="utf8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                continue
            rel_col = 3 if len(parts) >= 4 else 2
            try:
                rel = int(float(parts[rel_col]))
            except ValueError:
                if first_data_line:
                    first_data_line = False
                    continue
                raise
            first_data_line = False
            qid = parts[0]
            docid = parts[2] if len(parts) >= 4 else parts[1]
            add(qid, docid, rel)
    return qrels


def load_run(path: Path) -> dict[str, dict[str, float]]:
    run: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                raise ValueError(f"invalid TREC run line in {path}: {line}")
            qid, docid, score = parts[0], parts[2], float(parts[4])
            run.setdefault(qid, {})[docid] = score
    return run


def evaluate(qrels: dict[str, dict[str, int]], run: dict[str, dict[str, float]], cutoffs: list[int]) -> dict[str, float]:
    try:
        import pytrec_eval
    except ImportError as exc:
        raise SystemExit("missing dependency: pip install pytrec_eval") from exc

    measures: set[str] = set()
    for cutoff in cutoffs:
        measures.add(f"ndcg_cut_{cutoff}")
        measures.add(f"map_cut_{cutoff}")
        measures.add(f"recall_{cutoff}")

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    per_query = evaluator.evaluate(run)
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        metric_map = {
            f"ndcg@{cutoff}": f"ndcg_cut_{cutoff}",
            f"map@{cutoff}": f"map_cut_{cutoff}",
            f"recall@{cutoff}": f"recall_{cutoff}",
        }
        for out_name, pytrec_name in metric_map.items():
            values = [query_metrics[pytrec_name] for query_metrics in per_query.values()]
            metrics[out_name] = float(sum(values) / len(values)) if values else 0.0
    return metrics


def evaluate_run_group(qrels_path: Path, runs_path: Path, cutoffs: list[int]) -> list[dict]:
    qrels = load_qrels(qrels_path)
    results = []
    for path in iter_run_paths(runs_path):
        run = load_run(path)
        results.append(
            {
                "scorer": path.stem,
                "queries": len(qrels),
                "metrics": evaluate(qrels, run, cutoffs),
                "runFile": str(path),
            }
        )
    return results


def aggregate(dataset_runs: list[dict]) -> list[dict]:
    by_scorer: dict[str, dict] = {}
    for dataset_run in dataset_runs:
        for row in dataset_run["results"]:
            scorer = row["scorer"]
            agg = by_scorer.setdefault(
                scorer,
                {"scorer": scorer, "datasets": 0, "queries": 0, "metrics": {}},
            )
            agg["datasets"] += 1
            agg["queries"] += row["queries"]
            for metric, value in row["metrics"].items():
                agg["metrics"][metric] = agg["metrics"].get(metric, 0.0) + float(value)

    rows = []
    for row in by_scorer.values():
        for metric in list(row["metrics"].keys()):
            row["metrics"][metric] /= row["datasets"]
        rows.append(row)
    return sorted(rows, key=lambda row: row["scorer"])


def main() -> None:
    args = parse_args()
    cutoffs = [int(part.strip()) for part in args.cutoffs.split(",") if part.strip()]
    runs_path = Path(args.runs)

    if args.root is not None or args.datasets is not None:
        if args.root is None or args.datasets is None:
            raise SystemExit("--root and --datasets must be provided together")
        root = Path(args.root)
        datasets = [part.strip() for part in args.datasets.split(",") if part.strip()]
        dataset_runs = []
        for dataset in datasets:
            qrels_path = root / dataset / "qrels.tsv"
            dataset_runs.append(
                {
                    "dataset": dataset,
                    "qrels": str(qrels_path),
                    "runs": str(runs_path / dataset),
                    "results": evaluate_run_group(qrels_path, runs_path / dataset, cutoffs),
                }
            )
        payload = {
            "cutoffs": cutoffs,
            "root": str(root),
            "runsRoot": str(runs_path),
            "datasets": datasets,
            "runs": dataset_runs,
            "average": aggregate(dataset_runs),
        }
    else:
        if args.qrels is None:
            raise SystemExit("--qrels is required unless --root/--datasets are used")
        qrels_path = Path(args.qrels)
        payload = {
            "cutoffs": cutoffs,
            "qrels": str(qrels_path),
            "runs": str(runs_path),
            "results": evaluate_run_group(qrels_path, runs_path, cutoffs),
        }

    text = json.dumps(payload, indent=2) + "\n"
    if args.out:
        Path(args.out).write_text(text, encoding="utf8")
    print(text, end="")


if __name__ == "__main__":
    main()
