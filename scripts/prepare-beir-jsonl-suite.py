#!/usr/bin/env python3
"""Run BEIR JSONL export for multiple datasets and write a suite manifest."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Sequence


DEFAULT_DATASETS = ["arguana", "fiqa", "nfcorpus", "scidocs", "scifact"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", default="/tmp/beir-jsonl", help="Directory containing one subdirectory per dataset")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="BEIR datasets, comma-separated or space-separated",
    )
    parser.add_argument("--tokenizer", choices=["none", "split", "snowball"], default="snowball")
    parser.add_argument("--split", default="test", help="BEIR split for short dataset names")
    parser.add_argument("--embed-model", default=None, help="Optional sentence-transformers model")
    parser.add_argument(
        "--embed-cache-dir",
        default=None,
        help="Optional sentence-transformers cache directory for the embedding model",
    )
    parser.add_argument(
        "--embed-local-files-only",
        action="store_true",
        help="Load the embedding model from the local cache only",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--max-queries", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--prepare-script", default=None, help="Override prepare-beir-jsonl.py path for smoke tests")
    parser.add_argument("--manifest-out", default=None, help="Suite manifest path")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args([arg for arg in sys.argv[1:] if arg != "--"])


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def split_values(values: Sequence[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        for part in value.split(","):
            item = part.strip()
            if item:
                out.append(item)
    return out


def run(command: list[str], cwd: Path) -> dict:
    started_at = utc_now()
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        returncode = result.returncode
        stdout = result.stdout
        stderr = result.stderr
    except FileNotFoundError as exc:
        returncode = 127
        stdout = ""
        stderr = str(exc)
    return {
        "command": command,
        "cwd": str(cwd),
        "startedAt": started_at,
        "finishedAt": utc_now(),
        "durationSeconds": round(time.perf_counter() - start, 6),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
    }


def command_ok(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd) if cwd is not None else None,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def git_info(repo_root: Path) -> dict:
    return {
        "commit": command_ok(["git", "rev-parse", "HEAD"], repo_root),
        "statusShort": command_ok(["git", "status", "--short"], repo_root),
    }


def file_manifest(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def read_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf8"))


def env_value(name: str) -> str | None:
    value = os.environ.get(name)
    return value if value else None


def dataset_manifest(out_root: Path, dataset: str) -> dict:
    dataset_dir = out_root / dataset
    manifest_path = dataset_dir / "manifest.json"
    return {
        "dataset": dataset,
        "directory": str(dataset_dir),
        "files": {
            "docs": file_manifest(dataset_dir / "docs.jsonl"),
            "queries": file_manifest(dataset_dir / "queries.jsonl"),
            "qrels": file_manifest(dataset_dir / "qrels.tsv"),
            "manifest": file_manifest(manifest_path),
        },
        "exportManifest": read_json_if_exists(manifest_path),
    }


def build_command(args: argparse.Namespace, prepare_script: Path, out_root: Path, dataset: str) -> list[str]:
    command = [
        args.python,
        str(prepare_script),
        "--dataset",
        dataset,
        "--out",
        str(out_root / dataset),
        "--tokenizer",
        args.tokenizer,
        "--split",
        args.split,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.embed_model is not None:
        command.extend(["--embed-model", args.embed_model])
    if args.embed_cache_dir is not None:
        command.extend(["--embed-cache-dir", str(Path(args.embed_cache_dir).expanduser().resolve())])
    if args.embed_local_files_only:
        command.append("--embed-local-files-only")
    if args.max_docs is not None:
        command.extend(["--max-docs", str(args.max_docs)])
    if args.max_queries is not None:
        command.extend(["--max-queries", str(args.max_queries)])
    return command


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    prepare_script = Path(args.prepare_script).expanduser().resolve() if args.prepare_script else repo_root / "scripts" / "prepare-beir-jsonl.py"
    out_root = Path(args.out_root).expanduser().resolve()
    manifest_out = Path(args.manifest_out).expanduser().resolve() if args.manifest_out else out_root / "suite-manifest.json"
    datasets = split_values(args.datasets)

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    commands = []
    failed = False
    for dataset in datasets:
        command = build_command(args, prepare_script, out_root, dataset)
        if args.dry_run:
            record = {
                "dataset": dataset,
                "command": command,
                "cwd": str(repo_root),
                "startedAt": utc_now(),
                "finishedAt": utc_now(),
                "durationSeconds": 0.0,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "dryRun": True,
            }
        else:
            record = {"dataset": dataset, **run(command, repo_root), "dryRun": False}
        commands.append(record)
        if record["returncode"] != 0:
            failed = True
            if not args.continue_on_error:
                break

    generated_at = utc_now()
    manifest = {
        "generatedAt": generated_at,
        "kind": "bb25-beir-jsonl-suite",
        "dryRun": args.dry_run,
        "options": {
            "outRoot": str(out_root),
            "datasets": datasets,
            "tokenizer": args.tokenizer,
            "split": args.split,
            "embedModel": args.embed_model,
            "embedCacheDir": str(Path(args.embed_cache_dir).expanduser().resolve()) if args.embed_cache_dir is not None else None,
            "embedLocalFilesOnly": args.embed_local_files_only,
            "batchSize": args.batch_size,
            "maxDocs": args.max_docs,
            "maxQueries": args.max_queries,
            "prepareScript": str(prepare_script),
            "continueOnError": args.continue_on_error,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "git": git_info(repo_root),
            "cacheEnvironment": {
                "SENTENCE_TRANSFORMERS_HOME": env_value("SENTENCE_TRANSFORMERS_HOME"),
                "HF_HOME": env_value("HF_HOME"),
                "TRANSFORMERS_CACHE": env_value("TRANSFORMERS_CACHE"),
            },
        },
        "commands": commands,
        "datasets": [dataset_manifest(out_root, dataset) for dataset in datasets],
        "failed": failed,
    }
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_out),
                "outRoot": str(out_root),
                "datasets": datasets,
                "failed": failed,
                "dryRun": args.dry_run,
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
