#!/usr/bin/env python3
"""Run the Python reference benchmark scripts and write a manifest.

The reference implementation lives outside this repository. This runner keeps
the command contract, output paths, environment manifest, command logs, git
state, and output hashes in one place so Python-vs-TypeScript comparisons are
repeatable after the benchmark dependencies are installed.
"""

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
    parser.add_argument(
        "--reference-repo",
        default=os.environ.get("BAYESIAN_BM25_REF"),
        help="Path to a checkout of cognica-io/bayesian-bm25; defaults to BAYESIAN_BM25_REF",
    )
    parser.add_argument("--beir-dir", default="/tmp/beir", help="BEIR data directory")
    parser.add_argument("--out-dir", default="reference-results/python", help="Reference JSON output directory")
    parser.add_argument(
        "--manifest-dir",
        default="reference-results/manifests",
        help="Directory for runner and environment manifests",
    )
    parser.add_argument("--manifest-out", default=None, help="Optional runner manifest path")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        help="BEIR datasets, comma-separated or space-separated",
    )
    parser.add_argument("--model", default="all-MiniLM-L6-v2", help="sentence-transformers model")
    parser.add_argument("--retrieve-k", type=int, default=1000, help="Reference hybrid -R candidate depth")
    parser.add_argument("--top-k", type=int, default=10, help="Reference hybrid -k evaluation depth")
    parser.add_argument("--python", default=sys.executable, help="Python executable for reference commands")
    parser.add_argument("--download", action="store_true", help="Pass --download to hybrid_beir.py")
    parser.add_argument("--cache-dir", default=None, help="Optional embedding cache dir for hybrid_beir.py")
    parser.add_argument("--no-cache", action="store_true", help="Pass --no-cache to hybrid_beir.py")
    parser.add_argument("--tune", action="store_true", help="Pass --tune to hybrid_beir.py")
    parser.add_argument("--skip-sparse", action="store_true", help="Skip benchmarks/benchmark.py")
    parser.add_argument("--skip-base-rate", action="store_true", help="Skip benchmarks/base_rate.py")
    parser.add_argument("--skip-hybrid", action="store_true", help="Skip benchmarks/hybrid_beir.py")
    parser.add_argument("--dry-run", action="store_true", help="Record planned commands without executing them")
    parser.add_argument(
        "--require-env",
        action="store_true",
        help="Require benchmark dependencies in the environment manifest before running benchmarks",
    )
    return parser.parse_args()


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


def command_output(command: list[str], cwd: Path | None = None) -> str | None:
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


def git_info(path: Path) -> dict:
    return {
        "commit": command_output(["git", "rev-parse", "HEAD"], cwd=path),
        "statusShort": command_output(["git", "status", "--short"], cwd=path),
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


def run_command(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    dry_run: bool,
    env_updates: dict[str, str] | None = None,
) -> dict:
    started_at = utc_now()
    if dry_run:
        return {
            "name": name,
            "command": command,
            "cwd": str(cwd),
            "env": env_updates or {},
            "startedAt": started_at,
            "finishedAt": started_at,
            "durationSeconds": 0.0,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "dryRun": True,
        }

    env = os.environ.copy()
    if env_updates is not None:
        env.update(env_updates)
    start = time.perf_counter()
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
            env=env,
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
        "name": name,
        "command": command,
        "cwd": str(cwd),
        "env": env_updates or {},
        "startedAt": started_at,
        "finishedAt": utc_now(),
        "durationSeconds": round(time.perf_counter() - start, 6),
        "returncode": returncode,
        "stdout": stdout,
        "stderr": stderr,
        "dryRun": False,
    }


def reference_pythonpath(reference_repo: Path) -> str:
    existing = os.environ.get("PYTHONPATH")
    parts = [str(reference_repo)]
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def append_if_exists(command: list[str], flag: str, value: str | None) -> None:
    if value is not None:
        command.extend([flag, value])


def main() -> None:
    args = parse_args()
    if args.reference_repo is None:
        raise SystemExit("--reference-repo is required unless BAYESIAN_BM25_REF is set")

    repo_root = Path(__file__).resolve().parents[1]
    reference_repo = Path(args.reference_repo).expanduser().resolve()
    beir_dir = Path(args.beir_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_dir = Path(args.manifest_dir).expanduser().resolve()
    manifest_out = (
        Path(args.manifest_out).expanduser().resolve()
        if args.manifest_out
        else manifest_dir / "python-reference-benchmarks.json"
    )
    env_manifest = manifest_dir / "python-env.json"
    freeze_out = manifest_dir / "python-freeze.txt"
    datasets = split_values(args.datasets)

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)

    commands: list[tuple[str, list[str], Path, dict[str, str] | None]] = []
    env_command = [
        args.python,
        str(repo_root / "scripts" / "write-benchmark-env-manifest.py"),
        "--out",
        str(env_manifest),
        "--freeze-out",
        str(freeze_out),
    ]
    if args.require_env:
        env_command.append("--require")
    commands.append(("python-env", env_command, repo_root, None))

    reference_env = {"PYTHONPATH": reference_pythonpath(reference_repo)}
    if not args.skip_sparse:
        commands.append(
            (
                "reference-sparse",
                [
                    args.python,
                    str(reference_repo / "benchmarks" / "benchmark.py"),
                    "-o",
                    str(out_dir / "sparse-benchmark.json"),
                ],
                reference_repo,
                reference_env,
            )
        )
    if not args.skip_base_rate:
        commands.append(
            (
                "reference-base-rate",
                [
                    args.python,
                    str(reference_repo / "benchmarks" / "base_rate.py"),
                    "-o",
                    str(out_dir / "base-rate.json"),
                ],
                reference_repo,
                reference_env,
            )
        )
    if not args.skip_hybrid:
        hybrid_command = [
            args.python,
            str(reference_repo / "benchmarks" / "hybrid_beir.py"),
            "-d",
            str(beir_dir),
        ]
        if args.download:
            hybrid_command.append("--download")
        hybrid_command.extend(["--datasets", *datasets])
        hybrid_command.extend(["--model", args.model, "-R", str(args.retrieve_k), "-k", str(args.top_k)])
        append_if_exists(hybrid_command, "--cache-dir", args.cache_dir)
        if args.no_cache:
            hybrid_command.append("--no-cache")
        if args.tune:
            hybrid_command.append("--tune")
        hybrid_command.extend(["-o", str(out_dir / "hybrid-beir.json")])
        commands.append(("reference-hybrid-beir", hybrid_command, reference_repo, reference_env))

    records = []
    failed = False
    for name, command, cwd, env_updates in commands:
        record = run_command(name, command, cwd=cwd, dry_run=args.dry_run, env_updates=env_updates)
        records.append(record)
        if not args.dry_run and record["returncode"] != 0:
            failed = True
            if name == "python-env" and args.require_env:
                break

    generated_at = utc_now()
    manifest = {
        "generatedAt": generated_at,
        "kind": "bb25-python-reference-benchmarks",
        "dryRun": args.dry_run,
        "options": {
            "referenceRepo": str(reference_repo),
            "beirDir": str(beir_dir),
            "outDir": str(out_dir),
            "manifestDir": str(manifest_dir),
            "datasets": datasets,
            "model": args.model,
            "retrieveK": args.retrieve_k,
            "topK": args.top_k,
            "download": args.download,
            "cacheDir": args.cache_dir,
            "noCache": args.no_cache,
            "tune": args.tune,
            "skipSparse": args.skip_sparse,
            "skipBaseRate": args.skip_base_rate,
            "skipHybrid": args.skip_hybrid,
            "requireEnv": args.require_env,
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "runnerPython": args.python,
            "localRepo": {
                "path": str(repo_root),
                "git": git_info(repo_root),
            },
            "referenceRepo": {
                "path": str(reference_repo),
                "exists": reference_repo.exists(),
                "git": git_info(reference_repo) if reference_repo.exists() else None,
            },
            "envManifest": file_manifest(env_manifest),
            "freeze": file_manifest(freeze_out),
        },
        "commands": records,
        "outputs": {
            "sparseBenchmark": file_manifest(out_dir / "sparse-benchmark.json"),
            "baseRate": file_manifest(out_dir / "base-rate.json"),
            "hybridBeir": file_manifest(out_dir / "hybrid-beir.json"),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(
        json.dumps(
            {
                "manifest": str(manifest_out),
                "outputs": manifest["outputs"],
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
