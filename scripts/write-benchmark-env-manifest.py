#!/usr/bin/env python3
"""Write a reproducibility manifest for the Python benchmark environment.

The BEIR parity plan depends on optional Python packages that may not be
installed in a normal development shell. This script records the exact Python
runtime, platform, pip freeze output, git state, and import/version status for
the required benchmark packages before reference results are generated.
"""

from __future__ import annotations

import argparse
import datetime
import importlib.metadata
import importlib.util
import json
import platform
import subprocess
import sys
from pathlib import Path


DEFAULT_PACKAGES = [
    ("bayesian_bm25", ["bayesian-bm25"]),
    ("bm25s", ["bm25s"]),
    ("ir_datasets", ["ir-datasets", "ir_datasets"]),
    ("sentence_transformers", ["sentence-transformers"]),
    ("snowballstemmer", ["snowballstemmer"]),
    ("Stemmer", ["PyStemmer"]),
    ("pytrec_eval", ["pytrec_eval", "pytrec-eval"]),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out",
        default="reference-results/manifests/python-env.json",
        help="JSON manifest output path",
    )
    parser.add_argument(
        "--freeze-out",
        default=None,
        help="Optional pip freeze text output path",
    )
    parser.add_argument(
        "--require",
        action="store_true",
        help="Exit non-zero if any required benchmark dependency is missing",
    )
    return parser.parse_args()


def run(command: list[str]) -> dict:
    result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def version_for(distribution_names: list[str]) -> str | None:
    for name in distribution_names:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    return None


def package_status(import_name: str, distribution_names: list[str]) -> dict:
    spec = importlib.util.find_spec(import_name)
    return {
        "importName": import_name,
        "distributionNames": distribution_names,
        "importable": spec is not None,
        "version": version_for(distribution_names),
        "origin": None if spec is None else spec.origin,
    }


def git_info() -> dict:
    commit = run(["git", "rev-parse", "HEAD"])
    status = run(["git", "status", "--short"])
    return {
        "commit": commit["stdout"].strip() if commit["returncode"] == 0 else None,
        "statusShort": status["stdout"] if status["returncode"] == 0 else None,
    }


def main() -> None:
    args = parse_args()
    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    freeze = run([sys.executable, "-m", "pip", "freeze"])
    packages = [package_status(import_name, dist_names) for import_name, dist_names in DEFAULT_PACKAGES]
    missing = [pkg["importName"] for pkg in packages if not pkg["importable"]]

    manifest = {
        "generatedAt": generated_at,
        "kind": "bb25-python-benchmark-env",
        "python": {
            "version": sys.version,
            "executable": sys.executable,
            "prefix": sys.prefix,
            "basePrefix": sys.base_prefix,
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
        "git": git_info(),
        "packages": packages,
        "missing": missing,
        "pipFreeze": {
            "command": freeze["command"],
            "returncode": freeze["returncode"],
            "lines": freeze["stdout"].splitlines() if freeze["returncode"] == 0 else [],
            "stderr": freeze["stderr"],
        },
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")

    if args.freeze_out is not None:
        freeze_out = Path(args.freeze_out)
        freeze_out.parent.mkdir(parents=True, exist_ok=True)
        freeze_out.write_text(freeze["stdout"], encoding="utf8")

    print(json.dumps({"out": str(out), "missing": missing}, indent=2))
    if args.require and missing:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
