#!/usr/bin/env python3
"""Create and document the optional Python benchmark environment.

The reference benchmark path needs packages that are intentionally not part of
the TypeScript workspace. This helper makes the setup reproducible: create a
venv, install requirements, then write the same python-env manifest consumed by
the readiness audit.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_PYTHON_CANDIDATES = ["python3.12", "python3.11", "python3.10", "python3"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--venv", default=".venv-bench", help="Benchmark virtualenv directory")
    parser.add_argument(
        "--python",
        default=os.environ.get("BB25_BENCH_PYTHON"),
        help="Python executable to create the venv; defaults to BB25_BENCH_PYTHON or a compatible python3.x",
    )
    parser.add_argument("--requirements", default="requirements-bench.txt", help="Benchmark requirements file")
    parser.add_argument(
        "--out",
        default="reference-results/manifests/python-env-setup.json",
        help="Setup manifest output path",
    )
    parser.add_argument(
        "--env-manifest-out",
        default="reference-results/manifests/python-env.json",
        help="Python package environment manifest output path",
    )
    parser.add_argument(
        "--freeze-out",
        default="reference-results/manifests/python-freeze.txt",
        help="pip freeze text output path",
    )
    parser.add_argument("--upgrade-pip", action="store_true", help="Upgrade pip before installing requirements")
    parser.add_argument("--skip-install", action="store_true", help="Create/reuse venv and write manifests without installing requirements")
    parser.add_argument("--require", action="store_true", help="Fail if benchmark dependencies are missing after setup")
    parser.add_argument("--dry-run", action="store_true", help="Record planned commands without executing them")
    return parser.parse_args([arg for arg in sys.argv[1:] if arg != "--"])


def utc_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def run_command(command: list[str], *, cwd: Path, dry_run: bool) -> dict:
    started_at = utc_now()
    if dry_run:
        return {
            "command": command,
            "cwd": str(cwd),
            "startedAt": started_at,
            "finishedAt": started_at,
            "durationSeconds": 0.0,
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "dryRun": True,
        }

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
        "dryRun": False,
    }


def command_output(command: list[str]) -> str | None:
    try:
        result = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
    except FileNotFoundError:
        return None
    return result.stdout.strip() if result.returncode == 0 else None


def resolve_python(requested: str | None) -> str:
    candidates = [requested] if requested else []
    candidates.extend(DEFAULT_PYTHON_CANDIDATES)
    seen: set[str] = set()
    for candidate in candidates:
        if candidate is None or candidate in seen:
            continue
        seen.add(candidate)
        path = shutil.which(candidate) if os.sep not in candidate else candidate
        if path is None:
            continue
        if command_output([path, "--version"]) is not None:
            return path
    raise SystemExit(f"no usable Python found; tried: {', '.join(c for c in candidates if c)}")


def venv_python(venv: Path) -> Path:
    if platform.system() == "Windows":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def file_manifest(path: Path) -> dict:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_dir():
        return {"path": str(path), "exists": True, "type": "directory"}
    data = path.read_bytes()
    return {
        "path": str(path),
        "exists": True,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def command_ok(command: list[str], cwd: Path) -> str | None:
    try:
        result = subprocess.run(
            command,
            cwd=str(cwd),
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


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    python = resolve_python(args.python)
    venv = Path(args.venv).expanduser()
    if not venv.is_absolute():
        venv = repo_root / venv
    requirements = Path(args.requirements).expanduser()
    if not requirements.is_absolute():
        requirements = repo_root / requirements
    out = Path(args.out).expanduser()
    if not out.is_absolute():
        out = repo_root / out
    env_manifest = Path(args.env_manifest_out).expanduser()
    if not env_manifest.is_absolute():
        env_manifest = repo_root / env_manifest
    freeze_out = Path(args.freeze_out).expanduser()
    if not freeze_out.is_absolute():
        freeze_out = repo_root / freeze_out

    out.parent.mkdir(parents=True, exist_ok=True)
    env_manifest.parent.mkdir(parents=True, exist_ok=True)
    freeze_out.parent.mkdir(parents=True, exist_ok=True)

    py_in_venv = venv_python(venv)
    commands: list[dict] = []
    failed = False

    setup_commands: list[tuple[str, list[str]]] = [
        ("create-venv", [python, "-m", "venv", str(venv)]),
    ]
    if args.upgrade_pip:
        setup_commands.append(("upgrade-pip", [str(py_in_venv), "-m", "pip", "install", "--upgrade", "pip"]))
    if not args.skip_install:
        setup_commands.append(("install-requirements", [str(py_in_venv), "-m", "pip", "install", "-r", str(requirements)]))
    env_command = [
        str(py_in_venv),
        str(repo_root / "scripts" / "write-benchmark-env-manifest.py"),
        "--out",
        str(env_manifest),
        "--freeze-out",
        str(freeze_out),
    ]
    if args.require:
        env_command.append("--require")
    setup_commands.append(("write-python-env-manifest", env_command))

    for name, command in setup_commands:
        record = run_command(command, cwd=repo_root, dry_run=args.dry_run)
        record["name"] = name
        commands.append(record)
        if not args.dry_run and record["returncode"] != 0:
            failed = True
            break

    manifest = {
        "generatedAt": utc_now(),
        "kind": "bb25-python-benchmark-env-setup",
        "dryRun": args.dry_run,
        "options": {
            "venv": str(venv),
            "python": python,
            "requirements": str(requirements),
            "upgradePip": args.upgrade_pip,
            "skipInstall": args.skip_install,
            "require": args.require,
            "envManifestOut": str(env_manifest),
            "freezeOut": str(freeze_out),
        },
        "repo": {
            "root": str(repo_root),
            "git": git_info(repo_root),
        },
        "files": {
            "venv": file_manifest(venv),
            "venvPython": file_manifest(py_in_venv),
            "requirements": file_manifest(requirements),
            "envManifest": file_manifest(env_manifest),
            "freeze": file_manifest(freeze_out),
        },
        "commands": commands,
        "passed": not failed,
    }
    out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf8")
    print(
        json.dumps(
            {
                "manifest": str(out),
                "envManifest": str(env_manifest),
                "venv": str(venv),
                "passed": not failed,
                "dryRun": args.dry_run,
            },
            indent=2,
        )
    )
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
