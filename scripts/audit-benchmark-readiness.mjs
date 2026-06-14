#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { spawnSync } from "node:child_process";

const HYBRID_REQUIRED = [
  "python/sparse-benchmark.json",
  "python/base-rate.json",
  "python/hybrid-beir.json",
  "ts/hybrid-beir-internal.json",
  "ts/hybrid-beir-pytrec.json",
  "ts/baseline-parity.json",
  "manifests/python-env-setup.json",
  "manifests/python-env.json",
  "manifests/python-reference-benchmarks.json",
  "manifests/beir-jsonl-hybrid-export.json",
  "manifests/ts-hybrid-beir-internal.json",
  "manifests/ts-hybrid-beir-pytrec.json",
  "manifests/baseline-parity.json",
  "manifests/baseline-parity-runner.json",
];

const SPARSE_REQUIRED = [
  "python/sparse-benchmark.json",
  "python/base-rate.json",
  "ts/sparse-calibration-ts.json",
  "ts/sparse-ranking-ndcg_10-parity.json",
  "ts/sparse-ranking-map_10-parity.json",
  "ts/sparse-calibration-ece-parity.json",
  "ts/sparse-calibration-brier-parity.json",
  "ts/sparse-calibration-gate.json",
  "manifests/python-env-setup.json",
  "manifests/python-env.json",
  "manifests/python-reference-benchmarks.json",
  "manifests/beir-jsonl-sparse-export.json",
  "manifests/ts-sparse-calibration.json",
  "manifests/sparse-ranking-ndcg_10-parity.json",
  "manifests/sparse-ranking-map_10-parity.json",
  "manifests/sparse-calibration-ece-parity.json",
  "manifests/sparse-calibration-brier-parity.json",
  "manifests/sparse-calibration-gate.json",
  "manifests/sparse-calibration-parity-runner.json",
];

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
    if (key === "--") continue;
    if (!key.startsWith("--")) continue;
    const name = key.slice(2);
    const next = argv[i + 1];
    args[name] = next === undefined || next.startsWith("--") ? "true" : argv[++i];
  }
  return args;
}

function parseCsv(raw) {
  if (raw === undefined) return [];
  return raw.split(",").map((part) => part.trim()).filter(Boolean);
}

function parseProfiles(raw) {
  const profiles = parseCsv(raw ?? "hybrid");
  if (profiles.length === 0) return ["hybrid"];
  if (profiles.includes("all")) return ["hybrid", "sparse"];
  for (const profile of profiles) {
    if (profile !== "hybrid" && profile !== "sparse") {
      throw new Error(`unknown readiness profile: ${profile}`);
    }
  }
  return profiles;
}

function unique(values) {
  return [...new Set(values)];
}

function readJsonIfExists(path) {
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, "utf8"));
}

function fileManifest(path) {
  if (!existsSync(path)) return { path, exists: false };
  const data = readFileSync(path);
  return {
    path,
    exists: true,
    bytes: statSync(path).size,
    sha256: createHash("sha256").update(data).digest("hex"),
  };
}

function commandOk(command, options = {}) {
  const result = spawnSync(command[0], command.slice(1), {
    encoding: "utf8",
    ...options,
  });
  return result.status === 0 ? result.stdout.trim() : null;
}

function gitInfo(repoRoot) {
  return {
    commit: commandOk(["git", "rev-parse", "HEAD"], { cwd: repoRoot }),
    statusShort: commandOk(["git", "status", "--short"], { cwd: repoRoot }),
  };
}

function checkFile(root, relPath, required) {
  const path = join(root, relPath);
  const manifest = fileManifest(path);
  return {
    id: `file:${relPath}`,
    category: relPath.startsWith("manifests/") ? "manifest" : "result",
    required,
    passed: required ? manifest.exists : true,
    message: manifest.exists ? `${relPath} exists` : `${relPath} is missing`,
    file: manifest,
  };
}

function checkJson(root, relPath, predicate, message) {
  const path = join(root, relPath);
  const payload = readJsonIfExists(path);
  if (payload === null) {
    return {
      id: `json:${relPath}`,
      category: "content",
      required: true,
      passed: false,
      message: `${relPath} is missing`,
      file: fileManifest(path),
    };
  }
  let passed = false;
  let detail = message;
  try {
    const result = predicate(payload);
    passed = Boolean(result.passed);
    detail = result.message ?? message;
  } catch (error) {
    detail = `${message}: ${error instanceof Error ? error.message : String(error)}`;
  }
  return {
    id: `json:${relPath}`,
    category: "content",
    required: true,
    passed,
    message: detail,
    file: fileManifest(path),
  };
}

function pythonEnvCheck(payload) {
  const missing = Array.isArray(payload.missing) ? payload.missing : [];
  return {
    passed: missing.length === 0,
    message: missing.length === 0 ? "python benchmark dependencies importable" : `missing python packages: ${missing.join(",")}`,
  };
}

function commandManifestCheck(payload) {
  const commands = [];
  if (Array.isArray(payload.commands)) {
    commands.push(...payload.commands);
  } else if (Array.isArray(payload.datasetInputs)) {
    for (const input of payload.datasetInputs) {
      if (input?.commandRecord !== undefined) {
        commands.push(input.commandRecord);
      }
    }
  }

  const missingReturncode = commands.filter((command) => command.returncode === undefined);
  const failed = commands.filter((command) => command.returncode !== undefined && command.returncode !== 0);
  return {
    passed: commands.length > 0 && missingReturncode.length === 0 && failed.length === 0,
    message:
      commands.length === 0
        ? "manifest has no command records"
        : missingReturncode.length > 0
          ? `command records missing returncode: ${missingReturncode.map((command) => command.name ?? command.command?.[0] ?? "unknown").join(",")}`
          : failed.length === 0
            ? `${commands.length} command records passed`
            : `failed commands: ${failed.map((command) => command.name ?? command.command?.[0] ?? "unknown").join(",")}`,
  };
}

function parityCheck(payload) {
  const passed = payload.passed === true;
  const failures = Array.isArray(payload.failures) ? payload.failures : [];
  return {
    passed,
    message: passed ? "baseline parity passed" : `baseline parity failed with ${failures.length} failures`,
  };
}

function passedFieldCheck(label) {
  return (payload) => {
    const passed = payload.passed === true;
    const failures = Array.isArray(payload.failures) ? payload.failures : [];
    return {
      passed,
      message: passed ? `${label} passed` : `${label} failed with ${failures.length} failures`,
    };
  };
}

function datasetCoverageCheck(payload, expectedDatasets) {
  const actual = [];
  function addDatasetEntries(entries) {
    if (!Array.isArray(entries)) return;
    for (const entry of entries) {
      const dataset = typeof entry === "string" ? entry : entry?.dataset;
      if (dataset !== undefined && dataset !== null) actual.push(String(dataset));
    }
  }

  addDatasetEntries(payload.datasets);
  addDatasetEntries(payload.options?.datasets);
  addDatasetEntries(payload.inputs?.datasets);
  addDatasetEntries(payload.datasetInputs);

  const actualUnique = unique(actual);
  const missing = expectedDatasets.filter((dataset) => !actual.includes(dataset));
  return {
    passed: missing.length === 0,
    message: missing.length === 0 ? `datasets covered: ${actualUnique.join(",")}` : `missing datasets: ${missing.join(",")}`,
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const repoRoot = resolve(".");
  const root = resolve(args.root ?? "reference-results");
  const out = args.out === undefined ? null : resolve(args.out);
  const profiles = parseProfiles(args.profile);
  const datasets = parseCsv(args.datasets ?? "arguana,fiqa,nfcorpus,scidocs,scifact");
  const sparseDatasets = parseCsv(args["sparse-datasets"] ?? "nfcorpus,scifact");
  const warnOnly = args["warn-only"] !== undefined;
  const required = unique([
    ...(profiles.includes("hybrid") ? HYBRID_REQUIRED : []),
    ...(profiles.includes("sparse") ? SPARSE_REQUIRED : []),
    ...parseCsv(args.require),
  ]);

  const checks = [];
  for (const relPath of required) {
    checks.push(checkFile(root, relPath, true));
  }

  const jsonChecks = new Set();
  function addJsonCheck(relPath, predicate, message) {
    const key = `${relPath}:${message}`;
    if (jsonChecks.has(key)) return;
    jsonChecks.add(key);
    checks.push(checkJson(root, relPath, predicate, message));
  }

  addJsonCheck("manifests/python-env.json", pythonEnvCheck, "python env manifest valid");
  addJsonCheck("manifests/python-env-setup.json", commandManifestCheck, "python env setup commands valid");
  addJsonCheck("manifests/python-reference-benchmarks.json", commandManifestCheck, "python reference commands valid");

  if (profiles.includes("hybrid")) {
    addJsonCheck("manifests/beir-jsonl-hybrid-export.json", (payload) => datasetCoverageCheck(payload, datasets), "hybrid export dataset coverage");
    addJsonCheck("manifests/ts-hybrid-beir-internal.json", commandManifestCheck, "TS internal command manifest valid");
    addJsonCheck("manifests/ts-hybrid-beir-pytrec.json", (payload) => datasetCoverageCheck(payload.inputs ?? payload, datasets), "pytrec manifest dataset coverage");
    addJsonCheck("manifests/ts-hybrid-beir-pytrec.json", commandManifestCheck, "pytrec command manifest valid");
    addJsonCheck("manifests/baseline-parity-runner.json", commandManifestCheck, "baseline runner commands valid");
    addJsonCheck("ts/baseline-parity.json", parityCheck, "baseline parity result valid");
  }

  if (profiles.includes("sparse")) {
    addJsonCheck("manifests/beir-jsonl-sparse-export.json", (payload) => datasetCoverageCheck(payload, sparseDatasets), "sparse export dataset coverage");
    addJsonCheck("manifests/ts-sparse-calibration.json", commandManifestCheck, "TS sparse calibration commands valid");
    addJsonCheck("manifests/sparse-calibration-parity-runner.json", commandManifestCheck, "sparse parity runner commands valid");
    addJsonCheck("ts/sparse-ranking-ndcg_10-parity.json", passedFieldCheck("sparse NDCG parity"), "sparse NDCG parity result valid");
    addJsonCheck("ts/sparse-ranking-map_10-parity.json", passedFieldCheck("sparse MAP parity"), "sparse MAP parity result valid");
    addJsonCheck("ts/sparse-calibration-ece-parity.json", passedFieldCheck("sparse ECE parity"), "sparse ECE parity result valid");
    addJsonCheck("ts/sparse-calibration-brier-parity.json", passedFieldCheck("sparse Brier parity"), "sparse Brier parity result valid");
    addJsonCheck("ts/sparse-calibration-gate.json", passedFieldCheck("sparse calibration gate"), "sparse calibration gate result valid");
  }

  const failed = checks.filter((check) => check.required && !check.passed);
  const payload = {
    generatedAt: new Date().toISOString(),
    kind: "bb25-benchmark-readiness-audit",
    root,
    profiles,
    datasets,
    sparseDatasets,
    passed: failed.length === 0,
    summary: {
      checks: checks.length,
      failed: failed.length,
      missingFiles: failed.filter((check) => check.id.startsWith("file:")).map((check) => check.id.slice("file:".length)),
    },
    repo: {
      root: repoRoot,
      git: gitInfo(repoRoot),
      node: process.version,
      platform: process.platform,
      arch: process.arch,
    },
    checks,
  };

  if (out !== null) {
    mkdirSync(dirname(out), { recursive: true });
    writeFileSync(out, JSON.stringify(payload, null, 2) + "\n", "utf8");
  }

  process.stdout.write(
    JSON.stringify(
      {
        root,
        profiles,
        passed: payload.passed,
        checks: checks.length,
        failed: failed.length,
        missingFiles: payload.summary.missingFiles,
      },
      null,
      2,
    ) + "\n",
  );
  if (!payload.passed && !warnOnly) process.exitCode = 1;
}

main();
