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

const SUPPORTED_PROTOCOL_VERSION = 1;

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
  const requested = parseCsv(raw ?? "hybrid");
  const profiles = requested.length === 0 ? ["hybrid"] : requested;
  const expanded = [];
  for (const profile of profiles) {
    if (profile === "all") {
      expanded.push("hybrid", "sparse");
    } else if (profile === "release") {
      expanded.push("hybrid", "sparse", "hybrid-strict");
    } else if (profile === "hybrid-strict") {
      expanded.push("hybrid", "hybrid-strict");
    } else if (profile === "hybrid" || profile === "sparse") {
      expanded.push(profile);
    } else {
      throw new Error(`unknown readiness profile: ${profile}`);
    }
  }
  return unique(expanded);
}

function unique(values) {
  return [...new Set(values)];
}

function normalizeName(value) {
  return String(value).trim().toLowerCase().replace(/[-\s]+/g, "_");
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

function datasetParityCheck(payload, expectedDatasets, tolerance) {
  const comparisons = Array.isArray(payload.comparisons) ? payload.comparisons : [];
  const violations = [];
  const missing = [];
  let maxDiff = 0;
  let maxLabel = null;

  for (const comparison of comparisons) {
    const method = String(comparison?.method ?? "unknown");
    const datasets = Array.isArray(comparison?.datasets) ? comparison.datasets : [];
    const seen = new Set();
    for (const row of datasets) {
      const dataset = String(row?.dataset ?? "");
      if (dataset !== "") seen.add(dataset);
      const diff = Number(row?.diff);
      if (!Number.isFinite(diff)) continue;
      if (diff > maxDiff) {
        maxDiff = diff;
        maxLabel = `${method}/${dataset}`;
      }
      if (diff > tolerance) {
        violations.push({ method, dataset, diff });
      }
    }
    for (const dataset of expectedDatasets) {
      if (!seen.has(dataset)) missing.push(`${method}/${dataset}`);
    }
  }

  if (comparisons.length === 0) {
    return { passed: false, message: "baseline parity has no comparison rows" };
  }
  if (missing.length > 0) {
    return { passed: false, message: `baseline parity missing dataset rows: ${missing.slice(0, 5).join(",")}${missing.length > 5 ? ",..." : ""}` };
  }
  if (violations.length > 0) {
    return {
      passed: false,
      message: `dataset-level baseline parity failed: ${violations.length} diff(s) > ${tolerance}; max ${maxLabel}=${maxDiff.toFixed(6)}`,
    };
  }
  return {
    passed: true,
    message: `dataset-level baseline parity passed: max ${maxLabel ?? "n/a"}=${maxDiff.toFixed(6)} <= ${tolerance}`,
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

function fileRecordOk(record) {
  return record !== null && typeof record === "object" && record.exists === true && Number(record.bytes ?? 0) > 0;
}

function listMissing(expected, actual) {
  const actualSet = new Set(actual.map((value) => String(value)));
  return expected.filter((value) => !actualSet.has(value));
}

function commandFlag(command, flag) {
  if (!Array.isArray(command)) return undefined;
  const index = command.indexOf(flag);
  return index < 0 ? undefined : command[index + 1];
}

function firstCommandFlag(payload, flag) {
  const commands = Array.isArray(payload.commands) ? payload.commands : [];
  for (const command of commands) {
    const value = commandFlag(command.command, flag);
    if (value !== undefined) return value;
  }
  return undefined;
}

function cliArg(payload, key) {
  return payload.cliArgs?.[key] ?? firstCommandFlag(payload, `--${key}`);
}

function scorerListFrom(value) {
  if (Array.isArray(value)) return value.map(normalizeName).filter(Boolean);
  if (typeof value === "string") return parseCsv(value).map(normalizeName);
  return [];
}

function protocolResult(failures, passedMessage) {
  return {
    passed: failures.length === 0,
    message: failures.length === 0 ? passedMessage : failures.slice(0, 8).join("; "),
  };
}

function protocolVersionFailures(payload, label) {
  const version = Number(payload.protocolVersion ?? SUPPORTED_PROTOCOL_VERSION);
  if (Number.isInteger(version) && version === SUPPORTED_PROTOCOL_VERSION) return [];
  return [`${label} protocolVersion ${payload.protocolVersion ?? "missing"} is not supported by readiness schema v${SUPPORTED_PROTOCOL_VERSION}`];
}

function hybridExportProtocolCheck(payload, expectedDatasets) {
  const failures = protocolVersionFailures(payload, "hybrid export");
  if (payload.kind !== "bb25-beir-jsonl-suite") failures.push("hybrid export manifest kind is not bb25-beir-jsonl-suite");
  if (payload.options?.tokenizer === undefined) failures.push("missing export tokenizer provenance");
  if (payload.options?.split === undefined) failures.push("missing export split provenance");
  if (!Object.prototype.hasOwnProperty.call(payload.options ?? {}, "embedModel")) failures.push("missing embedding model provenance");
  if (!Object.prototype.hasOwnProperty.call(payload.options ?? {}, "embedCacheDir")) failures.push("missing embedding cache provenance");

  const datasetEntries = Array.isArray(payload.datasets) ? payload.datasets : [];
  const presentDatasets = datasetEntries.map((entry) => String(entry?.dataset ?? ""));
  const missingDatasets = listMissing(expectedDatasets, presentDatasets);
  if (missingDatasets.length > 0) failures.push(`export manifest missing datasets: ${missingDatasets.join(",")}`);

  for (const entry of datasetEntries) {
    const dataset = String(entry?.dataset ?? "unknown");
    const exportManifest = entry?.exportManifest;
    if (exportManifest === null || typeof exportManifest !== "object") {
      failures.push(`${dataset} missing per-dataset export manifest`);
      continue;
    }
    if (exportManifest.tokenizer === undefined) failures.push(`${dataset} missing tokenizer`);
    if (exportManifest.split === undefined) failures.push(`${dataset} missing split`);
    if (exportManifest.irDatasetId === undefined) failures.push(`${dataset} missing irDatasetId`);
    if (!fileRecordOk(entry?.files?.qrels) && !fileRecordOk(exportManifest.files?.qrels)) {
      failures.push(`${dataset} qrels file is not manifest-backed`);
    }
    const embedding = exportManifest.embedding;
    if (embedding === null || typeof embedding !== "object") {
      failures.push(`${dataset} missing embedding provenance`);
    } else {
      for (const key of ["model", "cacheDir", "localFilesOnly", "normalize"]) {
        if (!Object.prototype.hasOwnProperty.call(embedding, key)) failures.push(`${dataset} embedding.${key} missing`);
      }
    }
  }

  return protocolResult(failures, "hybrid export protocol provenance valid");
}

function tsHybridProtocolCheck(payload, expectedDatasets) {
  const failures = protocolVersionFailures(payload, "TS hybrid");
  const expectedScorers = ["bm25", "dense", "convex", "rrf"];
  if (payload.kind !== "bb25-beir-jsonl-bench") failures.push("TS hybrid manifest kind is not bb25-beir-jsonl-bench");
  if (String(cliArg(payload, "bm25-method") ?? "") !== "lucene") failures.push("TS hybrid bm25-method must be lucene");
  const candidateDepth = Number(cliArg(payload, "candidate-depth"));
  if (!Number.isInteger(candidateDepth) || candidateDepth <= 0) failures.push("TS hybrid candidate-depth must be a positive integer");
  const metricStyle = cliArg(payload, "metric-style") ?? "pytrec";
  if (metricStyle !== "pytrec") failures.push("TS hybrid evaluator metric-style must be pytrec");
  const topLevelScorers = scorerListFrom(cliArg(payload, "scorers"));
  const missingTopLevelScorers = listMissing(expectedScorers, topLevelScorers);
  if (missingTopLevelScorers.length > 0) failures.push(`TS hybrid scorer filter missing: ${missingTopLevelScorers.join(",")}`);

  const datasetInputs = Array.isArray(payload.datasetInputs) ? payload.datasetInputs : [];
  const presentDatasets = datasetInputs.map((entry) => String(entry?.dataset ?? ""));
  const missingDatasets = listMissing(expectedDatasets, presentDatasets);
  if (missingDatasets.length > 0) failures.push(`TS hybrid manifest missing datasets: ${missingDatasets.join(",")}`);

  for (const entry of datasetInputs) {
    const dataset = String(entry?.dataset ?? "unknown");
    if (!fileRecordOk(entry?.files?.qrels)) failures.push(`${dataset} qrels file is not manifest-backed`);
    const exportManifest = entry?.exportManifest;
    if (exportManifest === null || typeof exportManifest !== "object") {
      failures.push(`${dataset} missing export manifest snapshot`);
    } else {
      if (exportManifest.tokenizer === undefined) failures.push(`${dataset} missing tokenizer snapshot`);
      if (exportManifest.split === undefined) failures.push(`${dataset} missing split snapshot`);
      const embedding = exportManifest.embedding;
      if (embedding === null || typeof embedding !== "object") {
        failures.push(`${dataset} missing embedding/cache snapshot`);
      } else if (!Object.prototype.hasOwnProperty.call(embedding, "cacheDir")) {
        failures.push(`${dataset} missing embedding cache snapshot`);
      }
    }

    const options = entry?.resultSummary?.options;
    if (options?.bm25Method !== "lucene") failures.push(`${dataset} result bm25Method must be lucene`);
    if (Number(options?.candidateDepth) !== candidateDepth) failures.push(`${dataset} result candidateDepth does not match CLI`);
    if ((options?.metricStyle ?? "pytrec") !== "pytrec") failures.push(`${dataset} result evaluator metricStyle must be pytrec`);
    const resultScorers = scorerListFrom(options?.scorers);
    const missingScorers = listMissing(expectedScorers, resultScorers);
    if (missingScorers.length > 0) failures.push(`${dataset} result scorer filter missing: ${missingScorers.join(",")}`);
  }

  return protocolResult(failures, "TS hybrid protocol manifest valid");
}

function pytrecProtocolCheck(payload, expectedDatasets) {
  const failures = protocolVersionFailures(payload, "pytrec");
  if (payload.kind !== "bb25-pytrec-eval") failures.push("pytrec manifest kind is not bb25-pytrec-eval");
  if (payload.environment?.pytrecEvalVersion === undefined) failures.push("missing pytrec_eval version provenance");
  const inputDatasets = Array.isArray(payload.inputs?.datasets) ? payload.inputs.datasets.map(String) : [];
  const missingInputs = listMissing(expectedDatasets, inputDatasets);
  if (missingInputs.length > 0) failures.push(`pytrec inputs missing datasets: ${missingInputs.join(",")}`);

  const datasetInputs = Array.isArray(payload.datasetInputs) ? payload.datasetInputs : [];
  for (const entry of datasetInputs) {
    const dataset = String(entry?.dataset ?? "unknown");
    if (!fileRecordOk(entry?.qrels)) failures.push(`${dataset} pytrec qrels file is not manifest-backed`);
    const runFiles = Array.isArray(entry?.runFiles) ? entry.runFiles : [];
    if (runFiles.length === 0 || runFiles.some((file) => !fileRecordOk(file))) {
      failures.push(`${dataset} pytrec run files are not manifest-backed`);
    }
  }

  return protocolResult(failures, "pytrec evaluator protocol manifest valid");
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const repoRoot = resolve(".");
  const root = resolve(args.root ?? "reference-results");
  const out = args.out === undefined ? null : resolve(args.out);
  const profiles = parseProfiles(args.profile);
  const datasets = parseCsv(args.datasets ?? "arguana,fiqa,nfcorpus,scidocs,scifact");
  const sparseDatasets = parseCsv(args["sparse-datasets"] ?? "nfcorpus,scifact");
  const datasetTolerance = Number(args["dataset-tolerance"] ?? args["dataset-tolerance-points"] ?? "0.50");
  if (!Number.isFinite(datasetTolerance) || datasetTolerance < 0) {
    throw new Error(`invalid dataset tolerance: ${datasetTolerance}`);
  }
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
    addJsonCheck("manifests/beir-jsonl-hybrid-export.json", (payload) => hybridExportProtocolCheck(payload, datasets), "hybrid export protocol provenance");
    addJsonCheck("manifests/ts-hybrid-beir-internal.json", commandManifestCheck, "TS internal command manifest valid");
    addJsonCheck("manifests/ts-hybrid-beir-internal.json", (payload) => tsHybridProtocolCheck(payload, datasets), "TS hybrid protocol manifest valid");
    addJsonCheck("manifests/ts-hybrid-beir-pytrec.json", (payload) => datasetCoverageCheck(payload.inputs ?? payload, datasets), "pytrec manifest dataset coverage");
    addJsonCheck("manifests/ts-hybrid-beir-pytrec.json", commandManifestCheck, "pytrec command manifest valid");
    addJsonCheck("manifests/ts-hybrid-beir-pytrec.json", (payload) => pytrecProtocolCheck(payload, datasets), "pytrec protocol manifest valid");
    addJsonCheck("manifests/baseline-parity-runner.json", commandManifestCheck, "baseline runner commands valid");
    addJsonCheck("ts/baseline-parity.json", parityCheck, "baseline parity result valid");
  }

  if (profiles.includes("hybrid-strict")) {
    addJsonCheck(
      "ts/baseline-parity.json",
      (payload) => datasetParityCheck(payload, datasets, datasetTolerance),
      "baseline dataset-level parity valid",
    );
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
    datasetTolerance,
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
