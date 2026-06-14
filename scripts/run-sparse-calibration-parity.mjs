#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_DATASETS = "nfcorpus,scifact";
const DEFAULT_RANKING_METHODS = "bm25,bayesian_no_base_rate,bayesian_fitted_split";
const DEFAULT_CALIBRATION_METHODS = "bayesian_no_base_rate,bayesian,bayesian_fitted_split";
const DEFAULT_RANKING_REFERENCE_MAP =
  "bm25=Raw BM25,bayesian_no_base_rate=Bayesian (auto),bayesian_fitted_split=Bayesian (batch fit)";
const DEFAULT_CALIBRATION_REFERENCE_MAP =
  "bayesian_no_base_rate=Bayesian (no base rate),bayesian=Bayesian (base_rate=auto),bayesian_fitted_split=Batch fit (base_rate=auto)";

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

function splitCsv(raw) {
  if (raw === undefined || raw.trim() === "") return [];
  return raw
    .split(",")
    .map((part) => part.trim())
    .filter(Boolean);
}

function run(command, options = {}) {
  const startedAt = new Date().toISOString();
  const start = performance.now();
  const result = spawnSync(command[0], command.slice(1), {
    encoding: "utf8",
    ...options,
  });
  return {
    command,
    cwd: options.cwd ?? process.cwd(),
    startedAt,
    finishedAt: new Date().toISOString(),
    durationSeconds: Number(((performance.now() - start) / 1000).toFixed(6)),
    returncode: result.status ?? 1,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    error: result.error === undefined ? null : String(result.error.message ?? result.error),
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

function fileManifest(path) {
  const fullPath = resolve(path);
  if (!existsSync(fullPath)) return { path: fullPath, exists: false };
  const data = readFileSync(fullPath);
  return {
    path: fullPath,
    exists: true,
    bytes: statSync(fullPath).size,
    sha256: createHash("sha256").update(data).digest("hex"),
  };
}

function addIfDefined(command, flag, value) {
  if (value !== undefined) command.push(flag, value);
}

function recordCommand(name, command, repoRoot, dryRun) {
  if (dryRun) {
    return { name, command, cwd: repoRoot, dryRun: true, returncode: 0, stdout: "", stderr: "" };
  }
  return { name, ...run(command, { cwd: repoRoot }) };
}

function outputPath(args, outDir, name) {
  return resolve(args[name] ?? `${outDir}/${name}.json`);
}

function datasetTemplate(value, dataset) {
  return value.replaceAll("{dataset}", dataset);
}

function comparisonCommand(repoRoot, args, options) {
  const command = [
    "node",
    resolve(repoRoot, "scripts/check-bench-json.mjs"),
    "--reference",
    options.reference,
    "--actual",
    options.actual,
    "--methods",
    options.methods,
    "--metric",
    options.metric,
    "--metric-scale",
    "unit",
    "--tolerance",
    options.tolerance,
    "--datasets",
    options.datasets,
    "--out",
    options.out,
    "--manifest-out",
    options.manifestOut,
  ];
  addIfDefined(command, "--reference-method-map", options.referenceMethodMap);
  addIfDefined(command, "--actual-method-map", options.actualMethodMap);
  return command;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.help !== undefined || args.h !== undefined) {
    process.stdout.write(
      "usage: run-sparse-calibration-parity --root <beir-jsonl-sparse-dir> [--reference-ranking reference-results/python/sparse-benchmark.json] [--reference-calibration reference-results/python/base-rate.json] [--dry-run]\n",
    );
    process.exit(0);
  }
  if (args.root === undefined) {
    throw new Error("usage: run-sparse-calibration-parity --root <beir-jsonl-sparse-dir> [--dry-run]");
  }

  const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  const root = resolve(args.root);
  const datasets = args.datasets ?? DEFAULT_DATASETS;
  const outDir = resolve(args["out-dir"] ?? "reference-results/ts");
  const manifestDir = resolve(args["manifest-dir"] ?? "reference-results/manifests");
  const tsOut = resolve(args["ts-out"] ?? `${outDir}/sparse-calibration-ts.json`);
  const tsManifest = resolve(args["ts-manifest"] ?? `${manifestDir}/ts-sparse-calibration.json`);
  const runnerManifest = resolve(args["runner-manifest-out"] ?? `${manifestDir}/sparse-calibration-parity-runner.json`);
  const referenceRanking = resolve(args["reference-ranking"] ?? "reference-results/python/sparse-benchmark.json");
  const referenceCalibration = resolve(args["reference-calibration"] ?? "reference-results/python/base-rate.json");
  const dryRun = args["dry-run"] !== undefined;
  const datasetNames = splitCsv(datasets);

  mkdirSync(outDir, { recursive: true });
  mkdirSync(manifestDir, { recursive: true });

  const commands = [];
  let failed = false;
  const splitOutputs = [];
  const fitSplitSource = args["fit-split-source"] ?? "numpy";
  const generatedSplitPattern = resolve(args["fit-split-file"] ?? `${manifestDir}/fit-splits/{dataset}.json`);
  if (fitSplitSource !== "none" && args["fit-split-file"] === undefined) {
    const python = args.python ?? "python3";
    for (const dataset of datasetNames) {
      const splitOut = datasetTemplate(generatedSplitPattern, dataset);
      const splitCommand = [
        python,
        resolve(repoRoot, "scripts/write-numpy-query-split.py"),
        "--queries",
        resolve(root, dataset, "queries.jsonl"),
        "--out",
        splitOut,
        "--dataset",
        dataset,
        "--train-ratio",
        args["fit-train-ratio"] ?? "0.5",
        "--seed",
        args["fit-split-seed"] ?? "42",
      ];
      const record = recordCommand(`fit-split:${dataset}`, splitCommand, repoRoot, dryRun);
      commands.push(record);
      splitOutputs.push({ dataset, out: splitOut, returncode: record.returncode });
      if (record.returncode !== 0) failed = true;
    }
  }

  const benchCommand = [
    "node",
    resolve(repoRoot, "scripts/run-beir-jsonl-bench.mjs"),
    "--root",
    root,
    "--datasets",
    datasets,
    "--bm25-method",
    args["bm25-method"] ?? "lucene",
    "--metric-style",
    args["metric-style"] ?? "python-reference",
    "--alpha",
    args.alpha ?? "auto",
    "--beta",
    args.beta ?? "auto",
    "--base-rate",
    args["base-rate"] ?? "auto",
    "--base-rate-method",
    args["base-rate-method"] ?? "percentile",
    "--fit-split",
    "--fit-train-ratio",
    args["fit-train-ratio"] ?? "0.5",
    "--fit-split-seed",
    args["fit-split-seed"] ?? "42",
    "--calibration",
    "--cutoffs",
    args.cutoffs ?? "10",
    "--out",
    tsOut,
    "--manifest-out",
    tsManifest,
  ];
  if (fitSplitSource !== "none") {
    benchCommand.push("--fit-split-file", generatedSplitPattern);
  }
  addIfDefined(benchCommand, "--bb25", args.bb25);
  addIfDefined(benchCommand, "--doc-terms", args["doc-terms"]);
  addIfDefined(benchCommand, "--query-terms", args["query-terms"]);
  addIfDefined(benchCommand, "--calibration-bins", args["calibration-bins"]);
  addIfDefined(benchCommand, "--base-rate-sample-size", args["base-rate-sample-size"]);
  addIfDefined(benchCommand, "--base-rate-seed", args["base-rate-seed"]);

  if (!failed || args["continue-after-parity-failure"] !== undefined) {
    const benchRecord = recordCommand("ts-sparse-calibration-bench", benchCommand, repoRoot, dryRun);
    commands.push(benchRecord);
    if (benchRecord.returncode !== 0) failed = true;
  }

  const rankingOutputs = [];
  if (!failed && args["skip-ranking-parity"] === undefined) {
    const rankingMetrics = splitCsv(args["ranking-metrics"] ?? "ndcg@10,map@10");
    for (const metric of rankingMetrics) {
      const safeMetric = metric.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_+|_+$/g, "").toLowerCase();
      const out = outputPath(args, outDir, `sparse-ranking-${safeMetric}-parity`);
      const manifestOut = outputPath(args, manifestDir, `sparse-ranking-${safeMetric}-parity`);
      const command = comparisonCommand(repoRoot, args, {
        reference: referenceRanking,
        actual: tsOut,
        methods: args["ranking-methods"] ?? DEFAULT_RANKING_METHODS,
        metric,
        tolerance: args["ranking-tolerance"] ?? "0.005",
        datasets,
        out,
        manifestOut,
        referenceMethodMap: args["ranking-reference-method-map"] ?? DEFAULT_RANKING_REFERENCE_MAP,
        actualMethodMap: args["ranking-actual-method-map"],
      });
      const record = recordCommand(`sparse-ranking-parity:${metric}`, command, repoRoot, dryRun);
      commands.push(record);
      rankingOutputs.push({ metric, out, manifestOut, returncode: record.returncode });
      if (record.returncode !== 0) failed = true;
    }
  }

  const calibrationOutputs = [];
  if (!failed || args["continue-after-parity-failure"] !== undefined) {
    if (args["skip-calibration-parity"] === undefined) {
      const calibrationMetrics = splitCsv(args["calibration-metrics"] ?? "ece,brier");
      for (const metric of calibrationMetrics) {
        const safeMetric = metric.replace(/[^a-zA-Z0-9]+/g, "_").replace(/^_+|_+$/g, "").toLowerCase();
        const out = outputPath(args, outDir, `sparse-calibration-${safeMetric}-parity`);
        const manifestOut = outputPath(args, manifestDir, `sparse-calibration-${safeMetric}-parity`);
        const command = comparisonCommand(repoRoot, args, {
          reference: referenceCalibration,
          actual: tsOut,
          methods: args["calibration-methods"] ?? DEFAULT_CALIBRATION_METHODS,
          metric,
          tolerance: args["calibration-tolerance"] ?? "0.005",
          datasets,
          out,
          manifestOut,
          referenceMethodMap: args["calibration-reference-method-map"] ?? DEFAULT_CALIBRATION_REFERENCE_MAP,
          actualMethodMap: args["calibration-actual-method-map"],
        });
        const record = recordCommand(`sparse-calibration-parity:${metric}`, command, repoRoot, dryRun);
        commands.push(record);
        calibrationOutputs.push({ metric, out, manifestOut, returncode: record.returncode });
        if (record.returncode !== 0) failed = true;
      }
    }

    if (args["skip-calibration-gate"] === undefined) {
      const gateOut = resolve(args["gate-out"] ?? `${outDir}/sparse-calibration-gate.json`);
      const gateManifest = resolve(args["gate-manifest"] ?? `${manifestDir}/sparse-calibration-gate.json`);
      const gateCommand = [
        "node",
        resolve(repoRoot, "scripts/check-calibration-gate.mjs"),
        "--actual",
        tsOut,
        "--datasets",
        datasets,
        "--baseline",
        args.baseline ?? "bayesian_no_base_rate",
        "--calibrated",
        args.calibrated ?? "bayesian",
        "--fitted",
        args.fitted ?? "bayesian_fitted_split",
        "--metric",
        args["gate-metric"] ?? "ece",
        "--min-reduction",
        args["min-reduction"] ?? "0.50",
        "--fitted-max",
        args["fitted-max"] ?? "0.02",
        "--out",
        gateOut,
        "--manifest-out",
        gateManifest,
      ];
      const gateRecord = recordCommand("sparse-calibration-gate", gateCommand, repoRoot, dryRun);
      commands.push(gateRecord);
      calibrationOutputs.push({ metric: args["gate-metric"] ?? "ece", out: gateOut, manifestOut: gateManifest, returncode: gateRecord.returncode, gate: true });
      if (gateRecord.returncode !== 0) failed = true;
    }
  }

  const manifest = {
    generatedAt: new Date().toISOString(),
    kind: "bb25-sparse-calibration-parity-runner",
    dryRun,
    repo: {
      root: repoRoot,
      git: gitInfo(repoRoot),
      node: process.version,
      platform: process.platform,
      arch: process.arch,
    },
    options: {
      root,
      datasets: datasetNames,
      referenceRanking,
      referenceCalibration,
      outDir,
      manifestDir,
      bm25Method: args["bm25-method"] ?? "lucene",
      metricStyle: args["metric-style"] ?? "python-reference",
      alpha: args.alpha ?? "auto",
      beta: args.beta ?? "auto",
      baseRate: args["base-rate"] ?? "auto",
      baseRateMethod: args["base-rate-method"] ?? "percentile",
      fitTrainRatio: Number(args["fit-train-ratio"] ?? "0.5"),
      fitSplitSeed: Number(args["fit-split-seed"] ?? "42"),
      fitSplitSource,
      fitSplitFile: fitSplitSource === "none" ? null : generatedSplitPattern,
      rankingMethods: splitCsv(args["ranking-methods"] ?? DEFAULT_RANKING_METHODS),
      calibrationMethods: splitCsv(args["calibration-methods"] ?? DEFAULT_CALIBRATION_METHODS),
      rankingMetrics: splitCsv(args["ranking-metrics"] ?? "ndcg@10,map@10"),
      calibrationMetrics: splitCsv(args["calibration-metrics"] ?? "ece,brier"),
      rankingTolerance: Number(args["ranking-tolerance"] ?? "0.005"),
      calibrationTolerance: Number(args["calibration-tolerance"] ?? "0.005"),
      minReduction: Number(args["min-reduction"] ?? "0.50"),
      fittedMax: Number(args["fitted-max"] ?? "0.02"),
    },
    files: {
      referenceRanking: fileManifest(referenceRanking),
      referenceCalibration: fileManifest(referenceCalibration),
      ts: fileManifest(tsOut),
      tsManifest: fileManifest(tsManifest),
      fitSplits: Object.fromEntries(splitOutputs.map((row) => [row.dataset, fileManifest(row.out)])),
    },
    outputs: {
      runnerManifest,
      ts: tsOut,
      tsManifest,
      fitSplits: splitOutputs,
      ranking: rankingOutputs,
      calibration: calibrationOutputs,
    },
    commands,
    passed: !failed,
  };
  mkdirSync(dirname(runnerManifest), { recursive: true });
  writeFileSync(runnerManifest, JSON.stringify(manifest, null, 2) + "\n", "utf8");

  process.stdout.write(
    JSON.stringify(
      {
        manifest: runnerManifest,
        passed: !failed,
        actual: tsOut,
        ranking: rankingOutputs,
        calibration: calibrationOutputs,
      },
      null,
      2,
    ) + "\n",
  );
  if (failed) process.exitCode = 1;
}

main();
