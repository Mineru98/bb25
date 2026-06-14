#!/usr/bin/env node
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";

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

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function fileManifest(path) {
  const data = readFileSync(path);
  return {
    path: resolve(path),
    bytes: statSync(path).size,
    sha256: createHash("sha256").update(data).digest("hex"),
  };
}

function metricValue(row, metric) {
  const value = row?.[metric];
  if (value === undefined || value === null) return null;
  const n = Number(value);
  if (!Number.isFinite(n)) throw new Error(`invalid ${metric} value: ${value}`);
  return n;
}

function addCalibration(rows, dataset, row) {
  if (row === null || typeof row !== "object") return;
  const scorer = String(row.scorer ?? "");
  if (scorer === "") return;
  let datasetRows = rows.get(dataset);
  if (datasetRows === undefined) {
    datasetRows = new Map();
    rows.set(dataset, datasetRows);
  }
  datasetRows.set(scorer, row);
}

function extractCalibrationRows(payload) {
  const rows = new Map();

  if (Array.isArray(payload.calibration)) {
    for (const row of payload.calibration) addCalibration(rows, "average", row);
  }

  if (Array.isArray(payload.runs)) {
    for (const run of payload.runs) {
      const dataset = String(run.dataset ?? "dataset");
      if (Array.isArray(run.calibration)) {
        for (const row of run.calibration) addCalibration(rows, dataset, row);
      }
    }
  }

  for (const [dataset, methods] of Object.entries(payload)) {
    if (methods === null || typeof methods !== "object" || Array.isArray(methods)) continue;
    for (const [scorer, row] of Object.entries(methods)) {
      if (row !== null && typeof row === "object" && !Array.isArray(row)) {
        addCalibration(rows, dataset, { scorer, ...row });
      }
    }
  }

  return rows;
}

function selectedDatasets(rows, requested) {
  if (requested.length > 0) return requested;
  const nonAverage = [...rows.keys()].filter((dataset) => dataset !== "average");
  return nonAverage.length > 0 ? nonAverage : [...rows.keys()];
}

function checkCalibrationGate(args) {
  if (args.actual === undefined) {
    throw new Error(
      "usage: check-calibration-gate --actual results.json [--datasets nfcorpus,scifact] [--baseline bayesian_no_base_rate] [--calibrated bayesian] [--fitted bayesian_fitted_split] [--metric ece] [--min-reduction 0.50] [--fitted-max 0.02]",
    );
  }

  const metric = args.metric ?? "ece";
  const baseline = args.baseline ?? "bayesian_no_base_rate";
  const calibrated = args.calibrated ?? "bayesian";
  const fitted = args.fitted ?? "bayesian_fitted_split";
  const minReduction = args["min-reduction"] === undefined ? 0.5 : Number(args["min-reduction"]);
  const fittedMax = args["fitted-max"] === undefined ? 0.02 : Number(args["fitted-max"]);
  const requiredDatasets = parseCsv(args.datasets);
  const rows = extractCalibrationRows(readJson(args.actual));
  const datasets = selectedDatasets(rows, requiredDatasets);
  const checks = [];

  for (const dataset of datasets) {
    const datasetRows = rows.get(dataset);
    if (datasetRows === undefined) {
      checks.push({
        dataset,
        kind: "missing_dataset",
        passed: false,
        message: `missing dataset ${dataset}`,
      });
      continue;
    }

    const baselineRow = datasetRows.get(baseline);
    const calibratedRow = datasetRows.get(calibrated);
    const baselineValue = metricValue(baselineRow, metric);
    const calibratedValue = metricValue(calibratedRow, metric);
    if (baselineValue === null || calibratedValue === null) {
      checks.push({
        dataset,
        kind: "reduction",
        passed: false,
        baseline,
        calibrated,
        metric,
        message: `missing ${metric} for ${baselineValue === null ? baseline : calibrated}`,
      });
    } else {
      const reduction = baselineValue <= 0 ? (calibratedValue <= baselineValue ? 1 : 0) : (baselineValue - calibratedValue) / baselineValue;
      checks.push({
        dataset,
        kind: "reduction",
        passed: reduction >= minReduction,
        baseline,
        calibrated,
        metric,
        baselineValue,
        calibratedValue,
        reduction,
        minReduction,
        message: `${dataset} ${metric} reduction ${(reduction * 100).toFixed(2)}%`,
      });
    }

    if (fitted !== "none") {
      const fittedRow = datasetRows.get(fitted);
      const fittedValue = metricValue(fittedRow, metric);
      checks.push({
        dataset,
        kind: "fitted_threshold",
        passed: fittedValue !== null && fittedValue <= fittedMax,
        fitted,
        metric,
        fittedValue,
        fittedMax,
        message:
          fittedValue === null
            ? `missing ${metric} for ${fitted}`
            : `${dataset} ${fitted} ${metric}=${fittedValue.toFixed(6)} <= ${fittedMax}`,
      });
    }
  }

  const failures = checks.filter((check) => !check.passed);
  return {
    generatedAt: new Date().toISOString(),
    kind: "bb25-calibration-gate",
    passed: failures.length === 0,
    actual: resolve(args.actual),
    options: {
      datasets,
      requiredDatasets,
      metric,
      baseline,
      calibrated,
      fitted,
      minReduction,
      fittedMax,
    },
    checks,
    failures,
  };
}

const args = parseArgs(process.argv.slice(2));
if (args.help !== undefined || args.h !== undefined) {
  process.stdout.write(
    "usage: check-calibration-gate --actual results.json [--datasets nfcorpus,scifact] [--baseline bayesian_no_base_rate] [--calibrated bayesian] [--fitted bayesian_fitted_split] [--metric ece] [--min-reduction 0.50] [--fitted-max 0.02] [--out result.json] [--manifest-out manifest.json]\n",
  );
  process.exit(0);
}

const result = checkCalibrationGate(args);
if (args.out !== undefined) {
  mkdirSync(dirname(args.out), { recursive: true });
  writeFileSync(args.out, JSON.stringify(result, null, 2) + "\n", "utf8");
}
if (args["manifest-out"] !== undefined) {
  const manifest = {
    generatedAt: result.generatedAt,
    kind: "bb25-calibration-gate-manifest",
    command: process.argv.slice(1),
    inputs: {
      actual: fileManifest(args.actual),
    },
    summary: {
      passed: result.passed,
      checks: result.checks.length,
      failures: result.failures.map((failure) => failure.message),
    },
  };
  mkdirSync(dirname(args["manifest-out"]), { recursive: true });
  writeFileSync(args["manifest-out"], JSON.stringify(manifest, null, 2) + "\n", "utf8");
}

for (const check of result.checks) {
  process.stdout.write(`${check.passed ? "PASS" : "FAIL"}\t${check.kind}\t${check.message}\n`);
}
if (!result.passed) process.exitCode = 1;
