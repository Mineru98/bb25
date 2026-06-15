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

function normalizeMetricName(metric) {
  const lower = metric.toLowerCase();
  const match = lower.match(/^(ndcg|map|mrr|recall)@?(\d+)$/) ?? lower.match(/^(ndcg|map|mrr|recall)_cut_?(\d+)$/);
  if (match !== null) {
    return `${match[1]}@${match[2]}`;
  }
  return lower;
}

function metricAliases(metric) {
  const normalized = normalizeMetricName(metric);
  if (normalized === "ece") return ["ece", "ECE"];
  if (normalized === "brier") return ["brier", "Brier", "brierScore"];
  if (normalized === "samples") return ["samples", "Samples"];
  if (normalized === "bins") return ["bins", "Bins"];
  const [, family, cutoff] = normalized.match(/^(ndcg|map|mrr|recall)@(\d+)$/) ?? [];
  if (family === undefined || cutoff === undefined) {
    return [metric, normalized];
  }
  const title = family === "ndcg" ? "NDCG" : family === "map" ? "MAP" : family === "mrr" ? "MRR" : "Recall";
  const aliases = [normalized, `${title}@${cutoff}`, `${family}_cut_${cutoff}`, `${family}${cutoff}`];
  if (family === "map") aliases.push("MAP", "map");
  return aliases;
}

function defaultMetricScale(metric) {
  const normalized = normalizeMetricName(metric);
  return normalized === "ece" || normalized === "brier" || normalized === "samples" || normalized === "bins"
    ? "unit"
    : "points";
}

function metricUnitLabel(scale) {
  return scale === "points" ? "points" : "units";
}

function normalizeMetricValue(value, scale) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    throw new Error(`invalid metric value: ${value}`);
  }
  if (scale === "points") {
    return Math.abs(n) <= 1.0000001 ? n * 100.0 : n;
  }
  if (scale === "unit") {
    return n;
  }
  throw new Error(`invalid metric scale: ${scale}`);
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

function addMetric(rows, dataset, scorer, metrics) {
  if (metrics === undefined || metrics === null) return;
  let scorerMap = rows.get(scorer);
  if (scorerMap === undefined) {
    scorerMap = new Map();
    rows.set(scorer, scorerMap);
  }
  scorerMap.set(dataset, metrics);
}

function extractRows(payload) {
  const rows = new Map();

  if (Array.isArray(payload.results)) {
    for (const row of payload.results) {
      addMetric(rows, "average", String(row.scorer), row.metrics);
    }
  }

  if (Array.isArray(payload.calibration)) {
    for (const row of payload.calibration) {
      addMetric(rows, "average", String(row.scorer), {
        ece: row.ece,
        brier: row.brier,
        samples: row.samples,
        bins: row.bins,
      });
    }
  }

  if (Array.isArray(payload.average)) {
    for (const row of payload.average) {
      addMetric(rows, "average", String(row.scorer), row.metrics);
    }
  }

  if (Array.isArray(payload.runs)) {
    for (const run of payload.runs) {
      const dataset = String(run.dataset ?? "dataset");
      if (Array.isArray(run.results)) {
        for (const row of run.results) {
          addMetric(rows, dataset, String(row.scorer), row.metrics);
        }
      }
      if (Array.isArray(run.calibration)) {
        for (const row of run.calibration) {
          addMetric(rows, dataset, String(row.scorer), {
            ece: row.ece,
            brier: row.brier,
            samples: row.samples,
            bins: row.bins,
          });
        }
      }
    }
  }

  const pythonReferencePayload =
    payload.results !== null &&
    typeof payload.results === "object" &&
    !Array.isArray(payload.results) &&
    !Array.isArray(payload.runs)
      ? payload.results
      : payload;
  const looksLikePythonReference =
    !Array.isArray(pythonReferencePayload.results) &&
    !Array.isArray(pythonReferencePayload.runs) &&
    Object.values(pythonReferencePayload).some((value) => value !== null && typeof value === "object" && !Array.isArray(value));
  if (looksLikePythonReference) {
    for (const [dataset, methods] of Object.entries(pythonReferencePayload)) {
      if (methods === null || typeof methods !== "object" || Array.isArray(methods)) continue;
      for (const [method, metrics] of Object.entries(methods)) {
        if (metrics !== null && typeof metrics === "object" && !Array.isArray(metrics)) {
          addMetric(rows, dataset, method, metrics);
        }
      }
    }
  }

  return rows;
}

function parseMethodMap(raw) {
  const out = new Map();
  if (raw === undefined) return out;
  for (const pair of raw.split(",")) {
    if (pair.trim() === "") continue;
    const eq = pair.indexOf("=");
    if (eq <= 0) {
      throw new Error(`invalid method-map entry: ${pair}`);
    }
    const left = pair.slice(0, eq);
    const right = pair.slice(eq + 1);
    out.set(left.trim(), right.trim());
  }
  return out;
}

function getScorer(rows, method, explicitName = undefined) {
  const candidates = [
    explicitName,
    method,
    method.toLowerCase(),
    method.toUpperCase(),
    method.replace(/-/g, "_").toLowerCase(),
  ].filter(Boolean);
  for (const candidate of candidates) {
    if (rows.has(candidate)) {
      return { name: candidate, datasets: rows.get(candidate) };
    }
  }
  return null;
}

function getMetric(metrics, metric, scale) {
  for (const alias of metricAliases(metric)) {
    if (Object.prototype.hasOwnProperty.call(metrics, alias)) {
      return normalizeMetricValue(metrics[alias], scale);
    }
  }
  return null;
}

function averageMetric(datasetMetrics, metric, scale) {
  const values = [];
  for (const metrics of datasetMetrics.values()) {
    const value = getMetric(metrics, metric, scale);
    if (value !== null) values.push(value);
  }
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function parseCsv(raw) {
  if (raw === undefined) return [];
  return raw.split(",").map((part) => part.trim()).filter(Boolean);
}

function parseGateMode(raw, flagName) {
  const mode = raw ?? "off";
  if (mode === "true") return "strict";
  if (mode === "off" || mode === "warn" || mode === "strict") return mode;
  throw new Error(`${flagName} must be off, warn, or strict, got ${mode}`);
}

function methodFailureClass(method, kind, metric = "") {
  const normalized = method.toLowerCase().replace(/[-\s]+/g, "_");
  const normalizedMetric = normalizeMetricName(metric);
  if (normalizedMetric === "ece" || normalizedMetric === "brier") return "calibration";
  if (kind === "missing_metric") return "evaluator";
  if (normalized === "bm25" || normalized.includes("raw_bm25")) return "BM25/tokenizer";
  if (normalized === "dense" || normalized.includes("dense")) return "dense embedding";
  if (normalized === "convex" || normalized === "rrf" || normalized.includes("fusion_baseline")) return "candidate protocol";
  if (normalized.includes("bayesian") || normalized.includes("balanced") || normalized.includes("logodds")) {
    return "Bayesian fusion";
  }
  return "candidate protocol";
}

function failureMessage(failure) {
  switch (failure.kind) {
    case "missing_reference_method":
      return `missing reference method ${failure.method}`;
    case "missing_actual_method":
      return `missing actual method ${failure.method}`;
    case "missing_metric":
      return `${failure.side} method ${failure.method} has no ${failure.metric}`;
    case "missing_dataset":
      return `${failure.side} method ${failure.method} missing dataset ${failure.dataset}`;
    case "diff":
      return `${failure.method} ${failure.metric} diff ${failure.diff.toFixed(6)} ${failure.unitLabel ?? "points"} > ${failure.tolerance}`;
    case "dataset_diff":
      return `${failure.method} ${failure.dataset} ${failure.metric} dataset diff ${failure.diff.toFixed(6)} ${failure.unitLabel ?? "points"} > ${failure.tolerance}`;
    default:
      return failure.message ?? JSON.stringify(failure);
  }
}

function addFailure(failures, failure) {
  failures.push({
    classification: methodFailureClass(failure.method ?? "", failure.kind, failure.metric ?? ""),
    ...failure,
  });
}

function datasetValue(datasets, dataset, metric, scale) {
  const metrics = datasets.get(dataset);
  return metrics === undefined ? null : getMetric(metrics, metric, scale);
}

function compare(args) {
  if (args.reference === undefined || args.actual === undefined) {
    throw new Error(
      "usage: check-bench-json --reference ref.json --actual actual.json [--methods BM25,Dense,Convex,RRF] [--metric ndcg@10|ece|brier] [--metric-scale points|unit] [--tolerance 0.005] [--tolerance-points 0.50] [--datasets arguana,fiqa,...] [--dataset-gate off|warn|strict] [--dataset-tolerance-points 0.50] [--manifest-out manifest.json]",
    );
  }

  const methods = (args.methods ?? "BM25,Dense,Convex,RRF").split(",").map((s) => s.trim()).filter(Boolean);
  const metric = args.metric ?? "ndcg@10";
  const metricScale = args["metric-scale"] ?? defaultMetricScale(metric);
  const tolerance =
    args.tolerance !== undefined
      ? Number(args.tolerance)
      : args["tolerance-points"] === undefined
        ? metricScale === "points"
          ? 0.5
          : 0.005
        : Number(args["tolerance-points"]);
  const unitLabel = metricUnitLabel(metricScale);
  const datasetGateMode = parseGateMode(args["dataset-gate"], "--dataset-gate");
  const datasetTolerance =
    args["dataset-tolerance"] !== undefined
      ? Number(args["dataset-tolerance"])
      : args["dataset-tolerance-points"] !== undefined
        ? Number(args["dataset-tolerance-points"])
        : tolerance;
  if (!Number.isFinite(tolerance) || tolerance < 0) {
    throw new Error(`invalid tolerance: ${tolerance}`);
  }
  if (!Number.isFinite(datasetTolerance) || datasetTolerance < 0) {
    throw new Error(`invalid dataset tolerance: ${datasetTolerance}`);
  }
  const requiredDatasets = parseCsv(args.datasets);
  const refMap = parseMethodMap(args["reference-method-map"]);
  const actualMap = parseMethodMap(args["actual-method-map"]);
  const refRows = extractRows(readJson(args.reference));
  const actualRows = extractRows(readJson(args.actual));

  const failures = [];
  const warnings = [];
  const lines = [];
  const comparisons = [];
  for (const method of methods) {
    const ref = getScorer(refRows, method, refMap.get(method));
    const actual = getScorer(actualRows, method, actualMap.get(method));
    if (ref === null) {
      addFailure(failures, { kind: "missing_reference_method", method });
      continue;
    }
    if (actual === null) {
      addFailure(failures, { kind: "missing_actual_method", method });
      continue;
    }

    const refAvg = averageMetric(ref.datasets, metric, metricScale);
    const actualAvg = averageMetric(actual.datasets, metric, metricScale);
    if (refAvg === null) {
      addFailure(failures, { kind: "missing_metric", method, side: "reference", metric });
      continue;
    }
    if (actualAvg === null) {
      addFailure(failures, { kind: "missing_metric", method, side: "actual", metric });
      continue;
    }
    const diff = Math.abs(actualAvg - refAvg);
    lines.push(`${method}\tref=${refAvg.toFixed(6)}\tactual=${actualAvg.toFixed(6)}\tdiff=${diff.toFixed(6)} ${unitLabel}`);
    const row = {
      method,
      referenceName: ref.name,
      actualName: actual.name,
      metric,
      metricScale,
      referenceAverage: refAvg,
      actualAverage: actualAvg,
      diff,
      tolerance,
      datasets: [],
    };
    if (diff > tolerance) {
      addFailure(failures, {
        kind: "diff",
        method,
        metric,
        metricScale,
        unitLabel,
        reference: refAvg,
        actual: actualAvg,
        diff,
        tolerance,
      });
    }

    const datasetNames =
      requiredDatasets.length > 0
        ? requiredDatasets
        : [...ref.datasets.keys()].filter((dataset) => actual.datasets.has(dataset));
    for (const dataset of datasetNames) {
      if (!ref.datasets.has(dataset)) {
        addFailure(failures, { kind: "missing_dataset", method, side: "reference", dataset });
        continue;
      }
      if (!actual.datasets.has(dataset)) {
        addFailure(failures, { kind: "missing_dataset", method, side: "actual", dataset });
        continue;
      }
      const refValue = datasetValue(ref.datasets, dataset, metric, metricScale);
      const actualValue = datasetValue(actual.datasets, dataset, metric, metricScale);
      if (refValue === null || actualValue === null) continue;
      const datasetDiff = Math.abs(actualValue - refValue);
      lines.push(`  ${dataset}\tref=${refValue.toFixed(6)}\tactual=${actualValue.toFixed(6)}\tdiff=${datasetDiff.toFixed(6)} ${unitLabel}`);
      row.datasets.push({ dataset, reference: refValue, actual: actualValue, diff: datasetDiff });
      if (datasetGateMode !== "off" && datasetDiff > datasetTolerance) {
        const violation = {
          kind: "dataset_diff",
          method,
          dataset,
          metric,
          metricScale,
          unitLabel,
          reference: refValue,
          actual: actualValue,
          diff: datasetDiff,
          tolerance: datasetTolerance,
        };
        if (datasetGateMode === "strict") {
          addFailure(failures, violation);
        } else {
          warnings.push({
            classification: methodFailureClass(method, violation.kind, metric),
            ...violation,
          });
        }
      }
    }
    comparisons.push(row);
  }

  const classifications = {};
  for (const failure of failures) {
    classifications[failure.classification] = (classifications[failure.classification] ?? 0) + 1;
  }
  const warningClassifications = {};
  for (const warning of warnings) {
    warningClassifications[warning.classification] = (warningClassifications[warning.classification] ?? 0) + 1;
  }
  const datasetViolations = [
    ...failures.filter((failure) => failure.kind === "dataset_diff"),
    ...warnings.filter((warning) => warning.kind === "dataset_diff"),
  ];

  return {
    passed: failures.length === 0,
    methods,
    metric,
    metricScale,
    tolerancePoints: metricScale === "points" ? tolerance : null,
    toleranceUnits: metricScale === "unit" ? tolerance : null,
    tolerance,
    requiredDatasets,
    datasetGate: {
      mode: datasetGateMode,
      metric,
      metricScale,
      tolerancePoints: metricScale === "points" ? datasetTolerance : null,
      toleranceUnits: metricScale === "unit" ? datasetTolerance : null,
      tolerance: datasetTolerance,
      passed: datasetViolations.length === 0,
      violations: datasetViolations,
    },
    failures,
    warnings,
    classifications,
    warningClassifications,
    comparisons,
    lines,
  };
}

const args = parseArgs(process.argv.slice(2));
if (args.help !== undefined || args.h !== undefined) {
  process.stdout.write(
    "usage: check-bench-json --reference ref.json --actual actual.json [--methods BM25,Dense,Convex,RRF] [--metric ndcg@10|ece|brier] [--metric-scale points|unit] [--tolerance 0.005] [--tolerance-points 0.50] [--datasets arguana,fiqa,...] [--dataset-gate off|warn|strict] [--dataset-tolerance-points 0.50] [--out result.json] [--manifest-out manifest.json]\n",
  );
  process.exit(0);
}
const result = compare(args);
const { failures, lines, warnings } = result;
if (args.out !== undefined) {
  mkdirSync(dirname(args.out), { recursive: true });
  writeFileSync(args.out, JSON.stringify({ generatedAt: new Date().toISOString(), kind: "bb25-bench-json-check", ...result }, null, 2) + "\n", "utf8");
}
if (args["manifest-out"] !== undefined) {
  const manifest = {
    generatedAt: new Date().toISOString(),
    kind: "bb25-bench-json-check-manifest",
    command: process.argv.slice(1),
    inputs: {
      reference: fileManifest(args.reference),
      actual: fileManifest(args.actual),
    },
    summary: {
      passed: result.passed,
      methods: result.methods,
      metric: result.metric,
      metricScale: result.metricScale,
      tolerancePoints: result.tolerancePoints,
      toleranceUnits: result.toleranceUnits,
      tolerance: result.tolerance,
      requiredDatasets: result.requiredDatasets,
      datasetGate: result.datasetGate,
      classifications: result.classifications,
      warningClassifications: result.warningClassifications,
      failures: result.failures.map(failureMessage),
      warnings: result.warnings.map(failureMessage),
    },
  };
  mkdirSync(dirname(args["manifest-out"]), { recursive: true });
  writeFileSync(args["manifest-out"], JSON.stringify(manifest, null, 2) + "\n", "utf8");
}
if (lines.length > 0) {
  process.stdout.write(lines.join("\n") + "\n");
}
if (warnings.length > 0) {
  process.stderr.write(warnings.map((warning) => `WARN [${warning.classification}] ${failureMessage(warning)}`).join("\n") + "\n");
}
if (failures.length > 0) {
  process.stderr.write(failures.map((failure) => `FAIL [${failure.classification}] ${failureMessage(failure)}`).join("\n") + "\n");
  process.exitCode = 1;
} else {
  process.stdout.write("benchmark JSON matches expected thresholds\n");
}
