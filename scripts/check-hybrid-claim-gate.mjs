#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
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

function parseMethodMap(raw) {
  const out = new Map();
  if (raw === undefined) return out;
  for (const pair of raw.split(",")) {
    if (pair.trim() === "") continue;
    const eq = pair.indexOf("=");
    if (eq <= 0) throw new Error(`invalid method-map entry: ${pair}`);
    out.set(normalizeName(pair.slice(0, eq)), pair.slice(eq + 1).trim());
  }
  return out;
}

function normalizeName(value) {
  return String(value).trim().toLowerCase().replace(/[-\s]+/g, "_");
}

function normalizeMetricName(metric) {
  const lower = metric.toLowerCase();
  const match = lower.match(/^(ndcg|map|mrr|recall)@?(\d+)$/) ?? lower.match(/^(ndcg|map|mrr|recall)_cut_?(\d+)$/);
  return match === null ? lower : `${match[1]}@${match[2]}`;
}

function metricAliases(metric) {
  const normalized = normalizeMetricName(metric);
  const [, family, cutoff] = normalized.match(/^(ndcg|map|mrr|recall)@(\d+)$/) ?? [];
  if (family === undefined || cutoff === undefined) return [metric, normalized];
  const title = family === "ndcg" ? "NDCG" : family === "map" ? "MAP" : family === "mrr" ? "MRR" : "Recall";
  const aliases = [normalized, `${title}@${cutoff}`, `${family}_cut_${cutoff}`, `${family}${cutoff}`];
  if (family === "map") aliases.push("MAP", "map");
  return aliases;
}

function defaultMetricScale(metric) {
  const normalized = normalizeMetricName(metric);
  return normalized === "ece" || normalized === "brier" ? "unit" : "points";
}

function normalizeMetricValue(value, scale) {
  const n = Number(value);
  if (!Number.isFinite(n)) throw new Error(`invalid metric value: ${value}`);
  if (scale === "points") return Math.abs(n) <= 1.0000001 ? n * 100.0 : n;
  if (scale === "unit") return n;
  throw new Error(`invalid metric scale: ${scale}`);
}

function metricValue(metrics, metric, scale) {
  if (metrics === null || typeof metrics !== "object") return null;
  for (const alias of metricAliases(metric)) {
    if (Object.prototype.hasOwnProperty.call(metrics, alias)) {
      return normalizeMetricValue(metrics[alias], scale);
    }
  }
  return null;
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function readJsonIfExists(path) {
  if (path === undefined || path === null || !existsSync(path)) return null;
  return readJson(path);
}

function fileManifest(path) {
  if (path === undefined || path === null || !existsSync(path)) return { path: path ?? null, exists: false };
  const data = readFileSync(path);
  return {
    path: resolve(path),
    exists: true,
    bytes: statSync(path).size,
    sha256: createHash("sha256").update(data).digest("hex"),
  };
}

function addMetric(rows, scorer, dataset, value, kind = null) {
  if (value === null) return;
  const key = normalizeName(scorer);
  const current = rows.get(key) ?? { scorer: String(scorer), kind, average: null, datasets: new Map() };
  if (current.kind === null && kind !== null) current.kind = kind;
  if (dataset === "average") current.average = value;
  else current.datasets.set(dataset, value);
  rows.set(key, current);
}

function extractMetricRows(payload, metric, scale) {
  const rows = new Map();
  const kinds = new Map();
  if (Array.isArray(payload.scorers)) {
    for (const scorer of payload.scorers) {
      if (scorer?.scorer !== undefined && scorer?.kind !== undefined) kinds.set(normalizeName(scorer.scorer), String(scorer.kind));
    }
  }

  function addRow(dataset, row) {
    const scorer = row?.scorer;
    if (scorer === undefined || scorer === null) return;
    addMetric(rows, scorer, dataset, metricValue(row.metrics, metric, scale), kinds.get(normalizeName(scorer)) ?? null);
  }

  if (Array.isArray(payload.average)) {
    for (const row of payload.average) addRow("average", row);
  }
  if (Array.isArray(payload.results)) {
    for (const row of payload.results) addRow("average", row);
  }
  if (Array.isArray(payload.runs)) {
    for (const run of payload.runs) {
      const dataset = String(run.dataset ?? "dataset");
      if (Array.isArray(run.results)) {
        for (const row of run.results) addRow(dataset, row);
      }
    }
  }

  const looksLikePythonReference =
    !Array.isArray(payload.results) &&
    !Array.isArray(payload.runs) &&
    Object.values(payload).some((value) => value !== null && typeof value === "object" && !Array.isArray(value));
  if (looksLikePythonReference) {
    for (const [dataset, methods] of Object.entries(payload)) {
      if (methods === null || typeof methods !== "object" || Array.isArray(methods)) continue;
      for (const [method, metrics] of Object.entries(methods)) {
        addMetric(rows, method, dataset, metricValue(metrics, metric, scale), null);
      }
    }
  }

  for (const row of rows.values()) {
    if (row.average === null && row.datasets.size > 0) {
      row.average = [...row.datasets.values()].reduce((sum, value) => sum + value, 0.0) / row.datasets.size;
    }
  }
  return rows;
}

function findRow(rows, method) {
  return rows.get(normalizeName(method)) ?? null;
}

function resultCheck(kind, passed, message, extra = {}) {
  return { kind, passed, message, ...extra };
}

function sameSet(left, right) {
  const a = [...new Set(left.map(String))].sort();
  const b = [...new Set(right.map(String))].sort();
  return a.length === b.length && a.every((value, index) => value === b[index]);
}

function protocolChecks(baselineParity, baselineRunnerManifest, actual, actualManifest, metric) {
  const checks = [];
  const baselinePassed = baselineParity?.passed === true;
  checks.push(resultCheck("baseline_parity", baselinePassed, baselinePassed ? "baseline parity is green" : "baseline parity is not green"));
  const datasetGate = baselineParity?.datasetGate ?? null;
  const baselineDatasetGatePassed = datasetGate?.mode === "strict" && datasetGate?.passed === true;
  checks.push(resultCheck(
    "baseline_dataset_gate",
    baselineDatasetGatePassed,
    baselineDatasetGatePassed
      ? "strict dataset-level baseline parity is green"
      : "strict dataset-level baseline parity is missing or not green",
    { datasetGate },
  ));

  const actualDatasets = Array.isArray(actual.datasets)
    ? actual.datasets.map(String)
    : Array.isArray(actual.runs)
      ? actual.runs.map((run) => String(run.dataset))
      : [];
  const runnerDatasets = Array.isArray(baselineRunnerManifest?.options?.datasets)
    ? baselineRunnerManifest.options.datasets.map(String)
    : [];
  const sameDatasets = runnerDatasets.length > 0 && sameSet(runnerDatasets, actualDatasets);
  checks.push(resultCheck("same_inputs", sameDatasets, sameDatasets ? "baseline and claim use the same datasets" : "baseline and claim datasets differ", {
    baselineDatasets: runnerDatasets,
    actualDatasets,
  }));

  const judge = baselineRunnerManifest?.judge ?? null;
  const actualEvaluator = actualManifest?.kind ?? null;
  const sameEvaluator = judge === "pytrec" && actualEvaluator === "bb25-pytrec-eval";
  checks.push(resultCheck("same_evaluator", sameEvaluator, sameEvaluator ? "baseline and claim use pytrec evaluator" : "baseline and claim evaluator provenance differs", {
    judge,
    actualEvaluator,
  }));

  const runnerMetric = baselineRunnerManifest?.options?.metric ?? null;
  const sameMetric = typeof runnerMetric === "string" && runnerMetric.trim() !== "" && normalizeMetricName(runnerMetric) === normalizeMetricName(metric);
  checks.push(resultCheck("same_metric", sameMetric, sameMetric ? "baseline and claim metric matches" : `baseline metric ${runnerMetric} differs from ${metric}`, {
    baselineMetric: runnerMetric,
    metric,
  }));

  return checks;
}

function defaultReferenceMethod(method) {
  const defaults = new Map([
    ["bayesian_logodds", "Bayesian-LogOdds"],
    ["bayesian_logodds_br", "Bayesian-LogOdds-BR"],
    ["bayesian_vector_balanced", "Bayesian-Vector-Balanced"],
    ["balanced_fusion", "Bayesian-Balanced"],
    ["bayesian_attention", "Bayesian-Attention"],
    ["bayesian_attn_norm", "Bayesian-Attn-Norm"],
  ]);
  return defaults.get(normalizeName(method)) ?? method;
}

function checkHybridClaimGate(args) {
  if (args.actual === undefined || args.reference === undefined || args["baseline-parity"] === undefined) {
    throw new Error(
      "usage: check-hybrid-claim-gate --actual ts-hybrid.json --reference python-hybrid.json --baseline-parity baseline-parity.json [--actual-manifest pytrec-manifest.json] [--baseline-runner-manifest baseline-runner.json]",
    );
  }

  const metric = args.metric ?? "ndcg@10";
  const metricScale = args["metric-scale"] ?? defaultMetricScale(metric);
  const methods = parseCsv(args.methods ?? "bayesian_vector_balanced,bayesian_logodds");
  const baselines = parseMethodMap(args.baselines ?? "bm25=bm25,rrf=rrf");
  const referenceMethodMap = parseMethodMap(args["reference-method-map"]);
  const minBm25Delta = args["min-bm25-delta-points"] === undefined ? 4.0 : Number(args["min-bm25-delta-points"]);
  const minRrfDelta = args["min-rrf-delta-points"] === undefined ? -0.5 : Number(args["min-rrf-delta-points"]);
  const referenceTolerance = args["reference-tolerance-points"] === undefined ? 0.5 : Number(args["reference-tolerance-points"]);

  const actual = readJson(args.actual);
  const reference = readJson(args.reference);
  const baselineParity = readJson(args["baseline-parity"]);
  const actualManifest = readJsonIfExists(args["actual-manifest"]);
  const baselineRunnerManifest = readJsonIfExists(args["baseline-runner-manifest"]);
  const actualRows = extractMetricRows(actual, metric, metricScale);
  const referenceRows = extractMetricRows(reference, metric, metricScale);
  const protocol = protocolChecks(baselineParity, baselineRunnerManifest, actual, actualManifest, metric);

  const bm25 = findRow(actualRows, baselines.get("bm25") ?? "bm25");
  const rrf = findRow(actualRows, baselines.get("rrf") ?? "rrf");
  const claims = [];
  const excluded = [];

  for (const method of methods) {
    const actualRow = findRow(actualRows, method);
    const kind = actualRow?.kind ?? "unknown";
    if (kind !== "zero-shot") {
      excluded.push({ method, kind, reason: "not a zero-shot scorer kind" });
      continue;
    }
    const referenceMethod = referenceMethodMap.get(normalizeName(method)) ?? defaultReferenceMethod(method);
    const referenceRow = findRow(referenceRows, referenceMethod);
    const failures = [];
    if (actualRow?.average === null || actualRow?.average === undefined) failures.push(`missing actual ${metric}`);
    if (bm25?.average === null || bm25?.average === undefined) failures.push(`missing BM25 baseline ${metric}`);
    if (rrf?.average === null || rrf?.average === undefined) failures.push(`missing RRF baseline ${metric}`);
    if (referenceRow?.average === null || referenceRow?.average === undefined) failures.push(`missing Python reference ${referenceMethod} ${metric}`);

    const actualValue = actualRow?.average ?? null;
    const bm25Value = bm25?.average ?? null;
    const rrfValue = rrf?.average ?? null;
    const referenceValue = referenceRow?.average ?? null;
    const deltaVsBm25 = actualValue !== null && bm25Value !== null ? actualValue - bm25Value : null;
    const deltaVsRrf = actualValue !== null && rrfValue !== null ? actualValue - rrfValue : null;
    const deltaVsReference = actualValue !== null && referenceValue !== null ? actualValue - referenceValue : null;

    if (deltaVsBm25 !== null && deltaVsBm25 < minBm25Delta) failures.push(`delta vs BM25 ${deltaVsBm25.toFixed(6)} < ${minBm25Delta}`);
    if (deltaVsRrf !== null && deltaVsRrf < minRrfDelta) failures.push(`delta vs RRF ${deltaVsRrf.toFixed(6)} < ${minRrfDelta}`);
    if (deltaVsReference !== null && Math.abs(deltaVsReference) > referenceTolerance) {
      failures.push(`delta vs Python reference ${deltaVsReference.toFixed(6)} outside ±${referenceTolerance}`);
    }

    claims.push({
      method,
      kind,
      referenceMethod,
      metric,
      metricScale,
      value: actualValue,
      baselines: {
        bm25: bm25Value,
        rrf: rrfValue,
        pythonReference: referenceValue,
      },
      deltas: {
        vsBm25: deltaVsBm25,
        vsRrf: deltaVsRrf,
        vsPythonReference: deltaVsReference,
      },
      thresholds: {
        minBm25Delta,
        minRrfDelta,
        referenceTolerance,
      },
      passed: failures.length === 0,
      failures,
    });
  }

  const noEligible = claims.length === 0;
  const checks = [
    ...protocol,
    resultCheck("eligible_claim_rows", !noEligible, noEligible ? "no eligible zero-shot claim rows" : `${claims.length} eligible zero-shot claim row(s)`),
    ...claims.map((claim) => resultCheck("claim_thresholds", claim.passed, claim.passed ? `${claim.method} claim passed` : `${claim.method}: ${claim.failures.join("; ")}`, { method: claim.method })),
  ];
  const failures = checks.filter((check) => !check.passed);

  return {
    generatedAt: new Date().toISOString(),
    kind: "bb25-hybrid-claim-gate",
    passed: failures.length === 0,
    inputs: {
      actual: fileManifest(args.actual),
      reference: fileManifest(args.reference),
      baselineParity: fileManifest(args["baseline-parity"]),
      actualManifest: fileManifest(args["actual-manifest"]),
      baselineRunnerManifest: fileManifest(args["baseline-runner-manifest"]),
    },
    options: {
      methods,
      metric,
      metricScale,
      minBm25Delta,
      minRrfDelta,
      referenceTolerance,
    },
    protocol,
    claims,
    excluded,
    checks,
    failures,
  };
}

const args = parseArgs(process.argv.slice(2));
if (args.help !== undefined || args.h !== undefined) {
  process.stdout.write(
    "usage: check-hybrid-claim-gate --actual ts-hybrid.json --reference python-hybrid.json --baseline-parity baseline-parity.json [--methods bayesian_vector_balanced,bayesian_logodds] [--actual-manifest pytrec-manifest.json] [--baseline-runner-manifest baseline-runner.json] [--out result.json] [--manifest-out manifest.json]\n",
  );
  process.exit(0);
}

const result = checkHybridClaimGate(args);
if (args.out !== undefined) {
  mkdirSync(dirname(args.out), { recursive: true });
  writeFileSync(args.out, JSON.stringify(result, null, 2) + "\n", "utf8");
}
if (args["manifest-out"] !== undefined) {
  const manifest = {
    generatedAt: result.generatedAt,
    kind: "bb25-hybrid-claim-gate-manifest",
    command: process.argv.slice(1),
    inputs: result.inputs,
    summary: {
      passed: result.passed,
      checks: result.checks.length,
      failures: result.failures.map((failure) => failure.message),
      excluded: result.excluded,
    },
  };
  mkdirSync(dirname(args["manifest-out"]), { recursive: true });
  writeFileSync(args["manifest-out"], JSON.stringify(manifest, null, 2) + "\n", "utf8");
}

for (const check of result.checks) {
  process.stdout.write(`${check.passed ? "PASS" : "FAIL"}\t${check.kind}\t${check.message}\n`);
}
if (result.excluded.length > 0) {
  for (const row of result.excluded) {
    process.stdout.write(`SKIP\texcluded\t${row.method} (${row.kind}): ${row.reason}\n`);
  }
}
if (!result.passed) process.exitCode = 1;
