#!/usr/bin/env node
import { readFileSync } from "node:fs";

function parseArgs(argv) {
  const args = {};
  for (let i = 0; i < argv.length; i++) {
    const key = argv[i];
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
  const [, family, cutoff] = normalized.match(/^(ndcg|map|mrr|recall)@(\d+)$/) ?? [];
  if (family === undefined || cutoff === undefined) {
    return [metric, normalized];
  }
  const title = family === "ndcg" ? "NDCG" : family === "map" ? "MAP" : family === "mrr" ? "MRR" : "Recall";
  return [normalized, `${title}@${cutoff}`, `${family}_cut_${cutoff}`, `${family}${cutoff}`];
}

function toPoints(value) {
  const n = Number(value);
  if (!Number.isFinite(n)) {
    throw new Error(`invalid metric value: ${value}`);
  }
  return Math.abs(n) <= 1.0000001 ? n * 100.0 : n;
}

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
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

  if (Array.isArray(payload.average)) {
    for (const row of payload.average) {
      addMetric(rows, "average", String(row.scorer), row.metrics);
    }
  }

  if (Array.isArray(payload.runs)) {
    for (const run of payload.runs) {
      const dataset = String(run.dataset ?? "dataset");
      if (!Array.isArray(run.results)) continue;
      for (const row of run.results) {
        addMetric(rows, dataset, String(row.scorer), row.metrics);
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
    const [left, right] = pair.split("=");
    if (left === undefined || right === undefined) {
      throw new Error(`invalid method-map entry: ${pair}`);
    }
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

function getMetric(metrics, metric) {
  for (const alias of metricAliases(metric)) {
    if (Object.prototype.hasOwnProperty.call(metrics, alias)) {
      return toPoints(metrics[alias]);
    }
  }
  return null;
}

function averageMetric(datasetMetrics, metric) {
  const values = [];
  for (const metrics of datasetMetrics.values()) {
    const value = getMetric(metrics, metric);
    if (value !== null) values.push(value);
  }
  if (values.length === 0) return null;
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function compare(args) {
  if (args.reference === undefined || args.actual === undefined) {
    throw new Error(
      "usage: check-bench-json --reference ref.json --actual actual.json [--methods BM25,Dense,Convex,RRF] [--metric ndcg@10] [--tolerance-points 0.50]",
    );
  }

  const methods = (args.methods ?? "BM25,Dense,Convex,RRF").split(",").map((s) => s.trim()).filter(Boolean);
  const metric = args.metric ?? "ndcg@10";
  const tolerance = args["tolerance-points"] === undefined ? 0.5 : Number(args["tolerance-points"]);
  const refMap = parseMethodMap(args["reference-method-map"]);
  const actualMap = parseMethodMap(args["actual-method-map"]);
  const refRows = extractRows(readJson(args.reference));
  const actualRows = extractRows(readJson(args.actual));

  const failures = [];
  const lines = [];
  for (const method of methods) {
    const ref = getScorer(refRows, method, refMap.get(method));
    const actual = getScorer(actualRows, method, actualMap.get(method));
    if (ref === null) {
      failures.push(`missing reference method ${method}`);
      continue;
    }
    if (actual === null) {
      failures.push(`missing actual method ${method}`);
      continue;
    }

    const refAvg = averageMetric(ref.datasets, metric);
    const actualAvg = averageMetric(actual.datasets, metric);
    if (refAvg === null) {
      failures.push(`reference method ${method} has no ${metric}`);
      continue;
    }
    if (actualAvg === null) {
      failures.push(`actual method ${method} has no ${metric}`);
      continue;
    }
    const diff = Math.abs(actualAvg - refAvg);
    lines.push(`${method}\tref=${refAvg.toFixed(4)}\tactual=${actualAvg.toFixed(4)}\tdiff=${diff.toFixed(4)} points`);
    if (diff > tolerance) {
      failures.push(`${method} ${metric} diff ${diff.toFixed(4)} > ${tolerance}`);
    }

    const sharedDatasets = [...ref.datasets.keys()].filter((dataset) => actual.datasets.has(dataset));
    for (const dataset of sharedDatasets) {
      const refValue = getMetric(ref.datasets.get(dataset), metric);
      const actualValue = getMetric(actual.datasets.get(dataset), metric);
      if (refValue === null || actualValue === null) continue;
      const datasetDiff = Math.abs(actualValue - refValue);
      lines.push(`  ${dataset}\tref=${refValue.toFixed(4)}\tactual=${actualValue.toFixed(4)}\tdiff=${datasetDiff.toFixed(4)} points`);
    }
  }

  return { failures, lines };
}

const args = parseArgs(process.argv.slice(2));
if (args.help !== undefined || args.h !== undefined) {
  process.stdout.write(
    "usage: check-bench-json --reference ref.json --actual actual.json [--methods BM25,Dense,Convex,RRF] [--metric ndcg@10] [--tolerance-points 0.50]\n",
  );
  process.exit(0);
}
const { failures, lines } = compare(args);
if (lines.length > 0) {
  process.stdout.write(lines.join("\n") + "\n");
}
if (failures.length > 0) {
  process.stderr.write(failures.map((failure) => `FAIL ${failure}`).join("\n") + "\n");
  process.exitCode = 1;
} else {
  process.stdout.write("benchmark JSON matches expected thresholds\n");
}
