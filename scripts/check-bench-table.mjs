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

function parseTable(path) {
  const content = readFileSync(path, "utf8");
  const lines = content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line !== "" && !line.startsWith("==="));
  const headerIndex = lines.findIndex((line) => line.split(/\s+/)[0] === "scorer");
  if (headerIndex < 0) {
    throw new Error(`no scorer header found in ${path}`);
  }
  const headers = lines[headerIndex].split(/\s+/);
  const rows = new Map();
  for (const line of lines.slice(headerIndex + 1)) {
    const parts = line.split(/\s+/);
    if (parts.length < headers.length) continue;
    const row = {};
    for (let i = 0; i < headers.length; i++) {
      const header = headers[i];
      const value = parts[i];
      row[header] = i < 2 ? value : Number(value);
    }
    rows.set(String(row.scorer), row);
  }
  return { headers, rows };
}

function compareTables(expectedPath, actualPath, tolerance) {
  const expected = parseTable(expectedPath);
  const actual = parseTable(actualPath);
  const metricHeaders = expected.headers.filter((header) => header !== "scorer" && header !== "queries");
  const failures = [];

  for (const [scorer, expectedRow] of expected.rows) {
    const actualRow = actual.rows.get(scorer);
    if (actualRow === undefined) {
      failures.push(`missing scorer: ${scorer}`);
      continue;
    }
    if (expectedRow.queries !== actualRow.queries) {
      failures.push(`${scorer} queries expected ${expectedRow.queries}, got ${actualRow.queries}`);
    }
    for (const metric of metricHeaders) {
      if (!(metric in actualRow)) continue;
      const diff = Math.abs(Number(actualRow[metric]) - Number(expectedRow[metric]));
      if (diff > tolerance) {
        failures.push(`${scorer} ${metric} diff ${diff.toFixed(6)} > ${tolerance}`);
      }
    }
  }

  return { failures, expected, actual };
}

function checkOrdering(table, metric, order) {
  const failures = [];
  for (let i = 1; i < order.length; i++) {
    const prev = table.rows.get(order[i - 1]);
    const curr = table.rows.get(order[i]);
    if (prev === undefined || curr === undefined) continue;
    if (Number(prev[metric]) <= Number(curr[metric])) {
      failures.push(`${order[i - 1]} ${metric} must be greater than ${order[i]} ${metric}`);
    }
  }
  return failures;
}

const args = parseArgs(process.argv.slice(2));
if (args.expected === undefined || args.actual === undefined) {
  throw new Error("usage: check-bench-table --expected <table> --actual <table> [--tolerance 0.005] [--metric ndcg@10] [--order a,b,c]");
}

const tolerance = args.tolerance === undefined ? 0.005 : Number(args.tolerance);
const { failures, actual } = compareTables(args.expected, args.actual, tolerance);
if (args.metric !== undefined && args.order !== undefined) {
  failures.push(...checkOrdering(actual, args.metric, args.order.split(",")));
}

if (failures.length > 0) {
  process.stderr.write(failures.map((failure) => `FAIL ${failure}`).join("\n") + "\n");
  process.exitCode = 1;
} else {
  process.stdout.write("benchmark table matches expected thresholds\n");
}
