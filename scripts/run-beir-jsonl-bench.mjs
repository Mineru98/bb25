#!/usr/bin/env node
import { existsSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";

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

function runBench(bb25Path, datasetDir, args) {
  const command = [
    "node",
    bb25Path,
    "bench",
    "--docs",
    join(datasetDir, "docs.jsonl"),
    "--queries",
    join(datasetDir, "queries.jsonl"),
    "--qrels",
    join(datasetDir, "qrels.tsv"),
    "--bm25-method",
    args["bm25-method"] ?? "lucene",
    "--cutoffs",
    args.cutoffs ?? "10",
    "--json",
  ];

  if (args["doc-terms"] !== "false") command.push("--doc-terms", args["doc-terms"] ?? "terms");
  if (args["doc-fields"] !== undefined) command.push("--doc-fields", args["doc-fields"]);
  if (args["doc-field-terms"] !== undefined) command.push("--doc-field-terms", args["doc-field-terms"]);
  if (args["query-terms"] !== "false") command.push("--query-terms", args["query-terms"] ?? "terms");
  if (args["doc-embedding"] !== undefined) command.push("--doc-embedding", args["doc-embedding"]);
  if (args["query-embedding"] !== undefined) command.push("--query-embedding", args["query-embedding"]);
  if (args["candidate-depth"] !== undefined) command.push("--candidate-depth", args["candidate-depth"]);
  if (args["base-rate"] !== undefined) command.push("--base-rate", args["base-rate"]);
  if (args["base-rate-method"] !== undefined) command.push("--base-rate-method", args["base-rate-method"]);
  if (args["base-rate-sample-size"] !== undefined) command.push("--base-rate-sample-size", args["base-rate-sample-size"]);
  if (args["base-rate-seed"] !== undefined) command.push("--base-rate-seed", args["base-rate-seed"]);
  if (args["fit-split"] !== undefined) command.push("--fit-split");
  if (args["fit-train-ratio"] !== undefined) command.push("--fit-train-ratio", args["fit-train-ratio"]);
  if (args["fit-split-seed"] !== undefined) command.push("--fit-split-seed", args["fit-split-seed"]);
  if (args.calibration !== undefined) command.push("--calibration");
  if (args["calibration-bins"] !== undefined) command.push("--calibration-bins", args["calibration-bins"]);
  if (args["trec-run-dir"] !== undefined) command.push("--trec-run-dir", join(args["trec-run-dir"], basename(datasetDir)));
  if (args["trec-run-depth"] !== undefined) command.push("--trec-run-depth", args["trec-run-depth"]);

  const result = spawnSync(command[0], command.slice(1), { encoding: "utf8" });
  if (result.status !== 0) {
    process.stderr.write(result.stderr);
    throw new Error(`bench failed for ${datasetDir}`);
  }
  return JSON.parse(result.stdout);
}

function aggregate(runs) {
  const byScorer = new Map();
  for (const run of runs) {
    for (const row of run.results) {
      let agg = byScorer.get(row.scorer);
      if (agg === undefined) {
        agg = { scorer: row.scorer, datasets: 0, queries: 0, metrics: {} };
        byScorer.set(row.scorer, agg);
      }
      agg.datasets += 1;
      agg.queries += row.queries;
      for (const [metric, value] of Object.entries(row.metrics)) {
        agg.metrics[metric] = (agg.metrics[metric] ?? 0) + Number(value);
      }
    }
  }
  return [...byScorer.values()].map((row) => {
    for (const metric of Object.keys(row.metrics)) {
      row.metrics[metric] /= row.datasets;
    }
    return row;
  });
}

const args = parseArgs(process.argv.slice(2));
if (args.root === undefined || args.datasets === undefined) {
  throw new Error("usage: run-beir-jsonl-bench --root <dir> --datasets nfcorpus,scifact [--candidate-depth 1000] [--out results.json]");
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultBb25 = resolve(here, "../packages/cli/dist/cli.js");
const bb25Path = resolve(args.bb25 ?? defaultBb25);
if (!existsSync(bb25Path)) {
  throw new Error(`bb25 CLI not found at ${bb25Path}; run corepack pnpm --filter @bb25/cli build first`);
}

const root = resolve(args.root);
const datasets = args.datasets.split(",").map((name) => name.trim()).filter(Boolean);
const runs = datasets.map((dataset) => {
  const datasetDir = join(root, dataset);
  const result = runBench(bb25Path, datasetDir, args);
  return { dataset, ...result };
});
const output = {
  generatedAt: new Date().toISOString(),
  root,
  datasets,
  options: args,
  runs,
  average: aggregate(runs),
};

const json = JSON.stringify(output, null, 2) + "\n";
if (args.out !== undefined) {
  writeFileSync(args.out, json, "utf8");
}
process.stdout.write(json);
