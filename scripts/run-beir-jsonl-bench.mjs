#!/usr/bin/env node
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
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

function utcNow() {
  return new Date().toISOString();
}

function datasetValue(value, datasetDir) {
  return value
    .replaceAll("{dataset}", basename(datasetDir))
    .replaceAll("{datasetDir}", datasetDir);
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
  if (args["metric-style"] !== undefined) command.push("--metric-style", args["metric-style"]);
  if (args.scorers !== undefined) command.push("--scorers", args.scorers);
  if (args.alpha !== undefined) command.push("--alpha", args.alpha);
  if (args.beta !== undefined) command.push("--beta", args.beta);
  if (args["base-rate"] !== undefined) command.push("--base-rate", args["base-rate"]);
  if (args["base-rate-method"] !== undefined) command.push("--base-rate-method", args["base-rate-method"]);
  if (args["base-rate-sample-size"] !== undefined) command.push("--base-rate-sample-size", args["base-rate-sample-size"]);
  if (args["base-rate-seed"] !== undefined) command.push("--base-rate-seed", args["base-rate-seed"]);
  if (args["fit-split"] !== undefined) command.push("--fit-split");
  if (args["fit-train-ratio"] !== undefined) command.push("--fit-train-ratio", args["fit-train-ratio"]);
  if (args["fit-split-seed"] !== undefined) command.push("--fit-split-seed", args["fit-split-seed"]);
  if (args["fit-split-file"] !== undefined) command.push("--fit-split-file", datasetValue(args["fit-split-file"], datasetDir));
  if (args.calibration !== undefined) command.push("--calibration");
  if (args["calibration-bins"] !== undefined) command.push("--calibration-bins", args["calibration-bins"]);
  if (args["trec-run-dir"] !== undefined) command.push("--trec-run-dir", join(args["trec-run-dir"], basename(datasetDir)));
  if (args["trec-run-depth"] !== undefined) command.push("--trec-run-depth", args["trec-run-depth"]);

  const startedAt = utcNow();
  const start = performance.now();
  const result = spawnSync(command[0], command.slice(1), { encoding: "utf8" });
  const commandRecord = {
    name: `bb25-bench:${basename(datasetDir)}`,
    command,
    cwd: process.cwd(),
    startedAt,
    finishedAt: utcNow(),
    durationSeconds: Number(((performance.now() - start) / 1000).toFixed(6)),
    returncode: result.status ?? 1,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
    error: result.error === undefined ? null : String(result.error.message ?? result.error),
  };
  if (commandRecord.returncode !== 0) {
    process.stderr.write(commandRecord.stderr);
    throw new Error(`bench failed for ${datasetDir}`);
  }
  return { commandRecord, result: JSON.parse(commandRecord.stdout) };
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

function readJsonIfExists(path) {
  if (!existsSync(path)) return null;
  return JSON.parse(readFileSync(path, "utf8"));
}

function fileManifest(path) {
  if (!existsSync(path)) {
    return { path, exists: false };
  }
  const bytes = readFileSync(path);
  const stat = statSync(path);
  return {
    path,
    exists: true,
    bytes: stat.size,
    sha256: createHash("sha256").update(bytes).digest("hex"),
  };
}

function commandOk(command, options = {}) {
  const result = spawnSync(command[0], command.slice(1), {
    encoding: "utf8",
    ...options,
  });
  if (result.status !== 0) {
    return null;
  }
  return result.stdout.trim();
}

function gitInfo() {
  return {
    commit: commandOk(["git", "rev-parse", "HEAD"]),
    statusShort: commandOk(["git", "status", "--short"]),
  };
}

function packageInfo(rootDir) {
  const rootPackage = readJsonIfExists(join(rootDir, "package.json"));
  const cliPackage = readJsonIfExists(join(rootDir, "packages/cli/package.json"));
  const corePackage = readJsonIfExists(join(rootDir, "packages/core/package.json"));
  return {
    root: rootPackage === null ? null : { name: rootPackage.name, version: rootPackage.version, packageManager: rootPackage.packageManager },
    cli: cliPackage === null ? null : { name: cliPackage.name, version: cliPackage.version },
    core: corePackage === null ? null : { name: corePackage.name, version: corePackage.version },
  };
}

function datasetManifest(root, dataset, commandRecord, result) {
  const datasetDir = join(root, dataset);
  return {
    dataset,
    directory: datasetDir,
    files: {
      docs: fileManifest(join(datasetDir, "docs.jsonl")),
      queries: fileManifest(join(datasetDir, "queries.jsonl")),
      qrels: fileManifest(join(datasetDir, "qrels.tsv")),
      exportManifest: fileManifest(join(datasetDir, "manifest.json")),
    },
    exportManifest: readJsonIfExists(join(datasetDir, "manifest.json")),
    command: commandRecord.command,
    commandRecord,
    resultSummary: {
      cutoffs: result.cutoffs,
      options: result.options,
      scorers: result.scorers,
      fittedSplit: result.fittedSplit ?? null,
      attentionSplits: result.attentionSplits ?? [],
      denseCalibrationSplits: result.denseCalibrationSplits ?? [],
    },
  };
}

const args = parseArgs(process.argv.slice(2));
if (args.root === undefined || args.datasets === undefined) {
  throw new Error(
    "usage: run-beir-jsonl-bench --root <dir> --datasets nfcorpus,scifact [--candidate-depth 1000] [--out results.json] [--manifest-out manifest.json]",
  );
}

const here = dirname(fileURLToPath(import.meta.url));
const defaultBb25 = resolve(here, "../packages/cli/dist/cli.js");
const bb25Path = resolve(args.bb25 ?? defaultBb25);
if (!existsSync(bb25Path)) {
  throw new Error(`bb25 CLI not found at ${bb25Path}; run corepack pnpm --filter @bb25/cli build first`);
}

const root = resolve(args.root);
const datasets = args.datasets.split(",").map((name) => name.trim()).filter(Boolean);
const datasetRuns = datasets.map((dataset) => {
  const datasetDir = join(root, dataset);
  const { commandRecord, result } = runBench(bb25Path, datasetDir, args);
  return { dataset, commandRecord, result };
});
const runs = datasetRuns.map((run) => ({ dataset: run.dataset, command: run.commandRecord.command, ...run.result }));
const output = {
  generatedAt: utcNow(),
  root,
  datasets,
  options: args,
  runs,
  average: aggregate(runs),
};

const json = JSON.stringify(output, null, 2) + "\n";
if (args.out !== undefined) {
  mkdirSync(dirname(args.out), { recursive: true });
  writeFileSync(args.out, json, "utf8");
}
if (args["manifest-out"] !== undefined) {
  const repoRoot = resolve(here, "..");
  const manifest = {
    generatedAt: output.generatedAt,
    kind: "bb25-beir-jsonl-bench",
    root,
    datasets,
    bb25Path,
    output: args.out === undefined ? null : resolve(args.out),
    trecRunDir: args["trec-run-dir"] === undefined ? null : resolve(args["trec-run-dir"]),
    cliArgs: args,
    commands: datasetRuns.map((run) => run.commandRecord),
    environment: {
      node: process.version,
      platform: process.platform,
      arch: process.arch,
      git: gitInfo(),
      packages: packageInfo(repoRoot),
    },
    datasetInputs: datasetRuns.map((run) => datasetManifest(root, run.dataset, run.commandRecord, run.result)),
    average: output.average,
  };
  mkdirSync(dirname(args["manifest-out"]), { recursive: true });
  writeFileSync(args["manifest-out"], JSON.stringify(manifest, null, 2) + "\n", "utf8");
}
process.stdout.write(json);
