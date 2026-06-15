#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const DEFAULT_DATASETS = "arguana,fiqa,nfcorpus,scidocs,scifact";

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
  if (!existsSync(path)) return { path, exists: false };
  const data = readFileSync(path);
  return {
    path,
    exists: true,
    bytes: statSync(path).size,
    sha256: createHash("sha256").update(data).digest("hex"),
  };
}

function splitCsv(raw) {
  return raw.split(",").map((part) => part.trim()).filter(Boolean);
}

function scorerCsvFromMethods(raw) {
  return splitCsv(raw)
    .map((part) => part.toLowerCase().replace(/[-\s]+/g, "_"))
    .join(",");
}

function addIfDefined(command, flag, value) {
  if (value !== undefined) command.push(flag, value);
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  if (args.root === undefined) {
    throw new Error("usage: run-baseline-parity --root <beir-jsonl-dir> --reference reference-results/python/hybrid-beir.json [--judge pytrec|internal]");
  }
  if (args.reference === undefined) {
    throw new Error("--reference <stored-python-reference.json> is required");
  }

  const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  const root = resolve(args.root);
  const datasets = args.datasets ?? DEFAULT_DATASETS;
  const judge = args.judge ?? "pytrec";
  if (judge !== "pytrec" && judge !== "internal") {
    throw new Error(`--judge must be pytrec or internal, got ${judge}`);
  }

  const outDir = resolve(args["out-dir"] ?? "reference-results/ts");
  const manifestDir = resolve(args["manifest-dir"] ?? "reference-results/manifests");
  const runsDir = resolve(args["runs-dir"] ?? `${outDir}/runs`);
  const internalOut = resolve(args["internal-out"] ?? `${outDir}/hybrid-beir-internal.json`);
  const internalManifest = resolve(args["internal-manifest"] ?? `${manifestDir}/ts-hybrid-beir-internal.json`);
  const pytrecOut = resolve(args["pytrec-out"] ?? `${outDir}/hybrid-beir-pytrec.json`);
  const pytrecManifest = resolve(args["pytrec-manifest"] ?? `${manifestDir}/ts-hybrid-beir-pytrec.json`);
  const parityOut = resolve(args.out ?? `${outDir}/baseline-parity.json`);
  const parityManifest = resolve(args["manifest-out"] ?? `${manifestDir}/baseline-parity.json`);
  const runnerManifest = resolve(args["runner-manifest-out"] ?? `${manifestDir}/baseline-parity-runner.json`);
  const dryRun = args["dry-run"] !== undefined;

  mkdirSync(outDir, { recursive: true });
  mkdirSync(manifestDir, { recursive: true });
  mkdirSync(runsDir, { recursive: true });

  const commands = [];
  let failed = false;

  const benchCommand = [
    "node",
    resolve(repoRoot, "scripts/run-beir-jsonl-bench.mjs"),
    "--root",
    root,
    "--datasets",
    datasets,
    "--doc-embedding",
    args["doc-embedding"] ?? "embedding",
    "--query-embedding",
    args["query-embedding"] ?? "embedding",
    "--doc-fields",
    args["doc-fields"] ?? "title,body",
    "--doc-field-terms",
    args["doc-field-terms"] ?? "field_terms",
    "--candidate-depth",
    args["candidate-depth"] ?? "1000",
    "--trec-run-dir",
    runsDir,
    "--trec-run-depth",
    args["trec-run-depth"] ?? "1000",
    "--bm25-method",
    args["bm25-method"] ?? "lucene",
    "--cutoffs",
    args.cutoffs ?? "10",
    "--scorers",
    args.scorers ?? scorerCsvFromMethods(args.methods ?? "BM25,Dense,Convex,RRF"),
    "--out",
    internalOut,
    "--manifest-out",
    internalManifest,
  ];
  addIfDefined(benchCommand, "--bb25", args.bb25);
  const benchRecord = dryRun
    ? { name: "ts-beir-jsonl-bench", command: benchCommand, dryRun: true, returncode: 0, stdout: "", stderr: "" }
    : { name: "ts-beir-jsonl-bench", ...run(benchCommand, { cwd: repoRoot }) };
  commands.push(benchRecord);
  if (benchRecord.returncode !== 0) failed = true;

  let actualForParity = internalOut;
  if (!failed && judge === "pytrec") {
    const pytrecCommand = [
      args.python ?? "python3",
      resolve(repoRoot, "scripts/evaluate-trec-run.py"),
      "--root",
      root,
      "--datasets",
      datasets,
      "--runs",
      runsDir,
      "--cutoffs",
      args.cutoffs ?? "10",
      "--out",
      pytrecOut,
      "--manifest-out",
      pytrecManifest,
    ];
    const pytrecRecord = dryRun
      ? { name: "pytrec-eval", command: pytrecCommand, dryRun: true, returncode: 0, stdout: "", stderr: "" }
      : { name: "pytrec-eval", ...run(pytrecCommand, { cwd: repoRoot }) };
    commands.push(pytrecRecord);
    if (pytrecRecord.returncode !== 0) failed = true;
    actualForParity = pytrecOut;
  }

  if (!failed) {
    const checkCommand = [
      "node",
      resolve(repoRoot, "scripts/check-bench-json.mjs"),
      "--reference",
      resolve(args.reference),
      "--actual",
      actualForParity,
      "--methods",
      args.methods ?? "BM25,Dense,Convex,RRF",
      "--metric",
      args.metric ?? "ndcg@10",
      "--datasets",
      datasets,
      "--tolerance-points",
      args["tolerance-points"] ?? "0.50",
      "--out",
      parityOut,
      "--manifest-out",
      parityManifest,
    ];
    addIfDefined(checkCommand, "--reference-method-map", args["reference-method-map"]);
    addIfDefined(checkCommand, "--actual-method-map", args["actual-method-map"]);
    addIfDefined(checkCommand, "--dataset-gate", args["dataset-gate"]);
    addIfDefined(checkCommand, "--dataset-tolerance", args["dataset-tolerance"]);
    addIfDefined(checkCommand, "--dataset-tolerance-points", args["dataset-tolerance-points"]);
    const checkRecord = dryRun
      ? { name: "baseline-parity-check", command: checkCommand, dryRun: true, returncode: 0, stdout: "", stderr: "" }
      : { name: "baseline-parity-check", ...run(checkCommand, { cwd: repoRoot }) };
    commands.push(checkRecord);
    if (checkRecord.returncode !== 0) failed = true;
  }

  const manifest = {
    generatedAt: new Date().toISOString(),
    kind: "bb25-baseline-parity-runner",
    dryRun,
    judge,
    repo: {
      root: repoRoot,
      git: gitInfo(repoRoot),
      node: process.version,
      platform: process.platform,
      arch: process.arch,
    },
    options: {
      root,
      datasets: splitCsv(datasets),
      reference: resolve(args.reference),
      outDir,
      manifestDir,
      runsDir,
      methods: splitCsv(args.methods ?? "BM25,Dense,Convex,RRF"),
      scorers: splitCsv(args.scorers ?? scorerCsvFromMethods(args.methods ?? "BM25,Dense,Convex,RRF")),
      metric: args.metric ?? "ndcg@10",
      tolerancePoints: Number(args["tolerance-points"] ?? "0.50"),
      datasetGate: args["dataset-gate"] ?? "off",
      datasetTolerance:
        args["dataset-tolerance"] === undefined && args["dataset-tolerance-points"] === undefined
          ? null
          : Number(args["dataset-tolerance"] ?? args["dataset-tolerance-points"]),
    },
    files: {
      reference: fileManifest(resolve(args.reference)),
      internal: fileManifest(internalOut),
      internalManifest: fileManifest(internalManifest),
      pytrec: fileManifest(pytrecOut),
      pytrecManifest: fileManifest(pytrecManifest),
      parity: fileManifest(parityOut),
      parityManifest: fileManifest(parityManifest),
      runsDir: { path: runsDir, exists: existsSync(runsDir) },
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
        judge,
        passed: !failed,
        actual: actualForParity,
        parity: parityOut,
      },
      null,
      2,
    ) + "\n",
  );
  if (failed) process.exitCode = 1;
}

main();
