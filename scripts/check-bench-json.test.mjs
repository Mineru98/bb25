import assert from "node:assert/strict";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import test from "node:test";
import { spawnSync } from "node:child_process";

const repoRoot = dirname(dirname(fileURLToPath(import.meta.url)));
const node = process.execPath;

function writeJson(path, value) {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, JSON.stringify(value, null, 2) + "\n", "utf8");
}

function runScript(script, args) {
  return spawnSync(node, [join(repoRoot, script), ...args], {
    cwd: repoRoot,
    encoding: "utf8",
  });
}

function syntheticParityFixture(dir) {
  const reference = join(dir, "reference.json");
  const actual = join(dir, "actual.json");
  writeJson(reference, {
    d1: { BM25: { "NDCG@10": 0.5 } },
    d2: { BM25: { "NDCG@10": 0.6 } },
  });
  writeJson(actual, {
    runs: [
      { dataset: "d1", results: [{ scorer: "bm25", metrics: { "ndcg@10": 0.5 } }] },
      { dataset: "d2", results: [{ scorer: "bm25", metrics: { "ndcg@10": 0.594 } }] },
    ],
  });
  return { reference, actual };
}

function createHybridReadinessRoot(root, baselineParity) {
  const datasets = ["d1", "d2"];
  const embedding = {
    model: "sentence-transformers/all-MiniLM-L6-v2",
    cacheDir: "/tmp/bb25-embedding-cache",
    localFilesOnly: true,
    normalize: true,
  };
  const fileRecord = (dataset, name) => ({
    path: join(root, "beir-jsonl", dataset, name),
    exists: true,
    bytes: 10,
    sha256: "0".repeat(64),
  });
  const exportManifest = (dataset) => ({
    dataset,
    split: "test",
    irDatasetId: `beir/${dataset}/test`,
    tokenizer: "snowball",
    embedModel: embedding.model,
    embedCacheDir: embedding.cacheDir,
    embedLocalFilesOnly: embedding.localFilesOnly,
    embedding,
    files: {
      docs: fileRecord(dataset, "docs.jsonl"),
      queries: fileRecord(dataset, "queries.jsonl"),
      qrels: fileRecord(dataset, "qrels.tsv"),
    },
  });
  const datasetEntry = (dataset) => ({
    dataset,
    files: {
      docs: fileRecord(dataset, "docs.jsonl"),
      queries: fileRecord(dataset, "queries.jsonl"),
      qrels: fileRecord(dataset, "qrels.tsv"),
      manifest: fileRecord(dataset, "manifest.json"),
    },
    exportManifest: exportManifest(dataset),
  });
  const resultSummary = {
    cutoffs: [10],
    options: {
      bm25Method: "lucene",
      metricStyle: "pytrec",
      candidateDepth: 1000,
      scorers: ["bm25", "dense", "convex", "rrf"],
    },
    scorers: [
      { scorer: "bm25", kind: "zero-shot" },
      { scorer: "dense", kind: "zero-shot" },
      { scorer: "convex", kind: "zero-shot" },
      { scorer: "rrf", kind: "zero-shot" },
    ],
  };
  for (const relPath of [
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
  ]) {
    writeJson(join(root, relPath), {});
  }
  writeJson(join(root, "manifests/python-env.json"), { missing: [] });
  writeJson(join(root, "manifests/python-env-setup.json"), { commands: [{ returncode: 0 }] });
  writeJson(join(root, "manifests/python-reference-benchmarks.json"), { commands: [{ returncode: 0 }] });
  writeJson(join(root, "manifests/beir-jsonl-hybrid-export.json"), {
    kind: "bb25-beir-jsonl-suite",
    options: {
      datasets,
      tokenizer: "snowball",
      split: "test",
      embedModel: embedding.model,
      embedCacheDir: embedding.cacheDir,
      embedLocalFilesOnly: embedding.localFilesOnly,
    },
    commands: [{ returncode: 0 }],
    datasets: datasets.map(datasetEntry),
  });
  writeJson(join(root, "manifests/ts-hybrid-beir-internal.json"), {
    kind: "bb25-beir-jsonl-bench",
    datasets,
    cliArgs: {
      "bm25-method": "lucene",
      "candidate-depth": "1000",
      "metric-style": "pytrec",
      scorers: "bm25,dense,convex,rrf",
    },
    commands: [{ returncode: 0, command: ["node", "bb25", "bench"] }],
    datasetInputs: datasets.map((dataset) => ({
      dataset,
      files: datasetEntry(dataset).files,
      exportManifest: exportManifest(dataset),
      resultSummary,
    })),
  });
  writeJson(join(root, "manifests/ts-hybrid-beir-pytrec.json"), {
    kind: "bb25-pytrec-eval",
    inputs: { datasets },
    environment: { pytrecEvalVersion: "0.5" },
    datasetInputs: datasets.map((dataset) => ({
      dataset,
      qrels: fileRecord(dataset, "qrels.tsv"),
      runFiles: [
        fileRecord(dataset, "bm25.trec"),
        fileRecord(dataset, "dense.trec"),
        fileRecord(dataset, "convex.trec"),
        fileRecord(dataset, "rrf.trec"),
      ],
    })),
    commands: [{ returncode: 0 }],
  });
  writeJson(join(root, "manifests/baseline-parity-runner.json"), { commands: [{ returncode: 0 }] });
  writeJson(join(root, "ts/baseline-parity.json"), baselineParity);
}

function createHybridClaimGateFixture(dir, baselinePassed = true) {
  const actual = join(dir, "actual-hybrid.json");
  const reference = join(dir, "python-reference.json");
  const baselineParity = join(dir, "baseline-parity.json");
  const actualManifest = join(dir, "actual-manifest.json");
  const baselineRunnerManifest = join(dir, "baseline-runner-manifest.json");
  writeJson(actual, {
    datasets: ["d1", "d2"],
    average: [
      { scorer: "bm25", metrics: { "ndcg@10": 0.35 } },
      { scorer: "rrf", metrics: { "ndcg@10": 0.405 } },
      { scorer: "bayesian_vector_balanced", metrics: { "ndcg@10": 0.414 } },
      { scorer: "balanced_fusion", metrics: { "ndcg@10": 0.5 } },
    ],
    scorers: [
      { scorer: "bm25", kind: "zero-shot" },
      { scorer: "rrf", kind: "zero-shot" },
      { scorer: "bayesian_vector_balanced", kind: "zero-shot" },
      { scorer: "balanced_fusion", kind: "diagnostic" },
    ],
  });
  writeJson(reference, {
    d1: {
      "Bayesian-Vector-Balanced": { "NDCG@10": 0.415 },
    },
    d2: {
      "Bayesian-Vector-Balanced": { "NDCG@10": 0.413 },
    },
  });
  writeJson(baselineParity, {
    passed: baselinePassed,
    failures: baselinePassed ? [] : [{ kind: "diff" }],
    datasetGate: {
      mode: "strict",
      passed: baselinePassed,
      violations: baselinePassed ? [] : [{ method: "BM25", dataset: "d2", diff: 0.8 }],
    },
  });
  writeJson(actualManifest, {
    kind: "bb25-pytrec-eval",
    inputs: { datasets: ["d1", "d2"] },
  });
  writeJson(baselineRunnerManifest, {
    kind: "bb25-baseline-parity-runner",
    judge: "pytrec",
    options: {
      datasets: ["d1", "d2"],
      metric: "ndcg@10",
      methods: ["BM25", "Dense", "Convex", "RRF"],
    },
  });
  return { actual, reference, baselineParity, actualManifest, baselineRunnerManifest };
}

test("dataset parity gate can warn without failing and strict-fail per-dataset outliers", () => {
  const dir = mkdtempSync(join(tmpdir(), "bb25-check-bench-"));
  const { reference, actual } = syntheticParityFixture(dir);

  const commonArgs = [
    "--reference",
    reference,
    "--actual",
    actual,
    "--methods",
    "BM25",
    "--metric",
    "ndcg@10",
    "--datasets",
    "d1,d2",
    "--tolerance-points",
    "0.50",
  ];

  const devOut = join(dir, "dev.json");
  const dev = runScript("scripts/check-bench-json.mjs", [...commonArgs, "--out", devOut]);
  assert.equal(dev.status, 0, dev.stderr);
  assert.equal(JSON.parse(readFileSync(devOut, "utf8")).datasetGate.mode, "off");

  const warnOut = join(dir, "warn.json");
  const warn = runScript("scripts/check-bench-json.mjs", [
    ...commonArgs,
    "--dataset-gate",
    "warn",
    "--dataset-tolerance-points",
    "0.50",
    "--out",
    warnOut,
  ]);
  assert.equal(warn.status, 0, warn.stderr);
  const warnPayload = JSON.parse(readFileSync(warnOut, "utf8"));
  assert.equal(warnPayload.passed, true);
  assert.equal(warnPayload.datasetGate.passed, false);
  assert.equal(warnPayload.warnings.length, 1);
  assert.match(warn.stderr, /WARN .*dataset diff/);

  const strictOut = join(dir, "strict.json");
  const strict = runScript("scripts/check-bench-json.mjs", [
    ...commonArgs,
    "--dataset-gate",
    "strict",
    "--dataset-tolerance-points",
    "0.50",
    "--out",
    strictOut,
  ]);
  assert.equal(strict.status, 1);
  const strictPayload = JSON.parse(readFileSync(strictOut, "utf8"));
  assert.equal(strictPayload.passed, false);
  assert.equal(strictPayload.failures[0].kind, "dataset_diff");
});

test("baseline parity runner forwards dataset gate options into dry-run command and manifest", () => {
  const dir = mkdtempSync(join(tmpdir(), "bb25-run-baseline-"));
  const reference = join(dir, "reference.json");
  writeJson(reference, { d1: { BM25: { "NDCG@10": 0.5 } } });

  const result = runScript("scripts/run-baseline-parity.mjs", [
    "--dry-run",
    "--root",
    dir,
    "--reference",
    reference,
    "--datasets",
    "d1",
    "--methods",
    "BM25",
    "--out-dir",
    join(dir, "out"),
    "--manifest-dir",
    join(dir, "manifests"),
    "--dataset-gate",
    "warn",
    "--dataset-tolerance-points",
    "0.25",
  ]);
  assert.equal(result.status, 0, result.stderr);

  const manifest = JSON.parse(readFileSync(join(dir, "manifests", "baseline-parity-runner.json"), "utf8"));
  const checkCommand = manifest.commands.find((command) => command.name === "baseline-parity-check").command;
  assert.deepEqual(checkCommand.slice(checkCommand.indexOf("--dataset-gate"), checkCommand.indexOf("--dataset-gate") + 2), [
    "--dataset-gate",
    "warn",
  ]);
  assert.deepEqual(
    checkCommand.slice(checkCommand.indexOf("--dataset-tolerance-points"), checkCommand.indexOf("--dataset-tolerance-points") + 2),
    ["--dataset-tolerance-points", "0.25"],
  );
  assert.equal(manifest.options.datasetGate, "warn");
  assert.equal(manifest.options.datasetTolerance, 0.25);
});

test("hybrid-strict readiness profile fails average-green baseline parity with dataset outlier", () => {
  const dir = mkdtempSync(join(tmpdir(), "bb25-audit-readiness-"));
  const baselineParity = {
    passed: true,
    failures: [],
    comparisons: [
      {
        method: "BM25",
        datasets: [
          { dataset: "d1", reference: 50, actual: 50, diff: 0 },
          { dataset: "d2", reference: 60, actual: 59.4, diff: 0.6 },
        ],
      },
    ],
  };
  createHybridReadinessRoot(dir, baselineParity);

  const devOut = join(dir, "readiness-dev.json");
  const dev = runScript("scripts/audit-benchmark-readiness.mjs", [
    "--root",
    dir,
    "--profile",
    "hybrid",
    "--datasets",
    "d1,d2",
    "--out",
    devOut,
  ]);
  assert.equal(dev.status, 0, dev.stderr);
  assert.equal(JSON.parse(readFileSync(devOut, "utf8")).passed, true);

  const out = join(dir, "readiness.json");
  const result = runScript("scripts/audit-benchmark-readiness.mjs", [
    "--root",
    dir,
    "--profile",
    "hybrid-strict",
    "--datasets",
    "d1,d2",
    "--dataset-tolerance-points",
    "0.50",
    "--out",
    out,
  ]);
  assert.equal(result.status, 1);
  const payload = JSON.parse(readFileSync(out, "utf8"));
  assert.equal(payload.passed, false);
  assert.ok(payload.checks.some((check) => check.message.includes("dataset-level baseline parity failed")));
});

test("hybrid readiness protocol assertions catch manifest protocol drift", () => {
  const dir = mkdtempSync(join(tmpdir(), "bb25-audit-protocol-"));
  createHybridReadinessRoot(dir, {
    passed: true,
    failures: [],
    comparisons: [
      {
        method: "BM25",
        datasets: [
          { dataset: "d1", reference: 50, actual: 50, diff: 0 },
          { dataset: "d2", reference: 60, actual: 60, diff: 0 },
        ],
      },
    ],
  });

  const manifestPath = join(dir, "manifests", "ts-hybrid-beir-internal.json");
  const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
  manifest.protocolVersion = 99;
  manifest.cliArgs["bm25-method"] = "robertson";
  manifest.datasetInputs[0].resultSummary.options.scorers = ["bm25", "dense", "rrf"];
  writeJson(manifestPath, manifest);

  const out = join(dir, "readiness-protocol.json");
  const result = runScript("scripts/audit-benchmark-readiness.mjs", [
    "--root",
    dir,
    "--profile",
    "hybrid",
    "--datasets",
    "d1,d2",
    "--out",
    out,
  ]);
  assert.equal(result.status, 1);
  const payload = JSON.parse(readFileSync(out, "utf8"));
  assert.ok(payload.checks.some((check) => check.message.includes("protocolVersion 99 is not supported")));
  assert.ok(payload.checks.some((check) => check.message.includes("bm25-method must be lucene")));
  assert.ok(payload.checks.some((check) => check.message.includes("result scorer filter missing: convex")));
});

test("hybrid claim gate requires green baselines and excludes diagnostic claim rows", () => {
  const dir = mkdtempSync(join(tmpdir(), "bb25-hybrid-claim-"));
  const fixture = createHybridClaimGateFixture(dir, true);
  const out = join(dir, "claim-gate.json");
  const result = runScript("scripts/check-hybrid-claim-gate.mjs", [
    "--actual",
    fixture.actual,
    "--reference",
    fixture.reference,
    "--baseline-parity",
    fixture.baselineParity,
    "--actual-manifest",
    fixture.actualManifest,
    "--baseline-runner-manifest",
    fixture.baselineRunnerManifest,
    "--methods",
    "bayesian_vector_balanced,balanced_fusion",
    "--out",
    out,
  ]);
  assert.equal(result.status, 0, result.stderr);
  const payload = JSON.parse(readFileSync(out, "utf8"));
  assert.equal(payload.passed, true);
  assert.equal(payload.claims.length, 1);
  assert.equal(payload.claims[0].method, "bayesian_vector_balanced");
  assert.ok(Math.abs(payload.claims[0].deltas.vsBm25 - 6.4) < 1e-9);
  assert.deepEqual(payload.excluded, [{ method: "balanced_fusion", kind: "diagnostic", reason: "not a zero-shot scorer kind" }]);
});

test("hybrid claim gate blocks claims when baseline parity is not green", () => {
  const dir = mkdtempSync(join(tmpdir(), "bb25-hybrid-claim-blocked-"));
  const fixture = createHybridClaimGateFixture(dir, false);
  const out = join(dir, "claim-gate.json");
  const result = runScript("scripts/check-hybrid-claim-gate.mjs", [
    "--actual",
    fixture.actual,
    "--reference",
    fixture.reference,
    "--baseline-parity",
    fixture.baselineParity,
    "--actual-manifest",
    fixture.actualManifest,
    "--baseline-runner-manifest",
    fixture.baselineRunnerManifest,
    "--methods",
    "bayesian_vector_balanced",
    "--out",
    out,
  ]);
  assert.equal(result.status, 1);
  const payload = JSON.parse(readFileSync(out, "utf8"));
  assert.ok(payload.failures.some((check) => check.kind === "baseline_parity"));
});

test("hybrid claim gate requires strict dataset parity and metric provenance", () => {
  const dir = mkdtempSync(join(tmpdir(), "bb25-hybrid-claim-protocol-"));
  const fixture = createHybridClaimGateFixture(dir, true);

  const baselineParity = JSON.parse(readFileSync(fixture.baselineParity, "utf8"));
  baselineParity.datasetGate = { mode: "warn", passed: true, violations: [] };
  writeJson(fixture.baselineParity, baselineParity);

  const runnerManifest = JSON.parse(readFileSync(fixture.baselineRunnerManifest, "utf8"));
  delete runnerManifest.options.metric;
  writeJson(fixture.baselineRunnerManifest, runnerManifest);

  const out = join(dir, "claim-gate.json");
  const result = runScript("scripts/check-hybrid-claim-gate.mjs", [
    "--actual",
    fixture.actual,
    "--reference",
    fixture.reference,
    "--baseline-parity",
    fixture.baselineParity,
    "--actual-manifest",
    fixture.actualManifest,
    "--baseline-runner-manifest",
    fixture.baselineRunnerManifest,
    "--methods",
    "bayesian_vector_balanced",
    "--out",
    out,
  ]);
  assert.equal(result.status, 1);
  const payload = JSON.parse(readFileSync(out, "utf8"));
  assert.ok(payload.failures.some((check) => check.kind === "baseline_dataset_gate"));
  assert.ok(payload.failures.some((check) => check.kind === "same_metric"));
});
