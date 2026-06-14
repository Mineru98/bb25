#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, statSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const TARGETED_TESTS = [
  "test/modules/parameterLearner.test.ts",
  "test/modules/learnable.test.ts",
  "test/modules/calibration.test.ts",
  "test/modules/attention.test.ts",
  "test/modules/multiHead.test.ts",
  "test/modules/blockMaxIndex.test.ts",
  "test/modules/experiments.test.ts",
];

const FIXTURES = [
  "fixtures/golden_modules.json",
  "fixtures/golden_modules2.json",
  "fixtures/golden_modules3.json",
  "fixtures/golden_modules4.json",
];

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

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
}

function brier(probs, labels) {
  let sum = 0.0;
  for (let i = 0; i < Math.min(probs.length, labels.length); i++) {
    const diff = probs[i] - labels[i];
    sum += diff * diff;
  }
  return sum / Math.min(probs.length, labels.length);
}

function sigmoid(x) {
  return 1.0 / (1.0 + Math.exp(-x));
}

function blockMaxes(matrix, blockSize) {
  return matrix.map((scores) => {
    const blocks = [];
    for (let start = 0; start < scores.length; start += blockSize) {
      blocks.push(Math.max(...scores.slice(start, start + blockSize)));
    }
    return blocks;
  });
}

function check(name, passed, details, extra = {}) {
  return { name, passed, details, ...extra };
}

function syntheticChecks(repoRoot) {
  const modules = readJson(join(repoRoot, "fixtures/golden_modules.json"));
  const modules2 = readJson(join(repoRoot, "fixtures/golden_modules2.json"));
  const modules3 = readJson(join(repoRoot, "fixtures/golden_modules3.json"));

  const checks = [];

  const experiments = modules.experiments.results;
  checks.push(
    check(
      "exp1-exp13-all-pass",
      experiments.length === 13 && experiments.every((row) => row.passed === true),
      `${experiments.filter((row) => row.passed === true).length}/${experiments.length} experiments passed in fixture`,
    ),
  );

  const learners = modules.parameterLearner;
  const learnerFailures = [];
  for (const row of learners) {
    const first = row.lossHistory[0];
    const last = row.lossHistory[row.lossHistory.length - 1];
    if (!(last < first)) learnerFailures.push(`${row.name}: loss ${first} -> ${last}`);
    if (row.name.startsWith("synthetic") && !(row.alpha > 1.0 && row.beta > 0.0)) {
      learnerFailures.push(`${row.name}: alpha/beta did not move positive`);
    }
  }
  checks.push(
    check(
      "parameter-learner-loss-decreases",
      learnerFailures.length === 0,
      learnerFailures.length === 0 ? `${learners.length} seeded learners reduced loss` : learnerFailures.join("; "),
    ),
  );

  const learnable = modules3.learnable;
  const fitWeights = learnable.fit.weights;
  checks.push(
    check(
      "learnable-weights-move-to-reliable-signal",
      fitWeights[0] > 0.5 && fitWeights[0] > fitWeights[1],
      `fit weights=${fitWeights.map((v) => v.toFixed(6)).join(",")}`,
    ),
  );

  const platt = modules2.platt;
  const rawPlatt = platt.scores.map((score) => sigmoid(score));
  const rawBrier = brier(rawPlatt, platt.labels);
  const calibratedBrier = brier(platt.calibrated, platt.labels);
  checks.push(
    check(
      "platt-calibration-reduces-brier",
      calibratedBrier < rawBrier,
      `raw=${rawBrier.toFixed(6)} calibrated=${calibratedBrier.toFixed(6)}`,
      { rawBrier, calibratedBrier },
    ),
  );

  const isotonic = modules2.isotonic;
  const isoMonotonic = isotonic.calibrated.every((value, index, arr) => index === 0 || value >= arr[index - 1]);
  checks.push(
    check(
      "isotonic-calibration-monotonic",
      isoMonotonic,
      `probe outputs=${isotonic.calibrated.map((v) => v.toFixed(6)).join(",")}`,
    ),
  );

  const attention = modules3.attention;
  checks.push(
    check(
      "attention-fit-separates-positive-negative",
      attention.fit.combine[0] > attention.fit.combine[1],
      `fit combine=${attention.fit.combine.map((v) => v.toFixed(6)).join(",")}`,
    ),
  );

  const multiHead = modules3.multiHead;
  checks.push(
    check(
      "multi-head-fit-separates-positive-negative",
      multiHead.fitCombine[0] > multiHead.fitCombine[1],
      `fit combine=${multiHead.fitCombine.map((v) => v.toFixed(6)).join(",")}`,
    ),
  );

  const block = modules3.blockMaxIndex;
  const recomputed = blockMaxes(block.matrix, block.blockSize);
  const falsePrunes = [];
  for (let term = 0; term < block.matrix.length; term++) {
    for (let blockId = 0; blockId < recomputed[term].length; blockId++) {
      const expected = block.blockUpperBound[term][blockId];
      const actual = recomputed[term][blockId];
      if (Math.abs(expected - actual) > 1e-12) {
        falsePrunes.push(`term=${term} block=${blockId} expected=${expected} actual=${actual}`);
      }
    }
  }
  checks.push(
    check(
      "block-max-upper-bounds-are-exact",
      falsePrunes.length === 0,
      falsePrunes.length === 0 ? `${block.nBlocks} blocks verified` : falsePrunes.join("; "),
    ),
  );

  return checks;
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const repoRoot = resolve(dirname(fileURLToPath(import.meta.url)), "..");
  const out = resolve(args.out ?? "reference-results/ts/synthetic-smoke.json");
  const manifestOut = resolve(args["manifest-out"] ?? "reference-results/manifests/synthetic-smoke.json");
  const dryRun = args["dry-run"] !== undefined;
  const skipTests = args["skip-tests"] !== undefined;
  const commands = [];

  if (!skipTests) {
    const command = ["corepack", "pnpm", "--filter", "@bb25/core", "exec", "vitest", "run", ...TARGETED_TESTS];
    commands.push({
      name: "core-synthetic-targeted-tests",
      ...(dryRun ? { command, dryRun: true, returncode: 0, stdout: "", stderr: "" } : run(command, { cwd: repoRoot })),
    });
  }

  const checks = syntheticChecks(repoRoot);
  const failedChecks = checks.filter((row) => !row.passed);
  const failedCommands = commands.filter((row) => row.returncode !== 0);
  const payload = {
    generatedAt: new Date().toISOString(),
    kind: "bb25-synthetic-smoke",
    seed: 42,
    checks,
    passed: failedChecks.length === 0 && failedCommands.length === 0,
  };

  mkdirSync(dirname(out), { recursive: true });
  writeFileSync(out, JSON.stringify(payload, null, 2) + "\n", "utf8");

  const manifest = {
    generatedAt: payload.generatedAt,
    kind: "bb25-synthetic-smoke-manifest",
    dryRun,
    repo: {
      root: repoRoot,
      git: gitInfo(repoRoot),
      node: process.version,
      platform: process.platform,
      arch: process.arch,
    },
    fixtures: FIXTURES.map((path) => fileManifest(join(repoRoot, path))),
    output: fileManifest(out),
    commands,
    summary: {
      checks: checks.length,
      failedChecks: failedChecks.map((row) => row.name),
      failedCommands: failedCommands.map((row) => row.name),
    },
  };
  mkdirSync(dirname(manifestOut), { recursive: true });
  writeFileSync(manifestOut, JSON.stringify(manifest, null, 2) + "\n", "utf8");

  process.stdout.write(
    JSON.stringify(
      {
        out,
        manifest: manifestOut,
        passed: payload.passed,
        checks: checks.length,
        failedChecks: failedChecks.map((row) => row.name),
        failedCommands: failedCommands.map((row) => row.name),
      },
      null,
      2,
    ) + "\n",
  );
  if (!payload.passed) process.exitCode = 1;
}

main();
