#!/usr/bin/env node
import { spawnSync } from "node:child_process";
import { createHash } from "node:crypto";
import { existsSync, mkdirSync, readFileSync, readdirSync, statSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

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

function readJson(path) {
  return JSON.parse(readFileSync(path, "utf8"));
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

function directoryManifest(path, hashContents = false) {
  if (!existsSync(path)) return { path, exists: false };
  const stat = statSync(path);
  if (!stat.isDirectory()) return { ...fileManifest(path), type: "file" };

  let files = 0;
  let directories = 0;
  let bytes = 0;
  const treeHash = hashContents ? createHash("sha256") : null;

  function walk(dir, prefix = "") {
    directories += 1;
    const entries = readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
    for (const entry of entries) {
      const rel = prefix === "" ? entry.name : `${prefix}/${entry.name}`;
      const full = join(dir, entry.name);
      if (entry.isDirectory()) {
        walk(full, rel);
      } else if (entry.isFile()) {
        const data = hashContents ? readFileSync(full) : null;
        const fileStat = statSync(full);
        files += 1;
        bytes += fileStat.size;
        if (treeHash !== null) {
          treeHash.update(rel);
          treeHash.update("\0");
          treeHash.update(data);
          treeHash.update("\0");
        }
      }
    }
  }

  walk(path);
  return {
    path,
    exists: true,
    type: "directory",
    files,
    directories,
    bytes,
    ...(treeHash !== null ? { sha256: treeHash.digest("hex") } : {}),
  };
}

function resolveRepoPath(repoRoot, value) {
  return isAbsolute(value) ? value : join(repoRoot, value);
}

function replaceDataDir(command, fromDir, toDir) {
  return command.map((part) => {
    if (part === fromDir) return toDir;
    if (part.startsWith(`${fromDir}/`)) return `${toDir}${part.slice(fromDir.length)}`;
    return part;
  });
}

function benchCommandFromManifest(manifest, dataDir, repoRoot, bb25Path, args) {
  const original = manifest.benchmark.command.map(String);
  const defaultDataDir = dirname(original[original.indexOf("--docs") + 1] ?? "/tmp/squad/docs.jsonl");
  let command = replaceDataDir(original, defaultDataDir, dataDir);
  if (bb25Path !== undefined) {
    const benchIndex = command.indexOf("bench");
    if (benchIndex < 0) {
      throw new Error("benchmark.command must contain a bench subcommand when --bb25 is used");
    }
    command = ["node", resolveRepoPath(repoRoot, bb25Path), ...command.slice(benchIndex)];
  }
  if (args["embedding-cache-dir"] !== undefined) {
    command.push("--cache-dir", resolve(args["embedding-cache-dir"]));
  }
  if (args["embedding-local-only"] !== undefined) {
    command.push("--local-only");
  }
  return command;
}

function prepareCommandFromManifest(manifest, dataDir, repoRoot, args) {
  const dataset = manifest.dataset ?? {};
  const command = ["node", resolveRepoPath(repoRoot, dataset.prepareScript ?? "scripts/prepare-squad.mjs"), "--out", dataDir];
  if (dataset.maxQuestions !== undefined) command.push("--max-questions", String(dataset.maxQuestions));
  if (dataset.perContext !== undefined) command.push("--per-context", String(dataset.perContext));
  if (args.src !== undefined) {
    command.push("--src", resolve(args.src));
  } else if (args.url !== undefined) {
    command.push("--url", args.url);
  } else if (dataset.url !== undefined) {
    command.push("--url", dataset.url);
  }
  return command;
}

function compareCommand(manifest, actualOut, repoRoot, tolerance) {
  const script = resolveRepoPath(repoRoot, manifest.benchmark.comparisonScript ?? "scripts/check-bench-table.mjs");
  const expected = resolveRepoPath(repoRoot, manifest.benchmark.expectedTable);
  const command = ["node", script, "--expected", expected, "--actual", actualOut, "--tolerance", String(tolerance)];
  const ordering = manifest.ordering;
  if (ordering?.metric !== undefined && Array.isArray(ordering.descending)) {
    command.push("--metric", ordering.metric, "--order", ordering.descending.join(","));
  }
  return command;
}

function parseTolerance(manifest, args) {
  if (args.tolerance !== undefined) return Number(args.tolerance);
  const tolerance = manifest.tolerance ?? {};
  if (args["regenerated-embeddings"] !== undefined && tolerance.regeneratedEmbeddingAbs !== undefined) {
    return Number(tolerance.regeneratedEmbeddingAbs);
  }
  return tolerance.sameRuntimeAbs ?? 0.005;
}

function optionValue(command, flag) {
  const index = command.indexOf(flag);
  if (index < 0) return null;
  return command[index + 1] ?? null;
}

function embeddingInfo(command) {
  const cacheDir = optionValue(command, "--cache-dir");
  return {
    embed: command.includes("--embed"),
    model: optionValue(command, "--model") ?? "Xenova/bge-m3",
    dtype: optionValue(command, "--dtype") ?? "fp32",
    cacheDir,
    localOnly: command.includes("--local-only"),
    cacheEnvironment: {
      TRANSFORMERS_CACHE: process.env.TRANSFORMERS_CACHE ?? null,
      HF_HOME: process.env.HF_HOME ?? null,
      XDG_CACHE_HOME: process.env.XDG_CACHE_HOME ?? null,
    },
  };
}

function main() {
  const args = parseArgs(process.argv.slice(2));
  const here = dirname(fileURLToPath(import.meta.url));
  const repoRoot = resolve(here, "..");
  const fixtureManifestPath = resolve(args.manifest ?? join(repoRoot, "fixtures/bench/squad120-q8-manifest.json"));
  const fixtureManifest = readJson(fixtureManifestPath);
  const dataDir = resolve(args["data-dir"] ?? "/tmp/squad");
  const outDir = resolve(args["out-dir"] ?? "/tmp/bb25-squad-smoke");
  const actualOut = resolve(args["actual-out"] ?? join(outDir, "squad120-q8-results.txt"));
  const resultManifestOut = resolve(args["manifest-out"] ?? join(outDir, "squad120-q8-manifest.json"));
  const tolerance = parseTolerance(fixtureManifest, args);
  const embeddingCacheDir = args["embedding-cache-dir"] === undefined ? null : resolve(args["embedding-cache-dir"]);

  mkdirSync(dataDir, { recursive: true });
  mkdirSync(outDir, { recursive: true });
  mkdirSync(dirname(resultManifestOut), { recursive: true });
  if (embeddingCacheDir !== null && args["embedding-local-only"] === undefined) {
    mkdirSync(embeddingCacheDir, { recursive: true });
  }

  const records = [];
  let failed = false;
  const dryRun = args["dry-run"] !== undefined;
  const benchCommand = benchCommandFromManifest(fixtureManifest, dataDir, repoRoot, args.bb25, args);
  const cacheManifest = embeddingCacheDir === null ? null : directoryManifest(embeddingCacheDir, args["hash-embedding-cache"] !== undefined);

  if (args["require-embedding-cache"] !== undefined) {
    const cacheOk = cacheManifest !== null && cacheManifest.exists === true && cacheManifest.type === "directory" && cacheManifest.files > 0;
    records.push({
      name: "embedding-cache-preflight",
      command: ["embedding-cache-preflight", embeddingCacheDir ?? ""],
      cwd: repoRoot,
      dryRun,
      returncode: cacheOk ? 0 : 1,
      stdout: cacheOk ? "embedding cache present\n" : "",
      stderr:
        cacheManifest === null
          ? "--require-embedding-cache requires --embedding-cache-dir\n"
          : cacheManifest.exists
            ? "embedding cache directory is empty\n"
            : "embedding cache directory is missing\n",
    });
    if (!cacheOk) failed = true;
  }

  if (!failed && args["skip-prepare"] === undefined) {
    const prepare = prepareCommandFromManifest(fixtureManifest, dataDir, repoRoot, args);
    const record = dryRun ? { command: prepare, dryRun: true, returncode: 0, stdout: "", stderr: "" } : run(prepare, { cwd: repoRoot });
    records.push({ name: "prepare-squad", ...record });
    if (record.returncode !== 0) failed = true;
  }

  if (!failed && args["skip-bench"] === undefined) {
    const record = dryRun
      ? { command: benchCommand, dryRun: true, returncode: 0, stdout: "", stderr: "" }
      : run(benchCommand, { cwd: repoRoot });
    records.push({ name: "bench", ...record });
    if (!dryRun) {
      writeFileSync(actualOut, record.stdout, "utf8");
    }
    if (record.returncode !== 0) failed = true;
  }

  if (!failed && args["skip-compare"] === undefined) {
    const compare = compareCommand(fixtureManifest, actualOut, repoRoot, tolerance);
    const record = dryRun ? { command: compare, dryRun: true, returncode: 0, stdout: "", stderr: "" } : run(compare, { cwd: repoRoot });
    records.push({ name: "compare", ...record });
    if (record.returncode !== 0) failed = true;
  }

  const manifest = {
    generatedAt: new Date().toISOString(),
    kind: "bb25-squad-smoke",
    dryRun,
    fixtureManifest: fileManifest(fixtureManifestPath),
    repo: {
      root: repoRoot,
      git: gitInfo(repoRoot),
      node: process.version,
      platform: process.platform,
      arch: process.arch,
    },
    options: {
      dataDir,
      outDir,
      actualOut,
      tolerance,
      regeneratedEmbeddings: args["regenerated-embeddings"] !== undefined,
      skippedPrepare: args["skip-prepare"] !== undefined,
      skippedBench: args["skip-bench"] !== undefined,
      skippedCompare: args["skip-compare"] !== undefined,
      requireEmbeddingCache: args["require-embedding-cache"] !== undefined,
      hashEmbeddingCache: args["hash-embedding-cache"] !== undefined,
    },
    embedding: {
      ...embeddingInfo(benchCommand),
      cache: cacheManifest,
    },
    commands: records,
    files: {
      docs: fileManifest(join(dataDir, "docs.jsonl")),
      queries: fileManifest(join(dataDir, "queries.jsonl")),
      qrels: fileManifest(join(dataDir, "qrels.tsv")),
      expected: fileManifest(resolveRepoPath(repoRoot, fixtureManifest.benchmark.expectedTable)),
      actual: fileManifest(actualOut),
    },
  };
  writeFileSync(resultManifestOut, JSON.stringify(manifest, null, 2) + "\n", "utf8");
  process.stdout.write(
    JSON.stringify(
      {
        manifest: resultManifestOut,
        actual: actualOut,
        failed,
        dryRun,
      },
      null,
      2,
    ) + "\n",
  );
  if (failed) process.exitCode = 1;
}

main();
