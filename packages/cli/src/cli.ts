#!/usr/bin/env node
/** bb25 CLI: index / search / warmup / bench. */
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { parseArgs } from "node:util";
import {
  BM25Scorer,
  BayesianBM25Scorer,
  VectorScorer,
  HybridScorer,
  Tokenizer,
  type BM25Method,
} from "@bb25/core";
import {
  DEFAULT_PARAMS,
  corpusFromIndex,
  loadIndex,
  saveIndex,
  type EmbedderMeta,
  type IndexDoc,
  type IndexFile,
} from "./indexFile.js";
import { loadDocs, loadQueries, loadQrels } from "./jsonl.js";
import {
  runBenchWithDetails,
  formatTable,
  rankDocs,
  type BaseRateMethod,
  type BaseRateOption,
  type BayesianParameterOption,
  type BenchQuery,
  type CalibrationResult,
  type FitSplitOptions,
  type MetricStyle,
  type ScorerRun,
} from "./bench.js";

const USAGE = `bb25 — Bayesian BM25 CLI

Usage:
  bb25 index <corpus.jsonl> -o <index.json> [--embed] [--dtype fp32] [--model Xenova/bge-m3] [--cache-dir <dir>] [--local-only]
  bb25 search "<query>" --index <index.json> [--top-k 10] [--mode or|and|bm25|bayesian] [--embed] [--cache-dir <dir>] [--local-only]
  bb25 warmup [--dtype fp32] [--model Xenova/bge-m3] [--cache-dir <dir>] [--local-only]
  bb25 bench --docs <docs.jsonl> --queries <queries.jsonl> --qrels <qrels.tsv|.jsonl>
             [--embed] [--dtype fp32] [--model Xenova/bge-m3] [--cache-dir <dir>] [--local-only]
             [--cutoffs 5,10,20,100]
             [--scorers bm25,dense,convex,rrf]
             [--bm25-method robertson|lucene] [--query-terms terms] [--doc-terms terms] [--json]
             [--metric-style pytrec|python-reference]
             [--doc-fields title,body] [--doc-field-terms field_terms]
             [--alpha <number|auto>] [--beta <number|auto>]
             [--base-rate <number|auto>] [--base-rate-method percentile|mixture|elbow] [--calibration]
             [--fit-split] [--fit-train-ratio 0.5] [--fit-split-seed 42] [--fit-split-file split.json]
             [--output-json results.json]
             [--candidate-depth 1000] [--trec-run-dir runs] [--trec-run-depth 1000]
`;

function parseBm25Method(value: string | undefined): BM25Method {
  const method = value ?? "robertson";
  if (method === "robertson" || method === "lucene") {
    return method;
  }
  throw new Error(`invalid BM25 method '${method}' (robertson|lucene)`);
}

function parseBaseRate(value: string | undefined): BaseRateOption {
  if (value === undefined || value === "null" || value === "none") {
    return null;
  }
  if (value === "auto") {
    return "auto";
  }
  const n = Number(value);
  if (!(n > 0.0 && n < 1.0)) {
    throw new Error(`base-rate must be in (0, 1), "auto", "none", or "null"; got ${value}`);
  }
  return n;
}

function parseBaseRateMethod(value: string | undefined): BaseRateMethod {
  const method = value ?? "percentile";
  if (method === "percentile" || method === "mixture" || method === "elbow") {
    return method;
  }
  throw new Error(`invalid base-rate method '${method}' (percentile|mixture|elbow)`);
}

function parseBayesianParameter(name: "alpha" | "beta", value: string | undefined, fallback: number): BayesianParameterOption {
  if (value === undefined) {
    return fallback;
  }
  if (value === "auto") {
    return "auto";
  }
  const n = Number(value);
  if (!Number.isFinite(n)) {
    throw new Error(`${name} must be a finite number or "auto"; got ${value}`);
  }
  return n;
}

function parseMetricStyle(value: string | undefined): MetricStyle {
  const style = value ?? "pytrec";
  if (style === "pytrec" || style === "python-reference") {
    return style;
  }
  throw new Error(`invalid metric style '${style}' (pytrec|python-reference)`);
}

function sanitizeRunName(name: string): string {
  return name.replace(/[^A-Za-z0-9_.-]+/g, "_");
}

function parseCsv(value: string | undefined): string[] {
  if (value === undefined) {
    return [];
  }
  return value
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item.length > 0);
}

function readFitSplitFile(path: string): Pick<FitSplitOptions, "trainQueryIds" | "evalQueryIds" | "splitSource"> {
  const raw = JSON.parse(readFileSync(path, "utf8")) as {
    trainQueryIds?: unknown;
    evalQueryIds?: unknown;
    train?: unknown;
    eval?: unknown;
  };
  const train = raw.trainQueryIds ?? raw.train;
  const evalIds = raw.evalQueryIds ?? raw.eval;
  if (!Array.isArray(train) || !train.every((id) => typeof id === "string")) {
    throw new Error(`fit split file ${path} must contain trainQueryIds: string[]`);
  }
  if (!Array.isArray(evalIds) || !evalIds.every((id) => typeof id === "string")) {
    throw new Error(`fit split file ${path} must contain evalQueryIds: string[]`);
  }
  return {
    trainQueryIds: train,
    evalQueryIds: evalIds,
    splitSource: path,
  };
}

function writeTrecRunFiles(runDir: string, runs: ScorerRun[], maxDepth: number | null): void {
  mkdirSync(runDir, { recursive: true });
  const byScorer = new Map<string, ScorerRun[]>();
  for (const run of runs) {
    const existing = byScorer.get(run.scorer);
    if (existing === undefined) {
      byScorer.set(run.scorer, [run]);
    } else {
      existing.push(run);
    }
  }

  for (const [scorer, scorerRuns] of byScorer) {
    const lines: string[] = [];
    const tag = `bb25-${sanitizeRunName(scorer)}`;
    for (const run of scorerRuns) {
      const finiteScores = run.scores.filter(([, score]) => Number.isFinite(score));
      const ranked = rankDocs(finiteScores);
      const depth = maxDepth === null ? ranked.length : Math.min(maxDepth, ranked.length);
      const scoreById = new Map(finiteScores);
      for (let i = 0; i < depth; i++) {
        const docId = ranked[i]!;
        lines.push(`${run.queryId} Q0 ${docId} ${i + 1} ${scoreById.get(docId) ?? 0} ${tag}`);
      }
    }
    writeFileSync(`${runDir}/${sanitizeRunName(scorer)}.trec`, lines.join("\n") + "\n", "utf8");
  }
}

function formatCalibrationTable(results: CalibrationResult[]): string {
  const lines = ["scorer\tsamples\tbins\tece\tbrier"];
  for (const r of results) {
    lines.push([r.scorer, String(r.samples), String(r.bins), r.ece.toFixed(6), r.brier.toFixed(6)].join("\t"));
  }
  return lines.join("\n");
}

async function makeEmbedder(
  dtype: string,
  model: string,
  cacheDir?: string,
  localOnly = false,
): Promise<import("@bb25/embeddings").BgeM3Embedder> {
  const { BgeM3Embedder } = await import("@bb25/embeddings");
  return new BgeM3Embedder({ dtype: dtype as never, model, cacheDir, localOnly });
}

function embedderMeta(model: string, dtype: string, dim: number, cacheDir?: string, localOnly = false): EmbedderMeta {
  return {
    model,
    dim,
    dtype,
    pooling: "cls",
    normalize: true,
    ...(cacheDir !== undefined ? { cacheDir } : {}),
    ...(localOnly ? { localOnly } : {}),
  };
}

async function cmdIndex(argv: string[]): Promise<void> {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      out: { type: "string", short: "o" },
      embed: { type: "boolean", default: false },
      dtype: { type: "string", default: "fp32" },
      model: { type: "string", default: "Xenova/bge-m3" },
      "cache-dir": { type: "string" },
      "local-only": { type: "boolean", default: false },
      k1: { type: "string" },
      b: { type: "string" },
      "bm25-method": { type: "string", default: "robertson" },
      "metric-style": { type: "string", default: "pytrec" },
      "id-field": { type: "string", default: "doc_id" },
      "text-field": { type: "string", default: "text" },
      "terms-field": { type: "string" },
      "embedding-field": { type: "string" },
    },
  });
  const corpusPath = positionals[0];
  if (corpusPath === undefined || values.out === undefined) {
    throw new Error("index: requires <corpus.jsonl> and -o <index.json>");
  }

  const docs = loadDocs(
    corpusPath,
    values["id-field"],
    values["text-field"],
    values["embedding-field"] ?? null,
    values["terms-field"] ?? null,
  );
  const bm25Method = parseBm25Method(values["bm25-method"]);

  let embedder: EmbedderMeta | null = null;
  if (values.embed) {
    const e = await makeEmbedder(values.dtype!, values.model!, values["cache-dir"], values["local-only"]);
    process.stderr.write(`Embedding ${docs.length} documents...\n`);
    const vecs = await e.embed(docs.map((d) => d.text));
    for (let i = 0; i < docs.length; i++) docs[i]!.embedding = Array.from(vecs[i]!);
    embedder = embedderMeta(values.model!, values.dtype!, e.dim, values["cache-dir"], values["local-only"]);
  }

  const indexDocs: IndexDoc[] = docs.map((d) => ({
    id: d.docId,
    text: d.text,
    terms: d.terms,
    embedding: d.embedding.length > 0 ? d.embedding : null,
  }));

  // Compute stats via a transient corpus.
  const corpus = new (await import("@bb25/core")).Corpus();
  for (const d of indexDocs) {
    if (d.terms !== undefined && d.terms !== null) {
      corpus.addDocumentTokens(d.id, d.text, d.terms, d.embedding ?? []);
    } else {
      corpus.addDocument(d.id, d.text, d.embedding ?? []);
    }
  }
  corpus.buildIndex();

  const index: IndexFile = {
    version: 1,
    params: {
      ...DEFAULT_PARAMS,
      ...(values.k1 !== undefined ? { k1: Number(values.k1) } : {}),
      ...(values.b !== undefined ? { b: Number(values.b) } : {}),
      bm25Method,
    },
    embedder,
    documents: indexDocs,
    stats: { n: corpus.n, avgdl: corpus.avgdl },
  };
  saveIndex(values.out, index);
  process.stderr.write(`Wrote ${values.out} (${indexDocs.length} docs, avgdl=${corpus.avgdl.toFixed(2)})\n`);
}

async function cmdSearch(argv: string[]): Promise<void> {
  const { values, positionals } = parseArgs({
    args: argv,
    allowPositionals: true,
    options: {
      index: { type: "string" },
      "top-k": { type: "string", default: "10" },
      mode: { type: "string", default: "bm25" },
      embed: { type: "boolean", default: false },
      dtype: { type: "string", default: "fp32" },
      model: { type: "string", default: "Xenova/bge-m3" },
      "cache-dir": { type: "string" },
      "local-only": { type: "boolean", default: false },
    },
  });
  const query = positionals[0];
  if (query === undefined || values.index === undefined) {
    throw new Error('search: requires "<query>" and --index <index.json>');
  }
  const topK = Number(values["top-k"]);
  const mode = values.mode!;

  const index = loadIndex(values.index);
  const corpus = corpusFromIndex(index);
  const tokenizer = new Tokenizer();
  const terms = tokenizer.tokenize(query);

  const bm25 = new BM25Scorer(corpus, index.params.k1, index.params.b, index.params.bm25Method ?? "robertson");
  const bayes = new BayesianBM25Scorer(bm25, index.params.alpha, index.params.beta, index.params.baseRate);
  const vector = new VectorScorer();
  const hybrid = new HybridScorer(bayes, vector, index.params.hybridAlpha);

  let queryEmbedding: number[] | null = null;
  if (mode === "or" || mode === "and") {
    if (values.embed) {
      const e = await makeEmbedder(values.dtype!, values.model!, values["cache-dir"], values["local-only"]);
      const embs = (await e.embed([query])).map((v) => Array.from(v));
      queryEmbedding = embs[0] ?? null;
    } else {
      throw new Error(`search --mode ${mode} needs a query embedding; pass --embed`);
    }
  }

  const scores: [string, number][] = corpus.documents().map((d) => {
    switch (mode) {
      case "bm25":
        return [d.id, bm25.score(terms, d)];
      case "bayesian":
        return [d.id, bayes.score(terms, d)];
      case "or":
        return [d.id, hybrid.scoreOr(terms, queryEmbedding!, d)];
      case "and":
        return [d.id, hybrid.scoreAnd(terms, queryEmbedding!, d)];
      default:
        throw new Error(`unknown --mode '${mode}' (or|and|bm25|bayesian)`);
    }
  });

  const scoreById = new Map(scores);
  const ranked = rankDocs(scores).slice(0, topK);
  for (let i = 0; i < ranked.length; i++) {
    const id = ranked[i]!;
    process.stdout.write(`${i + 1}\t${id}\t${(scoreById.get(id) ?? 0).toFixed(6)}\n`);
  }
}

async function cmdWarmup(argv: string[]): Promise<void> {
  const { values } = parseArgs({
    args: argv,
    options: {
      dtype: { type: "string", default: "fp32" },
      model: { type: "string", default: "Xenova/bge-m3" },
      "cache-dir": { type: "string" },
      "local-only": { type: "boolean", default: false },
    },
  });
  const e = await makeEmbedder(values.dtype!, values.model!, values["cache-dir"], values["local-only"]);
  process.stderr.write(`Warming up ${values.model} (dtype=${values.dtype})...\n`);
  await e.warmup();
  process.stderr.write("Ready.\n");
}

async function cmdBench(argv: string[]): Promise<void> {
  const { values } = parseArgs({
    args: argv,
    options: {
      docs: { type: "string" },
      queries: { type: "string" },
      qrels: { type: "string" },
      embed: { type: "boolean", default: false },
      dtype: { type: "string", default: "fp32" },
      model: { type: "string", default: "Xenova/bge-m3" },
      "cache-dir": { type: "string" },
      "local-only": { type: "boolean", default: false },
      cutoffs: { type: "string", default: "5,10,20,100" },
      k1: { type: "string" },
      b: { type: "string" },
      alpha: { type: "string" },
      beta: { type: "string" },
      "base-rate": { type: "string" },
      "base-rate-method": { type: "string", default: "percentile" },
      "base-rate-sample-size": { type: "string" },
      "base-rate-seed": { type: "string" },
      "fit-split": { type: "boolean", default: false },
      "fit-train-ratio": { type: "string", default: "0.5" },
      "fit-split-seed": { type: "string", default: "42" },
      "fit-split-file": { type: "string" },
      "bm25-method": { type: "string", default: "robertson" },
      "metric-style": { type: "string", default: "pytrec" },
      scorers: { type: "string" },
      "doc-terms": { type: "string" },
      "doc-fields": { type: "string" },
      "doc-field-terms": { type: "string" },
      "query-terms": { type: "string" },
      "candidate-depth": { type: "string" },
      "doc-embedding": { type: "string" },
      "query-embedding": { type: "string" },
      "trec-run-dir": { type: "string" },
      "trec-run-depth": { type: "string" },
      calibration: { type: "boolean", default: false },
      "calibration-bins": { type: "string", default: "10" },
      json: { type: "boolean", default: false },
      "output-json": { type: "string" },
    },
  });
  if (values.docs === undefined || values.queries === undefined || values.qrels === undefined) {
    throw new Error("bench: requires --docs --queries --qrels");
  }
  const cutoffs = values.cutoffs!
    .split(",")
    .map((s) => parseInt(s.trim(), 10))
    .filter((n) => n > 0)
    .sort((a, b) => a - b);

  const k1 = values.k1 !== undefined ? Number(values.k1) : 1.2;
  const b = values.b !== undefined ? Number(values.b) : 0.75;
  const alpha = parseBayesianParameter("alpha", values.alpha, 1.0);
  const beta = parseBayesianParameter("beta", values.beta, 0.5);
  const baseRate = parseBaseRate(values["base-rate"]);
  const baseRateMethod = parseBaseRateMethod(values["base-rate-method"]);
  const baseRateSampleSize =
    values["base-rate-sample-size"] === undefined ? 50 : Number(values["base-rate-sample-size"]);
  const baseRateSeed = values["base-rate-seed"] === undefined ? 42 : Number(values["base-rate-seed"]);
  const fitTrainRatio = Number(values["fit-train-ratio"]);
  const fitSplitSeed = Number(values["fit-split-seed"]);
  const bm25Method = parseBm25Method(values["bm25-method"]);
  const metricStyle = parseMetricStyle(values["metric-style"]);
  const scorers = parseCsv(values.scorers);
  const candidateDepth = values["candidate-depth"] === undefined ? null : Number(values["candidate-depth"]);
  const trecRunDepth = values["trec-run-depth"] === undefined ? null : Number(values["trec-run-depth"]);
  const calibrationBins = values.calibration ? Number(values["calibration-bins"]) : null;
  const docFields = parseCsv(values["doc-fields"]);
  const explicitFitSplit = values["fit-split-file"] === undefined ? null : readFitSplitFile(values["fit-split-file"]);
  if (trecRunDepth !== null && (!(trecRunDepth > 0) || !Number.isInteger(trecRunDepth))) {
    throw new Error(`trec-run-depth must be a positive integer, got ${values["trec-run-depth"]}`);
  }
  if (!(baseRateSampleSize > 0) || !Number.isInteger(baseRateSampleSize)) {
    throw new Error(`base-rate-sample-size must be a positive integer, got ${values["base-rate-sample-size"]}`);
  }
  if (!Number.isInteger(baseRateSeed)) {
    throw new Error(`base-rate-seed must be an integer, got ${values["base-rate-seed"]}`);
  }
  if (!(fitTrainRatio > 0.0 && fitTrainRatio < 1.0)) {
    throw new Error(`fit-train-ratio must be in (0, 1), got ${values["fit-train-ratio"]}`);
  }
  if (!Number.isInteger(fitSplitSeed)) {
    throw new Error(`fit-split-seed must be an integer, got ${values["fit-split-seed"]}`);
  }
  if (calibrationBins !== null && (!(calibrationBins > 0) || !Number.isInteger(calibrationBins))) {
    throw new Error(`calibration-bins must be a positive integer, got ${values["calibration-bins"]}`);
  }

  const docs = loadDocs(
    values.docs,
    "doc_id",
    "text",
    values["doc-embedding"] ?? null,
    values["doc-terms"] ?? null,
    values["doc-field-terms"] ?? null,
  );
  const queries = loadQueries(values.queries, "query_id", "text", values["query-terms"] ?? null, values["query-embedding"] ?? null);
  const qrels = loadQrels(values.qrels);
  const multiField =
    docFields.length === 0
      ? null
      : {
          fields: docFields,
          docFields: new Map(
            docs.map((doc) => {
              if (doc.fields === null) {
                throw new Error(`doc ${doc.docId} is missing multi-field terms; pass --doc-field-terms`);
              }
              return [doc.docId, doc.fields] as const;
            }),
          ),
        };

  if (values.embed) {
    const e = await makeEmbedder(values.dtype!, values.model!, values["cache-dir"], values["local-only"]);
    process.stderr.write(`Embedding ${docs.length} docs + ${queries.length} queries...\n`);
    const docVecs = await e.embed(docs.map((d) => d.text));
    for (let i = 0; i < docs.length; i++) docs[i]!.embedding = Array.from(docVecs[i]!);
    const qVecs = await e.embed(queries.map((q) => q.text));
    for (let i = 0; i < queries.length; i++) queries[i]!.embedding = Array.from(qVecs[i]!);
  }

  const { Corpus } = await import("@bb25/core");
  const corpus = new Corpus();
  for (const d of docs) {
    if (d.terms !== null) {
      corpus.addDocumentTokens(d.docId, d.text, d.terms, d.embedding);
    } else {
      corpus.addDocument(d.docId, d.text, d.embedding);
    }
  }
  corpus.buildIndex();

  const benchQueries: BenchQuery[] = queries.map((q) => ({
    queryId: q.queryId,
    text: q.text,
    terms: q.terms,
    embedding: q.embedding,
  }));

  const scorerRuns: ScorerRun[] | undefined = values["trec-run-dir"] === undefined ? undefined : [];
  const details = runBenchWithDetails(corpus, benchQueries, qrels, {
    k1,
    b,
    alpha,
    beta,
    baseRate,
    baseRateMethod,
    baseRateSampleSize,
    baseRateSeed,
    bm25Method,
    metricStyle,
    scorers: scorers.length === 0 ? null : scorers,
    candidateDepth,
    cutoffs,
    calibrationBins,
    fitSplit:
      values["fit-split"] || explicitFitSplit !== null
        ? { trainRatio: fitTrainRatio, seed: fitSplitSeed, ...(explicitFitSplit ?? {}) }
        : null,
    multiField,
    runs: scorerRuns,
  });
  if (values["trec-run-dir"] !== undefined) {
    writeTrecRunFiles(values["trec-run-dir"], scorerRuns ?? [], trecRunDepth);
  }
  const payload = {
    cutoffs,
    options: details.options,
    ...(values["trec-run-dir"] !== undefined ? { trecRunDir: values["trec-run-dir"], trecRunDepth } : {}),
    results: details.results,
    scorers: details.scorers,
    ...(details.calibration.length > 0 ? { calibration: details.calibration } : {}),
    ...(details.fittedSplit !== null ? { fittedSplit: details.fittedSplit } : {}),
    ...(details.attentionSplits.length > 0 ? { attentionSplits: details.attentionSplits } : {}),
    ...(details.denseCalibrationSplits.length > 0
      ? { denseCalibrationSplits: details.denseCalibrationSplits }
      : {}),
  };
  const json = JSON.stringify(payload, null, 2) + "\n";
  if (values["output-json"] !== undefined) {
    writeFileSync(values["output-json"], json, "utf8");
  }
  if (values.json) {
    process.stdout.write(json);
  } else {
    process.stdout.write("=== Ranking Metrics ===\n");
    process.stdout.write(formatTable(details.results, cutoffs) + "\n");
    if (details.calibration.length > 0) {
      process.stdout.write("\n=== Calibration Metrics ===\n");
      process.stdout.write(formatCalibrationTable(details.calibration) + "\n");
    }
  }
}

async function main(): Promise<void> {
  const [, , cmd, ...rest] = process.argv;
  switch (cmd) {
    case "index":
      await cmdIndex(rest);
      break;
    case "search":
      await cmdSearch(rest);
      break;
    case "warmup":
      await cmdWarmup(rest);
      break;
    case "bench":
      await cmdBench(rest);
      break;
    case "-h":
    case "--help":
    case undefined:
      process.stdout.write(USAGE);
      break;
    default:
      process.stderr.write(`unknown command '${cmd}'\n\n${USAGE}`);
      process.exitCode = 1;
  }
}

main().catch((err: unknown) => {
  process.stderr.write(`error: ${err instanceof Error ? err.message : String(err)}\n`);
  process.exitCode = 1;
});
