#!/usr/bin/env node
/** bb25 CLI: index / search / warmup / bench. */
import { parseArgs } from "node:util";
import {
  BM25Scorer,
  BayesianBM25Scorer,
  VectorScorer,
  HybridScorer,
  Tokenizer,
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
import { runBench, formatTable, rankDocs, type BenchQuery } from "./bench.js";

const USAGE = `bb25 — Bayesian BM25 CLI

Usage:
  bb25 index <corpus.jsonl> -o <index.json> [--embed] [--dtype fp32] [--model Xenova/bge-m3]
  bb25 search "<query>" --index <index.json> [--top-k 10] [--mode or|and|bm25|bayesian] [--embed]
  bb25 warmup [--dtype fp32] [--model Xenova/bge-m3]
  bb25 bench --docs <docs.jsonl> --queries <queries.jsonl> --qrels <qrels.tsv|.jsonl>
             [--embed] [--dtype fp32] [--cutoffs 5,10,20,100]
`;

async function makeEmbedder(dtype: string, model: string): Promise<import("@bb25/embeddings").BgeM3Embedder> {
  const { BgeM3Embedder } = await import("@bb25/embeddings");
  return new BgeM3Embedder({ dtype: dtype as never, model });
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
      k1: { type: "string" },
      b: { type: "string" },
      "id-field": { type: "string", default: "doc_id" },
      "text-field": { type: "string", default: "text" },
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
  );

  let embedder: EmbedderMeta | null = null;
  if (values.embed) {
    const e = await makeEmbedder(values.dtype!, values.model!);
    process.stderr.write(`Embedding ${docs.length} documents...\n`);
    const vecs = await e.embed(docs.map((d) => d.text));
    for (let i = 0; i < docs.length; i++) docs[i]!.embedding = Array.from(vecs[i]!);
    embedder = { model: values.model!, dim: e.dim, dtype: values.dtype!, pooling: "cls", normalize: true };
  }

  const indexDocs: IndexDoc[] = docs.map((d) => ({
    id: d.docId,
    text: d.text,
    embedding: d.embedding.length > 0 ? d.embedding : null,
  }));

  // Compute stats via a transient corpus.
  const corpus = new (await import("@bb25/core")).Corpus();
  for (const d of indexDocs) corpus.addDocument(d.id, d.text, d.embedding ?? []);
  corpus.buildIndex();

  const index: IndexFile = {
    version: 1,
    params: {
      ...DEFAULT_PARAMS,
      ...(values.k1 !== undefined ? { k1: Number(values.k1) } : {}),
      ...(values.b !== undefined ? { b: Number(values.b) } : {}),
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

  const bm25 = new BM25Scorer(corpus, index.params.k1, index.params.b);
  const bayes = new BayesianBM25Scorer(bm25, index.params.alpha, index.params.beta, index.params.baseRate);
  const vector = new VectorScorer();
  const hybrid = new HybridScorer(bayes, vector, index.params.hybridAlpha);

  let queryEmbedding: number[] | null = null;
  if (mode === "or" || mode === "and") {
    if (values.embed) {
      const e = await makeEmbedder(values.dtype!, values.model!);
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
    },
  });
  const e = await makeEmbedder(values.dtype!, values.model!);
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
      cutoffs: { type: "string", default: "5,10,20,100" },
      "doc-embedding": { type: "string" },
      "query-embedding": { type: "string" },
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

  const docs = loadDocs(values.docs, "doc_id", "text", values["doc-embedding"] ?? null);
  const queries = loadQueries(values.queries, "query_id", "text", null, values["query-embedding"] ?? null);
  const qrels = loadQrels(values.qrels);

  if (values.embed) {
    const e = await makeEmbedder(values.dtype!, values.model!);
    process.stderr.write(`Embedding ${docs.length} docs + ${queries.length} queries...\n`);
    const docVecs = await e.embed(docs.map((d) => d.text));
    for (let i = 0; i < docs.length; i++) docs[i]!.embedding = Array.from(docVecs[i]!);
    const qVecs = await e.embed(queries.map((q) => q.text));
    for (let i = 0; i < queries.length; i++) queries[i]!.embedding = Array.from(qVecs[i]!);
  }

  const { Corpus } = await import("@bb25/core");
  const corpus = new Corpus();
  for (const d of docs) corpus.addDocument(d.docId, d.text, d.embedding);
  corpus.buildIndex();

  const benchQueries: BenchQuery[] = queries.map((q) => ({
    queryId: q.queryId,
    text: q.text,
    terms: q.terms,
    embedding: q.embedding,
  }));

  const results = runBench(corpus, benchQueries, qrels, { cutoffs });
  process.stdout.write("=== Ranking Metrics ===\n");
  process.stdout.write(formatTable(results, cutoffs) + "\n");
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
