#!/usr/bin/env node
// Prepare a SQuAD slice as bb25 bench inputs: docs.jsonl (unique contexts),
// queries.jsonl (questions), qrels.tsv (question -> its context, rel=1).
//
// Usage:
//   node scripts/prepare-squad.mjs --out <dir> [--max-questions 100]
//        [--src dev-v1.1.json | --url https://...dev-v1.1.json]
//
// Embeddings are NOT added here; run `bb25 bench --embed` (transformers.js)
// to generate dense vectors on both docs and queries.

import { writeFileSync, readFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";

const DEFAULT_URL =
  "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v1.1.json";

function parseArgs(argv) {
  const out = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (a.startsWith("--")) {
      const key = a.slice(2);
      const val = argv[i + 1] && !argv[i + 1].startsWith("--") ? argv[++i] : "true";
      out[key] = val;
    }
  }
  return out;
}

async function loadSquad(args) {
  if (args.src) {
    return JSON.parse(readFileSync(args.src, "utf8"));
  }
  const url = args.url || DEFAULT_URL;
  process.stderr.write(`Fetching SQuAD from ${url} ...\n`);
  const res = await fetch(url);
  if (!res.ok) throw new Error(`fetch failed: ${res.status} ${res.statusText}`);
  return res.json();
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const outDir = args.out;
  if (!outDir) throw new Error("--out <dir> is required");
  const maxQ = args["max-questions"] ? parseInt(args["max-questions"], 10) : Infinity;
  // Cap questions taken per paragraph so a slice spans many distinct contexts
  // (otherwise the first N questions all share a handful of paragraphs).
  const perContext = args["per-context"] ? parseInt(args["per-context"], 10) : 1;
  if (!existsSync(outDir)) mkdirSync(outDir, { recursive: true });

  const squad = await loadSquad(args);

  const contextToId = new Map();
  const docs = [];
  const queries = [];
  const qrels = [];

  let qCount = 0;
  outer: for (const article of squad.data) {
    for (const para of article.paragraphs) {
      const ctx = para.context;
      let docId = contextToId.get(ctx);
      if (docId === undefined) {
        docId = `c${docs.length}`;
        contextToId.set(ctx, docId);
        docs.push({ doc_id: docId, text: ctx });
      }
      let takenHere = 0;
      for (const qa of para.qas) {
        if (qCount >= maxQ) break outer;
        if (takenHere >= perContext) break;
        const queryId = qa.id;
        queries.push({ query_id: queryId, text: qa.question });
        qrels.push(`${queryId}\t${docId}\t1`);
        qCount += 1;
        takenHere += 1;
      }
    }
  }

  // Keep only docs referenced by the selected queries (when sliced).
  const usedDocs = new Set(qrels.map((l) => l.split("\t")[1]));
  const keptDocs = docs.filter((d) => usedDocs.has(d.doc_id));

  const docsPath = join(outDir, "docs.jsonl");
  const queriesPath = join(outDir, "queries.jsonl");
  const qrelsPath = join(outDir, "qrels.tsv");

  writeFileSync(docsPath, keptDocs.map((d) => JSON.stringify(d)).join("\n") + "\n");
  writeFileSync(queriesPath, queries.map((q) => JSON.stringify(q)).join("\n") + "\n");
  writeFileSync(qrelsPath, qrels.join("\n") + "\n");

  process.stderr.write(
    `Wrote ${keptDocs.length} docs, ${queries.length} queries, ${qrels.length} qrels to ${outDir}\n`,
  );
}

main().catch((e) => {
  process.stderr.write(`error: ${e.message}\n`);
  process.exitCode = 1;
});
