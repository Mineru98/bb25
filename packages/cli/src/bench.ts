/**
 * Ranking-quality benchmark harness. Direct port of the metric + evaluation
 * logic in benchmarks/run_benchmark.py.
 *
 * rankDocs sorts by (-score, docId) — the tie-break is part of the contract and
 * must match the reference exactly. NDCG uses exponential gain (2^rel - 1).
 */
import {
  BM25Scorer,
  BayesianBM25Scorer,
  VectorScorer,
  HybridScorer,
  BayesianProbabilityTransform,
  balancedLogOddsFusion,
  Tokenizer,
  type Corpus,
  type Document,
} from "@bb25/core";

export type RelMap = Map<string, number>;
export type Qrels = Map<string, RelMap>;

/** Sort (docId, score) pairs by descending score, ties broken by ascending docId. */
export function rankDocs(scores: [string, number][]): string[] {
  return scores
    .slice()
    .sort((a, b) => (b[1] - a[1] !== 0 ? b[1] - a[1] : a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([id]) => id);
}

export function averagePrecisionAtK(ranked: string[], relMap: RelMap, k: number): number {
  if (relMap.size === 0) {
    return 0.0;
  }
  let hits = 0;
  let precisionSum = 0.0;
  const top = ranked.slice(0, k);
  for (let idx = 0; idx < top.length; idx++) {
    if ((relMap.get(top[idx]!) ?? 0.0) > 0) {
      hits += 1;
      precisionSum += hits / (idx + 1);
    }
  }
  let relevant = 0;
  for (const r of relMap.values()) {
    if (r > 0) relevant += 1;
  }
  const denom = Math.min(relevant, k);
  return denom === 0 ? 0.0 : precisionSum / denom;
}

export function dcgAtK(ranked: string[], relMap: RelMap, k: number): number {
  let score = 0.0;
  const top = ranked.slice(0, k);
  for (let idx = 0; idx < top.length; idx++) {
    const rel = relMap.get(top[idx]!) ?? 0.0;
    if (rel <= 0) continue;
    score += (2 ** rel - 1) / Math.log2(idx + 2);
  }
  return score;
}

export function ndcgAtK(ranked: string[], relMap: RelMap, k: number): number {
  if (relMap.size === 0) {
    return 0.0;
  }
  const idealRels = [...relMap.values()].filter((r) => r > 0).sort((a, b) => b - a);
  let idealDcg = 0.0;
  for (let idx = 0; idx < Math.min(idealRels.length, k); idx++) {
    idealDcg += (2 ** idealRels[idx]! - 1) / Math.log2(idx + 2);
  }
  if (idealDcg === 0) {
    return 0.0;
  }
  return dcgAtK(ranked, relMap, k) / idealDcg;
}

export function mrrAtK(ranked: string[], relMap: RelMap, k: number): number {
  const top = ranked.slice(0, k);
  for (let idx = 0; idx < top.length; idx++) {
    if ((relMap.get(top[idx]!) ?? 0.0) > 0) {
      return 1.0 / (idx + 1);
    }
  }
  return 0.0;
}

export interface ScorerResult {
  scorer: string;
  queries: number;
  metrics: Record<string, number>;
}

export interface BenchQuery {
  queryId: string;
  text: string;
  terms: string[] | null;
  embedding: number[] | null;
}

function blankMetrics(cutoffs: number[]): Record<string, number> {
  const m: Record<string, number> = {};
  for (const k of cutoffs) {
    m[`ndcg@${k}`] = 0.0;
    m[`map@${k}`] = 0.0;
    m[`mrr@${k}`] = 0.0;
  }
  return m;
}

function accumulate(
  metrics: Record<string, number>,
  ranked: string[],
  relMap: RelMap,
  cutoffs: number[],
): void {
  for (const k of cutoffs) {
    metrics[`ndcg@${k}`]! += ndcgAtK(ranked, relMap, k);
    metrics[`map@${k}`]! += averagePrecisionAtK(ranked, relMap, k);
    metrics[`mrr@${k}`]! += mrrAtK(ranked, relMap, k);
  }
}

/** Evaluate a term-only scorer (terms, doc) -> score. */
export function evaluate(
  queries: BenchQuery[],
  docs: readonly Document[],
  scorerName: string,
  scoreFn: (terms: string[], doc: Document) => number,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
): ScorerResult {
  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    const scores: [string, number][] = docs.map((d) => [d.id, scoreFn(terms, d)]);
    accumulate(metrics, rankDocs(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: scorerName, queries: counted, metrics };
}

/** Evaluate a hybrid scorer (terms, embedding, doc) -> score. Skips embeddingless queries. */
export function evaluateHybrid(
  queries: BenchQuery[],
  docs: readonly Document[],
  scorerName: string,
  scoreFn: (terms: string[], embedding: number[], doc: Document) => number,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
): ScorerResult {
  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    if (q.embedding === null) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    const scores: [string, number][] = docs.map((d) => [d.id, scoreFn(terms, q.embedding!, d)]);
    accumulate(metrics, rankDocs(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: scorerName, queries: counted, metrics };
}

export interface BenchOptions {
  k1?: number;
  b?: number;
  alpha?: number;
  beta?: number;
  cutoffs?: number[];
}

/** Run the standard scorer comparison over a corpus + queries + qrels. */
export function runBench(
  corpus: Corpus,
  queries: BenchQuery[],
  qrels: Qrels,
  options: BenchOptions = {},
): ScorerResult[] {
  const k1 = options.k1 ?? 1.2;
  const b = options.b ?? 0.75;
  const alpha = options.alpha ?? 1.0;
  const beta = options.beta ?? 0.5;
  const cutoffs = options.cutoffs ?? [5, 10, 20, 100];

  const tokenizer = new Tokenizer();
  const docs = corpus.documents();
  const bm25 = new BM25Scorer(corpus, k1, b);
  const bayes = new BayesianBM25Scorer(bm25, alpha, beta, null);
  const vector = new VectorScorer();
  const hybrid = new HybridScorer(bayes, vector, 0.5);

  const hasEmbeddings = queries.some((q) => q.embedding !== null && q.embedding.length > 0);

  const results: ScorerResult[] = [
    evaluate(queries, docs, "bm25", (t, d) => bm25.score(t, d), qrels, tokenizer, cutoffs),
    evaluate(queries, docs, "bayesian", (t, d) => bayes.score(t, d), qrels, tokenizer, cutoffs),
    evaluateFittedBayesian(queries, docs, bm25, qrels, tokenizer, cutoffs, alpha, beta),
  ];

  if (hasEmbeddings) {
    results.push(
      evaluateHybrid(queries, docs, "hybrid_or", (t, e, d) => hybrid.scoreOr(t, e, d), qrels, tokenizer, cutoffs),
      evaluateHybrid(queries, docs, "hybrid_and", (t, e, d) => hybrid.scoreAnd(t, e, d), qrels, tokenizer, cutoffs),
      evaluateBalanced(queries, docs, bayes, vector, qrels, tokenizer, cutoffs),
      evaluateRrf(queries, docs, bm25, vector, qrels, tokenizer, cutoffs),
    );
  }

  return results;
}

function evaluateFittedBayesian(
  queries: BenchQuery[],
  docs: readonly Document[],
  bm25: BM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  alpha: number,
  beta: number,
): ScorerResult {
  const trainScores: number[] = [];
  const trainLabels: number[] = [];
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    for (const d of docs) {
      if (!relMap.has(d.id)) continue;
      const raw = bm25.score(terms, d);
      if (raw > 0.0) {
        trainScores.push(raw);
        trainLabels.push((relMap.get(d.id) ?? 0) > 0 ? 1.0 : 0.0);
      }
    }
  }
  if (trainScores.length < 4) {
    return { scorer: "bayesian_fitted", queries: 0, metrics: blankMetrics(cutoffs) };
  }
  const transform = new BayesianProbabilityTransform(alpha, beta, null);
  transform.fit(trainScores, trainLabels);
  const fitted = new BayesianBM25Scorer(bm25, transform.alpha, transform.beta, null);
  return evaluate(queries, docs, "bayesian_fitted", (t, d) => fitted.score(t, d), qrels, tokenizer, cutoffs);
}

function evaluateBalanced(
  queries: BenchQuery[],
  docs: readonly Document[],
  bayes: BayesianBM25Scorer,
  vector: VectorScorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  weight = 0.5,
): ScorerResult {
  const metrics = blankMetrics(cutoffs);
  const docIds = docs.map((d) => d.id);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    if (q.embedding === null) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    const sparse = docs.map((d) => bayes.score(terms, d));
    const dense = docs.map((d) => vector.score(q.embedding!, d));
    const fused = balancedLogOddsFusion(sparse, dense, weight);
    const scores: [string, number][] = docIds.map((id, i) => [id, fused[i]!]);
    accumulate(metrics, rankDocs(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: "balanced_fusion", queries: counted, metrics };
}

function evaluateRrf(
  queries: BenchQuery[],
  docs: readonly Document[],
  bm25: BM25Scorer,
  vector: VectorScorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  k = 60,
): ScorerResult {
  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    if (q.embedding === null) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    const bmRanked = rankDocs(docs.map((d) => [d.id, bm25.score(terms, d)]));
    const vecRanked = rankDocs(docs.map((d) => [d.id, vector.score(q.embedding!, d)]));
    const bmRank = new Map<string, number>();
    bmRanked.forEach((id, i) => bmRank.set(id, i + 1));
    const vecRank = new Map<string, number>();
    vecRanked.forEach((id, i) => vecRank.set(id, i + 1));
    const scores: [string, number][] = docs.map((d) => [
      d.id,
      1.0 / (k + bmRank.get(d.id)!) + 1.0 / (k + vecRank.get(d.id)!),
    ]);
    accumulate(metrics, rankDocs(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: "rrf", queries: counted, metrics };
}

/** Tab-separated table, mirroring run_benchmark.py format_table. */
export function formatTable(results: ScorerResult[], cutoffs: number[]): string {
  const headers = ["scorer", "queries"];
  for (const k of cutoffs) headers.push(`ndcg@${k}`, `map@${k}`, `mrr@${k}`);
  const lines = [headers.join("\t")];
  for (const r of results) {
    const row = [r.scorer, String(r.queries)];
    for (const k of cutoffs) {
      row.push(
        (r.metrics[`ndcg@${k}`] ?? 0).toFixed(4),
        (r.metrics[`map@${k}`] ?? 0).toFixed(4),
        (r.metrics[`mrr@${k}`] ?? 0).toFixed(4),
      );
    }
    lines.push(row.join("\t"));
  }
  return lines.join("\n");
}
