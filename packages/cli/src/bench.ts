/**
 * Ranking-quality benchmark harness.
 *
 * rankDocs sorts by (-score, docId) — the tie-break is part of the contract and
 * is deterministic. NDCG uses exponential gain (2^rel - 1).
 */
import {
  BM25Scorer,
  BayesianBM25Scorer,
  VectorScorer,
  HybridScorer,
  BayesianProbabilityTransform,
  balancedLogOddsFusion,
  calibrationReport,
  cosineSimilarity,
  minMaxNormalize,
  Tokenizer,
  type Corpus,
  type Document,
  type BM25Method,
} from "@bb25/core";

export type RelMap = Map<string, number>;
export type Qrels = Map<string, RelMap>;
export type BaseRateOption = number | "auto" | null;
export type BaseRateMethod = "percentile";

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
  return relevant === 0 ? 0.0 : precisionSum / relevant;
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

export function recallAtK(ranked: string[], relMap: RelMap, k: number): number {
  let relevant = 0;
  for (const r of relMap.values()) {
    if (r > 0) relevant += 1;
  }
  if (relevant === 0) {
    return 0.0;
  }
  let hits = 0;
  const top = ranked.slice(0, k);
  for (const docId of top) {
    if ((relMap.get(docId) ?? 0.0) > 0) {
      hits += 1;
    }
  }
  return hits / relevant;
}

export interface ScorerResult {
  scorer: string;
  queries: number;
  metrics: Record<string, number>;
}

export interface CalibrationResult {
  scorer: string;
  ece: number;
  brier: number;
  samples: number;
  bins: number;
}

export interface ScorerRun {
  scorer: string;
  queryId: string;
  scores: [string, number][];
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
    m[`recall@${k}`] = 0.0;
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
    metrics[`recall@${k}`]! += recallAtK(ranked, relMap, k);
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
  runs?: ScorerRun[],
): ScorerResult {
  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    const scores: [string, number][] = docs.map((d) => [d.id, scoreFn(terms, d)]);
    runs?.push({ scorer: scorerName, queryId: q.queryId, scores });
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
  candidateFn: ((terms: string[], embedding: number[]) => readonly Document[]) | null = null,
  runs?: ScorerRun[],
): ScorerResult {
  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    if (q.embedding === null) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    const candidateDocs = candidateFn === null ? docs : candidateFn(terms, q.embedding);
    const scores: [string, number][] = candidateDocs.map((d) => [d.id, scoreFn(terms, q.embedding!, d)]);
    runs?.push({ scorer: scorerName, queryId: q.queryId, scores });
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
  baseRate?: BaseRateOption;
  baseRateMethod?: BaseRateMethod;
  baseRateSampleSize?: number;
  baseRateSeed?: number;
  bm25Method?: BM25Method;
  candidateDepth?: number | null;
  cutoffs?: number[];
  calibrationBins?: number | null;
  runs?: ScorerRun[];
}

export interface BenchResolvedOptions {
  k1: number;
  b: number;
  alpha: number;
  beta: number;
  baseRate: number | null;
  requestedBaseRate: BaseRateOption;
  baseRateMethod: BaseRateMethod | null;
  baseRateSampleSize: number | null;
  baseRateSeed: number | null;
  bm25Method: BM25Method;
  candidateDepth: number | null;
  cutoffs: number[];
  calibrationBins: number | null;
}

export interface BenchDetails {
  results: ScorerResult[];
  options: BenchResolvedOptions;
  calibration: CalibrationResult[];
}

/** Run the standard scorer comparison over a corpus + queries + qrels. */
export function runBench(
  corpus: Corpus,
  queries: BenchQuery[],
  qrels: Qrels,
  options: BenchOptions = {},
): ScorerResult[] {
  return runBenchWithDetails(corpus, queries, qrels, options).results;
}

/** Run the scorer comparison and include resolved options/calibration metadata. */
export function runBenchWithDetails(
  corpus: Corpus,
  queries: BenchQuery[],
  qrels: Qrels,
  options: BenchOptions = {},
): BenchDetails {
  const k1 = options.k1 ?? 1.2;
  const b = options.b ?? 0.75;
  const alpha = options.alpha ?? 1.0;
  const beta = options.beta ?? 0.5;
  const requestedBaseRate = options.baseRate ?? null;
  const baseRateMethod = options.baseRateMethod ?? "percentile";
  const baseRateSampleSize = options.baseRateSampleSize ?? 50;
  const baseRateSeed = options.baseRateSeed ?? 42;
  const bm25Method = options.bm25Method ?? "robertson";
  const candidateDepth = options.candidateDepth ?? null;
  const cutoffs = options.cutoffs ?? [5, 10, 20, 100];
  const calibrationBins = options.calibrationBins ?? null;
  const runs = options.runs ?? (calibrationBins === null ? undefined : []);

  const tokenizer = new Tokenizer();
  const docs = corpus.documents();
  const bm25 = new BM25Scorer(corpus, k1, b, bm25Method);
  const baseRate =
    requestedBaseRate === "auto"
      ? estimateBaseRateFromPseudoQueries(docs, bm25, baseRateMethod, baseRateSampleSize, baseRateSeed)
      : requestedBaseRate;
  const bayes = new BayesianBM25Scorer(bm25, alpha, beta, baseRate);
  const vector = new VectorScorer();
  const hybrid = new HybridScorer(bayes, vector, 0.5);
  const hybridCandidates =
    candidateDepth === null
      ? null
      : (terms: string[], embedding: number[]) =>
          selectHybridCandidates(docs, terms, embedding, bm25, candidateDepth);

  const hasEmbeddings = queries.some((q) => q.embedding !== null && q.embedding.length > 0);

  const results: ScorerResult[] = [
    evaluate(queries, docs, "bm25", (t, d) => bm25.score(t, d), qrels, tokenizer, cutoffs, runs),
    evaluate(queries, docs, "bayesian", (t, d) => bayes.score(t, d), qrels, tokenizer, cutoffs, runs),
    evaluateFittedBayesian(queries, docs, bm25, qrels, tokenizer, cutoffs, alpha, beta, baseRate, runs),
  ];

  if (hasEmbeddings) {
    results.push(
      evaluateDense(queries, docs, qrels, cutoffs, candidateDepth, runs),
      evaluateConvex(queries, docs, bm25, qrels, tokenizer, cutoffs, candidateDepth, runs),
      evaluateHybrid(
        queries,
        docs,
        "hybrid_or",
        (t, e, d) => hybrid.scoreOr(t, e, d),
        qrels,
        tokenizer,
        cutoffs,
        hybridCandidates,
        runs,
      ),
      evaluateHybrid(
        queries,
        docs,
        "hybrid_and",
        (t, e, d) => hybrid.scoreAnd(t, e, d),
        qrels,
        tokenizer,
        cutoffs,
        hybridCandidates,
        runs,
      ),
      evaluateBalanced(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, runs),
      evaluateRrf(queries, docs, bm25, qrels, tokenizer, cutoffs, candidateDepth, runs),
    );
  }

  return {
    results,
    options: {
      k1,
      b,
      alpha,
      beta,
      baseRate,
      requestedBaseRate,
      baseRateMethod: requestedBaseRate === "auto" ? baseRateMethod : null,
      baseRateSampleSize: requestedBaseRate === "auto" ? baseRateSampleSize : null,
      baseRateSeed: requestedBaseRate === "auto" ? baseRateSeed : null,
      bm25Method,
      candidateDepth,
      cutoffs,
      calibrationBins,
    },
    calibration: calibrationBins === null || runs === undefined ? [] : evaluateCalibration(runs, qrels, calibrationBins),
  };
}

function estimateBaseRateFromPseudoQueries(
  docs: readonly Document[],
  bm25: BM25Scorer,
  method: BaseRateMethod,
  sampleSize: number,
  seed: number,
): number {
  if (method !== "percentile") {
    throw new Error(`unsupported base-rate method: ${method}`);
  }
  if (docs.length === 0) {
    return 1e-6;
  }
  const sampled = sampleDocumentIndices(docs.length, Math.min(Math.max(0, sampleSize), docs.length), seed);
  const ratios: number[] = [];
  for (const idx of sampled) {
    const queryTerms = docs[idx]!.tokens.slice(0, 5);
    if (queryTerms.length === 0) continue;
    const scores = docs.map((doc) => bm25.score(queryTerms, doc)).filter((score) => score > 0.0);
    if (scores.length === 0) continue;
    const threshold = percentile(scores, 95);
    let nAbove = 0;
    for (const score of scores) {
      if (score >= threshold) nAbove += 1;
    }
    ratios.push(nAbove / docs.length);
  }
  if (ratios.length === 0) {
    return 1e-6;
  }
  const mean = ratios.reduce((sum, value) => sum + value, 0.0) / ratios.length;
  return Math.min(0.5, Math.max(1e-6, mean));
}

function sampleDocumentIndices(n: number, sampleSize: number, seed: number): number[] {
  if (sampleSize >= n) {
    return Array.from({ length: n }, (_, i) => i);
  }
  const indices = Array.from({ length: n }, (_, i) => i);
  let state = seed >>> 0;
  for (let i = indices.length - 1; i > 0; i--) {
    state = (1664525 * state + 1013904223) >>> 0;
    const j = state % (i + 1);
    [indices[i], indices[j]] = [indices[j]!, indices[i]!];
  }
  return indices.slice(0, sampleSize);
}

function percentile(values: number[], pct: number): number {
  if (values.length === 0) {
    return 0.0;
  }
  const sorted = values.slice().sort((a, b) => a - b);
  if (sorted.length === 1) {
    return sorted[0]!;
  }
  const pos = (pct / 100) * (sorted.length - 1);
  const lo = Math.floor(pos);
  const hi = Math.ceil(pos);
  if (lo === hi) {
    return sorted[lo]!;
  }
  const frac = pos - lo;
  return sorted[lo]! * (1.0 - frac) + sorted[hi]! * frac;
}

function evaluateCalibration(runs: ScorerRun[], qrels: Qrels, bins: number): CalibrationResult[] {
  const calibrationScorers = new Set(["bayesian", "bayesian_fitted", "hybrid_or", "hybrid_and"]);
  const byScorer = new Map<string, { probs: number[]; labels: number[] }>();
  for (const run of runs) {
    if (!calibrationScorers.has(run.scorer)) continue;
    const relMap = qrels.get(run.queryId);
    if (relMap === undefined) continue;
    const bucket = byScorer.get(run.scorer) ?? { probs: [], labels: [] };
    for (const [docId, score] of run.scores) {
      if (!relMap.has(docId)) continue;
      bucket.probs.push(Math.min(1.0, Math.max(0.0, score)));
      bucket.labels.push((relMap.get(docId) ?? 0) > 0 ? 1.0 : 0.0);
    }
    byScorer.set(run.scorer, bucket);
  }

  const out: CalibrationResult[] = [];
  for (const [scorer, bucket] of byScorer) {
    if (bucket.probs.length === 0) continue;
    const report = calibrationReport(bucket.probs, bucket.labels, bins);
    out.push({ scorer, ece: report.ece, brier: report.brier, samples: report.nSamples, bins: report.nBins });
  }
  return out.sort((a, b) => (a.scorer < b.scorer ? -1 : a.scorer > b.scorer ? 1 : 0));
}

function selectHybridCandidates(
  docs: readonly Document[],
  terms: string[],
  embedding: number[],
  bm25: BM25Scorer,
  candidateDepth: number,
): readonly Document[] {
  if (candidateDepth <= 0 || candidateDepth >= docs.length) {
    return docs;
  }
  const sparseRanked = rankDocs(docs.map((d) => [d.id, bm25.score(terms, d)])).slice(0, candidateDepth);
  const denseRanked = rankDocs(docs.map((d) => [d.id, denseSimilarity(embedding, d)])).slice(0, candidateDepth);
  const candidateIds = new Set([...sparseRanked, ...denseRanked]);
  return docs.filter((d) => candidateIds.has(d.id));
}

function denseSimilarity(embedding: number[], doc: Document): number {
  return cosineSimilarity(embedding, doc.embedding);
}

function selectDenseCandidates(
  docs: readonly Document[],
  embedding: number[],
  candidateDepth: number | null,
): readonly Document[] {
  if (candidateDepth === null || candidateDepth <= 0 || candidateDepth >= docs.length) {
    return docs;
  }
  const denseRanked = rankDocs(docs.map((d) => [d.id, denseSimilarity(embedding, d)])).slice(0, candidateDepth);
  const candidateIds = new Set(denseRanked);
  return docs.filter((d) => candidateIds.has(d.id));
}

function evaluateDense(
  queries: BenchQuery[],
  docs: readonly Document[],
  qrels: Qrels,
  cutoffs: number[],
  candidateDepth: number | null,
  runs?: ScorerRun[],
): ScorerResult {
  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    if (q.embedding === null) continue;
    const candidateDocs = selectDenseCandidates(docs, q.embedding, candidateDepth);
    const scores: [string, number][] = candidateDocs.map((d) => [d.id, denseSimilarity(q.embedding!, d)]);
    runs?.push({ scorer: "dense", queryId: q.queryId, scores });
    accumulate(metrics, rankDocs(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: "dense", queries: counted, metrics };
}

function evaluateConvex(
  queries: BenchQuery[],
  docs: readonly Document[],
  bm25: BM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  candidateDepth: number | null,
  runs?: ScorerRun[],
  weight = 0.5,
): ScorerResult {
  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    if (q.embedding === null) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    const candidateDocs =
      candidateDepth === null ? docs : selectHybridCandidates(docs, terms, q.embedding, bm25, candidateDepth);
    const sparse = candidateDocs.map((d) => bm25.score(terms, d));
    const dense = candidateDocs.map((d) => denseSimilarity(q.embedding!, d));
    const sparseNorm = minMaxNormalize(sparse);
    const denseNorm = minMaxNormalize(dense);
    const scores: [string, number][] = candidateDocs.map((d, i) => [
      d.id,
      weight * denseNorm[i]! + (1.0 - weight) * sparseNorm[i]!,
    ]);
    runs?.push({ scorer: "convex", queryId: q.queryId, scores });
    accumulate(metrics, rankDocs(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: "convex", queries: counted, metrics };
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
  baseRate: number | null,
  runs?: ScorerRun[],
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
  const transform = new BayesianProbabilityTransform(alpha, beta, baseRate);
  transform.fit(trainScores, trainLabels);
  const fitted = new BayesianBM25Scorer(bm25, transform.alpha, transform.beta, baseRate);
  return evaluate(queries, docs, "bayesian_fitted", (t, d) => fitted.score(t, d), qrels, tokenizer, cutoffs, runs);
}

function evaluateBalanced(
  queries: BenchQuery[],
  docs: readonly Document[],
  bayes: BayesianBM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  bm25: BM25Scorer,
  candidateDepth: number | null,
  runs?: ScorerRun[],
  weight = 0.5,
): ScorerResult {
  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    if (q.embedding === null) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    const candidateDocs =
      candidateDepth === null ? docs : selectHybridCandidates(docs, terms, q.embedding, bm25, candidateDepth);
    const sparse = candidateDocs.map((d) => bayes.score(terms, d));
    const dense = candidateDocs.map((d) => denseSimilarity(q.embedding!, d));
    const fused = balancedLogOddsFusion(sparse, dense, weight);
    const scores: [string, number][] = candidateDocs.map((d, i) => [d.id, fused[i]!]);
    runs?.push({ scorer: "balanced_fusion", queryId: q.queryId, scores });
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
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  candidateDepth: number | null,
  runs?: ScorerRun[],
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
    const vecRanked = rankDocs(docs.map((d) => [d.id, denseSimilarity(q.embedding!, d)]));
    const effectiveDepth =
      candidateDepth === null || candidateDepth <= 0 || candidateDepth >= docs.length ? null : candidateDepth;
    const bmActiveRanked = effectiveDepth === null ? bmRanked : bmRanked.slice(0, effectiveDepth);
    const vecActiveRanked = effectiveDepth === null ? vecRanked : vecRanked.slice(0, effectiveDepth);
    const bmRank = new Map<string, number>();
    bmActiveRanked.forEach((id, i) => bmRank.set(id, i + 1));
    const vecRank = new Map<string, number>();
    vecActiveRanked.forEach((id, i) => vecRank.set(id, i + 1));
    const candidateDocs =
      effectiveDepth === null ? docs : selectHybridCandidates(docs, terms, q.embedding, bm25, effectiveDepth);
    const scores: [string, number][] = candidateDocs.map((d) => [
      d.id,
      (bmRank.has(d.id) ? 1.0 / (k + bmRank.get(d.id)!) : 0.0) +
        (vecRank.has(d.id) ? 1.0 / (k + vecRank.get(d.id)!) : 0.0),
    ]);
    runs?.push({ scorer: "rrf", queryId: q.queryId, scores });
    accumulate(metrics, rankDocs(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: "rrf", queries: counted, metrics };
}

/** Tab-separated benchmark table. */
export function formatTable(results: ScorerResult[], cutoffs: number[]): string {
  const headers = ["scorer", "queries"];
  for (const k of cutoffs) headers.push(`ndcg@${k}`, `map@${k}`, `mrr@${k}`, `recall@${k}`);
  const lines = [headers.join("\t")];
  for (const r of results) {
    const row = [r.scorer, String(r.queries)];
    for (const k of cutoffs) {
      row.push(
        (r.metrics[`ndcg@${k}`] ?? 0).toFixed(4),
        (r.metrics[`map@${k}`] ?? 0).toFixed(4),
        (r.metrics[`mrr@${k}`] ?? 0).toFixed(4),
        (r.metrics[`recall@${k}`] ?? 0).toFixed(4),
      );
    }
    lines.push(row.join("\t"));
  }
  return lines.join("\n");
}
