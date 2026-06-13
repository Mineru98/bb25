/**
 * Ranking-quality benchmark harness.
 *
 * rankDocs sorts by (-score, docId) — the tie-break is part of the contract and
 * is deterministic. NDCG uses exponential gain (2^rel - 1).
 */
import {
  AttentionLogOddsWeights,
  BM25Scorer,
  BayesianBM25Scorer,
  VectorScorer,
  HybridScorer,
  BayesianProbabilityTransform,
  balancedLogOddsFusion,
  calibrationReport,
  clamp,
  cosineToProbability,
  cosineSimilarity,
  Gating,
  logOddsConjunction,
  logit,
  minMaxNormalize,
  MultiFieldScorer,
  MultiHeadAttentionLogOddsWeights,
  sigmoid,
  Tokenizer,
  type Corpus,
  type Document,
  type BM25Method,
  type MultiFieldDocument,
} from "@bb25/core";

export type RelMap = Map<string, number>;
export type Qrels = Map<string, RelMap>;
export type BaseRateOption = number | "auto" | null;
export type BaseRateMethod = "percentile" | "mixture" | "elbow";
export type GatedLogOddsKind = "relu" | "swish" | "gelu" | "swish_b2" | "softplus";

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

export type ScorerKind = "zero-shot" | "calibration" | "diagnostic" | "tuned" | "smoke";

export interface ScorerMetadata {
  scorer: string;
  kind: ScorerKind;
}

export interface CalibrationResult {
  scorer: string;
  ece: number;
  brier: number;
  samples: number;
  bins: number;
}

export interface FitSplitOptions {
  trainRatio: number;
  seed: number;
}

export interface FittedSplitMetadata {
  scorer: "bayesian_fitted_split";
  trainRatio: number;
  seed: number;
  trainQueryIds: string[];
  evalQueryIds: string[];
  trainingPairs: number;
  alpha: number | null;
  beta: number | null;
}

export interface AttentionSplitMetadata {
  scorer:
    | "bayesian_attention_split"
    | "bayesian_attn_norm_split"
    | "bayesian_multihead_split"
    | "bayesian_multihead_norm_split";
  trainRatio: number;
  seed: number;
  trainQueryIds: string[];
  evalQueryIds: string[];
  trainingPairs: number;
  features: "basic" | "rich";
  normalize: boolean;
  heads: number;
  trained: boolean;
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
  fitSplit?: FitSplitOptions | null;
  multiField?: MultiFieldBenchOptions | null;
  runs?: ScorerRun[];
}

export interface MultiFieldBenchOptions {
  fields: string[];
  docFields: Map<string, Record<string, string[]>>;
  fieldWeights?: Record<string, number>;
}

export interface ResolvedMultiFieldOptions {
  fields: string[];
  fieldWeights: Record<string, number> | null;
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
  fitSplit: FitSplitOptions | null;
  multiField: ResolvedMultiFieldOptions | null;
}

export interface BenchDetails {
  results: ScorerResult[];
  options: BenchResolvedOptions;
  scorers: ScorerMetadata[];
  calibration: CalibrationResult[];
  fittedSplit: FittedSplitMetadata | null;
  attentionSplits: AttentionSplitMetadata[];
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
  const fitSplit = options.fitSplit ?? null;
  const multiField = options.multiField ?? null;
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
  const attentionSplits: AttentionSplitMetadata[] = [];
  const multiFieldScorer =
    multiField === null
      ? null
      : buildMultiFieldScorer(docs, multiField, k1, b, alpha, beta, baseRate, bm25Method);

  const results: ScorerResult[] = [
    evaluate(queries, docs, "bm25", (t, d) => bm25.score(t, d), qrels, tokenizer, cutoffs, runs),
    evaluate(queries, docs, "bayesian", (t, d) => bayes.score(t, d), qrels, tokenizer, cutoffs, runs),
    evaluateFittedBayesian(queries, docs, bm25, qrels, tokenizer, cutoffs, alpha, beta, baseRate, runs),
  ];
  let fittedSplit: FittedSplitMetadata | null = null;
  if (fitSplit !== null) {
    const split = evaluateSplitFittedBayesian(
      queries,
      docs,
      bm25,
      qrels,
      tokenizer,
      cutoffs,
      alpha,
      beta,
      baseRate,
      fitSplit,
      runs,
    );
    results.push(split.result);
    fittedSplit = split.metadata;
  }
  if (multiFieldScorer !== null) {
    results.push(evaluateMultiField(queries, docs, multiFieldScorer, qrels, tokenizer, cutoffs, candidateDepth, runs));
  }

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
      evaluateLogOdds(queries, docs, bm25, qrels, tokenizer, cutoffs, candidateDepth, alpha, beta, null, runs),
      ...(baseRate === null
        ? []
        : [evaluateLogOdds(queries, docs, bm25, qrels, tokenizer, cutoffs, candidateDepth, alpha, beta, baseRate, runs)]),
      evaluateGatedLogOdds(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, "relu", runs),
      evaluateGatedLogOdds(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, "swish", runs),
      evaluateGatedLogOdds(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, "gelu", runs),
      evaluateGatedLogOdds(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, "swish_b2", runs),
      evaluateGatedLogOdds(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, "softplus", runs),
      evaluateBalanced(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, runs),
      ...(multiFieldScorer === null
        ? []
        : [
            evaluateMultiFieldBalanced(
              queries,
              docs,
              multiFieldScorer,
              qrels,
              tokenizer,
              cutoffs,
              candidateDepth,
              runs,
            ),
          ]),
      evaluateRrf(queries, docs, bm25, qrels, tokenizer, cutoffs, candidateDepth, runs),
    );
    if (fitSplit !== null) {
      const splitAttention = evaluateSplitAttention(
        queries,
        docs,
        bm25,
        bayes,
        qrels,
        tokenizer,
        cutoffs,
        candidateDepth,
        fitSplit,
        runs,
      );
      results.push(...splitAttention.results);
      attentionSplits.push(...splitAttention.metadata);
    }
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
      fitSplit,
      multiField:
        multiField === null
          ? null
          : {
              fields: multiField.fields.slice(),
              fieldWeights: multiField.fieldWeights ?? null,
            },
    },
    scorers: buildScorerMetadata(results),
    calibration: calibrationBins === null || runs === undefined ? [] : evaluateCalibration(runs, qrels, calibrationBins),
    fittedSplit,
    attentionSplits,
  };
}

function buildScorerMetadata(results: ScorerResult[]): ScorerMetadata[] {
  return results.map((row) => {
    switch (row.scorer) {
      case "bayesian_fitted":
        return { scorer: row.scorer, kind: "smoke" };
      case "bayesian_fitted_split":
      case "bayesian_logodds_br":
        return { scorer: row.scorer, kind: "calibration" };
      case "bayesian_attention_split":
      case "bayesian_attn_norm_split":
      case "bayesian_multihead_split":
      case "bayesian_multihead_norm_split":
        return { scorer: row.scorer, kind: "tuned" };
      case "hybrid_or":
      case "hybrid_and":
      case "bayesian_gated_relu":
      case "bayesian_gated_swish":
      case "bayesian_gated_gelu":
      case "bayesian_gated_swish_b2":
      case "bayesian_gated_softplus":
      case "bayesian_multifield_bal":
      case "balanced_fusion":
        return { scorer: row.scorer, kind: "diagnostic" };
      default:
        return { scorer: row.scorer, kind: "zero-shot" };
    }
  });
}

function estimateBaseRateFromPseudoQueries(
  docs: readonly Document[],
  bm25: BM25Scorer,
  method: BaseRateMethod,
  sampleSize: number,
  seed: number,
): number {
  if (docs.length === 0) {
    return 1e-6;
  }
  const perQueryScores = collectPseudoQueryScores(docs, bm25, sampleSize, seed);
  if (perQueryScores.length === 0) {
    return 1e-6;
  }
  switch (method) {
    case "percentile":
      return estimateBaseRatePercentile(perQueryScores, docs.length);
    case "mixture":
      return estimateBaseRateMixture(perQueryScores);
    case "elbow":
      return estimateBaseRateElbow(perQueryScores);
  }
}

function collectPseudoQueryScores(
  docs: readonly Document[],
  bm25: BM25Scorer,
  sampleSize: number,
  seed: number,
): number[][] {
  const sampled = sampleDocumentIndices(docs.length, Math.min(Math.max(0, sampleSize), docs.length), seed);
  const perQueryScores: number[][] = [];
  for (const idx of sampled) {
    const queryTerms = docs[idx]!.tokens.slice(0, 5);
    if (queryTerms.length === 0) continue;
    const scores = docs.map((doc) => bm25.score(queryTerms, doc)).filter((score) => score > 0.0);
    if (scores.length === 0) continue;
    perQueryScores.push(scores);
  }
  return perQueryScores;
}

function clipBaseRate(baseRate: number): number {
  return Math.min(0.5, Math.max(1e-6, baseRate));
}

function estimateBaseRatePercentile(perQueryScores: number[][], nDocs: number): number {
  const ratios: number[] = [];
  for (const scores of perQueryScores) {
    const threshold = percentile(scores, 95);
    let nAbove = 0;
    for (const score of scores) {
      if (score >= threshold) nAbove += 1;
    }
    ratios.push(nAbove / nDocs);
  }
  const mean = ratios.reduce((sum, value) => sum + value, 0.0) / ratios.length;
  return clipBaseRate(mean);
}

function estimateBaseRateMixture(perQueryScores: number[][]): number {
  const allScores = perQueryScores.flat();
  if (allScores.length < 2) {
    return 1e-6;
  }

  const median = percentile(allScores, 50);
  const lo = allScores.filter((score) => score <= median);
  const hi = allScores.filter((score) => score > median);
  let mu0 = lo.length > 0 ? mean(lo) : median - 1.0;
  let mu1 = hi.length > 0 ? mean(hi) : median + 1.0;
  let var0 = Math.max(lo.length > 0 ? variance(lo, mu0) : 1.0, 1e-8);
  let var1 = Math.max(hi.length > 0 ? variance(hi, mu1) : 1.0, 1e-8);
  let pi1 = 0.5;

  for (let iter = 0; iter < 20; iter++) {
    const std0 = Math.sqrt(var0);
    const std1 = Math.sqrt(var1);
    const gamma: number[] = [];
    for (const score of allScores) {
      const logP0 = -0.5 * ((score - mu0) / std0) ** 2 - Math.log(std0);
      const logP1 = -0.5 * ((score - mu1) / std1) ** 2 - Math.log(std1);
      const logW0 = Math.log(Math.max(1.0 - pi1, 1e-10)) + logP0;
      const logW1 = Math.log(Math.max(pi1, 1e-10)) + logP1;
      gamma.push(Math.exp(logW1 - logAddExp(logW0, logW1)));
    }

    const nEff1 = gamma.reduce((sum, value) => sum + value, 0.0);
    const nEff0 = gamma.reduce((sum, value) => sum + (1.0 - value), 0.0);
    if (nEff0 < 1e-8 || nEff1 < 1e-8) {
      break;
    }

    let newMu0 = 0.0;
    let newMu1 = 0.0;
    for (let i = 0; i < allScores.length; i++) {
      const score = allScores[i]!;
      const g = gamma[i]!;
      newMu0 += (1.0 - g) * score;
      newMu1 += g * score;
    }
    mu0 = newMu0 / nEff0;
    mu1 = newMu1 / nEff1;

    let newVar0 = 0.0;
    let newVar1 = 0.0;
    for (let i = 0; i < allScores.length; i++) {
      const score = allScores[i]!;
      const g = gamma[i]!;
      newVar0 += (1.0 - g) * (score - mu0) ** 2;
      newVar1 += g * (score - mu1) ** 2;
    }
    var0 = Math.max(newVar0 / nEff0, 1e-8);
    var1 = Math.max(newVar1 / nEff1, 1e-8);
    pi1 = nEff1 / allScores.length;
  }

  return clipBaseRate(mu1 >= mu0 ? pi1 : 1.0 - pi1);
}

function estimateBaseRateElbow(perQueryScores: number[][]): number {
  const allScores = perQueryScores.flat().sort((a, b) => b - a);
  const n = allScores.length;
  if (n < 3) {
    return 1e-6;
  }

  const dx = n - 1;
  const dy = allScores[n - 1]! - allScores[0]!;
  const lineLen = Math.sqrt(dx * dx + dy * dy);
  if (lineLen < 1e-12) {
    return 1e-6;
  }

  let kneeIdx = 0;
  let maxDistance = Number.NEGATIVE_INFINITY;
  for (let i = 0; i < n; i++) {
    const distance = Math.abs(dy * i - dx * (allScores[i]! - allScores[0]!)) / lineLen;
    if (distance > maxDistance) {
      maxDistance = distance;
      kneeIdx = i;
    }
  }

  return clipBaseRate(Math.max(1, kneeIdx) / n);
}

function mean(values: number[]): number {
  return values.reduce((sum, value) => sum + value, 0.0) / values.length;
}

function variance(values: number[], mu: number): number {
  return values.reduce((sum, value) => sum + (value - mu) ** 2, 0.0) / values.length;
}

function logAddExp(a: number, b: number): number {
  const m = Math.max(a, b);
  return m + Math.log(Math.exp(a - m) + Math.exp(b - m));
}

function sampleDocumentIndices(n: number, sampleSize: number, seed: number): number[] {
  const indices = shuffleIndices(n, seed);
  if (sampleSize >= n) {
    return indices;
  }
  return indices.slice(0, sampleSize);
}

function shuffleIndices(n: number, seed: number): number[] {
  const indices = Array.from({ length: n }, (_, i) => i);
  let state = seed >>> 0;
  for (let i = indices.length - 1; i > 0; i--) {
    state = (1664525 * state + 1013904223) >>> 0;
    const j = state % (i + 1);
    [indices[i], indices[j]] = [indices[j]!, indices[i]!];
  }
  return indices;
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
  const calibrationScorers = new Set([
    "bayesian",
    "bayesian_fitted",
    "bayesian_fitted_split",
    "bayesian_logodds",
    "bayesian_logodds_br",
    "bayesian_multifield",
    "bayesian_gated_relu",
    "bayesian_gated_swish",
    "bayesian_gated_gelu",
    "bayesian_gated_swish_b2",
    "bayesian_gated_softplus",
    "bayesian_attention_split",
    "bayesian_attn_norm_split",
    "bayesian_multihead_split",
    "bayesian_multihead_norm_split",
    "hybrid_or",
    "hybrid_and",
  ]);
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

export function bayesianLogOddsFusionScores(
  bm25Scores: number[],
  denseSimilarities: number[],
  tfs: number[],
  docLenRatios: number[],
  bm25Alpha: number,
  bm25Beta: number,
  denseMedian: number,
  denseAlpha: number,
  baseRate: number | null,
): number[] {
  const nSignals = 2;
  const scale = Math.sqrt(nSignals);
  const wDense = 0.5;
  const wSparse = 0.5;
  const logitBase = baseRate === null ? 0.0 : logit(baseRate);
  const out: number[] = [];

  for (let i = 0; i < bm25Scores.length; i++) {
    const bm25Score = bm25Scores[i]!;
    const logitD = clamp(denseAlpha * (denseSimilarities[i]! - denseMedian), -500.0, 500.0);
    const logitLikelihood = bm25Alpha * (bm25Score - bm25Beta);
    const prior = BayesianProbabilityTransform.compositePrior(tfs[i]!, docLenRatios[i]!);
    const logitPrior = logit(prior);
    const logitS = clamp(logitLikelihood + logitPrior + logitBase, -500.0, 500.0);
    const lBar = wDense * logitD + wSparse * logitS;
    const raw = bm25Score > 0.0 ? lBar * scale : logitD * wDense;
    out.push(sigmoid(raw));
  }

  return out;
}

export function gatedLogOddsFusionScores(
  sparseProbs: number[],
  denseSimilarities: number[],
  gatingKind: GatedLogOddsKind,
): number[] {
  if (sparseProbs.length !== denseSimilarities.length) {
    throw new Error("sparseProbs and denseSimilarities must have the same length");
  }
  const gating = gatingForKind(gatingKind);
  return sparseProbs.map((prob, i) =>
    logOddsConjunction([prob, cosineToProbability(denseSimilarities[i]!)], null, null, gating),
  );
}

function gatingForKind(kind: GatedLogOddsKind) {
  switch (kind) {
    case "relu":
      return Gating.Relu;
    case "swish":
      return Gating.Swish;
    case "gelu":
      return Gating.Gelu;
    case "swish_b2":
      return Gating.generalizedSwish(2.0);
    case "softplus":
      return Gating.Softplus;
  }
}

function evaluateLogOdds(
  queries: BenchQuery[],
  docs: readonly Document[],
  bm25: BM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  candidateDepth: number | null,
  alpha: number,
  beta: number,
  baseRate: number | null,
  runs?: ScorerRun[],
): ScorerResult {
  const scorerName = baseRate === null ? "bayesian_logodds" : "bayesian_logodds_br";
  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    if (q.embedding === null) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    const candidateDocs =
      candidateDepth === null ? docs : selectHybridCandidates(docs, terms, q.embedding, bm25, candidateDepth);
    const bm25Scores = candidateDocs.map((d) => bm25.score(terms, d));
    const denseSimilarities = candidateDocs.map((d) => denseSimilarity(q.embedding!, d));
    const { median: denseMedian, alpha: denseAlpha } = denseCalibration(denseSimilarities);
    const tfs = candidateDocs.map((d) => queryTermOverlapCount(terms, d));
    const avgdl = bm25.avgdl();
    const docLenRatios = candidateDocs.map((d) => (avgdl > 0.0 ? d.length / avgdl : 1.0));
    const fused = bayesianLogOddsFusionScores(
      bm25Scores,
      denseSimilarities,
      tfs,
      docLenRatios,
      alpha,
      beta,
      denseMedian,
      denseAlpha,
      baseRate,
    );
    const scores: [string, number][] = candidateDocs.map((d, i) => [d.id, fused[i]!]);
    runs?.push({ scorer: scorerName, queryId: q.queryId, scores });
    accumulate(metrics, rankDocs(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: scorerName, queries: counted, metrics };
}

function evaluateGatedLogOdds(
  queries: BenchQuery[],
  docs: readonly Document[],
  bayes: BayesianBM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  bm25: BM25Scorer,
  candidateDepth: number | null,
  gatingKind: GatedLogOddsKind,
  runs?: ScorerRun[],
): ScorerResult {
  const scorerName = `bayesian_gated_${gatingKind}`;
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
    const fused = gatedLogOddsFusionScores(sparse, dense, gatingKind);
    const scores: [string, number][] = candidateDocs.map((d, i) => [d.id, fused[i]!]);
    runs?.push({ scorer: scorerName, queryId: q.queryId, scores });
    accumulate(metrics, rankDocs(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: scorerName, queries: counted, metrics };
}

function denseCalibration(values: number[]): { median: number; alpha: number } {
  const positive = values.filter((value) => value > 0.0);
  if (positive.length === 0) {
    return { median: 0.0, alpha: 1.0 };
  }
  const median = percentile(positive, 50);
  const std = Math.sqrt(variance(positive, mean(positive)));
  return { median, alpha: std > 0.0 ? 1.0 / std : 1.0 };
}

function queryTermOverlapCount(queryTerms: string[], doc: Document): number {
  const querySet = new Set(queryTerms);
  let count = 0;
  for (const term of querySet) {
    if (doc.termFreq.has(term)) {
      count += 1;
    }
  }
  return count;
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

function evaluateSplitFittedBayesian(
  queries: BenchQuery[],
  docs: readonly Document[],
  bm25: BM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  alpha: number,
  beta: number,
  baseRate: number | null,
  fitSplit: FitSplitOptions,
  runs?: ScorerRun[],
): { result: ScorerResult; metadata: FittedSplitMetadata } {
  const split = splitQueries(queries, qrels, fitSplit.trainRatio, fitSplit.seed);
  const trainQueries = split.trainIndices.map((idx) => queries[idx]!);
  const evalQueries = split.evalIndices.map((idx) => queries[idx]!);
  const train = collectFittedTrainingData(trainQueries, docs, bm25, qrels, tokenizer);

  const metadata: FittedSplitMetadata = {
    scorer: "bayesian_fitted_split",
    trainRatio: fitSplit.trainRatio,
    seed: fitSplit.seed,
    trainQueryIds: trainQueries.map((q) => q.queryId),
    evalQueryIds: evalQueries.map((q) => q.queryId),
    trainingPairs: train.scores.length,
    alpha: null,
    beta: null,
  };

  if (train.scores.length < 4 || evalQueries.length === 0) {
    return { result: { scorer: "bayesian_fitted_split", queries: 0, metrics: blankMetrics(cutoffs) }, metadata };
  }

  const transform = new BayesianProbabilityTransform(alpha, beta, baseRate);
  transform.fit(train.scores, train.labels);
  metadata.alpha = transform.alpha;
  metadata.beta = transform.beta;
  const fitted = new BayesianBM25Scorer(bm25, transform.alpha, transform.beta, baseRate);
  const result = evaluate(
    evalQueries,
    docs,
    "bayesian_fitted_split",
    (t, d) => fitted.score(t, d),
    qrels,
    tokenizer,
    cutoffs,
    runs,
  );
  return { result, metadata };
}

function splitQueries(
  queries: BenchQuery[],
  qrels: Qrels,
  trainRatio: number,
  seed: number,
): { trainIndices: number[]; evalIndices: number[] } {
  const eligible: number[] = [];
  for (let i = 0; i < queries.length; i++) {
    const relMap = qrels.get(queries[i]!.queryId);
    if (relMap !== undefined && relMap.size > 0) {
      eligible.push(i);
    }
  }
  if (eligible.length < 2) {
    return { trainIndices: eligible, evalIndices: [] };
  }
  const shuffled = shuffleIndices(eligible.length, seed).map((idx) => eligible[idx]!);
  const nTrain = Math.min(eligible.length - 1, Math.max(1, Math.round(eligible.length * trainRatio)));
  return { trainIndices: shuffled.slice(0, nTrain), evalIndices: shuffled.slice(nTrain) };
}

function collectFittedTrainingData(
  queries: BenchQuery[],
  docs: readonly Document[],
  bm25: BM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
): { scores: number[]; labels: number[] } {
  const scores: number[] = [];
  const labels: number[] = [];
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    for (const d of docs) {
      if (!relMap.has(d.id)) continue;
      const raw = bm25.score(terms, d);
      if (raw > 0.0) {
        scores.push(raw);
        labels.push((relMap.get(d.id) ?? 0) > 0 ? 1.0 : 0.0);
      }
    }
  }
  return { scores, labels };
}

function buildMultiFieldScorer(
  docs: readonly Document[],
  options: MultiFieldBenchOptions,
  k1: number,
  b: number,
  alpha: number,
  beta: number,
  baseRate: number | null,
  bm25Method: BM25Method,
): MultiFieldScorer | null {
  if (options.fields.length === 0) {
    return null;
  }

  const tokenCounts = new Map(options.fields.map((field) => [field, 0]));
  const multiDocs: MultiFieldDocument[] = docs.map((doc) => {
    const source = options.docFields.get(doc.id);
    if (source === undefined) {
      throw new Error(`multi-field document terms missing for "${doc.id}"`);
    }
    const fields: Record<string, string[]> = {};
    for (const field of options.fields) {
      const tokens = source[field];
      if (tokens === undefined) {
        throw new Error(`multi-field document "${doc.id}" missing field "${field}"`);
      }
      fields[field] = tokens.slice();
      tokenCounts.set(field, tokenCounts.get(field)! + tokens.length);
    }
    return { id: doc.id, fields };
  });

  for (const field of options.fields) {
    if ((tokenCounts.get(field) ?? 0) === 0) {
      return null;
    }
  }

  return new MultiFieldScorer(multiDocs, {
    fields: options.fields,
    fieldWeights: options.fieldWeights,
    k1,
    b,
    alpha,
    beta,
    baseRate,
    method: bm25Method,
  });
}

function evaluateMultiField(
  queries: BenchQuery[],
  docs: readonly Document[],
  scorer: MultiFieldScorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  candidateDepth: number | null,
  runs?: ScorerRun[],
): ScorerResult {
  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    const allScores: [string, number][] = docs.map((d) => [d.id, scorer.score(terms, d.id)]);
    const scores = limitScoresByDepth(allScores, candidateDepth);
    runs?.push({ scorer: "bayesian_multifield", queryId: q.queryId, scores });
    accumulate(metrics, rankDocs(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: "bayesian_multifield", queries: counted, metrics };
}

function evaluateMultiFieldBalanced(
  queries: BenchQuery[],
  docs: readonly Document[],
  scorer: MultiFieldScorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
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
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    const candidateDocs = selectMultiFieldDenseCandidates(docs, terms, q.embedding, scorer, candidateDepth);
    const sparse = candidateDocs.map((d) => scorer.score(terms, d.id));
    const dense = candidateDocs.map((d) => denseSimilarity(q.embedding!, d));
    const fused = balancedLogOddsFusion(sparse, dense, 0.5);
    const scores: [string, number][] = candidateDocs.map((d, i) => [d.id, fused[i]!]);
    runs?.push({ scorer: "bayesian_multifield_bal", queryId: q.queryId, scores });
    accumulate(metrics, rankDocs(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: "bayesian_multifield_bal", queries: counted, metrics };
}

function limitScoresByDepth(scores: [string, number][], candidateDepth: number | null): [string, number][] {
  if (candidateDepth === null || candidateDepth <= 0 || candidateDepth >= scores.length) {
    return scores;
  }
  const keep = new Set(rankDocs(scores).slice(0, candidateDepth));
  return scores.filter(([docId]) => keep.has(docId));
}

function selectMultiFieldDenseCandidates(
  docs: readonly Document[],
  terms: string[],
  embedding: number[],
  scorer: MultiFieldScorer,
  candidateDepth: number | null,
): readonly Document[] {
  if (candidateDepth === null || candidateDepth <= 0 || candidateDepth >= docs.length) {
    return docs;
  }
  const sparseRanked = rankDocs(docs.map((d) => [d.id, scorer.score(terms, d.id)])).slice(0, candidateDepth);
  const denseRanked = rankDocs(docs.map((d) => [d.id, denseSimilarity(embedding, d)])).slice(0, candidateDepth);
  const candidateIds = new Set([...sparseRanked, ...denseRanked]);
  return docs.filter((d) => candidateIds.has(d.id));
}

type AttentionFeatureSet = "basic" | "rich";
type AttentionScorerName =
  | "bayesian_attention_split"
  | "bayesian_attn_norm_split"
  | "bayesian_multihead_split"
  | "bayesian_multihead_norm_split";

interface AttentionFusionModel {
  fit(
    probs: number[],
    labels: number[],
    queryFeatures: number[],
    m: number,
    queryIds: number[] | null,
    lr: number,
    maxIter: number,
    tol: number,
  ): void;
  combine(probs: number[], m: number, queryFeatures: number[], mQ: number, useAveraged: boolean): number[];
}

interface AttentionQueryCache {
  query: BenchQuery;
  terms: string[];
  candidateDocs: readonly Document[];
  probs: number[];
  featuresBasic: number[];
  featuresRich: number[];
}

interface AttentionTrainingData {
  probs: number[];
  labels: number[];
  features: number[];
  queryIds: number[];
  pairs: number;
}

function evaluateSplitAttention(
  queries: BenchQuery[],
  docs: readonly Document[],
  bm25: BM25Scorer,
  bayes: BayesianBM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  candidateDepth: number | null,
  fitSplit: FitSplitOptions,
  runs?: ScorerRun[],
): { results: ScorerResult[]; metadata: AttentionSplitMetadata[] } {
  const basic = evaluateSplitAttentionVariant(
    "bayesian_attention_split",
    "basic",
    false,
    1,
    queries,
    docs,
    bm25,
    bayes,
    qrels,
    tokenizer,
    cutoffs,
    candidateDepth,
    fitSplit,
    runs,
  );
  const normalized = evaluateSplitAttentionVariant(
    "bayesian_attn_norm_split",
    "rich",
    true,
    1,
    queries,
    docs,
    bm25,
    bayes,
    qrels,
    tokenizer,
    cutoffs,
    candidateDepth,
    fitSplit,
    runs,
  );
  const multiHead = evaluateSplitAttentionVariant(
    "bayesian_multihead_split",
    "basic",
    false,
    4,
    queries,
    docs,
    bm25,
    bayes,
    qrels,
    tokenizer,
    cutoffs,
    candidateDepth,
    fitSplit,
    runs,
  );
  const multiHeadNorm = evaluateSplitAttentionVariant(
    "bayesian_multihead_norm_split",
    "rich",
    true,
    4,
    queries,
    docs,
    bm25,
    bayes,
    qrels,
    tokenizer,
    cutoffs,
    candidateDepth,
    fitSplit,
    runs,
  );
  return {
    results: [basic.result, normalized.result, multiHead.result, multiHeadNorm.result],
    metadata: [basic.metadata, normalized.metadata, multiHead.metadata, multiHeadNorm.metadata],
  };
}

function evaluateSplitAttentionVariant(
  scorerName: AttentionScorerName,
  featureSet: AttentionFeatureSet,
  normalize: boolean,
  heads: 1 | 4,
  queries: BenchQuery[],
  docs: readonly Document[],
  bm25: BM25Scorer,
  bayes: BayesianBM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  candidateDepth: number | null,
  fitSplit: FitSplitOptions,
  runs?: ScorerRun[],
): { result: ScorerResult; metadata: AttentionSplitMetadata } {
  const split = splitQueries(queries, qrels, fitSplit.trainRatio, fitSplit.seed);
  const trainQueries = split.trainIndices.map((idx) => queries[idx]!);
  const evalQueries = split.evalIndices.map((idx) => queries[idx]!);
  const nFeatures = featureSet === "basic" ? 3 : 7;
  const metadata: AttentionSplitMetadata = {
    scorer: scorerName,
    trainRatio: fitSplit.trainRatio,
    seed: fitSplit.seed,
    trainQueryIds: trainQueries.map((q) => q.queryId),
    evalQueryIds: evalQueries.map((q) => q.queryId),
    trainingPairs: 0,
    features: featureSet,
    normalize,
    heads,
    trained: false,
  };

  if (evalQueries.length === 0) {
    return { result: { scorer: scorerName, queries: 0, metrics: blankMetrics(cutoffs) }, metadata };
  }

  const trainCaches = trainQueries
    .filter((q) => q.embedding !== null)
    .map((q) => buildAttentionQueryCache(q, docs, bm25, bayes, tokenizer, candidateDepth));
  const trainQueryIds = new Map(trainQueries.map((q, i) => [q.queryId, i]));
  const train = collectAttentionTrainingData(trainCaches, qrels, trainQueryIds, featureSet, fitSplit.seed);
  metadata.trainingPairs = train.pairs;

  if (train.pairs < 10 || !hasBothBinaryClasses(train.labels)) {
    return { result: { scorer: scorerName, queries: 0, metrics: blankMetrics(cutoffs) }, metadata };
  }

  const model: AttentionFusionModel =
    heads === 1
      ? new AttentionLogOddsWeights(2, nFeatures, 0.5, normalize, 0, null)
      : new MultiHeadAttentionLogOddsWeights(heads, 2, nFeatures, 0.5, normalize);
  model.fit(
    train.probs,
    train.labels,
    train.features,
    train.pairs,
    normalize ? train.queryIds : null,
    0.01,
    500,
    1e-6,
  );
  metadata.trained = true;

  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of evalQueries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0 || q.embedding === null) continue;
    const cache = buildAttentionQueryCache(q, docs, bm25, bayes, tokenizer, candidateDepth);
    const features = featureSet === "basic" ? cache.featuresBasic : cache.featuresRich;
    const fused = model.combine(cache.probs, cache.candidateDocs.length, features, 1, true);
    const scores: [string, number][] = cache.candidateDocs.map((d, i) => [d.id, fused[i]!]);
    runs?.push({ scorer: scorerName, queryId: q.queryId, scores });
    accumulate(metrics, rankDocs(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { result: { scorer: scorerName, queries: counted, metrics }, metadata };
}

function buildAttentionQueryCache(
  q: BenchQuery,
  docs: readonly Document[],
  bm25: BM25Scorer,
  bayes: BayesianBM25Scorer,
  tokenizer: Tokenizer,
  candidateDepth: number | null,
): AttentionQueryCache {
  const embedding = q.embedding ?? [];
  const terms = q.terms ?? tokenizer.tokenize(q.text);
  const bm25Scores = docs.map((d) => bm25.score(terms, d));
  const denseScores = docs.map((d) => denseSimilarity(embedding, d));
  const candidateDocs =
    candidateDepth === null ? docs : selectHybridCandidates(docs, terms, embedding, bm25, candidateDepth);
  const probs: number[] = [];
  for (const d of candidateDocs) {
    probs.push(bayes.score(terms, d), cosineToProbability(denseSimilarity(embedding, d)));
  }

  const features = buildAttentionFeatures(terms, docs, bm25Scores, denseScores);
  return {
    query: q,
    terms,
    candidateDocs,
    probs,
    featuresBasic: features.basic,
    featuresRich: features.rich,
  };
}

function buildAttentionFeatures(
  terms: string[],
  docs: readonly Document[],
  bm25Scores: number[],
  denseScores: number[],
): { basic: number[]; rich: number[] } {
  const nDocs = Math.max(1, docs.length);
  const qlen = terms.length;
  const bm25HitRatio = bm25Scores.filter((score) => score > 0.0).length / nDocs;
  const maxBm25 = bm25Scores.reduce((maxValue, score) => Math.max(maxValue, score), 0.0);
  const maxBm25Log = maxBm25 > 0.0 ? Math.log1p(maxBm25) : 0.0;

  const denseRanked = denseScores.slice().sort((a, b) => b - a);
  const denseTop10 = denseRanked.slice(0, Math.min(10, denseRanked.length));
  const denseTop10Mean = denseTop10.length > 0 ? mean(denseTop10) : 0.0;
  const denseTop10Std = denseTop10.length > 1 ? Math.sqrt(variance(denseTop10, denseTop10Mean)) : 0.0;
  const maxDenseLog = denseTop10.length > 0 ? Math.log1p(Math.max(0.0, denseTop10[0]!)) : 0.0;

  const top100 = Math.min(100, docs.length);
  const bm25Top = new Set(rankDocs(docs.map((d, i) => [d.id, bm25Scores[i]!] as [string, number])).slice(0, top100));
  const denseTop = new Set(rankDocs(docs.map((d, i) => [d.id, denseScores[i]!] as [string, number])).slice(0, top100));
  const union = new Set([...bm25Top, ...denseTop]);
  let overlap = 0;
  for (const docId of bm25Top) {
    if (denseTop.has(docId)) overlap += 1;
  }
  const overlapRatio = union.size > 0 ? overlap / union.size : 0.0;

  const basic = [Math.log1p(qlen), bm25HitRatio, maxBm25Log];
  return {
    basic,
    rich: [...basic, denseTop10Mean, denseTop10Std, maxDenseLog, overlapRatio],
  };
}

function collectAttentionTrainingData(
  caches: AttentionQueryCache[],
  qrels: Qrels,
  queryIdOrdinals: Map<string, number>,
  featureSet: AttentionFeatureSet,
  seed: number,
): AttentionTrainingData {
  const probs: number[] = [];
  const labels: number[] = [];
  const features: number[] = [];
  const queryIds: number[] = [];

  for (let cacheIdx = 0; cacheIdx < caches.length; cacheIdx++) {
    const cache = caches[cacheIdx]!;
    const relMap = qrels.get(cache.query.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    const featureRow = featureSet === "basic" ? cache.featuresBasic : cache.featuresRich;
    const queryOrdinal = queryIdOrdinals.get(cache.query.queryId) ?? cacheIdx;
    const unjudged: number[] = [];
    let positives = 0;

    for (let i = 0; i < cache.candidateDocs.length; i++) {
      const docId = cache.candidateDocs[i]!.id;
      if (!relMap.has(docId)) {
        unjudged.push(i);
        continue;
      }
      const label = (relMap.get(docId) ?? 0) > 0 ? 1.0 : 0.0;
      if (label > 0) positives += 1;
      pushAttentionTrainingRow(probs, labels, features, queryIds, cache, featureRow, queryOrdinal, i, label);
    }

    const nNeg = Math.min(positives, unjudged.length);
    if (nNeg > 0) {
      const order = shuffleIndices(unjudged.length, seed + cacheIdx + 1).slice(0, nNeg);
      for (const orderIdx of order) {
        pushAttentionTrainingRow(
          probs,
          labels,
          features,
          queryIds,
          cache,
          featureRow,
          queryOrdinal,
          unjudged[orderIdx]!,
          0.0,
        );
      }
    }
  }

  return { probs, labels, features, queryIds, pairs: labels.length };
}

function pushAttentionTrainingRow(
  probs: number[],
  labels: number[],
  features: number[],
  queryIds: number[],
  cache: AttentionQueryCache,
  featureRow: number[],
  queryOrdinal: number,
  candidateIdx: number,
  label: number,
): void {
  const offset = candidateIdx * 2;
  probs.push(cache.probs[offset]!, cache.probs[offset + 1]!);
  labels.push(label);
  features.push(...featureRow);
  queryIds.push(queryOrdinal);
}

function hasBothBinaryClasses(labels: number[]): boolean {
  let positives = 0;
  for (const label of labels) {
    if (label > 0) positives += 1;
  }
  return positives > 0 && positives < labels.length;
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
