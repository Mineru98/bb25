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
  PlattCalibrator,
  sigmoid,
  Tokenizer,
  IsotonicCalibrator,
  VectorProbabilityTransform,
  type Corpus,
  type Document,
  type BM25Method,
  type MultiFieldDocument,
} from "@bb25/core";

export type RelMap = Map<string, number>;
export type Qrels = Map<string, RelMap>;
export type BayesianParameterOption = number | "auto";
export type BaseRateOption = number | "auto" | null;
export type BaseRateMethod = "percentile" | "mixture" | "elbow";
export type GatedLogOddsKind = "relu" | "swish" | "gelu" | "swish_b2" | "softplus";
export type MetricStyle = "pytrec" | "python-reference";

let activeMetricStyle: MetricStyle = "pytrec";

/** Sort (docId, score) pairs by descending score, ties broken by ascending docId. */
export function rankDocs(scores: [string, number][]): string[] {
  return scores
    .slice()
    .sort((a, b) => (b[1] - a[1] !== 0 ? b[1] - a[1] : a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0))
    .map(([id]) => id);
}

function rankDocsStable(scores: [string, number][]): string[] {
  return scores
    .slice()
    .sort((a, b) => b[1] - a[1])
    .map(([id]) => id);
}

function rankForMetrics(scores: [string, number][]): string[] {
  return activeMetricStyle === "python-reference" ? rankDocsStable(scores) : rankDocs(scores);
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

export function referenceAveragePrecisionAtK(ranked: string[], relMap: RelMap, k: number): number {
  let hits = 0;
  let precisionSum = 0.0;
  const top = ranked.slice(0, k);
  for (let idx = 0; idx < top.length; idx++) {
    if ((relMap.get(top[idx]!) ?? 0.0) > 0) {
      hits += 1;
      precisionSum += hits / (idx + 1);
    }
  }
  return hits === 0 ? 0.0 : precisionSum / hits;
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

function linearDcg(relevances: number[]): number {
  let score = 0.0;
  for (let idx = 0; idx < relevances.length; idx++) {
    const rel = relevances[idx]!;
    if (rel <= 0) continue;
    score += rel / Math.log2(idx + 2);
  }
  return score;
}

export function referenceNdcgAtK(ranked: string[], relMap: RelMap, k: number): number {
  const topRels = ranked.slice(0, k).map((docId) => relMap.get(docId) ?? 0.0);
  const actual = linearDcg(topRels);
  const ideal = linearDcg(topRels.slice().sort((a, b) => b - a));
  return ideal === 0.0 ? 0.0 : actual / ideal;
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
  trainQueryIds?: string[];
  evalQueryIds?: string[];
  splitSource?: string | null;
}

export interface FittedSplitMetadata {
  scorer: "bayesian_fitted_split";
  trainRatio: number;
  seed: number;
  splitSource: string | null;
  trainQueryIds: string[];
  evalQueryIds: string[];
  trainingPairs: number;
  alpha: number | null;
  beta: number | null;
}

export interface AttentionSplitMetadata {
  scorer:
    | "bayesian_attention"
    | "bayesian_attn_norm"
    | "bayesian_multihead"
    | "bayesian_multihead_norm"
    | "bayesian_attention_split"
    | "bayesian_attn_norm_split"
    | "bayesian_attn_norm_cv"
    | "bayesian_multihead_split"
    | "bayesian_multihead_norm_split"
    | "bayesian_vector_attn_split";
  trainRatio: number;
  seed: number;
  trainQueryIds: string[];
  evalQueryIds: string[];
  trainingPairs: number;
  features: "basic" | "rich";
  normalize: boolean;
  heads: number;
  trained: boolean;
  protocol?: "all-qrels" | "holdout" | "cross-validation";
  folds?: {
    fold: number;
    trainQueryIds: string[];
    evalQueryIds: string[];
    trainingPairs: number;
    trained: boolean;
  }[];
}

export type DenseCalibrationScorerName = "dense_platt_split" | "dense_isotonic_split";

export interface DenseCalibrationSplitMetadata {
  scorer: DenseCalibrationScorerName;
  trainRatio: number;
  seed: number;
  trainQueryIds: string[];
  evalQueryIds: string[];
  trainingPairs: number;
  trained: boolean;
  parameters: { a: number; b: number } | null;
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

interface ScoreRow {
  id: string;
  score: number;
  index: number;
}

function compareScoreRows(a: ScoreRow, b: ScoreRow): number {
  const diff = b.score - a.score;
  if (diff !== 0) return diff;
  return a.id < b.id ? -1 : a.id > b.id ? 1 : 0;
}

function rankedIds(rows: readonly ScoreRow[]): string[] {
  return rows.slice().sort(compareScoreRows).map((row) => row.id);
}

function runScores(rows: readonly ScoreRow[]): [string, number][] {
  return rows.map((row) => [row.id, row.score]);
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
    metrics[`ndcg@${k}`]! += activeMetricStyle === "python-reference" ? referenceNdcgAtK(ranked, relMap, k) : ndcgAtK(ranked, relMap, k);
    metrics[`map@${k}`]! +=
      activeMetricStyle === "python-reference" ? referenceAveragePrecisionAtK(ranked, relMap, k) : averagePrecisionAtK(ranked, relMap, k);
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
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
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
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
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
  alpha?: BayesianParameterOption;
  beta?: BayesianParameterOption;
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
  metricStyle?: MetricStyle;
  scorers?: string[] | null;
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
  requestedAlpha: BayesianParameterOption;
  requestedBeta: BayesianParameterOption;
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
  metricStyle: MetricStyle;
  scorers: string[] | null;
}

export interface BenchDetails {
  results: ScorerResult[];
  options: BenchResolvedOptions;
  scorers: ScorerMetadata[];
  calibration: CalibrationResult[];
  fittedSplit: FittedSplitMetadata | null;
  attentionSplits: AttentionSplitMetadata[];
  denseCalibrationSplits: DenseCalibrationSplitMetadata[];
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
  const requestedAlpha = options.alpha ?? 1.0;
  const requestedBeta = options.beta ?? 0.5;
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
  const metricStyle = options.metricStyle ?? "pytrec";
  const scorerFilter = buildScorerFilter(options.scorers ?? null);
  const wantsScorer = (name: string): boolean => scorerFilter === null || scorerFilter.has(normalizeScorerName(name));
  const wantsAny = (names: string[]): boolean => scorerFilter === null || names.some((name) => wantsScorer(name));
  const runs = options.runs ?? (calibrationBins === null ? undefined : []);
  const previousMetricStyle = activeMetricStyle;
  activeMetricStyle = metricStyle;
  try {

  const tokenizer = new Tokenizer();
  const docs = corpus.documents();
  const bm25 = new BM25Scorer(corpus, k1, b, bm25Method);
  const needsPseudoQueryScores = requestedAlpha === "auto" || requestedBeta === "auto" || requestedBaseRate === "auto";
  const pseudoQueryScores = needsPseudoQueryScores
    ? collectPseudoQueryScores(docs, bm25, baseRateSampleSize, baseRateSeed)
    : [];
  const { alpha, beta } = resolveBayesianParameters(pseudoQueryScores, requestedAlpha, requestedBeta);
  const baseRate =
    requestedBaseRate === "auto"
      ? estimateBaseRateFromPseudoQueryScores(pseudoQueryScores, docs.length, baseRateMethod)
      : requestedBaseRate;
  const bayes = new BayesianBM25Scorer(bm25, alpha, beta, baseRate);
  const bayesNoBaseRate = baseRate === null ? null : new BayesianBM25Scorer(bm25, alpha, beta, null);
  const vector = new VectorScorer();
  const hybrid = new HybridScorer(bayes, vector, 0.5);
  const hybridCandidates =
    candidateDepth === null
      ? null
      : (terms: string[], embedding: number[]) =>
          selectHybridCandidates(docs, terms, embedding, bm25, candidateDepth);

  const hasEmbeddings = queries.some((q) => q.embedding !== null && q.embedding.length > 0);
  const attentionSplits: AttentionSplitMetadata[] = [];
  const denseCalibrationSplits: DenseCalibrationSplitMetadata[] = [];
  const baselineOnly = hasEmbeddings && isBaselineOnlyScorerFilter(scorerFilter);
  const wantsMultiField = wantsAny(["bayesian_multifield", "bayesian_multifield_bal"]);
  const multiFieldScorer =
    multiField === null || !wantsMultiField
      ? null
      : buildMultiFieldScorer(docs, multiField, k1, b, alpha, beta, baseRate, bm25Method);

  const results: ScorerResult[] = [];
  if (baselineOnly) {
    results.push(
      ...evaluateHybridBaselineRows(
        queries,
        docs,
        bm25,
        qrels,
        tokenizer,
        cutoffs,
        candidateDepth,
        wantsScorer,
        runs,
      ),
    );
  }
  if (!baselineOnly && wantsScorer("bm25")) {
    results.push(evaluate(queries, docs, "bm25", (t, d) => bm25.score(t, d), qrels, tokenizer, cutoffs, runs));
  }
  if (!baselineOnly && bayesNoBaseRate !== null && wantsScorer("bayesian_no_base_rate")) {
    results.push(
      evaluate(
        queries,
        docs,
        "bayesian_no_base_rate",
        (t, d) => bayesianBenchmarkScore(bm25, bayesNoBaseRate, t, d),
        qrels,
        tokenizer,
        cutoffs,
        runs,
      ),
    );
  }
  if (!baselineOnly && wantsScorer("bayesian")) {
    results.push(
      evaluate(
        queries,
        docs,
        "bayesian",
        (t, d) => bayesianBenchmarkScore(bm25, bayes, t, d),
        qrels,
        tokenizer,
        cutoffs,
        runs,
      ),
    );
  }
  if (!baselineOnly && wantsScorer("bayesian_fitted")) {
    results.push(evaluateFittedBayesian(queries, docs, bm25, qrels, tokenizer, cutoffs, alpha, beta, baseRate, runs));
  }
  let fittedSplit: FittedSplitMetadata | null = null;
  if (!baselineOnly && fitSplit !== null && wantsScorer("bayesian_fitted_split")) {
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
  if (!baselineOnly && multiFieldScorer !== null && wantsScorer("bayesian_multifield")) {
    results.push(evaluateMultiField(queries, docs, multiFieldScorer, qrels, tokenizer, cutoffs, candidateDepth, runs));
  }

  if (!baselineOnly && hasEmbeddings) {
    if (wantsScorer("dense")) {
      results.push(evaluateDense(queries, docs, qrels, cutoffs, candidateDepth, runs));
    }
    if (wantsScorer("convex")) {
      results.push(evaluateConvex(queries, docs, bm25, qrels, tokenizer, cutoffs, candidateDepth, runs));
    }
    if (wantsScorer("hybrid_or")) {
      results.push(
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
      );
    }
    if (wantsScorer("hybrid_and")) {
      results.push(
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
      );
    }
    if (wantsScorer("bayesian_logodds")) {
      results.push(evaluateLogOdds(queries, docs, bm25, qrels, tokenizer, cutoffs, candidateDepth, alpha, beta, null, runs));
    }
    if (baseRate !== null && wantsScorer("bayesian_logodds_br")) {
      results.push(evaluateLogOdds(queries, docs, bm25, qrels, tokenizer, cutoffs, candidateDepth, alpha, beta, baseRate, runs));
    }
    if (wantsScorer("bayesian_gated_relu")) {
      results.push(evaluateGatedLogOdds(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, "relu", runs));
    }
    if (wantsScorer("bayesian_gated_swish")) {
      results.push(evaluateGatedLogOdds(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, "swish", runs));
    }
    if (wantsScorer("bayesian_gated_gelu")) {
      results.push(evaluateGatedLogOdds(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, "gelu", runs));
    }
    if (wantsScorer("bayesian_gated_swish_b2")) {
      results.push(evaluateGatedLogOdds(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, "swish_b2", runs));
    }
    if (wantsScorer("bayesian_gated_softplus")) {
      results.push(evaluateGatedLogOdds(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, "softplus", runs));
    }
    if (wantsScorer("balanced_fusion")) {
      results.push(evaluateBalanced(queries, docs, bayes, qrels, tokenizer, cutoffs, bm25, candidateDepth, runs));
    }
    if (wantsScorer("bayesian_vector_balanced")) {
      results.push(
        evaluateVectorProbabilityBalanced(
          queries,
          docs,
          bayes,
          qrels,
          tokenizer,
          cutoffs,
          bm25,
          candidateDepth,
          baseRate,
          runs,
        ),
      );
    }
    if (wantsScorer("bayesian_vector_softplus")) {
      results.push(
        evaluateVectorProbabilitySoftplus(
          queries,
          docs,
          bayes,
          qrels,
          tokenizer,
          cutoffs,
          bm25,
          candidateDepth,
          baseRate,
          runs,
        ),
      );
    }
    if (multiFieldScorer !== null && wantsScorer("bayesian_multifield_bal")) {
      results.push(
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
      );
    }
    if (wantsScorer("rrf")) {
      results.push(evaluateRrf(queries, docs, bm25, qrels, tokenizer, cutoffs, candidateDepth, runs));
    }

    const allQrelsAttentionScorers = [
      "bayesian_attention",
      "bayesian_attn_norm",
      "bayesian_multihead",
      "bayesian_multihead_norm",
    ];
    if (wantsAny(allQrelsAttentionScorers)) {
      const allQrelsAttention = evaluateAllQrelsAttention(
        queries,
        docs,
        bm25,
        bayes,
        qrels,
        tokenizer,
        cutoffs,
        candidateDepth,
        runs,
      );
      results.push(...allQrelsAttention.results.filter((row) => wantsScorer(row.scorer)));
      attentionSplits.push(...allQrelsAttention.metadata.filter((row) => wantsScorer(row.scorer)));
    }

    if (fitSplit !== null) {
      if (wantsAny(["dense_platt_split", "dense_isotonic_split"])) {
        const denseCalibration = evaluateSplitDenseCalibration(
          queries,
          docs,
          qrels,
          cutoffs,
          candidateDepth,
          fitSplit,
          runs,
        );
        results.push(...denseCalibration.results.filter((row) => wantsScorer(row.scorer)));
        denseCalibrationSplits.push(...denseCalibration.metadata.filter((row) => wantsScorer(row.scorer)));
      }

      const splitAttentionScorers = [
        "bayesian_attention_split",
        "bayesian_attn_norm_split",
        "bayesian_multihead_split",
        "bayesian_multihead_norm_split",
      ];
      if (wantsAny(splitAttentionScorers)) {
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
        results.push(...splitAttention.results.filter((row) => wantsScorer(row.scorer)));
        attentionSplits.push(...splitAttention.metadata.filter((row) => wantsScorer(row.scorer)));
      }

      if (wantsScorer("bayesian_attn_norm_cv")) {
        const cvAttention = evaluateCrossValidatedAttentionNorm(
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
        results.push(cvAttention.result);
        attentionSplits.push(cvAttention.metadata);
      }

      if (wantsScorer("bayesian_vector_attn_split")) {
        const vectorAttention = evaluateSplitVectorProbabilityAttention(
          queries,
          docs,
          bm25,
          bayes,
          qrels,
          tokenizer,
          cutoffs,
          candidateDepth,
          fitSplit,
          baseRate,
          runs,
        );
        results.push(vectorAttention.result);
        attentionSplits.push(vectorAttention.metadata);
      }
    }
  }

  const details: BenchDetails = {
    results,
    options: {
      k1,
      b,
      alpha,
      beta,
      requestedAlpha,
      requestedBeta,
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
      metricStyle,
      scorers: scorerFilter === null ? null : [...scorerFilter],
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
    denseCalibrationSplits,
  };
  return details;
  } finally {
    activeMetricStyle = previousMetricStyle;
  }
}

function normalizeScorerName(name: string): string {
  return name.trim().toLowerCase().replace(/[-\s]+/g, "_");
}

function buildScorerFilter(scorers: string[] | null): Set<string> | null {
  if (scorers === null || scorers.length === 0) return null;
  return new Set(scorers.map(normalizeScorerName).filter((name) => name.length > 0));
}

function isBaselineOnlyScorerFilter(filter: Set<string> | null): boolean {
  if (filter === null || filter.size === 0) return false;
  const baseline = new Set(["bm25", "dense", "convex", "rrf"]);
  return [...filter].every((name) => baseline.has(name));
}

function buildScorerMetadata(results: ScorerResult[]): ScorerMetadata[] {
  return results.map((row) => {
    switch (row.scorer) {
      case "bayesian_fitted":
      case "bayesian_attention":
      case "bayesian_attn_norm":
      case "bayesian_multihead":
      case "bayesian_multihead_norm":
        return { scorer: row.scorer, kind: "smoke" };
      case "bayesian_no_base_rate":
      case "bayesian_fitted_split":
      case "bayesian_logodds_br":
      case "dense_platt_split":
      case "dense_isotonic_split":
        return { scorer: row.scorer, kind: "calibration" };
      case "bayesian_attention_split":
      case "bayesian_attn_norm_split":
      case "bayesian_attn_norm_cv":
      case "bayesian_multihead_split":
      case "bayesian_multihead_norm_split":
      case "bayesian_vector_attn_split":
        return { scorer: row.scorer, kind: "tuned" };
      case "hybrid_or":
      case "hybrid_and":
      case "bayesian_gated_relu":
      case "bayesian_gated_swish":
      case "bayesian_gated_gelu":
      case "bayesian_gated_swish_b2":
      case "bayesian_gated_softplus":
      case "bayesian_vector_softplus":
      case "bayesian_multifield_bal":
      case "balanced_fusion":
        return { scorer: row.scorer, kind: "diagnostic" };
      default:
        return { scorer: row.scorer, kind: "zero-shot" };
    }
  });
}

function resolveBayesianParameters(
  perQueryScores: number[][],
  requestedAlpha: BayesianParameterOption,
  requestedBeta: BayesianParameterOption,
): { alpha: number; beta: number } {
  if (requestedAlpha !== "auto" && requestedBeta !== "auto") {
    return { alpha: requestedAlpha, beta: requestedBeta };
  }
  if (perQueryScores.length === 0) {
    return {
      alpha: requestedAlpha === "auto" ? 1.0 : requestedAlpha,
      beta: requestedBeta === "auto" ? 0.0 : requestedBeta,
    };
  }

  const allScores = perQueryScores.flat();
  const estimatedBeta = percentile(allScores, 50);
  const scoreStd = Math.sqrt(variance(allScores, mean(allScores)));
  const estimatedAlpha = scoreStd > 0.0 ? 1.0 / scoreStd : 1.0;
  return {
    alpha: requestedAlpha === "auto" ? estimatedAlpha : requestedAlpha,
    beta: requestedBeta === "auto" ? estimatedBeta : requestedBeta,
  };
}

function estimateBaseRateFromPseudoQueryScores(
  perQueryScores: number[][],
  nDocs: number,
  method: BaseRateMethod,
): number {
  if (nDocs === 0) {
    return 1e-6;
  }
  if (perQueryScores.length === 0) {
    return 1e-6;
  }
  switch (method) {
    case "percentile":
      return estimateBaseRatePercentile(perQueryScores, nDocs);
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
  const includeUnjudgedAsNegative = activeMetricStyle === "python-reference";
  const calibrationScorers = new Set([
    "bayesian_no_base_rate",
    "bayesian",
    "bayesian_fitted",
    "bayesian_fitted_split",
    "bayesian_logodds",
    "bayesian_logodds_br",
    "dense_platt_split",
    "dense_isotonic_split",
    "bayesian_multifield",
    "bayesian_gated_relu",
    "bayesian_gated_swish",
    "bayesian_gated_gelu",
    "bayesian_gated_swish_b2",
    "bayesian_gated_softplus",
    "bayesian_attention",
    "bayesian_attn_norm",
    "bayesian_multihead",
    "bayesian_multihead_norm",
    "bayesian_attention_split",
    "bayesian_attn_norm_split",
    "bayesian_attn_norm_cv",
    "bayesian_multihead_split",
    "bayesian_multihead_norm_split",
    "bayesian_vector_balanced",
    "bayesian_vector_softplus",
    "bayesian_vector_attn_split",
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
      if (includeUnjudgedAsNegative) {
        if (!(score > 0.0)) continue;
      } else if (!relMap.has(docId)) {
        continue;
      }
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

function evaluateHybridBaselineRows(
  queries: BenchQuery[],
  docs: readonly Document[],
  bm25: BM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  candidateDepth: number | null,
  wantsScorer: (name: string) => boolean,
  runs?: ScorerRun[],
): ScorerResult[] {
  const names = ["bm25", "dense", "convex", "rrf"].filter(wantsScorer);
  const state = new Map<string, { metrics: Record<string, number>; counted: number }>();
  for (const name of names) {
    state.set(name, { metrics: blankMetrics(cutoffs), counted: 0 });
  }
  const effectiveDepth =
    candidateDepth === null || candidateDepth <= 0 || candidateDepth >= docs.length ? null : candidateDepth;

  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    const sparseRows: ScoreRow[] = docs.map((d, index) => ({ id: d.id, score: bm25.score(terms, d), index }));
    const sparseRankedRows = sparseRows.slice().sort(compareScoreRows);
    const sparseActiveRows = effectiveDepth === null ? sparseRankedRows : sparseRankedRows.slice(0, effectiveDepth);

    const bm25State = state.get("bm25");
    if (bm25State !== undefined) {
      accumulate(bm25State.metrics, sparseRankedRows.map((row) => row.id), relMap, cutoffs);
      bm25State.counted += 1;
      runs?.push({ scorer: "bm25", queryId: q.queryId, scores: runScores(sparseActiveRows) });
    }

    if (q.embedding === null || (!wantsScorer("dense") && !wantsScorer("convex") && !wantsScorer("rrf"))) {
      continue;
    }

    const denseRows: ScoreRow[] = docs.map((d, index) => ({
      id: d.id,
      score: denseSimilarity(q.embedding!, d),
      index,
    }));
    const denseRankedRows = denseRows.slice().sort(compareScoreRows);
    const denseActiveRows = effectiveDepth === null ? denseRankedRows : denseRankedRows.slice(0, effectiveDepth);

    const denseState = state.get("dense");
    if (denseState !== undefined) {
      accumulate(denseState.metrics, denseActiveRows.map((row) => row.id), relMap, cutoffs);
      denseState.counted += 1;
      runs?.push({ scorer: "dense", queryId: q.queryId, scores: runScores(denseActiveRows) });
    }

    if (!wantsScorer("convex") && !wantsScorer("rrf")) {
      continue;
    }

    const candidateIds = new Set<string>();
    for (const row of sparseActiveRows) candidateIds.add(row.id);
    for (const row of denseActiveRows) candidateIds.add(row.id);
    const candidateRows = docs
      .map((doc, index) => ({ doc, index }))
      .filter(({ doc }) => candidateIds.has(doc.id));

    const convexState = state.get("convex");
    if (convexState !== undefined) {
      const sparse = candidateRows.map(({ index }) => sparseRows[index]!.score);
      const dense = candidateRows.map(({ index }) => denseRows[index]!.score);
      const sparseNorm = minMaxNormalize(sparse);
      const denseNorm = minMaxNormalize(dense);
      const scores: ScoreRow[] = candidateRows.map(({ doc, index }, i) => ({
        id: doc.id,
        score: 0.5 * denseNorm[i]! + 0.5 * sparseNorm[i]!,
        index,
      }));
      accumulate(convexState.metrics, rankedIds(scores), relMap, cutoffs);
      convexState.counted += 1;
      runs?.push({ scorer: "convex", queryId: q.queryId, scores: runScores(scores) });
    }

    const rrfState = state.get("rrf");
    if (rrfState !== undefined) {
      const sparseRanks = new Map<string, number>();
      sparseActiveRows.forEach((row, i) => sparseRanks.set(row.id, i + 1));
      const denseRanks = new Map<string, number>();
      denseActiveRows.forEach((row, i) => denseRanks.set(row.id, i + 1));
      const scores: ScoreRow[] = candidateRows.map(({ doc, index }) => ({
        id: doc.id,
        score:
          (sparseRanks.has(doc.id) ? 1.0 / (60 + sparseRanks.get(doc.id)!) : 0.0) +
          (denseRanks.has(doc.id) ? 1.0 / (60 + denseRanks.get(doc.id)!) : 0.0),
        index,
      }));
      accumulate(rrfState.metrics, rankedIds(scores), relMap, cutoffs);
      rrfState.counted += 1;
      runs?.push({ scorer: "rrf", queryId: q.queryId, scores: runScores(scores) });
    }
  }

  return names.map((name) => {
    const item = state.get(name)!;
    if (item.counted > 0) {
      for (const key of Object.keys(item.metrics)) item.metrics[key]! /= item.counted;
    }
    return { scorer: name, queries: item.counted, metrics: item.metrics };
  });
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
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: "dense", queries: counted, metrics };
}

interface DenseTrainingData {
  scores: number[];
  labels: number[];
}

function evaluateSplitDenseCalibration(
  queries: BenchQuery[],
  docs: readonly Document[],
  qrels: Qrels,
  cutoffs: number[],
  candidateDepth: number | null,
  fitSplit: FitSplitOptions,
  runs?: ScorerRun[],
): { results: ScorerResult[]; metadata: DenseCalibrationSplitMetadata[] } {
  const platt = evaluateSplitDenseCalibrationVariant(
    "dense_platt_split",
    queries,
    docs,
    qrels,
    cutoffs,
    candidateDepth,
    fitSplit,
    runs,
  );
  const isotonic = evaluateSplitDenseCalibrationVariant(
    "dense_isotonic_split",
    queries,
    docs,
    qrels,
    cutoffs,
    candidateDepth,
    fitSplit,
    runs,
  );
  return {
    results: [platt.result, isotonic.result],
    metadata: [platt.metadata, isotonic.metadata],
  };
}

function evaluateSplitDenseCalibrationVariant(
  scorerName: DenseCalibrationScorerName,
  queries: BenchQuery[],
  docs: readonly Document[],
  qrels: Qrels,
  cutoffs: number[],
  candidateDepth: number | null,
  fitSplit: FitSplitOptions,
  runs?: ScorerRun[],
): { result: ScorerResult; metadata: DenseCalibrationSplitMetadata } {
  const split = splitQueries(queries, qrels, fitSplit);
  const trainQueries = split.trainIndices.map((idx) => queries[idx]!);
  const evalQueries = split.evalIndices.map((idx) => queries[idx]!);
  const train = collectDenseCalibrationTrainingData(trainQueries, docs, qrels, fitSplit.seed);

  const metadata: DenseCalibrationSplitMetadata = {
    scorer: scorerName,
    trainRatio: fitSplit.trainRatio,
    seed: fitSplit.seed,
    trainQueryIds: trainQueries.map((q) => q.queryId),
    evalQueryIds: evalQueries.map((q) => q.queryId),
    trainingPairs: train.scores.length,
    trained: false,
    parameters: null,
  };

  if (train.scores.length < 10 || evalQueries.length === 0 || !hasBothBinaryClasses(train.labels)) {
    return { result: { scorer: scorerName, queries: 0, metrics: blankMetrics(cutoffs) }, metadata };
  }

  const calibrate =
    scorerName === "dense_platt_split"
      ? fitDensePlattCalibrator(train.scores, train.labels, metadata)
      : fitDenseIsotonicCalibrator(train.scores, train.labels);
  metadata.trained = true;

  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of evalQueries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0 || q.embedding === null) continue;
    const candidateDocs = selectDenseCandidates(docs, q.embedding, candidateDepth);
    const scores: [string, number][] = candidateDocs.map((d) => [
      d.id,
      Math.min(1.0, Math.max(0.0, calibrate(denseSimilarity(q.embedding!, d)))),
    ]);
    runs?.push({ scorer: scorerName, queryId: q.queryId, scores });
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { result: { scorer: scorerName, queries: counted, metrics }, metadata };
}

function fitDensePlattCalibrator(
  scores: number[],
  labels: number[],
  metadata: DenseCalibrationSplitMetadata,
): (score: number) => number {
  const calibrator = new PlattCalibrator();
  calibrator.fit(scores, labels, 0.01, 1000, 1e-6);
  metadata.parameters = { a: calibrator.a, b: calibrator.b };
  return (score: number) => calibrator.calibrate(score);
}

function fitDenseIsotonicCalibrator(scores: number[], labels: number[]): (score: number) => number {
  const calibrator = new IsotonicCalibrator();
  calibrator.fit(scores, labels);
  return (score: number) => calibrator.calibrate(score);
}

function collectDenseCalibrationTrainingData(
  queries: BenchQuery[],
  docs: readonly Document[],
  qrels: Qrels,
  seed: number,
): DenseTrainingData {
  const scores: number[] = [];
  const labels: number[] = [];

  for (let queryIdx = 0; queryIdx < queries.length; queryIdx++) {
    const q = queries[queryIdx]!;
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0 || q.embedding === null) continue;

    const unjudged: number[] = [];
    let positives = 0;
    for (let docIdx = 0; docIdx < docs.length; docIdx++) {
      const doc = docs[docIdx]!;
      if (!relMap.has(doc.id)) {
        unjudged.push(docIdx);
        continue;
      }
      const label = (relMap.get(doc.id) ?? 0) > 0 ? 1.0 : 0.0;
      if (label > 0) positives += 1;
      scores.push(denseSimilarity(q.embedding, doc));
      labels.push(label);
    }

    const nNeg = Math.min(positives, unjudged.length);
    if (nNeg > 0) {
      const order = shuffleIndices(unjudged.length, seed + queryIdx + 1).slice(0, nNeg);
      for (const orderIdx of order) {
        const doc = docs[unjudged[orderIdx]!]!;
        scores.push(denseSimilarity(q.embedding, doc));
        labels.push(0.0);
      }
    }
  }

  return { scores, labels };
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
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
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
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
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
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
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

function bayesianBenchmarkScore(
  bm25: BM25Scorer,
  scorer: BayesianBM25Scorer,
  queryTerms: string[],
  doc: Document,
): number {
  if (activeMetricStyle !== "python-reference") {
    return scorer.score(queryTerms, doc);
  }
  const rawScore = bm25.score(queryTerms, doc);
  if (rawScore === 0.0) {
    return 0.0;
  }
  const tf = queryTermOverlapCount(queryTerms, doc);
  const prior = scorer.compositePrior(tf, doc.length, bm25.avgdl());
  return scorer.posterior(rawScore, prior);
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
  const train = collectFittedTrainingData(queries, docs, bm25, qrels, tokenizer);
  if (train.scores.length < 4) {
    return { scorer: "bayesian_fitted", queries: 0, metrics: blankMetrics(cutoffs) };
  }
  const transform = new BayesianProbabilityTransform(alpha, beta, baseRate);
  fitBayesianTransform(transform, train.scores, train.labels);
  const fitted = new BayesianBM25Scorer(bm25, transform.alpha, transform.beta, baseRate);
  return evaluate(
    queries,
    docs,
    "bayesian_fitted",
    (t, d) => bayesianBenchmarkScore(bm25, fitted, t, d),
    qrels,
    tokenizer,
    cutoffs,
    runs,
  );
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
  const split = splitQueries(queries, qrels, fitSplit);
  const trainQueries = split.trainIndices.map((idx) => queries[idx]!);
  const evalQueries = split.evalIndices.map((idx) => queries[idx]!);
  const train = collectFittedTrainingData(trainQueries, docs, bm25, qrels, tokenizer);

  const metadata: FittedSplitMetadata = {
    scorer: "bayesian_fitted_split",
    trainRatio: fitSplit.trainRatio,
    seed: fitSplit.seed,
    splitSource: fitSplit.splitSource ?? null,
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
  fitBayesianTransform(transform, train.scores, train.labels);
  metadata.alpha = transform.alpha;
  metadata.beta = transform.beta;
  const fitted = new BayesianBM25Scorer(bm25, transform.alpha, transform.beta, baseRate);
  const result = evaluate(
    evalQueries,
    docs,
    "bayesian_fitted_split",
    (t, d) => bayesianBenchmarkScore(bm25, fitted, t, d),
    qrels,
    tokenizer,
    cutoffs,
    runs,
  );
  return { result, metadata };
}

function fitBayesianTransform(transform: BayesianProbabilityTransform, scores: number[], labels: number[]): void {
  if (activeMetricStyle === "python-reference") {
    transform.fit(scores, labels, 0.05, 3000);
    return;
  }
  transform.fit(scores, labels);
}

function splitQueries(
  queries: BenchQuery[],
  qrels: Qrels,
  fitSplitOrTrainRatio: FitSplitOptions | number,
  seedMaybe?: number,
): { trainIndices: number[]; evalIndices: number[] } {
  const fitSplit =
    typeof fitSplitOrTrainRatio === "number"
      ? { trainRatio: fitSplitOrTrainRatio, seed: seedMaybe ?? 42 }
      : fitSplitOrTrainRatio;
  if (fitSplit.trainQueryIds !== undefined || fitSplit.evalQueryIds !== undefined) {
    if (fitSplit.trainQueryIds === undefined || fitSplit.evalQueryIds === undefined) {
      throw new Error("fit split requires both trainQueryIds and evalQueryIds");
    }
    const queryIndexById = new Map(queries.map((query, idx) => [query.queryId, idx] as const));
    const seen = new Set<string>();
    const resolveIds = (ids: string[], label: string): number[] => {
      const indices: number[] = [];
      for (const id of ids) {
        if (seen.has(id)) {
          throw new Error(`duplicate query id in fit split: ${id}`);
        }
        seen.add(id);
        const idx = queryIndexById.get(id);
        if (idx === undefined) {
          throw new Error(`fit split ${label} query id not found: ${id}`);
        }
        indices.push(idx);
      }
      return indices;
    };
    return {
      trainIndices: resolveIds(fitSplit.trainQueryIds, "train"),
      evalIndices: resolveIds(fitSplit.evalQueryIds, "eval"),
    };
  }

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
  const shuffled = shuffleIndices(eligible.length, fitSplit.seed).map((idx) => eligible[idx]!);
  const nTrain = Math.min(eligible.length - 1, Math.max(1, Math.round(eligible.length * fitSplit.trainRatio)));
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
  const includeUnjudgedAsNegative = activeMetricStyle === "python-reference";
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0) continue;
    const terms = q.terms ?? tokenizer.tokenize(q.text);
    for (const d of docs) {
      const raw = bm25.score(terms, d);
      if (raw <= 0.0) continue;
      if (!includeUnjudgedAsNegative && !relMap.has(d.id)) continue;
      scores.push(raw);
      labels.push((relMap.get(d.id) ?? 0) > 0 ? 1.0 : 0.0);
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
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
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
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
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

type AllQrelsAttentionScorerName =
  | "bayesian_attention"
  | "bayesian_attn_norm"
  | "bayesian_multihead"
  | "bayesian_multihead_norm";

function evaluateAllQrelsAttention(
  queries: BenchQuery[],
  docs: readonly Document[],
  bm25: BM25Scorer,
  bayes: BayesianBM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  candidateDepth: number | null,
  runs?: ScorerRun[],
): { results: ScorerResult[]; metadata: AttentionSplitMetadata[] } {
  const basic = evaluateAllQrelsAttentionVariant(
    "bayesian_attention",
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
    runs,
  );
  const normalized = evaluateAllQrelsAttentionVariant(
    "bayesian_attn_norm",
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
    runs,
  );
  const multiHead = evaluateAllQrelsAttentionVariant(
    "bayesian_multihead",
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
    runs,
  );
  const multiHeadNorm = evaluateAllQrelsAttentionVariant(
    "bayesian_multihead_norm",
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
    runs,
  );
  return {
    results: [basic.result, normalized.result, multiHead.result, multiHeadNorm.result],
    metadata: [basic.metadata, normalized.metadata, multiHead.metadata, multiHeadNorm.metadata],
  };
}

function evaluateAllQrelsAttentionVariant(
  scorerName: AllQrelsAttentionScorerName,
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
  runs?: ScorerRun[],
): { result: ScorerResult; metadata: AttentionSplitMetadata } {
  const evalQueries = queries.filter((q) => {
    const relMap = qrels.get(q.queryId);
    return q.embedding !== null && relMap !== undefined && relMap.size > 0;
  });
  const nFeatures = featureSet === "basic" ? 3 : 7;
  const metadata: AttentionSplitMetadata = {
    scorer: scorerName,
    trainRatio: 1.0,
    seed: 0,
    trainQueryIds: evalQueries.map((q) => q.queryId),
    evalQueryIds: evalQueries.map((q) => q.queryId),
    trainingPairs: 0,
    features: featureSet,
    normalize,
    heads,
    trained: false,
    protocol: "all-qrels",
  };

  if (evalQueries.length === 0) {
    return { result: { scorer: scorerName, queries: 0, metrics: blankMetrics(cutoffs) }, metadata };
  }

  const trainCaches = evalQueries.map((q) => buildAttentionQueryCache(q, docs, bm25, bayes, tokenizer, candidateDepth));
  const trainQueryIds = new Map(evalQueries.map((q, i) => [q.queryId, i]));
  const train = collectAttentionTrainingData(trainCaches, qrels, trainQueryIds, featureSet, 0);
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
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { result: { scorer: scorerName, queries: counted, metrics }, metadata };
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
  const split = splitQueries(queries, qrels, fitSplit);
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
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { result: { scorer: scorerName, queries: counted, metrics }, metadata };
}

function evaluateCrossValidatedAttentionNorm(
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
  requestedFolds = 5,
): { result: ScorerResult; metadata: AttentionSplitMetadata } {
  const scorerName = "bayesian_attn_norm_cv";
  const eligibleIndices: number[] = [];
  for (let i = 0; i < queries.length; i++) {
    const q = queries[i]!;
    const relMap = qrels.get(q.queryId);
    if (q.embedding !== null && relMap !== undefined && relMap.size > 0) {
      eligibleIndices.push(i);
    }
  }
  const shuffled = shuffleIndices(eligibleIndices.length, fitSplit.seed).map((idx) => eligibleIndices[idx]!);
  const nFolds = Math.min(Math.max(2, requestedFolds), shuffled.length);
  const folds = Array.from({ length: nFolds }, () => [] as number[]);
  for (let i = 0; i < shuffled.length; i++) {
    folds[i % nFolds]!.push(shuffled[i]!);
  }

  const metadata: AttentionSplitMetadata = {
    scorer: scorerName,
    trainRatio: nFolds > 0 ? (nFolds - 1) / nFolds : 0,
    seed: fitSplit.seed,
    trainQueryIds: shuffled.map((idx) => queries[idx]!.queryId),
    evalQueryIds: shuffled.map((idx) => queries[idx]!.queryId),
    trainingPairs: 0,
    features: "rich",
    normalize: true,
    heads: 1,
    trained: false,
    protocol: "cross-validation",
    folds: [],
  };

  if (shuffled.length < 2) {
    return { result: { scorer: scorerName, queries: 0, metrics: blankMetrics(cutoffs) }, metadata };
  }

  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  let totalTrainingPairs = 0;

  for (let foldIdx = 0; foldIdx < folds.length; foldIdx++) {
    const evalSet = new Set(folds[foldIdx]!);
    const trainIndices = shuffled.filter((idx) => !evalSet.has(idx));
    const trainQueries = trainIndices.map((idx) => queries[idx]!);
    const evalQueries = folds[foldIdx]!.map((idx) => queries[idx]!);
    const foldMeta = {
      fold: foldIdx,
      trainQueryIds: trainQueries.map((q) => q.queryId),
      evalQueryIds: evalQueries.map((q) => q.queryId),
      trainingPairs: 0,
      trained: false,
    };

    const trainCaches = trainQueries.map((q) =>
      buildAttentionQueryCache(q, docs, bm25, bayes, tokenizer, candidateDepth),
    );
    const trainQueryIds = new Map(trainQueries.map((q, i) => [q.queryId, i]));
    const train = collectAttentionTrainingData(trainCaches, qrels, trainQueryIds, "rich", fitSplit.seed + foldIdx + 1);
    foldMeta.trainingPairs = train.pairs;
    totalTrainingPairs += train.pairs;

    if (train.pairs < 10 || !hasBothBinaryClasses(train.labels)) {
      metadata.folds!.push(foldMeta);
      continue;
    }

    const model = new AttentionLogOddsWeights(2, 7, 0.5, true, 0, null);
    model.fit(train.probs, train.labels, train.features, train.pairs, train.queryIds, 0.01, 500, 1e-6);
    foldMeta.trained = true;
    metadata.trained = true;

    for (const q of evalQueries) {
      const relMap = qrels.get(q.queryId);
      if (relMap === undefined || relMap.size === 0 || q.embedding === null) continue;
      const cache = buildAttentionQueryCache(q, docs, bm25, bayes, tokenizer, candidateDepth);
      const fused = model.combine(cache.probs, cache.candidateDocs.length, cache.featuresRich, 1, true);
      const scores: [string, number][] = cache.candidateDocs.map((d, i) => [d.id, fused[i]!]);
      runs?.push({ scorer: scorerName, queryId: q.queryId, scores });
      accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
      counted += 1;
    }

    metadata.folds!.push(foldMeta);
  }

  metadata.trainingPairs = totalTrainingPairs;
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
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: "balanced_fusion", queries: counted, metrics };
}

function evaluateVectorProbabilityBalanced(
  queries: BenchQuery[],
  docs: readonly Document[],
  bayes: BayesianBM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  bm25: BM25Scorer,
  candidateDepth: number | null,
  baseRate: number | null,
  runs?: ScorerRun[],
  weight = 0.5,
): ScorerResult {
  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0 || q.embedding === null) continue;
    const cache = buildVectorProbabilityQueryCache(q, docs, bm25, bayes, tokenizer, candidateDepth, baseRate);
    const sparse = signalColumn(cache.probs, 0);
    const vector = signalColumn(cache.probs, 1);
    const fused = balancedLogOddsFusion(sparse, vector, weight);
    const scores: [string, number][] = cache.candidateDocs.map((d, i) => [d.id, fused[i]!]);
    runs?.push({ scorer: "bayesian_vector_balanced", queryId: q.queryId, scores });
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: "bayesian_vector_balanced", queries: counted, metrics };
}

function evaluateVectorProbabilitySoftplus(
  queries: BenchQuery[],
  docs: readonly Document[],
  bayes: BayesianBM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  bm25: BM25Scorer,
  candidateDepth: number | null,
  baseRate: number | null,
  runs?: ScorerRun[],
): ScorerResult {
  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of queries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0 || q.embedding === null) continue;
    const cache = buildVectorProbabilityQueryCache(q, docs, bm25, bayes, tokenizer, candidateDepth, baseRate);
    const scores: [string, number][] = cache.candidateDocs.map((d, i) => [
      d.id,
      logOddsConjunction([cache.probs[i * 2]!, cache.probs[i * 2 + 1]!], null, null, Gating.Softplus),
    ]);
    runs?.push({ scorer: "bayesian_vector_softplus", queryId: q.queryId, scores });
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { scorer: "bayesian_vector_softplus", queries: counted, metrics };
}

function evaluateSplitVectorProbabilityAttention(
  queries: BenchQuery[],
  docs: readonly Document[],
  bm25: BM25Scorer,
  bayes: BayesianBM25Scorer,
  qrels: Qrels,
  tokenizer: Tokenizer,
  cutoffs: number[],
  candidateDepth: number | null,
  fitSplit: FitSplitOptions,
  baseRate: number | null,
  runs?: ScorerRun[],
): { result: ScorerResult; metadata: AttentionSplitMetadata } {
  const scorerName = "bayesian_vector_attn_split";
  const split = splitQueries(queries, qrels, fitSplit);
  const trainQueries = split.trainIndices.map((idx) => queries[idx]!);
  const evalQueries = split.evalIndices.map((idx) => queries[idx]!);
  const metadata: AttentionSplitMetadata = {
    scorer: scorerName,
    trainRatio: fitSplit.trainRatio,
    seed: fitSplit.seed,
    trainQueryIds: trainQueries.map((q) => q.queryId),
    evalQueryIds: evalQueries.map((q) => q.queryId),
    trainingPairs: 0,
    features: "basic",
    normalize: false,
    heads: 1,
    trained: false,
  };

  if (evalQueries.length === 0) {
    return { result: { scorer: scorerName, queries: 0, metrics: blankMetrics(cutoffs) }, metadata };
  }

  const trainCaches = trainQueries
    .filter((q) => q.embedding !== null)
    .map((q) => buildVectorProbabilityQueryCache(q, docs, bm25, bayes, tokenizer, candidateDepth, baseRate));
  const trainQueryIds = new Map(trainQueries.map((q, i) => [q.queryId, i]));
  const train = collectAttentionTrainingData(trainCaches, qrels, trainQueryIds, "basic", fitSplit.seed);
  metadata.trainingPairs = train.pairs;

  if (train.pairs < 10 || !hasBothBinaryClasses(train.labels)) {
    return { result: { scorer: scorerName, queries: 0, metrics: blankMetrics(cutoffs) }, metadata };
  }

  const model = new AttentionLogOddsWeights(2, 3, 0.5, false, 0, null);
  model.fit(train.probs, train.labels, train.features, train.pairs, null, 0.01, 500, 1e-6);
  metadata.trained = true;

  const metrics = blankMetrics(cutoffs);
  let counted = 0;
  for (const q of evalQueries) {
    const relMap = qrels.get(q.queryId);
    if (relMap === undefined || relMap.size === 0 || q.embedding === null) continue;
    const cache = buildVectorProbabilityQueryCache(q, docs, bm25, bayes, tokenizer, candidateDepth, baseRate);
    const fused = model.combine(cache.probs, cache.candidateDocs.length, cache.featuresBasic, 1, true);
    const scores: [string, number][] = cache.candidateDocs.map((d, i) => [d.id, fused[i]!]);
    runs?.push({ scorer: scorerName, queryId: q.queryId, scores });
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
    counted += 1;
  }
  if (counted > 0) {
    for (const key of Object.keys(metrics)) metrics[key]! /= counted;
  }
  return { result: { scorer: scorerName, queries: counted, metrics }, metadata };
}

function buildVectorProbabilityQueryCache(
  q: BenchQuery,
  docs: readonly Document[],
  bm25: BM25Scorer,
  bayes: BayesianBM25Scorer,
  tokenizer: Tokenizer,
  candidateDepth: number | null,
  baseRate: number | null,
): AttentionQueryCache {
  const embedding = q.embedding ?? [];
  const terms = q.terms ?? tokenizer.tokenize(q.text);
  const bm25Scores = docs.map((d) => bm25.score(terms, d));
  const denseScores = docs.map((d) => denseSimilarity(embedding, d));
  const sampleDistances = denseScores.map(cosineDistance);
  const sampleWeights = docs.map((d) => bayes.score(terms, d));
  const candidateDocs =
    candidateDepth === null ? docs : selectHybridCandidates(docs, terms, embedding, bm25, candidateDepth);
  const candidateDistances = candidateDocs.map((d) => cosineDistance(denseSimilarity(embedding, d)));
  const transform = VectorProbabilityTransform.fitBackground(sampleDistances, { baseRate });
  const vectorProbs = transform.calibrateWithSample(candidateDistances, sampleDistances, {
    weights: sampleWeights,
    method: "auto",
  }) as number[];
  const probs: number[] = [];
  for (let i = 0; i < candidateDocs.length; i++) {
    probs.push(bayes.score(terms, candidateDocs[i]!), vectorProbs[i]!);
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

function cosineDistance(similarity: number): number {
  return Math.max(0.0, 1.0 - Math.max(-1.0, Math.min(1.0, similarity)));
}

function signalColumn(probs: number[], offset: 0 | 1): number[] {
  const out: number[] = [];
  for (let i = offset; i < probs.length; i += 2) {
    out.push(probs[i]!);
  }
  return out;
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
    accumulate(metrics, rankForMetrics(scores), relMap, cutoffs);
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
