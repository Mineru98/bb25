/**
 * @bb25/core — pure-TypeScript port of the bb25 Rust core.
 * Zero runtime dependencies; inputs are strings and number[]/Float32Array only.
 */
export type { Vector } from "./mathUtils.js";
export {
  EPSILON,
  sigmoid,
  safeLog,
  logit,
  safeProb,
  clamp,
  dotProduct,
  vectorMagnitude,
  cosineSimilarity,
  softmax,
  softmaxRows,
  minMaxNormalize,
} from "./mathUtils.js";

export { Tokenizer } from "./tokenizer.js";
export { Corpus } from "./corpus.js";
export type { Document, CorpusStats } from "./corpus.js";
export { BM25Scorer } from "./bm25.js";
export { BayesianBM25Scorer } from "./bayesian.js";
export { VectorScorer } from "./vector.js";
export { HybridScorer } from "./hybrid.js";

export {
  Gating,
  cosineToProbability,
  probNot,
  probAnd,
  probOr,
  logOddsConjunction,
  balancedLogOddsFusion,
} from "./fusion.js";

export type { Embedder } from "./embedder.js";
export {
  buildDefaultCorpus,
  buildDefaultQueries,
} from "./defaults.js";
export type { DefaultQuery } from "./defaults.js";

export {
  expectedCalibrationError,
  brierScore,
  reliabilityDiagram,
  calibrationReport,
  summarizeCalibration,
} from "./metrics.js";
export type { CalibrationReport, ReliabilityBin } from "./metrics.js";

export { ParameterLearner } from "./parameterLearner.js";
export type { ParameterLearnerResult } from "./parameterLearner.js";

export { ExperimentRunner, runExperiments } from "./experiments.js";
export type { Query, ExperimentResult } from "./experiments.js";

export {
  BayesianProbabilityTransform,
  TemporalBayesianTransform,
  TrainingMode,
} from "./probability.js";

export { PlattCalibrator, IsotonicCalibrator } from "./calibration.js";

export { LearnableLogOddsWeights } from "./learnableWeights.js";
export { AttentionLogOddsWeights } from "./attentionWeights.js";
export { MultiHeadAttentionLogOddsWeights } from "./multiHeadAttention.js";
export { BlockMaxIndex } from "./blockMaxIndex.js";

export { FusionDebugger } from "./debug.js";
export type {
  BM25SignalTrace,
  VectorSignalTrace,
  NotTrace,
  FusionTrace,
  SignalTrace,
  DocumentTrace,
  ComparisonResult,
} from "./debug.js";
