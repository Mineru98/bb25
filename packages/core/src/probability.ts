/**
 * Bayesian probability transforms. Direct port of `src/probability.rs`.
 *
 * Sigmoid likelihood + composite prior + Bayesian posterior with optional
 * base-rate correction (two-step Bayes update). Supports batch fitting
 * (gradient descent) and online learning (SGD with EMA, Polyak averaging,
 * gradient clipping). Temporal variant applies exponential decay weighting.
 *
 * JS `number` is IEEE-754 f64, identical to Rust `f64`. Every accumulation
 * follows the reference's left-to-right order so results match bit-for-bit.
 */

import { sigmoid, safeProb, clamp } from "./mathUtils.js";

const ALPHA_MIN = 0.01;

/** Training mode for parameter learning (C1/C2/C3 conditions). */
export type TrainingMode = "balanced" | "priorAware" | "priorFree";

export const TrainingMode = {
  /** C1: Train on sigmoid likelihood pred = sigmoid(alpha*(s-beta)). */
  Balanced: "balanced",
  /** C2: Train on full Bayesian posterior with composite prior. */
  PriorAware: "priorAware",
  /** C3: Same training as balanced, but at inference prior=0.5. */
  PriorFree: "priorFree",
} as const;

type PriorFn = (score: number, tf: number, docLenRatio: number) => number;

/**
 * Transforms raw BM25 scores into calibrated probabilities.
 */
export class BayesianProbabilityTransform {
  public alpha: number;
  public beta: number;
  public baseRate: number | null;
  // NOTE: the reference carries a `logit_base_rate` field marked #[allow(dead_code)];
  // it is never read in either language, so it is omitted here (no behavioral effect).
  private trainingModeValue: TrainingMode;
  private nUpdates: number;
  private gradAlphaEma: number;
  private gradBetaEma: number;
  private alphaAvg: number;
  private betaAvg: number;
  private priorFn: PriorFn | null;

  constructor(alpha = 1.0, beta = 0.0, baseRate: number | null = null) {
    if (baseRate !== null) {
      if (!(baseRate > 0.0 && baseRate < 1.0)) {
        throw new Error(`base_rate must be in (0, 1), got ${baseRate}`);
      }
    }
    this.alpha = alpha;
    this.beta = beta;
    this.baseRate = baseRate;
    this.trainingModeValue = TrainingMode.Balanced;
    this.nUpdates = 0;
    this.gradAlphaEma = 0.0;
    this.gradBetaEma = 0.0;
    this.alphaAvg = alpha;
    this.betaAvg = beta;
    this.priorFn = null;
  }

  /**
   * Create a new transform with a custom prior function.
   *
   * The priorFn receives (score, tf, docLenRatio) and returns a prior probability.
   */
  static withPriorFn(
    alpha: number,
    beta: number,
    baseRate: number | null,
    priorFn: PriorFn,
  ): BayesianProbabilityTransform {
    const s = new BayesianProbabilityTransform(alpha, beta, baseRate);
    s.priorFn = priorFn;
    return s;
  }

  /** EMA-averaged alpha for stable inference after online updates. */
  averagedAlpha(): number {
    return this.alphaAvg;
  }

  /** EMA-averaged beta for stable inference after online updates. */
  averagedBeta(): number {
    return this.betaAvg;
  }

  /** Current training mode. */
  trainingMode(): TrainingMode {
    return this.trainingModeValue;
  }

  /** Sigmoid likelihood: sigma(alpha * (score - beta)). */
  likelihood(score: number): number {
    return sigmoid(this.alpha * (score - this.beta));
  }

  /** Term-frequency prior: 0.2 + 0.7 * min(1, tf / 10). */
  static tfPrior(tf: number): number {
    return 0.2 + 0.7 * Math.min(tf / 10.0, 1.0);
  }

  /**
   * Document-length normalization prior (Eq. 26).
   *
   * P_norm = 0.3 + 0.6 * (1 - min(1, |docLenRatio - 0.5| * 2))
   */
  static normPrior(docLenRatio: number): number {
    return 0.3 + 0.6 * (1.0 - Math.min(Math.abs(docLenRatio - 0.5) * 2.0, 1.0));
  }

  /** Composite prior: clamp(0.7 * P_tf + 0.3 * P_norm, 0.1, 0.9). */
  static compositePrior(tf: number, docLenRatio: number): number {
    const pTf = BayesianProbabilityTransform.tfPrior(tf);
    const pNorm = BayesianProbabilityTransform.normPrior(docLenRatio);
    return clamp(0.7 * pTf + 0.3 * pNorm, 0.1, 0.9);
  }

  /**
   * Bayesian posterior via two-step Bayes update.
   *
   * Without baseRate: P = L*p / (L*p + (1-L)*(1-p))
   * With baseRate: second Bayes update using baseRate as corpus-level prior.
   */
  static posterior(
    likelihoodVal: number,
    prior: number,
    baseRate: number | null,
  ): number {
    const l = safeProb(likelihoodVal);
    const p = safeProb(prior);
    const numerator = l * p;
    const denominator = numerator + (1.0 - l) * (1.0 - p);
    let result = safeProb(numerator / denominator);

    if (baseRate !== null) {
      const br = baseRate;
      const numBr = result * br;
      const denBr = numBr + (1.0 - result) * (1.0 - br);
      result = safeProb(numBr / denBr);
    }

    return result;
  }

  /** Full pipeline: BM25 score -> calibrated probability. */
  scoreToProbability(score: number, tf: number, docLenRatio: number): number {
    const lVal = this.likelihood(score);

    let prior: number;
    if (this.trainingModeValue === TrainingMode.PriorFree) {
      prior = 0.5;
    } else if (this.priorFn !== null) {
      prior = this.priorFn(score, tf, docLenRatio);
    } else {
      prior = BayesianProbabilityTransform.compositePrior(tf, docLenRatio);
    }

    return BayesianProbabilityTransform.posterior(lVal, prior, this.baseRate);
  }

  /** WAND upper bound for safe document pruning (Theorem 6.1.2). */
  wandUpperBound(bm25UpperBound: number, pMax = 0.9): number {
    const lMax = this.likelihood(bm25UpperBound);
    return BayesianProbabilityTransform.posterior(lMax, pMax, this.baseRate);
  }

  /** Batch gradient descent to learn alpha and beta (Algorithm 8.3.1). */
  fit(
    scores: number[],
    labels: number[],
    learningRate = 0.01,
    maxIterations = 1000,
    tolerance = 1e-6,
    mode: TrainingMode = "balanced",
    tfs?: number[],
    docLenRatios?: number[],
  ): void {
    if (mode === TrainingMode.PriorAware) {
      if (tfs === undefined || docLenRatios === undefined) {
        throw new Error(
          "tfs and doc_len_ratios are required when mode is PriorAware",
        );
      }
    }

    let priors: number[] | null = null;
    if (mode === TrainingMode.PriorAware) {
      const tfsRef = tfs as number[];
      const dlrs = docLenRatios as number[];
      const m = Math.min(tfsRef.length, dlrs.length);
      priors = [];
      for (let i = 0; i < m; i++) {
        priors.push(
          BayesianProbabilityTransform.compositePrior(
            tfsRef[i] as number,
            dlrs[i] as number,
          ),
        );
      }
    }

    let alpha = this.alpha;
    let beta = this.beta;
    const n = scores.length;

    for (let iter = 0; iter < maxIterations; iter++) {
      let gradAlpha: number;
      let gradBeta: number;
      if (mode === TrainingMode.PriorAware) {
        [gradAlpha, gradBeta] = computePriorAwareGradients(
          scores,
          labels,
          priors as number[],
          alpha,
          beta,
          n,
        );
      } else {
        [gradAlpha, gradBeta] = computeBalancedGradients(
          scores,
          labels,
          alpha,
          beta,
          n,
        );
      }

      const newAlpha = alpha - learningRate * gradAlpha;
      const newBeta = beta - learningRate * gradBeta;

      if (
        Math.abs(newAlpha - alpha) < tolerance &&
        Math.abs(newBeta - beta) < tolerance
      ) {
        alpha = newAlpha;
        beta = newBeta;
        break;
      }

      alpha = newAlpha;
      beta = newBeta;
    }

    this.alpha = alpha;
    this.beta = beta;
    this.trainingModeValue = mode;
    this.nUpdates = 0;
    this.gradAlphaEma = 0.0;
    this.gradBetaEma = 0.0;
    this.alphaAvg = alpha;
    this.betaAvg = beta;
  }

  /**
   * Reset online learning state after a batch fit (internal helper used by
   * both `fit` and the temporal wrapper). Mirrors the field resets at the end
   * of Rust `fit`: training_mode, n_updates, grad EMAs, and Polyak averages.
   */
  resetAfterFit(mode: TrainingMode, alpha: number, beta: number): void {
    this.alpha = alpha;
    this.beta = beta;
    this.trainingModeValue = mode;
    this.nUpdates = 0;
    this.gradAlphaEma = 0.0;
    this.gradBetaEma = 0.0;
    this.alphaAvg = alpha;
    this.betaAvg = beta;
  }

  /** Online SGD update from a single observation or mini-batch. */
  update(
    scores: number[],
    labels: number[],
    learningRate = 0.01,
    momentum = 0.9,
    decayTau = 1000,
    maxGradNorm = 1.0,
    avgDecay = 0.995,
    mode: TrainingMode | null = null,
    tfs?: number[],
    docLenRatios?: number[],
  ): void {
    const effectiveMode = mode ?? this.trainingModeValue;
    if (effectiveMode === TrainingMode.PriorAware) {
      if (tfs === undefined || docLenRatios === undefined) {
        throw new Error(
          "tfs and doc_len_ratios are required when mode is PriorAware",
        );
      }
    }

    const n = scores.length;

    let gradAlpha: number;
    let gradBeta: number;
    if (effectiveMode === TrainingMode.PriorAware) {
      const tfsRef = tfs as number[];
      const dlrs = docLenRatios as number[];
      const m = Math.min(tfsRef.length, dlrs.length);
      const priors: number[] = [];
      for (let i = 0; i < m; i++) {
        priors.push(
          BayesianProbabilityTransform.compositePrior(
            tfsRef[i] as number,
            dlrs[i] as number,
          ),
        );
      }
      [gradAlpha, gradBeta] = computePriorAwareGradients(
        scores,
        labels,
        priors,
        this.alpha,
        this.beta,
        n,
      );
    } else {
      [gradAlpha, gradBeta] = computeBalancedGradients(
        scores,
        labels,
        this.alpha,
        this.beta,
        n,
      );
    }

    if (mode !== null) {
      this.trainingModeValue = effectiveMode;
    }

    // EMA smoothing
    this.gradAlphaEma = momentum * this.gradAlphaEma + (1.0 - momentum) * gradAlpha;
    this.gradBetaEma = momentum * this.gradBetaEma + (1.0 - momentum) * gradBeta;

    // Bias correction
    this.nUpdates += 1;
    const correction = 1.0 - Math.pow(momentum, this.nUpdates);
    let correctedAlpha = this.gradAlphaEma / correction;
    let correctedBeta = this.gradBetaEma / correction;

    // Gradient clipping
    const gradNorm = Math.sqrt(
      correctedAlpha * correctedAlpha + correctedBeta * correctedBeta,
    );
    if (gradNorm > maxGradNorm) {
      const scale = maxGradNorm / gradNorm;
      correctedAlpha *= scale;
      correctedBeta *= scale;
    }

    // Learning rate decay
    const effectiveLr = learningRate / (1.0 + this.nUpdates / decayTau);

    this.alpha -= effectiveLr * correctedAlpha;
    this.beta -= effectiveLr * correctedBeta;

    // Alpha must stay positive
    if (this.alpha < ALPHA_MIN) {
      this.alpha = ALPHA_MIN;
    }

    // Polyak parameter averaging
    this.alphaAvg = avgDecay * this.alphaAvg + (1.0 - avgDecay) * this.alpha;
    this.betaAvg = avgDecay * this.betaAvg + (1.0 - avgDecay) * this.beta;
  }
}

/** Compute gradients for balanced/prior_free training mode. */
function computeBalancedGradients(
  scores: number[],
  labels: number[],
  alpha: number,
  beta: number,
  n: number,
): [number, number] {
  let gradAlpha = 0.0;
  let gradBeta = 0.0;
  const m = Math.min(scores.length, labels.length);
  for (let i = 0; i < m; i++) {
    const s = scores[i] as number;
    const y = labels[i] as number;
    const l = safeProb(sigmoid(alpha * (s - beta)));
    const error = l - y;
    gradAlpha += error * (s - beta);
    gradBeta += error * -alpha;
  }
  return [gradAlpha / n, gradBeta / n];
}

/** Compute gradients for prior_aware training mode. */
function computePriorAwareGradients(
  scores: number[],
  labels: number[],
  priors: number[],
  alpha: number,
  beta: number,
  n: number,
): [number, number] {
  let gradAlpha = 0.0;
  let gradBeta = 0.0;
  const m = Math.min(scores.length, labels.length);
  for (let i = 0; i < m; i++) {
    const s = scores[i] as number;
    const y = labels[i] as number;
    const l = safeProb(sigmoid(alpha * (s - beta)));
    const p = priors[i] as number;
    const denom = l * p + (1.0 - l) * (1.0 - p);
    const predicted = safeProb((l * p) / denom);

    const dpDl = (p * (1.0 - p)) / (denom * denom);
    const dlDalpha = l * (1.0 - l) * (s - beta);
    const dlDbeta = -l * (1.0 - l) * alpha;

    const error = predicted - y;
    gradAlpha += error * dpDl * dlDalpha;
    gradBeta += error * dpDl * dlDbeta;
  }
  return [gradAlpha / n, gradBeta / n];
}

/** Compute weighted gradients for balanced/prior_free training mode. */
function computeWeightedBalancedGradients(
  scores: number[],
  labels: number[],
  weights: number[],
  alpha: number,
  beta: number,
  n: number,
): [number, number] {
  let gradAlpha = 0.0;
  let gradBeta = 0.0;
  const m = Math.min(scores.length, labels.length);
  for (let i = 0; i < m; i++) {
    const s = scores[i] as number;
    const y = labels[i] as number;
    const l = safeProb(sigmoid(alpha * (s - beta)));
    const error = l - y;
    gradAlpha += (weights[i] as number) * error * (s - beta);
    gradBeta += (weights[i] as number) * error * -alpha;
  }
  return [gradAlpha / n, gradBeta / n];
}

/** Compute weighted gradients for prior_aware training mode. */
function computeWeightedPriorAwareGradients(
  scores: number[],
  labels: number[],
  priors: number[],
  weights: number[],
  alpha: number,
  beta: number,
  n: number,
): [number, number] {
  let gradAlpha = 0.0;
  let gradBeta = 0.0;
  const m = Math.min(scores.length, labels.length);
  for (let i = 0; i < m; i++) {
    const s = scores[i] as number;
    const y = labels[i] as number;
    const l = safeProb(sigmoid(alpha * (s - beta)));
    const p = priors[i] as number;
    const denom = l * p + (1.0 - l) * (1.0 - p);
    const predicted = safeProb((l * p) / denom);

    const dpDl = (p * (1.0 - p)) / (denom * denom);
    const dlDalpha = l * (1.0 - l) * (s - beta);
    const dlDbeta = -l * (1.0 - l) * alpha;

    const error = predicted - y;
    gradAlpha += (weights[i] as number) * error * dpDl * dlDalpha;
    gradBeta += (weights[i] as number) * error * dpDl * dlDbeta;
  }
  return [gradAlpha / n, gradBeta / n];
}

/**
 * Wraps a BayesianProbabilityTransform with temporal decay weighting.
 *
 * Recent observations receive higher weight during fitting and updating,
 * controlled by an exponential decay with configurable half-life.
 */
export class TemporalBayesianTransform {
  public transform: BayesianProbabilityTransform;
  private decayHalfLifeValue: number;
  private decayRate: number;
  private timestampValue: number;

  constructor(
    alpha = 1.0,
    beta = 0.0,
    baseRate: number | null = null,
    decayHalfLife = 100.0,
  ) {
    if (!(decayHalfLife > 0.0)) {
      throw new Error(`decay_half_life must be > 0, got ${decayHalfLife}`);
    }
    const decayRate = Math.log(2.0) / decayHalfLife;
    this.transform = new BayesianProbabilityTransform(alpha, beta, baseRate);
    this.decayHalfLifeValue = decayHalfLife;
    this.decayRate = decayRate;
    this.timestampValue = 0;
  }

  /** Half-life of the temporal decay. */
  decayHalfLife(): number {
    return this.decayHalfLifeValue;
  }

  /** Current timestamp counter. */
  timestamp(): number {
    return this.timestampValue;
  }

  /** Delegate to inner transform's likelihood. */
  likelihood(score: number): number {
    return this.transform.likelihood(score);
  }

  /** Delegate to inner transform's scoreToProbability. */
  scoreToProbability(score: number, tf: number, docLenRatio: number): number {
    return this.transform.scoreToProbability(score, tf, docLenRatio);
  }

  /** Delegate to inner transform's wandUpperBound. */
  wandUpperBound(bm25UpperBound: number, pMax = 0.9): number {
    return this.transform.wandUpperBound(bm25UpperBound, pMax);
  }

  /** Delegate to inner transform's averagedAlpha. */
  averagedAlpha(): number {
    return this.transform.averagedAlpha();
  }

  /** Delegate to inner transform's averagedBeta. */
  averagedBeta(): number {
    return this.transform.averagedBeta();
  }

  /**
   * Batch gradient descent with temporal sample weighting.
   *
   * When timestamps are provided, each sample is weighted by
   * exp(-decayRate * (maxTs - ts_i)), normalized so that weights sum to n.
   */
  fit(
    scores: number[],
    labels: number[],
    timestamps?: number[],
    learningRate = 0.01,
    maxIterations = 1000,
    tolerance = 1e-6,
    mode: TrainingMode = "balanced",
    tfs?: number[],
    docLenRatios?: number[],
  ): void {
    if (mode === TrainingMode.PriorAware) {
      if (tfs === undefined || docLenRatios === undefined) {
        throw new Error(
          "tfs and doc_len_ratios are required when mode is PriorAware",
        );
      }
    }

    let sampleWeights: number[] | null = null;
    if (timestamps !== undefined) {
      const ts = timestamps;
      let maxTs = 0;
      for (let i = 0; i < ts.length; i++) {
        const t = ts[i] as number;
        if (t > maxTs) {
          maxTs = t;
        }
      }
      const maxTsF = maxTs;
      const raw: number[] = [];
      for (let i = 0; i < ts.length; i++) {
        raw.push(Math.exp(-this.decayRate * (maxTsF - (ts[i] as number))));
      }
      let sum = 0.0;
      for (let i = 0; i < raw.length; i++) {
        sum += raw[i] as number;
      }
      const len = raw.length;
      sampleWeights = [];
      for (let i = 0; i < raw.length; i++) {
        sampleWeights.push(((raw[i] as number) * len) / sum);
      }
    }

    let priors: number[] | null = null;
    if (mode === TrainingMode.PriorAware) {
      const tfsRef = tfs as number[];
      const dlrs = docLenRatios as number[];
      const m = Math.min(tfsRef.length, dlrs.length);
      priors = [];
      for (let i = 0; i < m; i++) {
        priors.push(
          BayesianProbabilityTransform.compositePrior(
            tfsRef[i] as number,
            dlrs[i] as number,
          ),
        );
      }
    }

    let alpha = this.transform.alpha;
    let beta = this.transform.beta;
    const n = scores.length;

    for (let iter = 0; iter < maxIterations; iter++) {
      let gradAlpha: number;
      let gradBeta: number;
      if (sampleWeights !== null && priors !== null) {
        [gradAlpha, gradBeta] = computeWeightedPriorAwareGradients(
          scores,
          labels,
          priors,
          sampleWeights,
          alpha,
          beta,
          n,
        );
      } else if (sampleWeights !== null && priors === null) {
        [gradAlpha, gradBeta] = computeWeightedBalancedGradients(
          scores,
          labels,
          sampleWeights,
          alpha,
          beta,
          n,
        );
      } else if (sampleWeights === null && priors !== null) {
        [gradAlpha, gradBeta] = computePriorAwareGradients(
          scores,
          labels,
          priors,
          alpha,
          beta,
          n,
        );
      } else {
        [gradAlpha, gradBeta] = computeBalancedGradients(
          scores,
          labels,
          alpha,
          beta,
          n,
        );
      }

      const newAlpha = alpha - learningRate * gradAlpha;
      const newBeta = beta - learningRate * gradBeta;

      if (
        Math.abs(newAlpha - alpha) < tolerance &&
        Math.abs(newBeta - beta) < tolerance
      ) {
        alpha = newAlpha;
        beta = newBeta;
        break;
      }

      alpha = newAlpha;
      beta = newBeta;
    }

    // Replicate Rust: directly mutate inner state to mirror fit() resets.
    this.transform.resetAfterFit(mode, alpha, beta);
  }

  /**
   * Online SGD update with temporal decay applied to the averaging parameter.
   *
   * Increments the internal timestamp, computes an effective avgDecay that
   * ramps up with timestamp count, then delegates to the inner transform.
   */
  update(
    scores: number[],
    labels: number[],
    learningRate = 0.01,
    momentum = 0.9,
    decayTau = 1000,
    maxGradNorm = 1.0,
    avgDecay = 0.995,
    mode: TrainingMode | null = null,
    tfs?: number[],
    docLenRatios?: number[],
  ): void {
    this.timestampValue += 1;
    const effectiveAvgDecay =
      avgDecay * (1.0 - 1.0 / (1.0 + this.timestampValue));
    this.transform.update(
      scores,
      labels,
      learningRate,
      momentum,
      decayTau,
      maxGradNorm,
      effectiveAvgDecay,
      mode,
      tfs,
      docLenRatios,
    );
  }
}
