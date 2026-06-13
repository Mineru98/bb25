/**
 * Learnable per-signal reliability weights for log-odds conjunction.
 * Direct port of `src/learnable_weights.rs`.
 *
 * Learns weights that map from the Naive Bayes uniform initialization
 * (w_i = 1/n) to per-signal reliability weights via softmax parameterization.
 *
 * The gradient dL/dz_j = n^alpha * (p - y) * w_j * (x_j - x_bar_w)
 * is Hebbian: the product of pre-synaptic activity (signal deviation
 * from weighted mean) and post-synaptic error (prediction minus label).
 */
import { logit, safeProb, sigmoid, softmax } from "./mathUtils.js";
import { Gating, logOddsConjunction } from "./fusion.js";

export class LearnableLogOddsWeights {
  private readonly nSignalsValue: number;
  private readonly alphaValue: number;
  private logits: number[];
  private nUpdates: number;
  private gradLogitsEma: number[];
  private weightsAvg: number[];
  private readonly baseRateValue: number | null;
  private readonly logitBaseRate: number | null;

  constructor(nSignals: number, alpha = 0.0, baseRate: number | null = null) {
    if (!(nSignals >= 1)) {
      throw new Error(`n_signals must be >= 1, got ${nSignals}`);
    }
    if (baseRate != null) {
      if (!(baseRate > 0.0 && baseRate < 1.0)) {
        throw new Error(`base_rate must be in (0, 1), got ${baseRate}`);
      }
    }
    this.logitBaseRate = baseRate != null ? logit(safeProb(baseRate)) : null;
    const uniform = 1.0 / nSignals;
    this.nSignalsValue = nSignals;
    this.alphaValue = alpha;
    this.logits = new Array<number>(nSignals).fill(0.0);
    this.nUpdates = 0;
    this.gradLogitsEma = new Array<number>(nSignals).fill(0.0);
    this.weightsAvg = new Array<number>(nSignals).fill(uniform);
    this.baseRateValue = baseRate;
  }

  baseRate(): number | null {
    return this.baseRateValue;
  }

  nSignals(): number {
    return this.nSignalsValue;
  }

  alpha(): number {
    return this.alphaValue;
  }

  /** Current weights: softmax of internal logits. */
  weights(): number[] {
    return softmax(this.logits);
  }

  /** Polyak-averaged weights for stable inference. */
  averagedWeights(): number[] {
    return this.weightsAvg.slice();
  }

  /** Combine probability signals via weighted log-odds conjunction. */
  combine(probs: number[], useAveraged = false): number {
    if (this.logitBaseRate != null) {
      const w = useAveraged ? this.weightsAvg.slice() : this.weights();
      const n = probs.length;
      const scale = Math.pow(n, this.alphaValue);
      let lWeighted = 0.0;
      for (let i = 0; i < w.length; i++) {
        lWeighted += w[i]! * logit(safeProb(probs[i]!));
      }
      return sigmoid(scale * lWeighted + this.logitBaseRate);
    } else {
      const w = useAveraged ? this.weightsAvg.slice() : this.weights();
      return logOddsConjunction(probs, this.alphaValue, w, Gating.None);
    }
  }

  /** Batch gradient descent on BCE loss to learn weights. */
  fit(
    probs: number[][],
    labels: number[],
    learningRate = 0.01,
    maxIterations = 1000,
    tolerance = 1e-6,
  ): void {
    const m = probs.length;
    const n = this.nSignalsValue;
    const scale = Math.pow(n, this.alphaValue);

    // Precompute log-odds of input signals
    const x: number[][] = [];
    for (let i = 0; i < m; i++) {
      const row = probs[i]!;
      if (row.length !== n) {
        throw new Error(`probs row length ${row.length} != n_signals ${n}`);
      }
      const xRow = new Array<number>(n);
      for (let j = 0; j < n; j++) {
        xRow[j] = logit(safeProb(row[j]!));
      }
      x.push(xRow);
    }

    for (let iter = 0; iter < maxIterations; iter++) {
      const w = softmax(this.logits);

      const gradLogits = new Array<number>(n).fill(0.0);

      for (let i = 0; i < m; i++) {
        const xi = x[i]!;
        // Weighted mean log-odds
        let xBarW = 0.0;
        for (let j = 0; j < n; j++) {
          xBarW += w[j]! * xi[j]!;
        }

        // Predicted probability (add logit_base_rate if present)
        const lWeighted = scale * xBarW + (this.logitBaseRate ?? 0.0);
        const p = sigmoid(lWeighted);
        const error = p - labels[i]!;

        // Gradient for each logit z_j
        for (let j = 0; j < n; j++) {
          gradLogits[j]! += scale * error * w[j]! * (xi[j]! - xBarW);
        }
      }

      // Average over samples
      let maxChange = 0.0;
      for (let j = 0; j < n; j++) {
        gradLogits[j]! /= m;
        const delta = learningRate * gradLogits[j]!;
        this.logits[j]! -= delta;
        maxChange = Math.max(maxChange, Math.abs(delta));
      }

      if (maxChange < tolerance) {
        break;
      }
    }

    // Reset online state
    this.nUpdates = 0;
    this.gradLogitsEma = new Array<number>(n).fill(0.0);
    this.weightsAvg = softmax(this.logits);
  }

  /** Online SGD update from a single observation or mini-batch. */
  update(
    probs: number[][],
    labels: number[],
    learningRate = 0.01,
    momentum = 0.9,
    decayTau = 1000,
    maxGradNorm = 1.0,
    avgDecay = 0.995,
  ): void {
    const m = probs.length;
    const n = this.nSignalsValue;
    const scale = Math.pow(n, this.alphaValue);
    const w = softmax(this.logits);

    const gradLogits = new Array<number>(n).fill(0.0);

    for (let i = 0; i < m; i++) {
      const row = probs[i]!;
      if (row.length !== n) {
        throw new Error(`probs row length ${row.length} != n_signals ${n}`);
      }
      const x = new Array<number>(n);
      for (let j = 0; j < n; j++) {
        x[j] = logit(safeProb(row[j]!));
      }
      let xBarW = 0.0;
      for (let j = 0; j < n; j++) {
        xBarW += w[j]! * x[j]!;
      }
      const lWeighted = scale * xBarW + (this.logitBaseRate ?? 0.0);
      const p = sigmoid(lWeighted);
      const error = p - labels[i]!;

      for (let j = 0; j < n; j++) {
        gradLogits[j]! += scale * error * w[j]! * (x[j]! - xBarW);
      }
    }

    // Average over mini-batch
    for (let j = 0; j < n; j++) {
      gradLogits[j]! /= m;
    }

    // EMA smoothing
    for (let j = 0; j < n; j++) {
      this.gradLogitsEma[j]! =
        momentum * this.gradLogitsEma[j]! + (1.0 - momentum) * gradLogits[j]!;
    }

    // Bias correction
    this.nUpdates += 1;
    const correction = 1.0 - Math.pow(momentum, this.nUpdates);
    const corrected = new Array<number>(n);
    for (let j = 0; j < n; j++) {
      corrected[j] = this.gradLogitsEma[j]! / correction;
    }

    // L2 gradient clipping
    let gradNormSq = 0.0;
    for (let j = 0; j < n; j++) {
      gradNormSq += corrected[j]! * corrected[j]!;
    }
    const gradNorm = Math.sqrt(gradNormSq);
    if (gradNorm > maxGradNorm) {
      const clipScale = maxGradNorm / gradNorm;
      for (let j = 0; j < n; j++) {
        corrected[j]! *= clipScale;
      }
    }

    // Learning rate decay
    const effectiveLr = learningRate / (1.0 + this.nUpdates / decayTau);

    for (let j = 0; j < n; j++) {
      this.logits[j]! -= effectiveLr * corrected[j]!;
    }

    // Polyak averaging of weights in the simplex
    const rawWeights = softmax(this.logits);
    for (let j = 0; j < n; j++) {
      this.weightsAvg[j]! =
        avgDecay * this.weightsAvg[j]! + (1.0 - avgDecay) * rawWeights[j]!;
    }
  }
}
