/**
 * Query-dependent signal weighting via attention (Paper 2, Section 8).
 *
 * Direct port of `src/attention_weights.rs`. Computes per-signal softmax
 * attention weights from query features, then combines probability signals
 * via weighted log-odds conjunction.
 *
 * JS `number` is IEEE-754 f64, identical to Rust `f64`. Every accumulation
 * follows the reference's left-to-right order so results match bit-for-bit.
 */
import { Gating, logOddsConjunction } from "./fusion.js";
import { logit, minMaxNormalize, safeProb, sigmoid, softmaxRows } from "./mathUtils.js";

const MASK64 = (1n << 64n) - 1n;

/** Port of simple_normal_init (LCG + Box-Muller). seed is a non-negative integer. */
function simpleNormalInit(n: number, scale: number, seed: number): number[] {
  let state = (BigInt(seed) + 1n) & MASK64;
  const result: number[] = [];
  const pairs = Math.floor((n + 1) / 2);
  for (let i = 0; i < pairs; i++) {
    state = (state * 6364136223846793005n + 1442695040888963407n) & MASK64;
    let u1 = Number(state >> 11n) / 2 ** 53;
    u1 = Math.max(u1, 1e-15);
    state = (state * 6364136223846793005n + 1442695040888963407n) & MASK64;
    const u2 = Number(state >> 11n) / 2 ** 53;
    const r = Math.sqrt(-2.0 * Math.log(u1));
    const theta = 2.0 * Math.PI * u2;
    result.push(r * Math.cos(theta) * scale);
    result.push(r * Math.sin(theta) * scale);
  }
  return result.slice(0, n);
}

export class AttentionLogOddsWeights {
  private readonly nSignalsVal: number;
  private readonly nQueryFeaturesVal: number;
  private readonly alpha: number;
  private readonly normalize: boolean;
  // W: (nSignals, nQueryFeatures) stored row-major
  private wMatrix: number[];
  // b: (nSignals,)
  private bias: number[];
  // Online learning state
  private nUpdates: number;
  private gradWEma: number[];
  private gradBEma: number[];
  // Polyak averaging
  private wAvg: number[];
  private bAvg: number[];
  // Base rate
  private readonly logitBaseRate: number | null;

  /** Create new attention weights with Xavier initialization. */
  constructor(
    nSignals: number,
    nQueryFeatures: number,
    alpha: number,
    normalize: boolean,
    seed: number,
    baseRate: number | null,
  ) {
    if (!(nSignals >= 1)) {
      throw new Error(`n_signals must be >= 1, got ${nSignals}`);
    }
    if (!(nQueryFeatures >= 1)) {
      throw new Error(`n_query_features must be >= 1, got ${nQueryFeatures}`);
    }
    if (baseRate !== null) {
      if (!(baseRate > 0.0 && baseRate < 1.0)) {
        throw new Error(`base_rate must be in (0, 1), got ${baseRate}`);
      }
    }

    const logitBr =
      baseRate !== null ? logit(safeProb(baseRate)) : null;

    // Xavier-style initialization using a simple PRNG
    const scale = 1.0 / Math.sqrt(nQueryFeatures);
    const total = nSignals * nQueryFeatures;
    const wMatrix = simpleNormalInit(total, scale, seed);

    this.nSignalsVal = nSignals;
    this.nQueryFeaturesVal = nQueryFeatures;
    this.alpha = alpha;
    this.normalize = normalize;
    this.wMatrix = wMatrix.slice();
    this.bias = new Array<number>(nSignals).fill(0.0);
    this.nUpdates = 0;
    this.gradWEma = new Array<number>(total).fill(0.0);
    this.gradBEma = new Array<number>(nSignals).fill(0.0);
    this.wAvg = wMatrix.slice();
    this.bAvg = new Array<number>(nSignals).fill(0.0);
    this.logitBaseRate = logitBr;
  }

  /** Number of probability signals (Rust `n_signals()`). */
  nSignals(): number {
    return this.nSignalsVal;
  }

  /** Number of query features (Rust `n_query_features()`). */
  nQueryFeatures(): number {
    return this.nQueryFeaturesVal;
  }

  /** Weight matrix W of shape (nSignals, nQueryFeatures). */
  weightsMatrix(): number[] {
    return this.wMatrix.slice();
  }

  /**
   * Compute softmax attention weights from query features.
   *
   * queryFeatures: flat array of shape (m * nQueryFeatures)
   * Returns flat array of shape (m * nSignals)
   */
  private computeWeights(
    queryFeatures: number[],
    m: number,
    useAveraged: boolean,
  ): number[] {
    const n = this.nSignalsVal;
    const nqf = this.nQueryFeaturesVal;
    const w = useAveraged ? this.wAvg : this.wMatrix;
    const b = useAveraged ? this.bAvg : this.bias;

    // z = queryFeatures @ W^T + b
    // queryFeatures: (m, nqf), W: (n, nqf), result: (m, n)
    const z = new Array<number>(m * n).fill(0.0);
    for (let row = 0; row < m; row++) {
      for (let col = 0; col < n; col++) {
        let val = b[col]!;
        for (let k = 0; k < nqf; k++) {
          val += queryFeatures[row * nqf + k]! * w[col * nqf + k]!;
        }
        z[row * n + col] = val;
      }
    }

    return softmaxRows(z, n);
  }

  /** Per-column min-max normalization on logit array (m rows, nSignals cols). */
  private normalizeLogitsColumns(x: number[], m: number): void {
    const n = this.nSignalsVal;
    for (let col = 0; col < n; col++) {
      const column: number[] = [];
      for (let row = 0; row < m; row++) {
        column.push(x[row * n + col]!);
      }
      const normalized = minMaxNormalize(column);
      for (let row = 0; row < m; row++) {
        x[row * n + col] = normalized[row]!;
      }
    }
  }

  /**
   * Combine probability signals via query-dependent weighted log-odds.
   *
   * probs: flat array of shape (m * nSignals) for m candidates
   * queryFeatures: flat array of shape (mQ * nQueryFeatures)
   * If mQ < m, the last query feature row is broadcast.
   */
  combine(
    probs: number[],
    m: number,
    queryFeatures: number[],
    mQ: number,
    useAveraged = false,
  ): number[] {
    const n = this.nSignalsVal;
    const weights = this.computeWeights(queryFeatures, mQ, useAveraged);

    const lbr = this.logitBaseRate ?? 0.0;

    if (m === 1 && !this.normalize) {
      if (this.logitBaseRate !== null) {
        const scale = Math.pow(n, this.alpha);
        const wFlat: number[] = [];
        for (let j = 0; j < n; j++) {
          wFlat.push(weights[j]!);
        }
        let lWeighted = 0.0;
        for (let j = 0; j < n; j++) {
          lWeighted += wFlat[j]! * logit(safeProb(probs[j]!));
        }
        return [sigmoid(scale * lWeighted + lbr)];
      }
      const wFlat: number[] = [];
      const rowProbs: number[] = [];
      for (let j = 0; j < n; j++) {
        wFlat.push(weights[j]!);
      }
      for (let j = 0; j < n; j++) {
        rowProbs.push(probs[j]!);
      }
      return [logOddsConjunction(rowProbs, this.alpha, wFlat, Gating.None)];
    }

    if (this.normalize) {
      const scale = Math.pow(n, this.alpha);
      const x: number[] = probs.map((p) => logit(safeProb(p)));
      this.normalizeLogitsColumns(x, m);

      const results = new Array<number>(m).fill(0.0);
      for (let i = 0; i < m; i++) {
        const wiRow = Math.min(i, mQ - 1);
        let lWeighted = 0.0;
        for (let j = 0; j < n; j++) {
          lWeighted += weights[wiRow * n + j]! * x[i * n + j]!;
        }
        results[i] = sigmoid(scale * lWeighted + lbr);
      }
      return results;
    }

    // Batched: each row has its own query-dependent weights
    const results = new Array<number>(m).fill(0.0);
    for (let i = 0; i < m; i++) {
      const wiRow = Math.min(i, mQ - 1);
      if (this.logitBaseRate !== null) {
        const scale = Math.pow(n, this.alpha);
        let lWeighted = 0.0;
        for (let j = 0; j < n; j++) {
          lWeighted += weights[wiRow * n + j]! * logit(safeProb(probs[i * n + j]!));
        }
        results[i] = sigmoid(scale * lWeighted + lbr);
      } else {
        const wSlice: number[] = [];
        const rowProbs: number[] = [];
        for (let j = 0; j < n; j++) {
          wSlice.push(weights[wiRow * n + j]!);
        }
        for (let j = 0; j < n; j++) {
          rowProbs.push(probs[i * n + j]!);
        }
        results[i] = logOddsConjunction(rowProbs, this.alpha, wSlice, Gating.None);
      }
    }
    return results;
  }

  /** Batch gradient descent on BCE loss to learn W and b. */
  fit(
    probs: number[],
    labels: number[],
    queryFeatures: number[],
    m: number,
    queryIds: number[] | null,
    learningRate: number,
    maxIterations: number,
    tolerance: number,
  ): void {
    const n = this.nSignalsVal;
    const nqf = this.nQueryFeaturesVal;
    const scale = Math.pow(n, this.alpha);

    // Compute logits of input signals
    const x: number[] = probs.map((p) => logit(safeProb(p)));

    if (this.normalize) {
      if (queryIds !== null) {
        // Per-query group normalization
        const uniqueIds: number[] = queryIds.slice();
        uniqueIds.sort((a, b) => a - b);
        const deduped: number[] = [];
        for (let i = 0; i < uniqueIds.length; i++) {
          if (i === 0 || uniqueIds[i] !== uniqueIds[i - 1]) {
            deduped.push(uniqueIds[i]!);
          }
        }
        for (const qid of deduped) {
          const indices: number[] = [];
          for (let i = 0; i < m; i++) {
            if (queryIds[i] === qid) {
              indices.push(i);
            }
          }
          for (let col = 0; col < n; col++) {
            const column: number[] = [];
            for (const i of indices) {
              column.push(x[i * n + col]!);
            }
            const normalized = minMaxNormalize(column);
            for (let idx = 0; idx < indices.length; idx++) {
              const i = indices[idx]!;
              x[i * n + col] = normalized[idx]!;
            }
          }
        }
      } else {
        this.normalizeLogitsColumns(x, m);
      }
    }

    for (let iter = 0; iter < maxIterations; iter++) {
      // Compute per-sample attention weights: z = qf @ W^T + b
      const z = new Array<number>(m * n).fill(0.0);
      for (let row = 0; row < m; row++) {
        for (let col = 0; col < n; col++) {
          let val = this.bias[col]!;
          for (let k = 0; k < nqf; k++) {
            val += queryFeatures[row * nqf + k]! * this.wMatrix[col * nqf + k]!;
          }
          z[row * n + col] = val;
        }
      }
      const w = softmaxRows(z, n);

      // Compute predictions and gradients
      const gradW = new Array<number>(n * nqf).fill(0.0);
      const gradB = new Array<number>(n).fill(0.0);

      const lbr = this.logitBaseRate ?? 0.0;
      for (let i = 0; i < m; i++) {
        let xBarW = 0.0;
        for (let j = 0; j < n; j++) {
          xBarW += w[i * n + j]! * x[i * n + j]!;
        }
        const p = sigmoid(scale * xBarW + lbr);
        const error = p - labels[i]!;

        // gradZ_j = scale * error * w_j * (x_j - xBarW)
        for (let j = 0; j < n; j++) {
          const gz = scale * error * w[i * n + j]! * (x[i * n + j]! - xBarW);
          // dL/dW_jk = gz * qf_k
          for (let k = 0; k < nqf; k++) {
            gradW[j * nqf + k] = gradW[j * nqf + k]! + gz * queryFeatures[i * nqf + k]!;
          }
          gradB[j] = gradB[j]! + gz;
        }
      }

      // Average over samples
      const mF = m;
      const oldW = this.wMatrix.slice();
      const oldB = this.bias.slice();

      for (let idx = 0; idx < gradW.length; idx++) {
        gradW[idx] = gradW[idx]! / mF;
        this.wMatrix[idx] = this.wMatrix[idx]! - learningRate * gradW[idx]!;
      }
      for (let j = 0; j < n; j++) {
        gradB[j] = gradB[j]! / mF;
        this.bias[j] = this.bias[j]! - learningRate * gradB[j]!;
      }

      // Check convergence
      let maxChangeW = 0.0;
      for (let idx = 0; idx < oldW.length; idx++) {
        maxChangeW = Math.max(maxChangeW, Math.abs(oldW[idx]! - this.wMatrix[idx]!));
      }
      let maxChangeB = 0.0;
      for (let j = 0; j < oldB.length; j++) {
        maxChangeB = Math.max(maxChangeB, Math.abs(oldB[j]! - this.bias[j]!));
      }

      if (Math.max(maxChangeW, maxChangeB) < tolerance) {
        break;
      }
    }

    // Reset online state
    this.nUpdates = 0;
    this.gradWEma = new Array<number>(n * nqf).fill(0.0);
    this.gradBEma = new Array<number>(n).fill(0.0);
    this.wAvg = this.wMatrix.slice();
    this.bAvg = this.bias.slice();
  }

  /** Online SGD update from a single observation or mini-batch. */
  update(
    probs: number[],
    labels: number[],
    queryFeatures: number[],
    m: number,
    learningRate: number,
    momentum: number,
    decayTau: number,
    maxGradNorm: number,
    avgDecay: number,
  ): void {
    const n = this.nSignalsVal;
    const nqf = this.nQueryFeaturesVal;
    const scale = Math.pow(n, this.alpha);

    const x: number[] = probs.map((p) => logit(safeProb(p)));

    if (this.normalize && m > 1) {
      this.normalizeLogitsColumns(x, m);
    }

    // Compute attention weights
    const z = new Array<number>(m * n).fill(0.0);
    for (let row = 0; row < m; row++) {
      for (let col = 0; col < n; col++) {
        let val = this.bias[col]!;
        for (let k = 0; k < nqf; k++) {
          val += queryFeatures[row * nqf + k]! * this.wMatrix[col * nqf + k]!;
        }
        z[row * n + col] = val;
      }
    }
    const w = softmaxRows(z, n);

    // Compute gradients
    const gradW = new Array<number>(n * nqf).fill(0.0);
    const gradB = new Array<number>(n).fill(0.0);

    const lbr = this.logitBaseRate ?? 0.0;
    for (let i = 0; i < m; i++) {
      let xBarW = 0.0;
      for (let j = 0; j < n; j++) {
        xBarW += w[i * n + j]! * x[i * n + j]!;
      }
      const p = sigmoid(scale * xBarW + lbr);
      const error = p - labels[i]!;

      for (let j = 0; j < n; j++) {
        const gz = scale * error * w[i * n + j]! * (x[i * n + j]! - xBarW);
        for (let k = 0; k < nqf; k++) {
          gradW[j * nqf + k] = gradW[j * nqf + k]! + gz * queryFeatures[i * nqf + k]!;
        }
        gradB[j] = gradB[j]! + gz;
      }
    }

    const mF = m;
    for (let idx = 0; idx < gradW.length; idx++) {
      gradW[idx] = gradW[idx]! / mF;
    }
    for (let j = 0; j < n; j++) {
      gradB[j] = gradB[j]! / mF;
    }

    // EMA smoothing
    for (let idx = 0; idx < gradW.length; idx++) {
      this.gradWEma[idx] =
        momentum * this.gradWEma[idx]! + (1.0 - momentum) * gradW[idx]!;
    }
    for (let j = 0; j < n; j++) {
      this.gradBEma[j] =
        momentum * this.gradBEma[j]! + (1.0 - momentum) * gradB[j]!;
    }

    // Bias correction
    this.nUpdates += 1;
    const correction = 1.0 - Math.pow(momentum, this.nUpdates);
    const correctedW: number[] = this.gradWEma.map((g) => g / correction);
    const correctedB: number[] = this.gradBEma.map((g) => g / correction);

    // L2 gradient clipping (joint norm)
    let normSum = 0.0;
    for (const g of correctedW) {
      normSum += g * g;
    }
    for (const g of correctedB) {
      normSum += g * g;
    }
    const gradNorm = Math.sqrt(normSum);
    if (gradNorm > maxGradNorm) {
      const clipScale = maxGradNorm / gradNorm;
      for (let idx = 0; idx < correctedW.length; idx++) {
        correctedW[idx] = correctedW[idx]! * clipScale;
      }
      for (let idx = 0; idx < correctedB.length; idx++) {
        correctedB[idx] = correctedB[idx]! * clipScale;
      }
    }

    // Learning rate decay
    const effectiveLr = learningRate / (1.0 + this.nUpdates / decayTau);

    for (let idx = 0; idx < this.wMatrix.length; idx++) {
      this.wMatrix[idx] = this.wMatrix[idx]! - effectiveLr * correctedW[idx]!;
    }
    for (let j = 0; j < n; j++) {
      this.bias[j] = this.bias[j]! - effectiveLr * correctedB[j]!;
    }

    // Polyak averaging
    for (let idx = 0; idx < this.wMatrix.length; idx++) {
      this.wAvg[idx] =
        avgDecay * this.wAvg[idx]! + (1.0 - avgDecay) * this.wMatrix[idx]!;
    }
    for (let j = 0; j < n; j++) {
      this.bAvg[j] = avgDecay * this.bAvg[j]! + (1.0 - avgDecay) * this.bias[j]!;
    }
  }

  /**
   * Compute upper bounds on fused probabilities using per-signal upper bound probs.
   *
   * upperBoundProbs: flat array of shape (m * nSignals)
   * Returns an array of length m with upper bound fused probabilities.
   */
  computeUpperBounds(
    upperBoundProbs: number[],
    m: number,
    queryFeatures: number[],
    mQ: number,
    useAveraged = false,
  ): number[] {
    const n = this.nSignalsVal;
    const scale = Math.pow(n, this.alpha);
    const weights = this.computeWeights(queryFeatures, mQ, useAveraged);
    const lbr = this.logitBaseRate ?? 0.0;

    const results = new Array<number>(m).fill(0.0);
    for (let i = 0; i < m; i++) {
      const wiRow = Math.min(i, mQ - 1);
      let lWeighted = 0.0;
      for (let j = 0; j < n; j++) {
        lWeighted += weights[wiRow * n + j]! * logit(safeProb(upperBoundProbs[i * n + j]!));
      }
      results[i] = sigmoid(scale * lWeighted + lbr);
    }
    return results;
  }

  /**
   * Prune candidates whose upper bound fused probability is below threshold.
   *
   * Returns { surviving, fused } for candidates that survive pruning.
   */
  prune(
    probs: number[],
    m: number,
    queryFeatures: number[],
    mQ: number,
    threshold: number,
    upperBoundProbs: number[] | null,
    useAveraged = false,
  ): { surviving: number[]; fused: number[] } {
    const n = this.nSignalsVal;

    // Compute upper bounds to determine surviving candidates
    const ubProbs = upperBoundProbs ?? probs;
    const upperBounds = this.computeUpperBounds(ubProbs, m, queryFeatures, mQ, useAveraged);

    // Filter by threshold
    const surviving: number[] = [];
    for (let i = 0; i < m; i++) {
      if (upperBounds[i]! >= threshold) {
        surviving.push(i);
      }
    }

    // Compute actual fused probabilities for survivors
    const survivorM = surviving.length;
    if (survivorM === 0) {
      return { surviving: [], fused: [] };
    }

    // Gather survivor probs into a flat array
    const survivorProbs = new Array<number>(survivorM * n).fill(0.0);
    for (let si = 0; si < surviving.length; si++) {
      const origI = surviving[si]!;
      for (let j = 0; j < n; j++) {
        survivorProbs[si * n + j] = probs[origI * n + j]!;
      }
    }

    // Gather survivor query features
    const nqf = this.nQueryFeaturesVal;
    const survivorQf = new Array<number>(survivorM * nqf).fill(0.0);
    for (let si = 0; si < surviving.length; si++) {
      const origI = surviving[si]!;
      const qi = Math.min(origI, mQ - 1);
      for (let k = 0; k < nqf; k++) {
        survivorQf[si * nqf + k] = queryFeatures[qi * nqf + k]!;
      }
    }

    const fused = this.combine(survivorProbs, survivorM, survivorQf, survivorM, useAveraged);
    return { surviving, fused };
  }
}
