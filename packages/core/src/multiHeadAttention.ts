/**
 * Multi-head attention over log-odds signals. Direct port of
 * `src/multi_head_attention.rs`.
 *
 * Each head independently computes query-dependent attention weights over the
 * same probability signals. The final fused result averages the per-head
 * outputs in log-odds space before applying sigmoid.
 */
import { logit, safeProb, sigmoid } from "./mathUtils.js";
import { AttentionLogOddsWeights } from "./attentionWeights.js";

export class MultiHeadAttentionLogOddsWeights {
  private readonly nHeadsVal: number;
  private readonly headsVal: AttentionLogOddsWeights[];

  /**
   * Create a new multi-head attention module.
   *
   * Each head is initialized with a different seed (0..nHeads-1).
   */
  constructor(
    nHeads: number,
    nSignals: number,
    nQueryFeatures: number,
    alpha = 0.5,
    normalize = false,
  ) {
    if (!(nHeads >= 1)) {
      throw new Error(`n_heads must be >= 1, got ${nHeads}`);
    }
    this.nHeadsVal = nHeads;
    const heads: AttentionLogOddsWeights[] = [];
    for (let h = 0; h < nHeads; h++) {
      heads.push(
        new AttentionLogOddsWeights(nSignals, nQueryFeatures, alpha, normalize, h, null),
      );
    }
    this.headsVal = heads;
  }

  nHeads(): number {
    return this.nHeadsVal;
  }

  heads(): AttentionLogOddsWeights[] {
    return this.headsVal;
  }

  /**
   * Combine probability signals by averaging per-head outputs in log-odds space.
   *
   * probs: flat array of shape (m * nSignals)
   * queryFeatures: flat array of shape (mQ * nQueryFeatures)
   * Returns array of length m.
   */
  combine(
    probs: number[],
    m: number,
    queryFeatures: number[],
    mQ: number,
    useAveraged = false,
  ): number[] {
    const headResults: number[][] = [];
    for (let h = 0; h < this.headsVal.length; h++) {
      headResults.push(this.headsVal[h]!.combine(probs, m, queryFeatures, mQ, useAveraged));
    }

    const results = new Array<number>(m).fill(0.0);
    const nH = this.nHeadsVal;
    for (let i = 0; i < m; i++) {
      let sum = 0.0;
      for (let h = 0; h < headResults.length; h++) {
        sum += logit(safeProb(headResults[h]![i]!));
      }
      const avgLogit = sum / nH;
      results[i] = sigmoid(avgLogit);
    }
    return results;
  }

  /** Batch gradient descent to train all heads. */
  fit(
    probs: number[],
    labels: number[],
    queryFeatures: number[],
    m: number,
    queryIds: number[] | null,
    lr = 0.01,
    maxIter = 1000,
    tol = 1e-6,
  ): void {
    for (let h = 0; h < this.headsVal.length; h++) {
      this.headsVal[h]!.fit(probs, labels, queryFeatures, m, queryIds, lr, maxIter, tol);
    }
  }

  /** Online SGD update for all heads. */
  update(
    probs: number[],
    labels: number[],
    queryFeatures: number[],
    m: number,
    lr = 0.01,
    momentum = 0.9,
    decayTau = 1000,
    maxGradNorm = 1.0,
    avgDecay = 0.995,
  ): void {
    for (let h = 0; h < this.headsVal.length; h++) {
      this.headsVal[h]!.update(
        probs,
        labels,
        queryFeatures,
        m,
        lr,
        momentum,
        decayTau,
        maxGradNorm,
        avgDecay,
      );
    }
  }

  /** Compute upper bounds by averaging per-head upper bound log-odds. */
  computeUpperBounds(
    upperBoundProbs: number[],
    m: number,
    queryFeatures: number[],
    mQ: number,
    useAveraged = false,
  ): number[] {
    const headUbs: number[][] = [];
    for (let h = 0; h < this.headsVal.length; h++) {
      headUbs.push(
        this.headsVal[h]!.computeUpperBounds(upperBoundProbs, m, queryFeatures, mQ, useAveraged),
      );
    }

    const results = new Array<number>(m).fill(0.0);
    const nH = this.nHeadsVal;
    for (let i = 0; i < m; i++) {
      let sum = 0.0;
      for (let h = 0; h < headUbs.length; h++) {
        sum += logit(safeProb(headUbs[h]![i]!));
      }
      const avgLogit = sum / nH;
      results[i] = sigmoid(avgLogit);
    }
    return results;
  }

  /**
   * Prune candidates using multi-head upper bounds.
   *
   * Returns { surviving, fused }.
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
    if (this.headsVal.length === 0) {
      return { surviving: [], fused: [] };
    }
    const n = this.headsVal[0]!.nSignals();
    const nqf = this.headsVal[0]!.nQueryFeatures();

    const ubProbs = upperBoundProbs ?? probs;
    const upperBounds = this.computeUpperBounds(ubProbs, m, queryFeatures, mQ, useAveraged);

    const surviving: number[] = [];
    for (let i = 0; i < m; i++) {
      if (upperBounds[i]! >= threshold) {
        surviving.push(i);
      }
    }

    if (surviving.length === 0) {
      return { surviving: [], fused: [] };
    }

    const survivorM = surviving.length;
    const survivorProbs = new Array<number>(survivorM * n).fill(0.0);
    for (let si = 0; si < surviving.length; si++) {
      const origI = surviving[si]!;
      for (let j = 0; j < n; j++) {
        survivorProbs[si * n + j] = probs[origI * n + j]!;
      }
    }

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
