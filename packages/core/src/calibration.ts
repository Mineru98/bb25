/**
 * Score calibration helpers.
 *
 * Uses deterministic left-to-right accumulation order with explicit clamps,
 * defaults, and edge-case handling.
 */

import { safeProb, sigmoid } from "./mathUtils.js";

/**
 * Platt scaling calibrator: P(y=1|s) = sigmoid(a*s + b).
 *
 * Learns parameters a and b via gradient descent on binary cross-entropy loss.
 */
export class PlattCalibrator {
  public a: number;
  public b: number;

  constructor(a = 1.0, b = 0.0) {
    this.a = a;
    this.b = b;
  }

  /** Fit Platt scaling parameters from scores and binary labels. */
  fit(
    scores: number[],
    labels: number[],
    learningRate = 0.01,
    maxIterations = 1000,
    tolerance = 1e-6,
  ): void {
    const n = scores.length;
    let a = this.a;
    let b = this.b;

    for (let iter = 0; iter < maxIterations; iter++) {
      let gradA = 0.0;
      let gradB = 0.0;

      // Pair scores and labels up to the shorter input length.
      const m = Math.min(scores.length, labels.length);
      for (let i = 0; i < m; i++) {
        const s = scores[i] as number;
        const y = labels[i] as number;
        const p = safeProb(sigmoid(a * s + b));
        const error = p - y;
        gradA += error * s;
        gradB += error;
      }

      gradA /= n;
      gradB /= n;

      const newA = a - learningRate * gradA;
      const newB = b - learningRate * gradB;

      if (
        Math.abs(newA - a) < tolerance &&
        Math.abs(newB - b) < tolerance
      ) {
        a = newA;
        b = newB;
        break;
      }

      a = newA;
      b = newB;
    }

    this.a = a;
    this.b = b;
  }

  /** Calibrate a single score. */
  calibrate(score: number): number {
    return sigmoid(this.a * score + this.b);
  }

  /** Calibrate a batch of scores. */
  calibrateBatch(scores: number[]): number[] {
    return scores.map((s) => this.calibrate(s));
  }
}

/**
 * Isotonic regression calibrator using the Pool Adjacent Violators Algorithm (PAVA).
 *
 * Fits a non-decreasing step function from scores to probabilities,
 * then uses binary search with linear interpolation for prediction.
 */
export class IsotonicCalibrator {
  private xBreakpoints: number[] | null;
  private yBreakpoints: number[] | null;

  constructor() {
    this.xBreakpoints = null;
    this.yBreakpoints = null;
  }

  /**
   * Fit isotonic regression using the PAVA algorithm.
   *
   * Sorts (score, label) pairs by score, then merges adjacent blocks
   * that violate the non-decreasing constraint.
   */
  fit(scores: number[], labels: number[]): void {
    if (scores.length !== labels.length) {
      throw new Error("scores and labels must have the same length");
    }
    if (scores.length === 0) {
      this.xBreakpoints = [];
      this.yBreakpoints = [];
      return;
    }

    // Sort by score (stable; partial_cmp -> Equal on NaN).
    const indices: number[] = [];
    for (let i = 0; i < scores.length; i++) {
      indices.push(i);
    }
    // Stable sort: compare by score, treating NaN comparisons as Equal.
    indices.sort((ia, ib) => {
      const sa = scores[ia] as number;
      const sb = scores[ib] as number;
      if (sa < sb) {
        return -1;
      } else if (sa > sb) {
        return 1;
      } else {
        // Equal, or NaN involved -> Ordering::Equal.
        return 0;
      }
    });

    const sortedX: number[] = indices.map((i) => scores[i] as number);
    const sortedY: number[] = indices.map((i) => labels[i] as number);

    // PAVA: maintain blocks of (sum_y, count, x_start, x_end).
    let blockSum: number[] = sortedY.slice();
    let blockCount: number[] = sortedY.map(() => 1.0);
    let blockXStart: number[] = sortedX.slice();
    let blockXEnd: number[] = sortedX.slice();
    let nBlocks = sortedY.length;

    // Pool adjacent violators.
    let changed = true;
    while (changed) {
      changed = false;
      let i = 0;
      const newSum: number[] = [];
      const newCount: number[] = [];
      const newXStart: number[] = [];
      const newXEnd: number[] = [];

      while (i < nBlocks) {
        let s = blockSum[i] as number;
        let c = blockCount[i] as number;
        const xs = blockXStart[i] as number;
        let xe = blockXEnd[i] as number;

        // Merge forward while violating non-decreasing constraint.
        while (
          i + 1 < nBlocks &&
          s / c > (blockSum[i + 1] as number) / (blockCount[i + 1] as number)
        ) {
          i += 1;
          s += blockSum[i] as number;
          c += blockCount[i] as number;
          xe = blockXEnd[i] as number;
          changed = true;
        }

        newSum.push(s);
        newCount.push(c);
        newXStart.push(xs);
        newXEnd.push(xe);
        i += 1;
      }

      blockSum = newSum;
      blockCount = newCount;
      blockXStart = newXStart;
      blockXEnd = newXEnd;
      nBlocks = blockSum.length;
    }

    // Build breakpoints: midpoint of each block's x range as the representative x.
    const xBp: number[] = [];
    const yBp: number[] = [];
    for (let i = 0; i < nBlocks; i++) {
      xBp.push(((blockXStart[i] as number) + (blockXEnd[i] as number)) / 2.0);
      yBp.push((blockSum[i] as number) / (blockCount[i] as number));
    }

    this.xBreakpoints = xBp;
    this.yBreakpoints = yBp;
  }

  /** Calibrate a single score using binary search and linear interpolation. */
  calibrate(score: number): number {
    if (this.xBreakpoints === null) {
      throw new Error("IsotonicCalibrator has not been fitted");
    }
    const xBp = this.xBreakpoints;
    const yBp = this.yBreakpoints as number[];

    if (xBp.length === 0) {
      return 0.5;
    }

    if (xBp.length === 1) {
      return yBp[0] as number;
    }

    // Clamp to boundary values.
    if (score <= (xBp[0] as number)) {
      return yBp[0] as number;
    }
    if (score >= (xBp[xBp.length - 1] as number)) {
      return yBp[yBp.length - 1] as number;
    }

    // Binary search for the interval.
    let lo = 0;
    let hi = xBp.length - 1;
    while (lo + 1 < hi) {
      const mid = Math.floor((lo + hi) / 2);
      if ((xBp[mid] as number) <= score) {
        lo = mid;
      } else {
        hi = mid;
      }
    }

    // Linear interpolation between breakpoints.
    const x0 = xBp[lo] as number;
    const x1 = xBp[hi] as number;
    const y0 = yBp[lo] as number;
    const y1 = yBp[hi] as number;

    const range = x1 - x0;
    if (Math.abs(range) < 1e-15) {
      return (y0 + y1) / 2.0;
    }

    const t = (score - x0) / range;
    return y0 + t * (y1 - y0);
  }

  /** Calibrate a batch of scores. */
  calibrateBatch(scores: number[]): number[] {
    return scores.map((s) => this.calibrate(s));
  }
}
