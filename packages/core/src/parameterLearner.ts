/**
 * Logistic parameter learner.
 *
 * Learns `alpha`/`beta` for the calibration `p = sigmoid(alpha * (s - beta))`
 * via gradient descent on cross-entropy. Accumulations use a deterministic
 * left-to-right order.
 */

import { sigmoid, EPSILON } from "./mathUtils.js";

export interface ParameterLearnerResult {
  alpha: number;
  beta: number;
  lossHistory: number[];
  converged: boolean;
}

/**
 * Clamp probabilities while mapping NaN to EPSILON. This keeps divergent
 * non-finite gradient descent trajectories deterministic.
 */
function clampProbability(p: number): number {
  if (Number.isNaN(p)) {
    return EPSILON;
  }
  return Math.min(Math.max(p, EPSILON), 1.0 - EPSILON);
}

export class ParameterLearner {
  private readonly learningRate: number;
  private readonly maxIterations: number;
  private readonly tolerance: number;

  constructor(learningRate = 0.01, maxIterations = 1000, tolerance = 1e-6) {
    this.learningRate = learningRate;
    this.maxIterations = maxIterations;
    this.tolerance = tolerance;
  }

  crossEntropyLoss(
    scores: number[],
    labels: number[],
    alpha: number,
    beta: number,
  ): number {
    // Pair scores and labels up to the shorter input length.
    const n = scores.length;
    let totalLoss = 0.0;
    const m = Math.min(scores.length, labels.length);
    for (let i = 0; i < m; i++) {
      const s = scores[i] as number;
      const y = labels[i] as number;
      let p = sigmoid(alpha * (s - beta));
      p = clampProbability(p);
      totalLoss -= y * Math.log(p) + (1.0 - y) * Math.log(1.0 - p);
    }
    return totalLoss / n;
  }

  learn(scores: number[], labels: number[]): ParameterLearnerResult {
    let alpha = 1.0;
    let beta = 0.0;
    const n = scores.length;
    const lossHistory: number[] = [];

    const m = Math.min(scores.length, labels.length);

    for (let iteration = 0; iteration < this.maxIterations; iteration++) {
      const loss = this.crossEntropyLoss(scores, labels, alpha, beta);
      lossHistory.push(loss);

      if (iteration > 0) {
        const prev = lossHistory[lossHistory.length - 2] as number;
        if (Math.abs(prev - loss) < this.tolerance) {
          return {
            alpha,
            beta,
            lossHistory,
            converged: true,
          };
        }
      }

      let gradAlpha = 0.0;
      let gradBeta = 0.0;
      for (let i = 0; i < m; i++) {
        const s = scores[i] as number;
        const y = labels[i] as number;
        let p = sigmoid(alpha * (s - beta));
        p = Math.min(Math.max(p, EPSILON), 1.0 - EPSILON);
        const error = p - y;
        gradAlpha += error * (s - beta);
        gradBeta += error * -alpha;
      }
      gradAlpha /= n;
      gradBeta /= n;

      alpha -= this.learningRate * gradAlpha;
      beta -= this.learningRate * gradBeta;
    }

    const finalLoss = this.crossEntropyLoss(scores, labels, alpha, beta);
    lossHistory.push(finalLoss);

    return {
      alpha,
      beta,
      lossHistory,
      converged: false,
    };
  }
}
