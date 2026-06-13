/**
 * Probabilistic fusion primitives.
 */
import { logit, minMaxNormalize, safeProb, sigmoid } from "./mathUtils.js";

/** Gating function for sparse-signal logits before aggregation. */
export type Gating =
  | { kind: "none" }
  | { kind: "relu" }
  | { kind: "swish" }
  | { kind: "generalizedSwish"; beta: number }
  | { kind: "gelu" };

export const Gating = {
  None: { kind: "none" } as Gating,
  Relu: { kind: "relu" } as Gating,
  Swish: { kind: "swish" } as Gating,
  generalizedSwish: (beta: number): Gating => ({ kind: "generalizedSwish", beta }),
  Gelu: { kind: "gelu" } as Gating,
};

function applyGating(logitVal: number, gating: Gating): number {
  switch (gating.kind) {
    case "none":
      return logitVal;
    case "relu":
      return Math.max(logitVal, 0.0);
    case "swish":
      return logitVal * sigmoid(logitVal);
    case "generalizedSwish":
      return logitVal * sigmoid(gating.beta * logitVal);
    case "gelu":
      return logitVal * sigmoid(1.702 * logitVal);
  }
}

/** Maps cosine similarity [-1, 1] to probability (0, 1). */
export function cosineToProbability(score: number): number {
  return safeProb((1.0 + score) / 2.0);
}

/** Probabilistic complement: P(not A) = 1 - P(A). */
export function probNot(prob: number): number {
  return safeProb(1.0 - safeProb(prob));
}

/** Probabilistic AND via product rule in log-space. */
export function probAnd(probs: number[]): number {
  let logSum = 0.0;
  for (const p of probs) {
    logSum += Math.log(safeProb(p));
  }
  return Math.exp(logSum);
}

/** Probabilistic OR via complement rule in log-space. */
export function probOr(probs: number[]): number {
  let logComplementSum = 0.0;
  for (const p of probs) {
    logComplementSum += Math.log(1.0 - safeProb(p));
  }
  return 1.0 - Math.exp(logComplementSum);
}

/**
 * Log-odds conjunction (paper Eq. 20/23).
 *
 * Unweighted (weights = null): sigmoid(mean(logit(p_i)) * n^alpha), default alpha = 0.5.
 * Weighted: sigmoid(n^alpha * sum(w_i * logit(p_i))), default alpha = 0.0,
 *   requires all w_i >= 0 and sum(w_i) = 1.
 */
export function logOddsConjunction(
  probs: number[],
  alpha: number | null = null,
  weights: number[] | null = null,
  gating: Gating = Gating.None,
): number {
  if (probs.length === 0) {
    return 0.5;
  }
  const n = probs.length;

  const gatedLogits = probs.map((p) => applyGating(logit(safeProb(p)), gating));

  if (weights === null) {
    const effectiveAlpha = alpha ?? 0.5;
    let sum = 0.0;
    for (const l of gatedLogits) {
      sum += l;
    }
    const lBar = sum / n;
    return sigmoid(lBar * Math.pow(n, effectiveAlpha));
  } else {
    if (weights.length !== probs.length) {
      throw new Error("weights length must match probs length");
    }
    if (!weights.every((wi) => wi >= 0.0)) {
      throw new Error("all weights must be non-negative");
    }
    let wsum = 0.0;
    for (const wi of weights) {
      wsum += wi;
    }
    if (Math.abs(wsum - 1.0) >= 1e-6) {
      throw new Error("weights must sum to 1.0");
    }
    const effectiveAlpha = alpha ?? 0.0;
    let weightedLogitSum = 0.0;
    for (let i = 0; i < gatedLogits.length; i++) {
      weightedLogitSum += weights[i]! * gatedLogits[i]!;
    }
    return sigmoid(Math.pow(n, effectiveAlpha) * weightedLogitSum);
  }
}

/**
 * Balanced log-odds fusion for hybrid sparse-dense retrieval.
 * Converts to logit-space, min-max normalizes each, then linearly blends.
 */
export function balancedLogOddsFusion(
  sparseProbs: number[],
  denseSimilarities: number[],
  weight: number,
): number[] {
  const n = sparseProbs.length;
  const logitSparse = sparseProbs.map((p) => logit(safeProb(p)));
  const logitDense = denseSimilarities.map((s) => logit(cosineToProbability(s)));

  const sparseNorm = minMaxNormalize(logitSparse);
  const denseNorm = minMaxNormalize(logitDense);

  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    out.push(weight * denseNorm[i]! + (1.0 - weight) * sparseNorm[i]!);
  }
  return out;
}
