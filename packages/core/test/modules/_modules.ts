/** Shared loader for the additional-module golden fixtures (Phase A). */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

export { expectClose } from "../_golden.js";

const here = dirname(fileURLToPath(import.meta.url));
const goldenPath = resolve(here, "../../../../fixtures/golden_modules.json");

export interface GoldenMetricCase {
  probs: number[];
  labels: number[];
  nBins: number;
  ece: number;
  brier: number;
  reliability: [number, number, number][];
}
export interface GoldenPlCase {
  name: string;
  scores: number[];
  labels: number[];
  lr: number;
  maxIter: number;
  tol: number;
  alpha: number;
  beta: number;
  converged: boolean;
  lossHistory: number[];
  crossEntropy: { alpha: number; beta: number; loss: number }[];
}
export interface GoldenModules {
  metrics: GoldenMetricCase[];
  parameterLearner: GoldenPlCase[];
  experiments: { params: { k1: number; b: number }; results: { name: string; passed: boolean }[] };
}

export const goldenModules: GoldenModules = JSON.parse(
  readFileSync(goldenPath, "utf8"),
) as GoldenModules;

// --- Phase A2 fixtures (probability transforms + calibrators) ---
const golden2Path = resolve(here, "../../../../fixtures/golden_modules2.json");

export interface GoldenModules2 {
  probabilityTransform: {
    likelihood: [number, number][];
    tfPrior: [number, number][];
    normPrior: [number, number][];
    compositePrior: { tf: number; dlr: number; v: number }[];
    posterior: { likelihood: number; prior: number; baseRate: number | null; v: number }[];
    scoreToProbability: { score: number; tf: number; dlr: number; v: number }[];
    scoreToProbabilityBaseRate: { score: number; tf: number; dlr: number; v: number }[];
    wandUpperBound: { ub: number; pMax: number; v: number }[];
    fit: {
      scores: number[];
      labels: number[];
      tfs: number[];
      dlrs: number[];
      balanced: { alpha: number; beta: number };
      priorFree: { alpha: number; beta: number };
      priorAware: { alpha: number; beta: number };
    };
    update: {
      scores: number[];
      labels: number[];
      alpha: number;
      beta: number;
      averagedAlpha: number;
      averagedBeta: number;
    };
  };
  temporalTransform: {
    scores: number[];
    labels: number[];
    timestamps: number[];
    decayHalfLife: number;
    fit: { alpha: number; beta: number };
    update: {
      timestamp: number;
      alpha: number;
      beta: number;
      averagedAlpha: number;
      averagedBeta: number;
    };
  };
  platt: {
    scores: number[];
    labels: number[];
    lr: number;
    maxIter: number;
    tol: number;
    a: number;
    b: number;
    calibrated: number[];
  };
  isotonic: {
    scores: number[];
    labels: number[];
    probe: number[];
    calibrated: number[];
  };
}

export const goldenModules2: GoldenModules2 = JSON.parse(
  readFileSync(golden2Path, "utf8"),
) as GoldenModules2;

// --- Phase A3 fixtures (learnable/attention/multi-head weights + block-max) ---
const golden3Path = resolve(here, "../../../../fixtures/golden_modules3.json");

export interface GoldenModules3 {
  learnable: {
    initWeights: number[];
    initAveraged: number[];
    combinePlain: number;
    combineBaseRate: number;
    fit: { probs: number[][]; labels: number[]; weights: number[]; combine: number };
    update: { weights: number[]; averaged: number[] };
  };
  attention: {
    init: { nSignals: number; nQueryFeatures: number; seed: number; weightsMatrix: number[] }[];
    combineBroadcast: number[];
    combineFull: number[];
    combineSingle: number[];
    combineNormalize: number[];
    combineBaseRate: number[];
    fit: { weightsMatrix: number[]; combine: number[] };
    computeUpperBounds: number[];
    prune: { surviving: number[]; fused: number[] };
  };
  multiHead: {
    nHeads: number;
    combineSingle: number[];
    combineMulti: number[];
    fitCombine: number[];
  };
  blockMaxIndex: {
    blockSize: number;
    matrix: number[][];
    nBlocks: number;
    blockUpperBound: number[][];
    bayesianBlockUpperBound00: number;
  };
}

export const goldenModules3: GoldenModules3 = JSON.parse(
  readFileSync(golden3Path, "utf8"),
) as GoldenModules3;
