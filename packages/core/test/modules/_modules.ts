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
