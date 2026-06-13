/**
 * Loads committed golden fixtures and provides a numeric-parity helper.
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";
import { expect } from "vitest";

const here = dirname(fileURLToPath(import.meta.url));
const goldenPath = resolve(here, "../../../fixtures/golden.json");

export interface GoldenTermScore {
  term: string;
  bm25Term: number;
  bayesianTerm: number;
}
export interface GoldenDocScore {
  id: string;
  bm25: number;
  bayesian: number;
  vector: number;
  hybridOr: number;
  hybridAnd: number;
  terms: GoldenTermScore[];
}
export interface GoldenDoc {
  id: string;
  text: string;
  embedding: number[];
  tokens: string[];
  length: number;
  termFreq: Record<string, number>;
}
export interface Golden {
  params: { k1: number; b: number; alpha: number; beta: number; hybridAlpha: number; epsilon: number };
  corpus: { n: number; avgdl: number; df: Record<string, number> };
  documents: GoldenDoc[];
  idf: Record<string, number>;
  queries: { text: string; terms: string[]; embedding: number[] | null; relevant: string[] }[];
  scores: { query: string; perDoc: GoldenDocScore[] }[];
  tokenizer: { input: string; tokens: string[] }[];
  math: {
    sigmoid: [number, number][];
    logit: [number, number][];
    cosineToProbability: [number, number][];
    fusion: {
      probs: number[];
      probOr: number;
      probAnd: number;
      logOddsConjDefault: number;
      logOddsConjAlpha05: number;
    }[];
  };
}

export const golden: Golden = JSON.parse(readFileSync(goldenPath, "utf8")) as Golden;

/** Tolerance well inside the design budget (1e-9 .. 1e-6); transcendental fns
 * (exp/ln/pow) are not bit-identical across language runtimes, so we allow a
 * tiny absolute + relative slack. */
export function expectClose(actual: number, expected: number, label = ""): void {
  const tol = 1e-12 + 1e-9 * Math.abs(expected);
  const diff = Math.abs(actual - expected);
  expect(
    diff <= tol,
    `${label}: |${actual} - ${expected}| = ${diff} > ${tol}`,
  ).toBe(true);
}
