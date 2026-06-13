import { describe, expect, it } from "vitest";
import {
  BM25Scorer,
  BayesianProbabilityTransform,
  Corpus,
  cosineToProbability,
  logOddsConjunction,
  logit,
  probAnd,
  probNot,
  probOr,
  safeProb,
  sigmoid,
} from "../src/index.js";

function makeRng(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (1664525 * state + 1013904223) >>> 0;
    return state / 0x100000000;
  };
}

function posteriorByLogOdds(likelihood: number, prior: number, baseRate: number | null): number {
  let z = logit(likelihood) + logit(prior);
  if (baseRate !== null) {
    z += logit(baseRate);
  }
  return sigmoid(z);
}

describe("Tier 0 seeded probability properties", () => {
  it("sigmoid/logit are finite, monotonic, symmetric, and round-trip away from clamps", () => {
    const xs = [-40, -20, -8, -2, 0, 2, 8, 20, 40];
    for (const x of xs) {
      const y = sigmoid(x);
      expect(Number.isFinite(y)).toBe(true);
      expect(y).toBeGreaterThanOrEqual(0);
      expect(y).toBeLessThanOrEqual(1);
      expect(sigmoid(-x)).toBeCloseTo(1 - y, 12);
    }

    for (let i = 1; i < xs.length; i++) {
      expect(sigmoid(xs[i]!)).toBeGreaterThan(sigmoid(xs[i - 1]!));
    }

    const rng = makeRng(42);
    for (let i = 0; i < 200; i++) {
      const p = 1e-6 + rng() * (1 - 2e-6);
      expect(sigmoid(logit(p))).toBeCloseTo(p, 10);
    }
  });

  it("posterior equals log-odds addition with and without base rate", () => {
    const rng = makeRng(4242);
    for (let i = 0; i < 200; i++) {
      const likelihood = 1e-5 + rng() * (1 - 2e-5);
      const prior = 1e-5 + rng() * (1 - 2e-5);
      for (const baseRate of [null, 0.05, 0.2, 0.8] as const) {
        const got = BayesianProbabilityTransform.posterior(likelihood, prior, baseRate);
        const expected = safeProb(posteriorByLogOdds(likelihood, prior, baseRate));
        expect(got).toBeCloseTo(expected, 10);
      }
    }
  });

  it("scoreToProbability is monotonic when priors are fixed", () => {
    const transform = new BayesianProbabilityTransform(1.3, 0.2, null);
    let prev = 0.0;
    for (let i = 0; i <= 100; i++) {
      const score = -4 + i * 0.1;
      const got = transform.scoreToProbability(score, 4, 1.0);
      expect(got).toBeGreaterThanOrEqual(prev - 1e-12);
      prev = got;
    }
  });

  it("fusion primitives preserve probability identities on seeded samples", () => {
    const rng = makeRng(7);
    for (let i = 0; i < 100; i++) {
      const probs = [0.01 + rng() * 0.98, 0.01 + rng() * 0.98, 0.01 + rng() * 0.98];
      const minProb = Math.min(...probs);
      const maxProb = Math.max(...probs);

      expect(probAnd(probs)).toBeLessThanOrEqual(minProb + 1e-12);
      expect(probOr(probs)).toBeGreaterThanOrEqual(maxProb - 1e-12);
      expect(probNot(probNot(probs[0]!))).toBeCloseTo(probs[0]!, 10);
      expect(probOr(probs)).toBeCloseTo(probNot(probAnd(probs.map(probNot))), 10);

      const uniform = probs.map(() => 1 / probs.length);
      expect(logOddsConjunction(probs, 0.0, uniform)).toBeCloseTo(
        logOddsConjunction(probs, 0.0, null),
        12,
      );
    }

    expect(cosineToProbability(-1)).toBeCloseTo(1e-10, 12);
    expect(cosineToProbability(0)).toBeCloseTo(0.5, 12);
    expect(cosineToProbability(1)).toBeCloseTo(1 - 1e-10, 12);
    expect(() => logOddsConjunction([0.3, 0.7], null, [0.6, -0.1])).toThrow(
      /non-negative/,
    );
    expect(() => logOddsConjunction([0.3, 0.7], null, [0.6, 0.6])).toThrow(
      /sum to 1/,
    );
  });

  it("Lucene BM25 WAND upper bound is never below observed term probabilities", () => {
    const corpus = new Corpus();
    corpus.addDocumentTokens("d1", "doc one", ["alpha", "alpha", "beta"]);
    corpus.addDocumentTokens("d2", "doc two", ["alpha", "gamma", "gamma", "gamma"]);
    corpus.addDocumentTokens("d3", "doc three", ["delta", "epsilon"]);
    corpus.buildIndex();

    const bm25 = new BM25Scorer(corpus, 1.2, 0.75, "lucene");
    const transform = new BayesianProbabilityTransform(1.0, 0.5, null);

    for (const term of ["alpha", "beta", "gamma", "delta"]) {
      const upper = transform.wandUpperBound(bm25.upperBound(term), 0.9);
      for (const doc of corpus.documents()) {
        const tf = doc.termFreq.get(term) ?? 0;
        if (tf === 0) continue;
        const raw = bm25.scoreTermStandard(term, doc);
        const actual = transform.scoreToProbability(raw, tf, doc.length / corpus.avgdl);
        expect(upper).toBeGreaterThanOrEqual(actual - 1e-12);
      }
    }
  });
});
