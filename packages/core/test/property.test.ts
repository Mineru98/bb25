import { describe, expect, it } from "vitest";
import {
  BM25Scorer,
  BayesianProbabilityTransform,
  BlockMaxIndex,
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

  it("composite priors stay bounded, symmetric, and monotonic in term frequency", () => {
    const rng = makeRng(99);

    for (let i = 0; i < 200; i++) {
      const tf = rng() * 30.0;
      const docLenRatio = rng() * 2.0;
      const pTf = BayesianProbabilityTransform.tfPrior(tf);
      const pNorm = BayesianProbabilityTransform.normPrior(docLenRatio);
      const prior = BayesianProbabilityTransform.compositePrior(tf, docLenRatio);

      expect(pTf).toBeGreaterThanOrEqual(0.2 - 1e-12);
      expect(pTf).toBeLessThanOrEqual(0.9 + 1e-12);
      expect(pNorm).toBeGreaterThanOrEqual(0.3 - 1e-12);
      expect(pNorm).toBeLessThanOrEqual(0.9 + 1e-12);
      expect(prior).toBeGreaterThanOrEqual(0.1 - 1e-12);
      expect(prior).toBeLessThanOrEqual(0.9 + 1e-12);
    }

    for (let i = 0; i <= 20; i++) {
      const delta = i / 40;
      expect(BayesianProbabilityTransform.normPrior(0.5 - delta)).toBeCloseTo(
        BayesianProbabilityTransform.normPrior(0.5 + delta),
        12,
      );
    }

    for (let i = 0; i < 50; i++) {
      const docLenRatio = rng() * 2.0;
      let prev = BayesianProbabilityTransform.compositePrior(0, docLenRatio);
      for (let tf = 1; tf <= 30; tf++) {
        const next = BayesianProbabilityTransform.compositePrior(tf, docLenRatio);
        expect(next).toBeGreaterThanOrEqual(prev - 1e-12);
        prev = next;
      }
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

  it("block-max Bayesian upper bounds are never below seeded document probabilities", () => {
    const rng = makeRng(314159);
    const nTerms = 5;
    const nDocs = 31;
    const blockSize = 7;
    const scoreMatrix: number[][] = [];
    const tfs: number[][] = [];
    const docLenRatios = Array.from({ length: nDocs }, () => 0.1 + rng() * 1.9);

    for (let term = 0; term < nTerms; term++) {
      const scores: number[] = [];
      const termTfs: number[] = [];
      for (let doc = 0; doc < nDocs; doc++) {
        scores.push(rng() * 6.0);
        termTfs.push(1 + Math.floor(rng() * 25));
      }
      scoreMatrix.push(scores);
      tfs.push(termTfs);
    }

    const index = new BlockMaxIndex(blockSize);
    index.build(scoreMatrix);
    const transform = new BayesianProbabilityTransform(1.1, 0.2, 0.3);
    const pMax = 0.9;

    for (let term = 0; term < nTerms; term++) {
      for (let block = 0; block < index.nBlocks(); block++) {
        const scoreBound = index.blockUpperBound(term, block);
        const probabilityBound = index.bayesianBlockUpperBound(term, block, transform, pMax);
        const start = block * blockSize;
        const end = Math.min(start + blockSize, nDocs);

        for (let doc = start; doc < end; doc++) {
          const score = scoreMatrix[term]![doc]!;
          expect(scoreBound).toBeGreaterThanOrEqual(score - 1e-12);

          const prior = BayesianProbabilityTransform.compositePrior(
            tfs[term]![doc]!,
            docLenRatios[doc]!,
          );
          const actual = BayesianProbabilityTransform.posterior(
            transform.likelihood(score),
            prior,
            transform.baseRate,
          );
          expect(probabilityBound).toBeGreaterThanOrEqual(actual - 1e-12);
        }
      }
    }
  });
});
