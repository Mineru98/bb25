/**
 * Bayesian BM25 scorer.
 *
 * `alpha`/`beta` parameterize the sigmoid likelihood, not a Beta prior.
 * Defaults are (1.0, 0.5). `posterior()` intentionally leaves clamping to
 * callers that need it.
 */
import type { BM25Scorer } from "./bm25.js";
import type { Document } from "./corpus.js";
import { probOr } from "./fusion.js";
import { clamp, safeProb, sigmoid } from "./mathUtils.js";

export class BayesianBM25Scorer {
  private readonly bm25: BM25Scorer;
  private readonly alpha: number;
  private readonly beta: number;
  private readonly baseRateVal: number | null;

  constructor(bm25: BM25Scorer, alpha = 1.0, beta = 0.5, baseRate: number | null = null) {
    if (baseRate !== null) {
      if (!(baseRate > 0.0 && baseRate < 1.0)) {
        throw new Error(`base_rate must be in (0, 1), got ${baseRate}`);
      }
    }
    this.bm25 = bm25;
    this.alpha = alpha;
    this.beta = beta;
    this.baseRateVal = baseRate;
  }

  baseRate(): number | null {
    return this.baseRateVal;
  }

  likelihood(score: number): number {
    return sigmoid(this.alpha * (score - this.beta));
  }

  tfPrior(tf: number): number {
    return 0.2 + 0.7 * Math.min(tf / 10.0, 1.0);
  }

  /** Document-length normalization prior (Eq. 26). Bell curve centered at ratio=0.5. */
  normPrior(docLength: number, avgDocLength: number): number {
    if (avgDocLength < 1.0) {
      return 0.5;
    }
    const ratio = docLength / avgDocLength;
    return 0.3 + 0.6 * (1.0 - Math.min(Math.abs(ratio - 0.5) * 2.0, 1.0));
  }

  compositePrior(tf: number, docLength: number, avgDocLength: number): number {
    const pTf = this.tfPrior(tf);
    const pNorm = this.normPrior(docLength, avgDocLength);
    return clamp(0.7 * pTf + 0.3 * pNorm, 0.1, 0.9);
  }

  /** Two-step Bayesian posterior update (Remark 4.4.5). */
  posterior(score: number, prior: number): number {
    const lik = safeProb(this.likelihood(score));
    const p = safeProb(prior);

    // Step 1: standard Bayes update
    const numerator = lik * p;
    const denominator = numerator + (1.0 - lik) * (1.0 - p);
    const p1 = numerator / denominator;

    // Step 2: base rate adjustment
    if (this.baseRateVal !== null) {
      const br = this.baseRateVal;
      const num2 = p1 * br;
      const den2 = num2 + (1.0 - p1) * (1.0 - br);
      return num2 / den2;
    }
    return p1;
  }

  scoreTerm(term: string, doc: Document): number {
    const rawScore = this.bm25.scoreTermStandard(term, doc);
    if (rawScore === 0.0) {
      return 0.0;
    }
    const tf = doc.termFreq.get(term) ?? 0;
    const prior = this.compositePrior(tf, doc.length, this.bm25.avgdl());
    return this.posterior(rawScore, prior);
  }

  score(queryTerms: string[], doc: Document): number {
    const posteriors: number[] = [];
    for (const term of queryTerms) {
      const p = this.scoreTerm(term, doc);
      if (p > 0.0) {
        posteriors.push(p);
      }
    }

    if (posteriors.length === 0) {
      return 0.0;
    }

    return probOr(posteriors);
  }
}
