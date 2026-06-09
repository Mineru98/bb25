/**
 * HybridScorer. Direct port of `src/hybrid_scorer.rs`.
 *
 * `scoreAnd` uses log-odds conjunction (NOT the product rule), per §14 Q3:
 *   probabilisticAnd(probs) = logOddsConjunction(probs, alpha=0.5, NoGating).
 * `scoreAnd` short-circuits to 0 when both signals are below EPSILON.
 */
import type { BayesianBM25Scorer } from "./bayesian.js";
import type { Document } from "./corpus.js";
import { Gating, logOddsConjunction, probOr } from "./fusion.js";
import { EPSILON, type Vector } from "./mathUtils.js";
import type { VectorScorer } from "./vector.js";

export class HybridScorer {
  private readonly bayesian: BayesianBM25Scorer;
  private readonly vector: VectorScorer;
  private readonly alpha: number;

  constructor(bayesian: BayesianBM25Scorer, vector: VectorScorer, alpha = 0.5) {
    this.bayesian = bayesian;
    this.vector = vector;
    this.alpha = alpha;
  }

  probabilisticAnd(probs: number[]): number {
    return logOddsConjunction(probs, this.alpha, null, Gating.None);
  }

  probabilisticOr(probs: number[]): number {
    return probOr(probs);
  }

  scoreAnd(queryTerms: string[], queryEmbedding: Vector, doc: Document): number {
    const bayesianProb = this.bayesian.score(queryTerms, doc);
    const vectorProb = this.vector.score(queryEmbedding, doc);
    if (bayesianProb < EPSILON && vectorProb < EPSILON) {
      return 0.0;
    }
    return this.probabilisticAnd([bayesianProb, vectorProb]);
  }

  scoreOr(queryTerms: string[], queryEmbedding: Vector, doc: Document): number {
    const bayesianProb = this.bayesian.score(queryTerms, doc);
    const vectorProb = this.vector.score(queryEmbedding, doc);
    return this.probabilisticOr([bayesianProb, vectorProb]);
  }

  naiveSum(scores: number[]): number {
    let sum = 0.0;
    for (const s of scores) {
      sum += s;
    }
    return sum;
  }

  rrfScore(ranks: number[], k = 60): number {
    let sum = 0.0;
    for (const rank of ranks) {
      sum += 1.0 / (k + rank);
    }
    return sum;
  }
}
