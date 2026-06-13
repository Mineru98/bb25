/**
 * Vector scorer.
 *
 * Dimension-agnostic: no dim assertion. `cosineSimilarity` zips to the shorter
 * vector, so query/doc embeddings of mismatched length truncate rather than
 * error.
 */
import type { Document } from "./corpus.js";
import { clamp, cosineSimilarity, type Vector } from "./mathUtils.js";

export class VectorScorer {
  scoreToProbability(sim: number): number {
    return clamp((1.0 + sim) / 2.0, 0.0, 1.0);
  }

  score(queryEmbedding: Vector, doc: Document): number {
    const sim = cosineSimilarity(queryEmbedding, doc.embedding);
    return this.scoreToProbability(sim);
  }
}
