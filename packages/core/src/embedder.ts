/**
 * Embedder interface. The core defines only the contract; concrete
 * implementations (e.g. BGE-M3 via transformers.js) live in @bb25/embeddings
 * so that @bb25/core stays dependency-free.
 */
import type { Vector } from "./mathUtils.js";

export interface Embedder {
  /** Output dimensionality (e.g. 1024 for BGE-M3). */
  readonly dim: number;
  /** Embed a batch of texts. */
  embed(texts: string[]): Promise<Vector[]>;
}
