/**
 * Block-max index for efficient top-k retrieval with early termination.
 * Direct port of `src/block_max_index.rs`.
 *
 * Stores precomputed block-level maximum scores for each term, enabling safe
 * pruning of document blocks that cannot contribute to the top-k results.
 *
 * JS `number` is IEEE-754 f64, identical to Rust `f64`. Every accumulation
 * follows the reference's left-to-right order so results match bit-for-bit.
 */

import type { BayesianProbabilityTransform } from "./probability.js";

export class BlockMaxIndex {
  private blockSizeValue: number;
  private blockMaxes: number[][] | null; // [n_terms][n_blocks]
  private nDocs: number;

  /** Create a new block-max index with the specified block size. */
  constructor(blockSize = 128) {
    if (!(blockSize >= 1)) {
      throw new Error(`block_size must be >= 1, got ${blockSize}`);
    }
    this.blockSizeValue = blockSize;
    this.blockMaxes = null;
    this.nDocs = 0;
  }

  /**
   * Build block-max structures from a score matrix.
   *
   * scoreMatrix[term][doc] gives the score of the document for that term.
   */
  build(scoreMatrix: number[][]): void {
    if (scoreMatrix.length === 0) {
      this.blockMaxes = [];
      this.nDocs = 0;
      return;
    }

    this.nDocs = (scoreMatrix[0] as number[]).length;
    const nBlocks = Math.ceil(this.nDocs / this.blockSizeValue);

    const allMaxes: number[][] = [];

    for (let term = 0; term < scoreMatrix.length; term++) {
      const termScores = scoreMatrix[term] as number[];
      if (termScores.length !== this.nDocs) {
        throw new Error("All term score vectors must have the same length");
      }

      const termBlockMaxes: number[] = [];

      for (let blockId = 0; blockId < nBlocks; blockId++) {
        const start = blockId * this.blockSizeValue;
        const end = Math.min(start + this.blockSizeValue, this.nDocs);
        let maxVal = Number.NEGATIVE_INFINITY;
        for (let i = start; i < end; i++) {
          // Rust folds with f64::max, which IGNORES a NaN operand; `>` is false
          // for NaN so maxVal is left unchanged, replicating that NaN-skip.
          const v = termScores[i] as number;
          if (v > maxVal) {
            maxVal = v;
          }
        }
        termBlockMaxes.push(maxVal);
      }

      allMaxes.push(termBlockMaxes);
    }

    this.blockMaxes = allMaxes;
  }

  /** Get the block-level upper bound score for a given term and block. */
  blockUpperBound(termIdx: number, blockId: number): number {
    if (this.blockMaxes === null) {
      throw new Error("BlockMaxIndex has not been built");
    }
    const maxes = this.blockMaxes;
    if (!(termIdx < maxes.length)) {
      throw new Error(
        `term_idx ${termIdx} out of range (n_terms=${maxes.length})`,
      );
    }
    const termMaxes = maxes[termIdx] as number[];
    if (!(blockId < termMaxes.length)) {
      throw new Error(
        `block_id ${blockId} out of range (n_blocks=${termMaxes.length})`,
      );
    }
    return termMaxes[blockId] as number;
  }

  /**
   * Compute a Bayesian upper bound probability for a term in a block.
   *
   * Uses the block-max score as the BM25 upper bound and the given pMax as the
   * prior upper bound, then delegates to the transform's wandUpperBound method.
   */
  bayesianBlockUpperBound(
    termIdx: number,
    blockId: number,
    transform: BayesianProbabilityTransform,
    pMax: number,
  ): number {
    const bm25Ub = this.blockUpperBound(termIdx, blockId);
    return transform.wandUpperBound(bm25Ub, pMax);
  }

  /** Block size used for this index. */
  blockSize(): number {
    return this.blockSizeValue;
  }

  /** Number of blocks (computed from nDocs and blockSize). */
  nBlocks(): number {
    if (this.nDocs === 0) {
      return 0;
    }
    return Math.ceil(this.nDocs / this.blockSizeValue);
  }
}
