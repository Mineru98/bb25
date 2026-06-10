import { describe, it } from "vitest";
import { goldenModules3, expectClose } from "./_modules.js";
import { BlockMaxIndex } from "../../src/blockMaxIndex.js";
import { BayesianProbabilityTransform } from "../../src/probability.js";

describe("BlockMaxIndex golden parity", () => {
  const golden = goldenModules3.blockMaxIndex;

  it("matches block upper bounds and n_blocks", () => {
    const index = new BlockMaxIndex(3);
    index.build(golden.matrix);

    expectClose(index.nBlocks(), golden.nBlocks, "n_blocks");

    for (let term = 0; term <= 1; term++) {
      for (let block = 0; block < golden.nBlocks; block++) {
        expectClose(
          index.blockUpperBound(term, block),
          (golden.blockUpperBound[term] as number[])[block] as number,
          `block_upper_bound[${term}][${block}]`,
        );
      }
    }
  });

  it("matches bayesian block upper bound", () => {
    const index = new BlockMaxIndex(3);
    index.build(golden.matrix);

    const transform = new BayesianProbabilityTransform(1.0, 0.5, null);
    expectClose(
      index.bayesianBlockUpperBound(0, 0, transform, 0.9),
      golden.bayesianBlockUpperBound00,
      "bayesian_block_upper_bound(0,0)",
    );
  });
});
