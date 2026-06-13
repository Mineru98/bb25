import { describe, it } from "vitest";
import { goldenModules3, expectClose } from "./_modules.js";
import { MultiHeadAttentionLogOddsWeights } from "../../src/multiHeadAttention.js";

describe("multi-head attention golden parity", () => {
  it("combine single candidate", () => {
    const g = goldenModules3.multiHead;
    const mh = new MultiHeadAttentionLogOddsWeights(4, 2, 1, 0.5, false);
    const out = mh.combine([0.8, 0.9], 1, [1.0], 1);
    for (let i = 0; i < g.combineSingle.length; i++) {
      expectClose(out[i]!, g.combineSingle[i]!, `combineSingle[${i}]`);
    }
  });

  it("combine multiple candidates", () => {
    const g = goldenModules3.multiHead;
    const mh = new MultiHeadAttentionLogOddsWeights(4, 2, 1, 0.5, false);
    const out = mh.combine([0.9, 0.2, 0.3, 0.8, 0.6, 0.6], 3, [1, 2, 3], 3);
    for (let i = 0; i < g.combineMulti.length; i++) {
      expectClose(out[i]!, g.combineMulti[i]!, `combineMulti[${i}]`);
    }
  });

  it("fit then combine", () => {
    const g = goldenModules3.multiHead;
    const mh = new MultiHeadAttentionLogOddsWeights(4, 2, 1, 0.5, false);
    mh.fit(
      [0.9, 0.2, 0.8, 0.3, 0.2, 0.9, 0.3, 0.85, 0.6, 0.6, 0.1, 0.1],
      [1, 1, 0, 0, 1, 0],
      [1, 2, 1, 3, 2, 1],
      6,
      null,
      0.1,
      200,
      1e-9,
    );
    const out = mh.combine([0.9, 0.2, 0.3, 0.8, 0.6, 0.6], 3, [1, 2, 3], 3);
    for (let i = 0; i < g.fitCombine.length; i++) {
      expectClose(out[i]!, g.fitCombine[i]!, `fitCombine[${i}]`);
    }
  });
});
