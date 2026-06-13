import { describe, it } from "vitest";
import { goldenModules3, expectClose } from "./_modules.js";
import { LearnableLogOddsWeights } from "../../src/learnableWeights.js";

const golden = goldenModules3.learnable;

describe("LearnableLogOddsWeights", () => {
  it("init weights and averaged weights", () => {
    const m = new LearnableLogOddsWeights(3, 0, null);
    const w = m.weights();
    const a = m.averagedWeights();
    for (let i = 0; i < golden.initWeights.length; i++) {
      expectClose(w[i]!, golden.initWeights[i]!, `initWeights[${i}]`);
      expectClose(a[i]!, golden.initAveraged[i]!, `initAveraged[${i}]`);
    }
  });

  it("combine plain (no base rate)", () => {
    const m = new LearnableLogOddsWeights(2, 0, null);
    expectClose(m.combine([0.7, 0.8]), golden.combinePlain, "combinePlain");
  });

  it("combine with base rate", () => {
    const m = new LearnableLogOddsWeights(2, 0, 0.3);
    expectClose(m.combine([0.7, 0.8]), golden.combineBaseRate, "combineBaseRate");
  });

  it("fit", () => {
    const m = new LearnableLogOddsWeights(2, 0, null);
    m.fit(golden.fit.probs, golden.fit.labels, 0.1, 300, 1e-9);
    const w = m.weights();
    for (let i = 0; i < golden.fit.weights.length; i++) {
      expectClose(w[i]!, golden.fit.weights[i]!, `fit.weights[${i}]`);
    }
    expectClose(m.combine([0.7, 0.8]), golden.fit.combine, "fit.combine");
  });

  it("update (per-row sequence)", () => {
    const m = new LearnableLogOddsWeights(2, 0.5, null);
    const probs = golden.fit.probs;
    const labels = golden.fit.labels;
    for (let i = 0; i < probs.length; i++) {
      m.update([probs[i]!], [labels[i]!], 0.05, 0.9, 1000, 1, 0.99);
    }
    const w = m.weights();
    const a = m.averagedWeights();
    for (let i = 0; i < golden.update.weights.length; i++) {
      expectClose(w[i]!, golden.update.weights[i]!, `update.weights[${i}]`);
      expectClose(a[i]!, golden.update.averaged[i]!, `update.averaged[${i}]`);
    }
  });
});
