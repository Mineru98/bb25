import { describe, expect, it } from "vitest";
import { AttentionLogOddsWeights } from "../../src/attentionWeights.js";
import { expectClose, goldenModules3 } from "./_modules.js";

const g = goldenModules3.attention;

describe("AttentionLogOddsWeights", () => {
  it("PRNG gate: Xavier init weights match golden element-wise", () => {
    for (const cfg of g.init) {
      const attn = new AttentionLogOddsWeights(
        cfg.nSignals,
        cfg.nQueryFeatures,
        0.5,
        false,
        cfg.seed,
        null,
      );
      const wm = attn.weightsMatrix();
      expect(wm.length).toBe(cfg.weightsMatrix.length);
      for (let i = 0; i < wm.length; i++) {
        expectClose(wm[i]!, cfg.weightsMatrix[i]!, `init seed=${cfg.seed} [${i}]`);
      }
    }
  });

  const cProbs = [0.9, 0.2, 0.3, 0.8, 0.6, 0.6];

  it("combine broadcast (mQ=1)", () => {
    const attn = new AttentionLogOddsWeights(2, 1, 0.5, false, 0, null);
    const out = attn.combine(cProbs, 3, [2.0], 1);
    expect(out.length).toBe(g.combineBroadcast.length);
    for (let i = 0; i < out.length; i++) {
      expectClose(out[i]!, g.combineBroadcast[i]!, `combineBroadcast[${i}]`);
    }
  });

  it("combine full (mQ=3)", () => {
    const attn = new AttentionLogOddsWeights(2, 1, 0.5, false, 0, null);
    const out = attn.combine(cProbs, 3, [1, 2, 3], 3);
    for (let i = 0; i < out.length; i++) {
      expectClose(out[i]!, g.combineFull[i]!, `combineFull[${i}]`);
    }
  });

  it("combine single (m=1)", () => {
    const attn = new AttentionLogOddsWeights(2, 1, 0.5, false, 0, null);
    const out = attn.combine([0.8, 0.9], 1, [1.5], 1);
    for (let i = 0; i < out.length; i++) {
      expectClose(out[i]!, g.combineSingle[i]!, `combineSingle[${i}]`);
    }
  });

  it("combine normalize", () => {
    const attn = new AttentionLogOddsWeights(2, 1, 0.5, true, 0, null);
    const out = attn.combine(cProbs, 3, [1, 2, 3], 3);
    for (let i = 0; i < out.length; i++) {
      expectClose(out[i]!, g.combineNormalize[i]!, `combineNormalize[${i}]`);
    }
  });

  it("combine base rate", () => {
    const attn = new AttentionLogOddsWeights(2, 1, 0.5, false, 0, 0.3);
    const out = attn.combine(cProbs, 3, [1, 2, 3], 3);
    for (let i = 0; i < out.length; i++) {
      expectClose(out[i]!, g.combineBaseRate[i]!, `combineBaseRate[${i}]`);
    }
  });

  it("fit then combine matches golden", () => {
    const attn = new AttentionLogOddsWeights(2, 1, 0.5, false, 0, null);
    attn.fit(
      [0.9, 0.2, 0.8, 0.3, 0.2, 0.9, 0.3, 0.85, 0.6, 0.6, 0.1, 0.1],
      [1, 1, 0, 0, 1, 0],
      [1, 2, 1, 3, 2, 1],
      6,
      null,
      0.1,
      200,
      1e-9,
    );
    const wm = attn.weightsMatrix();
    for (let i = 0; i < wm.length; i++) {
      expectClose(wm[i]!, g.fit.weightsMatrix[i]!, `fit.weightsMatrix[${i}]`);
    }
    const out = attn.combine(cProbs, 3, [1, 2, 3], 3);
    for (let i = 0; i < out.length; i++) {
      expectClose(out[i]!, g.fit.combine[i]!, `fit.combine[${i}]`);
    }
  });

  it("computeUpperBounds matches golden", () => {
    const attn = new AttentionLogOddsWeights(2, 1, 0.5, false, 0, null);
    const out = attn.computeUpperBounds([0.95, 0.9, 0.4, 0.5, 0.2, 0.1], 3, [1, 2, 3], 3);
    for (let i = 0; i < out.length; i++) {
      expectClose(out[i]!, g.computeUpperBounds[i]!, `computeUpperBounds[${i}]`);
    }
  });

  it("prune matches golden", () => {
    const attn = new AttentionLogOddsWeights(2, 1, 0.5, false, 0, null);
    const { surviving, fused } = attn.prune(
      cProbs,
      3,
      [1, 2, 3],
      3,
      0.5,
      [0.95, 0.9, 0.4, 0.5, 0.2, 0.1],
    );
    expect(surviving).toEqual(g.prune.surviving);
    expect(fused.length).toBe(g.prune.fused.length);
    for (let i = 0; i < fused.length; i++) {
      expectClose(fused[i]!, g.prune.fused[i]!, `prune.fused[${i}]`);
    }
  });
});
