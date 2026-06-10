import { describe, it, expect } from "vitest";
import { goldenModules4, expectClose } from "./_modules.js";
import { FusionDebugger } from "../../src/debug.js";
import { BayesianProbabilityTransform } from "../../src/probability.js";

describe("FusionDebugger numeric trace parity", () => {
  const dbg = new FusionDebugger(new BayesianProbabilityTransform(1.0, 0.5, null));
  const dbgBr = new FusionDebugger(new BayesianProbabilityTransform(1.0, 0.5, 0.2));
  const g = goldenModules4;

  it("traceBm25 (no base rate)", () => {
    const b = dbg.traceBm25(2.0, 5.0, 0.6);
    expectClose(b.likelihood, g.traceBm25.likelihood, "likelihood");
    expectClose(b.tfPrior, g.traceBm25.tfPrior, "tfPrior");
    expectClose(b.normPrior, g.traceBm25.normPrior, "normPrior");
    expectClose(b.compositePrior, g.traceBm25.compositePrior, "compositePrior");
    expectClose(b.logitLikelihood, g.traceBm25.logitLikelihood, "logitLikelihood");
    expectClose(b.logitPrior, g.traceBm25.logitPrior, "logitPrior");
    expectClose(b.posterior, g.traceBm25.posterior, "posterior");
  });

  it("traceBm25 (base rate)", () => {
    const b = dbgBr.traceBm25(2.0, 5.0, 0.6);
    expectClose(b.posterior, g.traceBm25BaseRate.posterior, "posterior(br)");
    expectClose(b.logitBaseRate!, g.traceBm25BaseRate.logitBaseRate, "logitBaseRate");
  });

  it("traceVector / traceNot", () => {
    const v = dbg.traceVector(0.6);
    expectClose(v.probability, g.traceVector.probability, "v.probability");
    expectClose(v.logitProbability, g.traceVector.logitProbability, "v.logitProbability");
    const nt = dbg.traceNot(0.3, "x");
    expectClose(nt.complement, g.traceNot.complement, "not.complement");
    expectClose(nt.logitInput, g.traceNot.logitInput, "not.logitInput");
    expectClose(nt.logitComplement, g.traceNot.logitComplement, "not.logitComplement");
  });

  it("traceFusion variants", () => {
    const probs = [0.8, 0.6];
    const lo = dbg.traceFusion(probs, null, "log_odds", null, null);
    expectClose(lo.meanLogit!, g.traceFusion.logOdds.meanLogit, "lo.meanLogit");
    expectClose(lo.nAlphaScale!, g.traceFusion.logOdds.nAlphaScale, "lo.nAlphaScale");
    expectClose(lo.scaledLogit!, g.traceFusion.logOdds.scaledLogit, "lo.scaledLogit");
    expectClose(lo.fusedProbability, g.traceFusion.logOdds.fused, "lo.fused");

    const low = dbg.traceFusion(probs, null, "log_odds", 0.0, [0.3, 0.7]);
    expectClose(low.meanLogit!, g.traceFusion.logOddsWeighted.meanLogit, "low.meanLogit");
    expectClose(low.scaledLogit!, g.traceFusion.logOddsWeighted.scaledLogit, "low.scaledLogit");
    expectClose(low.fusedProbability, g.traceFusion.logOddsWeighted.fused, "low.fused");

    const pa = dbg.traceFusion(probs, null, "prob_and", null, null);
    expectClose(pa.logProbSum!, g.traceFusion.probAnd.logProbSum, "pa.logProbSum");
    expectClose(pa.fusedProbability, g.traceFusion.probAnd.fused, "pa.fused");

    const po = dbg.traceFusion(probs, null, "prob_or", null, null);
    expectClose(po.logComplementSum!, g.traceFusion.probOr.logComplementSum, "po.logComplementSum");
    expectClose(po.fusedProbability, g.traceFusion.probOr.fused, "po.fused");

    const pn = dbg.traceFusion(probs, null, "prob_not", null, null);
    expectClose(pn.fusedProbability, g.traceFusion.probNot.fused, "pn.fused");
  });

  it("traceDocument + compare", () => {
    const da = dbg.traceDocument(2.0, 5.0, 0.6, 0.6, "log_odds", null, null, "dA");
    const db = dbg.traceDocument(0.5, 1.0, 0.4, 0.2, "log_odds", null, null, "dB");
    expectClose(da.finalProbability, g.traceDocument.finalA, "finalA");
    expectClose(db.finalProbability, g.traceDocument.finalB, "finalB");

    const cmp = dbg.compare(da, db);
    expect(cmp.dominantSignal).toBe(g.compare.dominant);
    expect(cmp.signalDeltas.length).toBe(g.compare.deltas.length);
    for (let i = 0; i < g.compare.deltas.length; i++) {
      expect(cmp.signalDeltas[i]![0]).toBe(g.compare.deltas[i]!.name);
      expectClose(cmp.signalDeltas[i]![1], g.compare.deltas[i]!.delta, `delta ${i}`);
    }
  });
});
