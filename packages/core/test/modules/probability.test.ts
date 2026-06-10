/**
 * Golden-parity tests for the probability transforms.
 * Mirrors `examples/extract_golden_modules2.rs` exactly.
 */
import { describe, it } from "vitest";
import { goldenModules2, expectClose } from "./_modules.js";
import {
  BayesianProbabilityTransform,
  TemporalBayesianTransform,
} from "../../src/probability.js";

const g = goldenModules2.probabilityTransform;
const gt = goldenModules2.temporalTransform;

describe("BayesianProbabilityTransform — deterministic surface", () => {
  it("likelihood", () => {
    const pt = new BayesianProbabilityTransform(1.0, 0.5, null);
    for (const [score, v] of g.likelihood) {
      expectClose(pt.likelihood(score), v, `likelihood(${score})`);
    }
  });

  it("tfPrior", () => {
    for (const [tf, v] of g.tfPrior) {
      expectClose(BayesianProbabilityTransform.tfPrior(tf), v, `tfPrior(${tf})`);
    }
  });

  it("normPrior", () => {
    for (const [dlr, v] of g.normPrior) {
      expectClose(
        BayesianProbabilityTransform.normPrior(dlr),
        v,
        `normPrior(${dlr})`,
      );
    }
  });

  it("compositePrior", () => {
    for (const c of g.compositePrior) {
      expectClose(
        BayesianProbabilityTransform.compositePrior(c.tf, c.dlr),
        c.v,
        `compositePrior(${c.tf},${c.dlr})`,
      );
    }
  });

  it("posterior", () => {
    for (const c of g.posterior) {
      expectClose(
        BayesianProbabilityTransform.posterior(c.likelihood, c.prior, c.baseRate),
        c.v,
        `posterior(${c.likelihood},${c.prior},${c.baseRate})`,
      );
    }
  });

  it("scoreToProbability (no baseRate, Balanced)", () => {
    const pt = new BayesianProbabilityTransform(1.0, 0.5, null);
    for (const c of g.scoreToProbability) {
      expectClose(
        pt.scoreToProbability(c.score, c.tf, c.dlr),
        c.v,
        `scoreToProbability(${c.score},${c.tf},${c.dlr})`,
      );
    }
  });

  it("scoreToProbability with baseRate", () => {
    const pt = new BayesianProbabilityTransform(1.0, 0.5, 0.2);
    for (const c of g.scoreToProbabilityBaseRate) {
      expectClose(
        pt.scoreToProbability(c.score, c.tf, c.dlr),
        c.v,
        `scoreToProbabilityBaseRate(${c.score},${c.tf},${c.dlr})`,
      );
    }
  });

  it("wandUpperBound", () => {
    const pt = new BayesianProbabilityTransform(1.0, 0.5, null);
    for (const c of g.wandUpperBound) {
      expectClose(
        pt.wandUpperBound(c.ub, c.pMax),
        c.v,
        `wandUpperBound(${c.ub},${c.pMax})`,
      );
    }
  });
});

describe("BayesianProbabilityTransform — fit", () => {
  it("balanced", () => {
    const pt = new BayesianProbabilityTransform(1.0, 0.0, null);
    pt.fit(g.fit.scores, g.fit.labels, 0.1, 300, 1e-9, "balanced");
    expectClose(pt.alpha, g.fit.balanced.alpha, "fit.balanced.alpha");
    expectClose(pt.beta, g.fit.balanced.beta, "fit.balanced.beta");
  });

  it("priorFree", () => {
    const pt = new BayesianProbabilityTransform(1.0, 0.0, null);
    pt.fit(g.fit.scores, g.fit.labels, 0.1, 300, 1e-9, "priorFree");
    expectClose(pt.alpha, g.fit.priorFree.alpha, "fit.priorFree.alpha");
    expectClose(pt.beta, g.fit.priorFree.beta, "fit.priorFree.beta");
  });

  it("priorAware", () => {
    const pt = new BayesianProbabilityTransform(1.0, 0.0, null);
    pt.fit(
      g.fit.scores,
      g.fit.labels,
      0.1,
      300,
      1e-9,
      "priorAware",
      g.fit.tfs,
      g.fit.dlrs,
    );
    expectClose(pt.alpha, g.fit.priorAware.alpha, "fit.priorAware.alpha");
    expectClose(pt.beta, g.fit.priorAware.beta, "fit.priorAware.beta");
  });
});

describe("BayesianProbabilityTransform — update", () => {
  it("per-sample balanced sequence", () => {
    const pt = new BayesianProbabilityTransform(1.0, 0.0, null);
    for (let i = 0; i < g.update.scores.length; i++) {
      pt.update(
        [g.update.scores[i] as number],
        [g.update.labels[i] as number],
        0.05,
        0.9,
        1000.0,
        1.0,
        0.99,
        "balanced",
      );
    }
    expectClose(pt.alpha, g.update.alpha, "update.alpha");
    expectClose(pt.beta, g.update.beta, "update.beta");
    expectClose(pt.averagedAlpha(), g.update.averagedAlpha, "update.averagedAlpha");
    expectClose(pt.averagedBeta(), g.update.averagedBeta, "update.averagedBeta");
  });
});

describe("TemporalBayesianTransform", () => {
  it("fit with timestamps (balanced)", () => {
    const tt = new TemporalBayesianTransform(1.0, 0.0, null, gt.decayHalfLife);
    tt.fit(gt.scores, gt.labels, gt.timestamps, 0.1, 300, 1e-9, "balanced");
    expectClose(tt.transform.alpha, gt.fit.alpha, "temporal.fit.alpha");
    expectClose(tt.transform.beta, gt.fit.beta, "temporal.fit.beta");
  });

  it("update sequence", () => {
    const tt = new TemporalBayesianTransform(1.0, 0.0, null, gt.decayHalfLife);
    for (let i = 0; i < gt.scores.length; i++) {
      tt.update(
        [gt.scores[i] as number],
        [gt.labels[i] as number],
        0.05,
        0.9,
        1000.0,
        1.0,
        0.995,
        "balanced",
      );
    }
    expectClose(tt.timestamp(), gt.update.timestamp, "temporal.update.timestamp");
    expectClose(tt.transform.alpha, gt.update.alpha, "temporal.update.alpha");
    expectClose(tt.transform.beta, gt.update.beta, "temporal.update.beta");
    expectClose(
      tt.averagedAlpha(),
      gt.update.averagedAlpha,
      "temporal.update.averagedAlpha",
    );
    expectClose(
      tt.averagedBeta(),
      gt.update.averagedBeta,
      "temporal.update.averagedBeta",
    );
  });
});
