import { describe, expect, it } from "vitest";
import {
  VectorProbabilityTransform,
  gaussianPDF,
  ivfDensityPrior,
  knnDensityPrior,
  silvermanBandwidth,
} from "../../src/vectorProbability.js";

describe("VectorProbabilityTransform", () => {
  it("calibrates closer distances to higher probabilities with weighted KDE", () => {
    const sample = [0.05, 0.08, 0.12, 0.2, 0.8, 0.95, 1.1, 1.25];
    const weights = [1, 1, 0.9, 0.7, 0.05, 0.02, 0.01, 0.01];
    const transform = VectorProbabilityTransform.fitBackground(sample, { baseRate: 0.2 });

    const probs = transform.calibrateWithSample([0.07, 0.4, 1.15], sample, {
      weights,
      method: "kde",
      bandwidthFactor: 1.2,
    });

    expect(probs[0]!).toBeGreaterThan(probs[1]!);
    expect(probs[1]!).toBeGreaterThan(probs[2]!);
    for (const prob of probs) {
      expect(Number.isFinite(prob)).toBe(true);
      expect(prob).toBeGreaterThan(0);
      expect(prob).toBeLessThan(1);
    }
  });

  it("supports auto routing and density priors with finite probabilities", () => {
    const sample = [0.03, 0.04, 0.06, 0.7, 0.8, 0.9];
    const prior = knnDensityPrior(sample, 0.4);
    const transform = VectorProbabilityTransform.fitBackground(sample);
    const probs = transform.calibrate([0.04, 0.85], { densityPrior: prior });

    expect(probs[0]!).toBeGreaterThan(probs[1]!);
    expect(transform.detectGap(sample)).not.toBeNull();
    for (const prob of probs) {
      expect(Number.isFinite(prob)).toBe(true);
      expect(prob).toBeGreaterThan(0);
      expect(prob).toBeLessThan(1);
    }
  });

  it("density priors reward sparse vector neighborhoods", () => {
    expect(ivfDensityPrior(5, 10)).toBeGreaterThan(ivfDensityPrior(20, 10));
    expect(knnDensityPrior(0.8, 0.4)).toBeGreaterThan(knnDensityPrior(0.2, 0.4));
  });

  it("basic density helpers stay positive and finite", () => {
    expect(gaussianPDF(0, 0, 1)).toBeGreaterThan(0);
    expect(silvermanBandwidth([0.1, 0.2, 0.4], [1, 0.5, 0.2])).toBeGreaterThan(0);
  });
});
