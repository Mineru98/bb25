import { describe, it, expect } from "vitest";
import { goldenModules, expectClose } from "./_modules.js";
import {
  expectedCalibrationError,
  brierScore,
  reliabilityDiagram,
  calibrationReport,
} from "../../src/metrics.js";

describe("metrics golden parity", () => {
  for (let c = 0; c < goldenModules.metrics.length; c++) {
    const tc = goldenModules.metrics[c]!;
    it(`case ${c} (nBins=${tc.nBins})`, () => {
      const ece = expectedCalibrationError(tc.probs, tc.labels, tc.nBins);
      const brier = brierScore(tc.probs, tc.labels);
      expectClose(ece, tc.ece, `case ${c} ece`);
      expectClose(brier, tc.brier, `case ${c} brier`);

      const reliability = reliabilityDiagram(tc.probs, tc.labels, tc.nBins);
      expect(reliability.length).toBe(tc.reliability.length);
      for (let b = 0; b < tc.reliability.length; b++) {
        const got = reliability[b]!;
        const want = tc.reliability[b]!;
        expectClose(got[0], want[0], `case ${c} bin ${b} avgPred`);
        expectClose(got[1], want[1], `case ${c} bin ${b} avgActual`);
        expect(got[2]).toBe(want[2]);
      }
    });
  }

  it("calibrationReport aggregates and maps fields (nSamples/nBins)", () => {
    const tc = goldenModules.metrics[0]!;
    const report = calibrationReport(tc.probs, tc.labels, tc.nBins);
    expectClose(report.ece, tc.ece, "report ece");
    expectClose(report.brier, tc.brier, "report brier");
    expect(report.nSamples).toBe(tc.probs.length);
    expect(report.nBins).toBe(tc.nBins);
    expect(report.reliability.length).toBe(tc.reliability.length);
  });
});
