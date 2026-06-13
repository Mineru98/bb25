import { describe, it } from "vitest";
import { goldenModules2, expectClose } from "./_modules.js";
import { PlattCalibrator, IsotonicCalibrator } from "../../src/calibration.js";

describe("calibration golden parity", () => {
  it("Platt scaling fit + calibrateBatch", () => {
    const g = goldenModules2.platt;
    const cal = new PlattCalibrator(1.0, 0.0);
    cal.fit(g.scores, g.labels, g.lr, g.maxIter, g.tol);

    expectClose(cal.a, g.a, "platt a");
    expectClose(cal.b, g.b, "platt b");

    const calibrated = cal.calibrateBatch(g.scores);
    for (let i = 0; i < g.calibrated.length; i++) {
      expectClose(calibrated[i]!, g.calibrated[i]!, `platt calibrated[${i}]`);
    }
  });

  it("Isotonic regression fit + calibrateBatch", () => {
    const g = goldenModules2.isotonic;
    const cal = new IsotonicCalibrator();
    cal.fit(g.scores, g.labels);

    const calibrated = cal.calibrateBatch(g.probe);
    for (let i = 0; i < g.calibrated.length; i++) {
      expectClose(calibrated[i]!, g.calibrated[i]!, `isotonic calibrated[${i}]`);
    }
  });
});
