import { describe, it } from "vitest";
import { sigmoid, logit } from "../src/mathUtils.js";
import { cosineToProbability, probOr, probAnd, logOddsConjunction } from "../src/fusion.js";
import { golden, expectClose } from "./_golden.js";

describe("Math + fusion primitive parity", () => {
  it("sigmoid", () => {
    for (const [x, y] of golden.math.sigmoid) {
      expectClose(sigmoid(x), y, `sigmoid(${x})`);
    }
  });

  it("logit", () => {
    for (const [p, y] of golden.math.logit) {
      expectClose(logit(p), y, `logit(${p})`);
    }
  });

  it("cosineToProbability", () => {
    for (const [s, y] of golden.math.cosineToProbability) {
      expectClose(cosineToProbability(s), y, `cosineToProbability(${s})`);
    }
  });

  it("prob_or / prob_and / log_odds_conjunction", () => {
    for (const f of golden.math.fusion) {
      expectClose(probOr(f.probs), f.probOr, `probOr(${f.probs})`);
      expectClose(probAnd(f.probs), f.probAnd, `probAnd(${f.probs})`);
      expectClose(
        logOddsConjunction(f.probs, null, null),
        f.logOddsConjDefault,
        `logOddsConjDefault(${f.probs})`,
      );
      expectClose(
        logOddsConjunction(f.probs, 0.5, null),
        f.logOddsConjAlpha05,
        `logOddsConjAlpha05(${f.probs})`,
      );
    }
  });
});
