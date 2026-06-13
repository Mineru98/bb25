import { describe, expect, it } from "vitest";

import { runExperiments } from "../../src/experiments.js";
import { goldenModules } from "./_modules.js";

describe("experiments harness (exp1..exp13)", () => {
  const results = runExperiments();
  const golden = goldenModules.experiments.results;

  it("produces the same number of experiments as the golden fixture", () => {
    expect(results.length).toBe(golden.length);
    expect(results.length).toBe(13);
  });

  for (let i = 0; i < golden.length; i++) {
    const expected = golden[i]!;
    it(`experiment ${i + 1} matches golden name and passes: ${expected.name}`, () => {
      const actual = results[i]!;
      expect(actual.name).toBe(expected.name);
      expect(actual.passed).toBe(true);
      expect(expected.passed).toBe(true);
    });
  }
});
