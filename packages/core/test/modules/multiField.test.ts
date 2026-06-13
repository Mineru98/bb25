import { describe, expect, it } from "vitest";
import { MultiFieldScorer } from "../../src/multiField.js";

describe("MultiFieldScorer", () => {
  it("fuses field-level Bayesian probabilities with deterministic ranking", () => {
    const scorer = new MultiFieldScorer(
      [
        { id: "title-hit", fields: { title: ["needle"], body: ["common"] } },
        { id: "body-hit", fields: { title: ["other"], body: ["needle", "common"] } },
        { id: "miss", fields: { title: ["other"], body: ["common"] } },
      ],
      {
        fields: ["title", "body"],
        fieldWeights: { title: 0.8, body: 0.2 },
        method: "lucene",
      },
    );

    const scores = new Map(scorer.scores(["needle"]));
    expect(scores.get("title-hit")!).toBeGreaterThan(scores.get("body-hit")!);
    expect(scores.get("body-hit")!).toBeGreaterThan(scores.get("miss")!);
    for (const score of scores.values()) {
      expect(Number.isFinite(score)).toBe(true);
      expect(score).toBeGreaterThanOrEqual(0);
      expect(score).toBeLessThanOrEqual(1);
    }
  });

  it("validates field weights", () => {
    expect(
      () =>
        new MultiFieldScorer([{ id: "d1", fields: { title: ["a"], body: ["b"] } }], {
          fields: ["title", "body"],
          fieldWeights: { title: 0.7, body: 0.7 },
        }),
    ).toThrow("fieldWeights must sum to 1.0");
  });
});
