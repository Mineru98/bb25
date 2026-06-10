import { describe, it, expect } from "vitest";
import { BgeM3Embedder } from "../src/index.js";

// Fast, offline checks — no model download.
describe("BgeM3Embedder (unit)", () => {
  it("reports dim 1024", () => {
    expect(new BgeM3Embedder().dim).toBe(1024);
    expect(new BgeM3Embedder({ dtype: "q8" }).dim).toBe(1024);
  });

  it("embed([]) returns [] without loading the model", async () => {
    const e = new BgeM3Embedder({ localOnly: true });
    await expect(e.embed([])).resolves.toEqual([]);
  });
});
