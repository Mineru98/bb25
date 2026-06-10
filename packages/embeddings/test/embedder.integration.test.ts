import { describe, it, expect } from "vitest";
import { BgeM3Embedder, type Dtype } from "../src/index.js";

/**
 * Model integration test. Downloads Xenova/bge-m3 on first run, so it is gated
 * behind BB25_EMBED_IT=1 to keep the default suite fast and offline-safe.
 *   BB25_EMBED_IT=1 [BB25_EMBED_DTYPE=q8] pnpm --filter @bb25/embeddings test
 */
const RUN = process.env.BB25_EMBED_IT === "1";
const DTYPE = (process.env.BB25_EMBED_DTYPE as Dtype | undefined) ?? "q8";

function l2(v: Float32Array): number {
  let s = 0;
  for (const x of v) s += x * x;
  return Math.sqrt(s);
}
function cos(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += (a[i] as number) * (b[i] as number);
  return dot / (l2(a) * l2(b));
}

describe.runIf(RUN)(`BgeM3Embedder (integration, dtype=${DTYPE})`, () => {
  const embedder = new BgeM3Embedder({ dtype: DTYPE });

  it("produces normalized 1024-d embeddings with sensible semantics", async () => {
    const texts = [
      "machine learning algorithms learn patterns from data",
      "deep neural networks for supervised learning",
      "a recipe for banana bread with walnuts",
    ];
    const v = await embedder.embed(texts);

    expect(v.length).toBe(3);
    for (const e of v) {
      expect(e.length).toBe(1024);
      expect(l2(e)).toBeCloseTo(1.0, 3); // normalize: true
    }

    // related (ml vs dnn) should be closer than unrelated (ml vs banana)
    expect(cos(v[0]!, v[1]!)).toBeGreaterThan(cos(v[0]!, v[2]!));
  });

  it("is deterministic for identical input", async () => {
    const [a] = await embedder.embed(["reproducibility check"]);
    const [b] = await embedder.embed(["reproducibility check"]);
    expect(Array.from(a!)).toEqual(Array.from(b!));
  });
});
