import { describe, it, expect } from "vitest";
import { buildDefaultCorpus } from "../src/defaults.js";
import { BM25Scorer } from "../src/bm25.js";
import { golden, expectClose } from "./_golden.js";

describe("Corpus + index parity", () => {
  const corpus = buildDefaultCorpus();

  it("matches corpus stats (n, avgdl)", () => {
    expect(corpus.n).toBe(golden.corpus.n);
    expectClose(corpus.avgdl, golden.corpus.avgdl, "avgdl");
  });

  it("matches document frequency map", () => {
    expect(corpus.df.size).toBe(Object.keys(golden.corpus.df).length);
    for (const [term, df] of Object.entries(golden.corpus.df)) {
      expect(corpus.df.get(term), `df[${term}]`).toBe(df);
    }
  });

  it("matches per-document tokens, length, term frequencies and embeddings", () => {
    expect(corpus.documents().length).toBe(golden.documents.length);
    for (const gd of golden.documents) {
      const doc = corpus.getDocument(gd.id);
      expect(doc, `doc ${gd.id}`).toBeDefined();
      expect(doc!.tokens).toEqual(gd.tokens);
      expect(doc!.length).toBe(gd.length);
      expect(doc!.embedding).toEqual(gd.embedding);
      expect(doc!.text).toBe(gd.text);
      expect(doc!.termFreq.size).toBe(Object.keys(gd.termFreq).length);
      for (const [term, tf] of Object.entries(gd.termFreq)) {
        expect(doc!.termFreq.get(term), `tf[${gd.id}][${term}]`).toBe(tf);
      }
    }
  });

  it("matches IDF for every vocabulary term", () => {
    const bm25 = new BM25Scorer(corpus, golden.params.k1, golden.params.b);
    for (const [term, idf] of Object.entries(golden.idf)) {
      expectClose(bm25.idf(term), idf, `idf[${term}]`);
    }
  });
});
