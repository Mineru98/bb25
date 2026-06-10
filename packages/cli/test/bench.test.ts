import { describe, it, expect } from "vitest";
import {
  rankDocs,
  ndcgAtK,
  mrrAtK,
  averagePrecisionAtK,
  runBench,
  type RelMap,
  type Qrels,
  type BenchQuery,
} from "../src/bench.js";
import { buildDefaultCorpus, buildDefaultQueries } from "@bb25/core";

describe("bench metrics", () => {
  it("rankDocs sorts by -score then ascending id (tie-break)", () => {
    expect(
      rankDocs([
        ["b", 1.0],
        ["a", 1.0],
        ["c", 2.0],
      ]),
    ).toEqual(["c", "a", "b"]);
  });

  it("perfect ranking => ndcg=mrr=ap=1", () => {
    const rel: RelMap = new Map([
      ["d1", 1],
      ["d2", 1],
    ]);
    const ranked = ["d1", "d2", "d3", "d4"];
    expect(ndcgAtK(ranked, rel, 10)).toBeCloseTo(1.0, 12);
    expect(mrrAtK(ranked, rel, 10)).toBeCloseTo(1.0, 12);
    expect(averagePrecisionAtK(ranked, rel, 10)).toBeCloseTo(1.0, 12);
  });

  it("mrr reflects first relevant rank", () => {
    const rel: RelMap = new Map([["x", 1]]);
    expect(mrrAtK(["a", "b", "x", "d"], rel, 10)).toBeCloseTo(1 / 3, 12);
  });

  it("ndcg uses exponential gain and log2(idx+1) discount", () => {
    // single relevant doc with rel=1 at rank 2: dcg = (2^1-1)/log2(3) = 1/1.585
    const rel: RelMap = new Map([["x", 1]]);
    const got = ndcgAtK(["a", "x"], rel, 10);
    expect(got).toBeCloseTo(1 / Math.log2(3), 12); // ideal dcg = 1
  });

  it("runBench on the default corpus produces sane bm25/bayesian/hybrid rows", () => {
    const corpus = buildDefaultCorpus();
    const dq = buildDefaultQueries();
    const queries: BenchQuery[] = dq.map((q, i) => ({
      queryId: `q${i}`,
      text: q.text,
      terms: q.terms,
      embedding: q.embedding,
    }));
    const qrels: Qrels = new Map(
      dq.map((q, i) => [`q${i}`, new Map(q.relevant.map((d) => [d, 1])) as RelMap]),
    );

    const results = runBench(corpus, queries, qrels, { cutoffs: [5, 10] });
    const names = results.map((r) => r.scorer);
    expect(names).toContain("bm25");
    expect(names).toContain("bayesian");
    expect(names).toContain("hybrid_or");
    expect(names).toContain("rrf");
    for (const r of results) {
      for (const k of [5, 10]) {
        const ndcg = r.metrics[`ndcg@${k}`]!;
        expect(ndcg).toBeGreaterThanOrEqual(0);
        expect(ndcg).toBeLessThanOrEqual(1 + 1e-12);
      }
    }
    // hybrid_or should rank the toy corpus well (relevant docs are semantically clustered)
    const hybridOr = results.find((r) => r.scorer === "hybrid_or")!;
    expect(hybridOr.metrics["ndcg@10"]!).toBeGreaterThan(0.5);
  });
});
