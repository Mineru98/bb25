import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  rankDocs,
  ndcgAtK,
  mrrAtK,
  averagePrecisionAtK,
  recallAtK,
  runBench,
  runBenchWithDetails,
  type RelMap,
  type Qrels,
  type BenchQuery,
} from "../src/bench.js";
import { Corpus, buildDefaultCorpus, buildDefaultQueries } from "@bb25/core";

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
    expect(recallAtK(ranked, rel, 10)).toBeCloseTo(1.0, 12);
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

  it("map@k uses pytrec-style full relevant denominator", () => {
    const rel: RelMap = new Map([
      ["d1", 1],
      ["d2", 1],
      ["d3", 1],
    ]);
    expect(averagePrecisionAtK(["d1", "d2"], rel, 2)).toBeCloseTo(2 / 3, 12);
    expect(recallAtK(["d1", "d2"], rel, 2)).toBeCloseTo(2 / 3, 12);
  });

  it("matches the binary pytrec-style metric parity fixture", () => {
    const fixturePath = fileURLToPath(new URL("../../../fixtures/bench/pytrec-parity.json", import.meta.url));
    const fixture = JSON.parse(readFileSync(fixturePath, "utf8")) as {
      cutoff: number;
      qrels: Record<string, Record<string, number>>;
      rankings: Record<string, string[]>;
      expected: Record<string, Record<string, number>>;
    };

    const means = { ndcg: 0, map: 0, mrr: 0, recall: 0 };
    for (const qid of ["q1", "q2"]) {
      const rel = new Map(Object.entries(fixture.qrels[qid] ?? {})) as RelMap;
      const ranked = fixture.rankings[qid]!;
      const expected = fixture.expected[qid]!;
      const k = fixture.cutoff;
      const ndcg = ndcgAtK(ranked, rel, k);
      const map = averagePrecisionAtK(ranked, rel, k);
      const mrr = mrrAtK(ranked, rel, k);
      const recall = recallAtK(ranked, rel, k);

      expect(ndcg).toBeCloseTo(expected[`ndcg@${k}`]!, 12);
      expect(map).toBeCloseTo(expected[`map@${k}`]!, 12);
      expect(mrr).toBeCloseTo(expected[`mrr@${k}`]!, 12);
      expect(recall).toBeCloseTo(expected[`recall@${k}`]!, 12);

      means.ndcg += ndcg / 2;
      means.map += map / 2;
      means.mrr += mrr / 2;
      means.recall += recall / 2;
    }

    const expectedMean = fixture.expected.mean!;
    const k = fixture.cutoff;
    expect(means.ndcg).toBeCloseTo(expectedMean[`ndcg@${k}`]!, 12);
    expect(means.map).toBeCloseTo(expectedMean[`map@${k}`]!, 12);
    expect(means.mrr).toBeCloseTo(expectedMean[`mrr@${k}`]!, 12);
    expect(means.recall).toBeCloseTo(expectedMean[`recall@${k}`]!, 12);
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

    const results = runBench(corpus, queries, qrels, { cutoffs: [5, 10], bm25Method: "lucene" });
    const names = results.map((r) => r.scorer);
    expect(names).toContain("bm25");
    expect(names).toContain("bayesian");
    expect(names).toContain("dense");
    expect(names).toContain("convex");
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

  it("RRF over candidate-depth union gives no contribution to unretrieved signals", () => {
    const corpus = new Corpus();
    corpus.addDocumentTokens("sparse-only", "", ["needle"], [0, 1]);
    corpus.addDocumentTokens("dense-only", "", ["other"], [1, 0]);
    corpus.addDocumentTokens("neither", "", ["other"], [-1, 0]);
    corpus.buildIndex();

    const queries: BenchQuery[] = [
      { queryId: "q1", text: "needle", terms: ["needle"], embedding: [1, 0] },
    ];
    const qrels: Qrels = new Map([["q1", new Map([["dense-only", 1]]) as RelMap]]);
    const runs: NonNullable<Parameters<typeof runBench>[3]>["runs"] = [];

    runBench(corpus, queries, qrels, {
      bm25Method: "lucene",
      candidateDepth: 1,
      cutoffs: [1],
      runs,
    });

    const rrf = runs!.find((run) => run.scorer === "rrf" && run.queryId === "q1")!;
    expect(rrf.scores.map(([docId]) => docId).sort()).toEqual(["dense-only", "sparse-only"]);
    const scores = new Map(rrf.scores);
    expect(scores.get("dense-only")).toBeCloseTo(1 / 61, 12);
    expect(scores.get("sparse-only")).toBeCloseTo(1 / 61, 12);
  });

  it("resolves percentile base-rate auto from pseudo-query BM25 scores", () => {
    const corpus = new Corpus();
    corpus.addDocumentTokens("d1", "", ["alpha"], []);
    corpus.addDocumentTokens("d2", "", ["beta"], []);
    corpus.addDocumentTokens("d3", "", ["gamma"], []);
    corpus.addDocumentTokens("d4", "", ["delta"], []);
    corpus.buildIndex();

    const queries: BenchQuery[] = [{ queryId: "q1", text: "alpha", terms: ["alpha"], embedding: null }];
    const qrels: Qrels = new Map([["q1", new Map([["d1", 1]]) as RelMap]]);
    const details = runBenchWithDetails(corpus, queries, qrels, {
      bm25Method: "lucene",
      baseRate: "auto",
      cutoffs: [1],
    });

    expect(details.options.requestedBaseRate).toBe("auto");
    expect(details.options.baseRateMethod).toBe("percentile");
    expect(details.options.baseRate).toBeCloseTo(0.25, 12);
  });

  it("reports calibration metrics for probabilistic scorer rows", () => {
    const corpus = new Corpus();
    corpus.addDocumentTokens("d1", "", ["alpha"], []);
    corpus.addDocumentTokens("d2", "", ["alpha"], []);
    corpus.addDocumentTokens("d3", "", ["beta"], []);
    corpus.addDocumentTokens("d4", "", ["gamma"], []);
    corpus.buildIndex();

    const queries: BenchQuery[] = [{ queryId: "q1", text: "alpha", terms: ["alpha"], embedding: null }];
    const qrels: Qrels = new Map([
      [
        "q1",
        new Map([
          ["d1", 1],
          ["d2", 1],
          ["d3", 0],
          ["d4", 0],
        ]) as RelMap,
      ],
    ]);
    const details = runBenchWithDetails(corpus, queries, qrels, {
      bm25Method: "lucene",
      calibrationBins: 2,
      cutoffs: [1],
    });

    const bayesian = details.calibration.find((row) => row.scorer === "bayesian");
    expect(bayesian).toBeDefined();
    expect(bayesian!.samples).toBe(4);
    expect(bayesian!.bins).toBe(2);
    expect(bayesian!.ece).toBeGreaterThanOrEqual(0);
    expect(bayesian!.brier).toBeGreaterThanOrEqual(0);
  });
});
