import { describe, it, expect } from "vitest";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  rankDocs,
  ndcgAtK,
  mrrAtK,
  averagePrecisionAtK,
  recallAtK,
  bayesianLogOddsFusionScores,
  gatedLogOddsFusionScores,
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

  it("supports mixture and elbow base-rate auto estimators", () => {
    const corpus = new Corpus();
    for (const [id, terms] of [
      ["d1", ["alpha", "alpha", "shared"]],
      ["d2", ["alpha", "shared"]],
      ["d3", ["beta", "beta", "shared"]],
      ["d4", ["beta", "shared"]],
      ["d5", ["gamma", "shared"]],
      ["d6", ["delta"]],
      ["d7", ["epsilon"]],
      ["d8", ["zeta"]],
    ] as const) {
      corpus.addDocumentTokens(id, "", terms, []);
    }
    corpus.buildIndex();

    const queries: BenchQuery[] = [{ queryId: "q1", text: "alpha", terms: ["alpha"], embedding: null }];
    const qrels: Qrels = new Map([["q1", new Map([["d1", 1]]) as RelMap]]);

    const percentile = runBenchWithDetails(corpus, queries, qrels, {
      bm25Method: "lucene",
      baseRate: "auto",
      baseRateMethod: "percentile",
      baseRateSampleSize: 8,
      cutoffs: [1],
    });
    const mixture = runBenchWithDetails(corpus, queries, qrels, {
      bm25Method: "lucene",
      baseRate: "auto",
      baseRateMethod: "mixture",
      baseRateSampleSize: 8,
      cutoffs: [1],
    });
    const elbow = runBenchWithDetails(corpus, queries, qrels, {
      bm25Method: "lucene",
      baseRate: "auto",
      baseRateMethod: "elbow",
      baseRateSampleSize: 8,
      cutoffs: [1],
    });

    expect(percentile.options.baseRate).toBeCloseTo(0.125, 12);
    expect(mixture.options.baseRate).toBeCloseTo(0.42878639704500715, 12);
    expect(elbow.options.baseRate).toBeCloseTo(0.42857142857142855, 12);
    expect(mixture.options.baseRateMethod).toBe("mixture");
    expect(elbow.options.baseRateMethod).toBe("elbow");
  });

  it("Bayesian LogOdds fusion handles agreement, disagreement, and base-rate correction", () => {
    const agreement = bayesianLogOddsFusionScores(
      [2.0],
      [0.8],
      [2],
      [1.0],
      1.0,
      0.5,
      0.0,
      2.0,
      null,
    )[0]!;
    const disagreement = bayesianLogOddsFusionScores(
      [2.0],
      [-0.8],
      [2],
      [1.0],
      1.0,
      0.5,
      0.0,
      2.0,
      null,
    )[0]!;
    const br = bayesianLogOddsFusionScores(
      [2.0],
      [0.8],
      [2],
      [1.0],
      1.0,
      0.5,
      0.0,
      2.0,
      0.05,
    )[0]!;
    const denseOnly = bayesianLogOddsFusionScores(
      [0.0],
      [0.8],
      [0],
      [1.0],
      1.0,
      0.5,
      0.0,
      2.0,
      null,
    )[0]!;

    expect(agreement).toBeGreaterThan(disagreement);
    expect(br).toBeLessThan(agreement);
    expect(denseOnly).toBeCloseTo(1 / (1 + Math.exp(-0.8)), 12);
    for (const value of [agreement, disagreement, br, denseOnly]) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThan(0);
      expect(value).toBeLessThan(1);
    }
  });

  it("gated LogOdds fusion handles agreement and finite probability bounds", () => {
    const relu = gatedLogOddsFusionScores([0.9, 0.2], [0.8, -0.8], "relu");
    const swish = gatedLogOddsFusionScores([0.9, 0.2], [0.8, -0.8], "swish");
    const gelu = gatedLogOddsFusionScores([0.9, 0.2], [0.8, -0.8], "gelu");
    const swishB2 = gatedLogOddsFusionScores([0.9, 0.2], [0.8, -0.8], "swish_b2");
    const softplus = gatedLogOddsFusionScores([0.9, 0.2], [0.8, -0.8], "softplus");

    expect(relu[0]!).toBeGreaterThan(relu[1]!);
    expect(swish[0]!).toBeGreaterThan(swish[1]!);
    expect(gelu[0]!).toBeGreaterThan(gelu[1]!);
    expect(swishB2[0]!).toBeGreaterThan(swishB2[1]!);
    expect(softplus[0]!).toBeGreaterThan(softplus[1]!);
    for (const value of [...relu, ...swish, ...gelu, ...swishB2, ...softplus]) {
      expect(Number.isFinite(value)).toBe(true);
      expect(value).toBeGreaterThan(0);
      expect(value).toBeLessThan(1);
    }
  });

  it("emits Bayesian LogOdds hybrid rows and marks calibrated/diagnostic metadata", () => {
    const corpus = new Corpus();
    corpus.addDocumentTokens("sparse-dense", "", ["needle"], [1, 0]);
    corpus.addDocumentTokens("sparse-only", "", ["needle"], [0, 1]);
    corpus.addDocumentTokens("dense-only", "", ["other"], [1, 0]);
    corpus.addDocumentTokens("neither", "", ["other"], [-1, 0]);
    corpus.buildIndex();

    const queries: BenchQuery[] = [
      { queryId: "q1", text: "needle", terms: ["needle"], embedding: [1, 0] },
    ];
    const qrels: Qrels = new Map([["q1", new Map([["sparse-dense", 1], ["dense-only", 0]]) as RelMap]]);
    const runs: NonNullable<Parameters<typeof runBench>[3]>["runs"] = [];
    const details = runBenchWithDetails(corpus, queries, qrels, {
      bm25Method: "lucene",
      baseRate: "auto",
      candidateDepth: 2,
      cutoffs: [1],
      calibrationBins: 2,
      runs,
    });

    expect(details.results.map((row) => row.scorer)).toContain("bayesian_logodds");
    expect(details.results.map((row) => row.scorer)).toContain("bayesian_logodds_br");
    expect(details.results.map((row) => row.scorer)).toContain("bayesian_gated_relu");
    expect(details.results.map((row) => row.scorer)).toContain("bayesian_gated_swish");
    expect(details.results.map((row) => row.scorer)).toContain("bayesian_gated_gelu");
    expect(details.results.map((row) => row.scorer)).toContain("bayesian_gated_swish_b2");
    expect(details.results.map((row) => row.scorer)).toContain("bayesian_gated_softplus");
    expect(details.scorers.find((row) => row.scorer === "bayesian_logodds")!.kind).toBe("zero-shot");
    expect(details.scorers.find((row) => row.scorer === "bayesian_logodds_br")!.kind).toBe("calibration");
    expect(details.scorers.find((row) => row.scorer === "bayesian_gated_relu")!.kind).toBe("diagnostic");
    expect(details.scorers.find((row) => row.scorer === "bayesian_gated_swish")!.kind).toBe("diagnostic");
    expect(details.scorers.find((row) => row.scorer === "bayesian_gated_gelu")!.kind).toBe("diagnostic");
    expect(details.scorers.find((row) => row.scorer === "bayesian_gated_swish_b2")!.kind).toBe("diagnostic");
    expect(details.scorers.find((row) => row.scorer === "bayesian_gated_softplus")!.kind).toBe("diagnostic");
    expect(runs!.some((run) => run.scorer === "bayesian_logodds_br")).toBe(true);
    expect(details.calibration.some((row) => row.scorer === "bayesian_logodds_br")).toBe(true);
    expect(details.calibration.some((row) => row.scorer === "bayesian_gated_relu")).toBe(true);
    expect(details.calibration.some((row) => row.scorer === "bayesian_gated_softplus")).toBe(true);
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

  it("emits MultiField sparse and balanced hybrid rows when field terms are provided", () => {
    const corpus = new Corpus();
    corpus.addDocumentTokens("title-hit", "", ["needle", "common"], [0, 1]);
    corpus.addDocumentTokens("body-hit", "", ["needle", "common"], [1, 0]);
    corpus.addDocumentTokens("miss", "", ["common"], [-1, 0]);
    corpus.buildIndex();

    const queries: BenchQuery[] = [
      { queryId: "q1", text: "needle", terms: ["needle"], embedding: [1, 0] },
    ];
    const qrels: Qrels = new Map([
      ["q1", new Map([["title-hit", 1], ["body-hit", 0], ["miss", 0]]) as RelMap],
    ]);
    const details = runBenchWithDetails(corpus, queries, qrels, {
      bm25Method: "lucene",
      cutoffs: [1],
      calibrationBins: 2,
      candidateDepth: 2,
      multiField: {
        fields: ["title", "body"],
        fieldWeights: { title: 0.8, body: 0.2 },
        docFields: new Map([
          ["title-hit", { title: ["needle"], body: ["common"] }],
          ["body-hit", { title: ["other"], body: ["needle", "common"] }],
          ["miss", { title: ["other"], body: ["common"] }],
        ]),
      },
    });

    const names = details.results.map((row) => row.scorer);
    expect(names).toContain("bayesian_multifield");
    expect(names).toContain("bayesian_multifield_bal");
    expect(details.results.find((row) => row.scorer === "bayesian_multifield")!.queries).toBe(1);
    expect(details.results.find((row) => row.scorer === "bayesian_multifield_bal")!.queries).toBe(1);
    expect(details.scorers.find((row) => row.scorer === "bayesian_multifield")!.kind).toBe("zero-shot");
    expect(details.scorers.find((row) => row.scorer === "bayesian_multifield_bal")!.kind).toBe("diagnostic");
    expect(details.calibration.some((row) => row.scorer === "bayesian_multifield")).toBe(true);
    expect(details.options.multiField).toEqual({
      fields: ["title", "body"],
      fieldWeights: { title: 0.8, body: 0.2 },
    });
  });

  it("adds split-aware fitted Bayesian row with train/eval query metadata", () => {
    const corpus = new Corpus();
    corpus.addDocumentTokens("a1", "", ["alpha", "one"], []);
    corpus.addDocumentTokens("a2", "", ["alpha", "two"], []);
    corpus.addDocumentTokens("b1", "", ["beta", "one"], []);
    corpus.addDocumentTokens("b2", "", ["beta", "two"], []);
    corpus.buildIndex();

    const queries: BenchQuery[] = [
      { queryId: "q1", text: "alpha", terms: ["alpha"], embedding: null },
      { queryId: "q2", text: "alpha", terms: ["alpha"], embedding: null },
      { queryId: "q3", text: "beta", terms: ["beta"], embedding: null },
      { queryId: "q4", text: "beta", terms: ["beta"], embedding: null },
    ];
    const qrels: Qrels = new Map([
      ["q1", new Map([["a1", 1], ["a2", 0]]) as RelMap],
      ["q2", new Map([["a1", 1], ["a2", 0]]) as RelMap],
      ["q3", new Map([["b1", 1], ["b2", 0]]) as RelMap],
      ["q4", new Map([["b1", 1], ["b2", 0]]) as RelMap],
    ]);

    const details = runBenchWithDetails(corpus, queries, qrels, {
      bm25Method: "lucene",
      cutoffs: [1],
      fitSplit: { trainRatio: 0.5, seed: 42 },
      calibrationBins: 2,
    });

    const splitRow = details.results.find((row) => row.scorer === "bayesian_fitted_split");
    expect(splitRow).toBeDefined();
    expect(splitRow!.queries).toBe(2);
    expect(details.fittedSplit).not.toBeNull();
    expect(details.fittedSplit!.trainQueryIds).toHaveLength(2);
    expect(details.fittedSplit!.evalQueryIds).toHaveLength(2);
    expect(new Set([...details.fittedSplit!.trainQueryIds, ...details.fittedSplit!.evalQueryIds]).size).toBe(4);
    expect(details.fittedSplit!.trainingPairs).toBe(4);
    expect(details.fittedSplit!.alpha).not.toBeNull();
    expect(details.fittedSplit!.beta).not.toBeNull();
    expect(details.calibration.some((row) => row.scorer === "bayesian_fitted_split")).toBe(true);
  });

  it("adds split-aware tuned attention and multi-head rows with train/eval metadata", () => {
    const corpus = new Corpus();
    for (let i = 1; i <= 6; i++) {
      corpus.addDocumentTokens(`d${i}`, "", [`topic${i}`, "shared"], [1, 0]);
      corpus.addDocumentTokens(`n${i}`, "", [`noise${i}`, "shared"], [-1, 0]);
    }
    corpus.buildIndex();

    const queries: BenchQuery[] = Array.from({ length: 6 }, (_, idx) => {
      const i = idx + 1;
      return { queryId: `q${i}`, text: `topic${i}`, terms: [`topic${i}`], embedding: [1, 0] };
    });
    const qrels: Qrels = new Map(
      Array.from({ length: 6 }, (_, idx) => {
        const i = idx + 1;
        return [`q${i}`, new Map([[`d${i}`, 1], [`n${i}`, 0]]) as RelMap];
      }),
    );

    const details = runBenchWithDetails(corpus, queries, qrels, {
      bm25Method: "lucene",
      baseRate: "auto",
      cutoffs: [1],
      fitSplit: { trainRatio: 0.67, seed: 42 },
      calibrationBins: 2,
    });

    const attention = details.results.find((row) => row.scorer === "bayesian_attention_split");
    const norm = details.results.find((row) => row.scorer === "bayesian_attn_norm_split");
    const multiHead = details.results.find((row) => row.scorer === "bayesian_multihead_split");
    const multiHeadNorm = details.results.find((row) => row.scorer === "bayesian_multihead_norm_split");
    expect(attention).toBeDefined();
    expect(norm).toBeDefined();
    expect(multiHead).toBeDefined();
    expect(multiHeadNorm).toBeDefined();
    expect(attention!.queries).toBe(2);
    expect(norm!.queries).toBe(2);
    expect(multiHead!.queries).toBe(2);
    expect(multiHeadNorm!.queries).toBe(2);
    expect(details.scorers.find((row) => row.scorer === "bayesian_attention_split")!.kind).toBe("tuned");
    expect(details.scorers.find((row) => row.scorer === "bayesian_attn_norm_split")!.kind).toBe("tuned");
    expect(details.scorers.find((row) => row.scorer === "bayesian_multihead_split")!.kind).toBe("tuned");
    expect(details.scorers.find((row) => row.scorer === "bayesian_multihead_norm_split")!.kind).toBe("tuned");

    const attentionMeta = details.attentionSplits.find((row) => row.scorer === "bayesian_attention_split");
    const normMeta = details.attentionSplits.find((row) => row.scorer === "bayesian_attn_norm_split");
    const multiHeadMeta = details.attentionSplits.find((row) => row.scorer === "bayesian_multihead_split");
    const multiHeadNormMeta = details.attentionSplits.find((row) => row.scorer === "bayesian_multihead_norm_split");
    expect(attentionMeta).toBeDefined();
    expect(normMeta).toBeDefined();
    expect(multiHeadMeta).toBeDefined();
    expect(multiHeadNormMeta).toBeDefined();
    expect(attentionMeta!.trained).toBe(true);
    expect(normMeta!.trained).toBe(true);
    expect(multiHeadMeta!.trained).toBe(true);
    expect(multiHeadNormMeta!.trained).toBe(true);
    expect(attentionMeta!.trainingPairs).toBe(12);
    expect(normMeta!.trainingPairs).toBe(12);
    expect(multiHeadMeta!.trainingPairs).toBe(12);
    expect(multiHeadNormMeta!.trainingPairs).toBe(12);
    expect(attentionMeta!.features).toBe("basic");
    expect(normMeta!.features).toBe("rich");
    expect(multiHeadMeta!.features).toBe("basic");
    expect(multiHeadNormMeta!.features).toBe("rich");
    expect(attentionMeta!.heads).toBe(1);
    expect(normMeta!.heads).toBe(1);
    expect(multiHeadMeta!.heads).toBe(4);
    expect(multiHeadNormMeta!.heads).toBe(4);
    expect(normMeta!.normalize).toBe(true);
    expect(multiHeadNormMeta!.normalize).toBe(true);
    expect(new Set([...attentionMeta!.trainQueryIds, ...attentionMeta!.evalQueryIds]).size).toBe(6);
    expect(details.calibration.some((row) => row.scorer === "bayesian_attention_split")).toBe(true);
    expect(details.calibration.some((row) => row.scorer === "bayesian_attn_norm_split")).toBe(true);
    expect(details.calibration.some((row) => row.scorer === "bayesian_multihead_split")).toBe(true);
    expect(details.calibration.some((row) => row.scorer === "bayesian_multihead_norm_split")).toBe(true);
  });
});
