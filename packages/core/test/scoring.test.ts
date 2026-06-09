import { describe, it } from "vitest";
import { buildDefaultCorpus, buildDefaultQueries } from "../src/defaults.js";
import { BM25Scorer } from "../src/bm25.js";
import { BayesianBM25Scorer } from "../src/bayesian.js";
import { VectorScorer } from "../src/vector.js";
import { HybridScorer } from "../src/hybrid.js";
import { golden, expectClose } from "./_golden.js";

describe("Full scoring matrix parity (BM25 / Bayesian / Vector / Hybrid)", () => {
  const corpus = buildDefaultCorpus();
  const queries = buildDefaultQueries();
  const bm25 = new BM25Scorer(corpus, golden.params.k1, golden.params.b);
  const bayesian = new BayesianBM25Scorer(bm25, golden.params.alpha, golden.params.beta, null);
  const vector = new VectorScorer();
  const hybrid = new HybridScorer(bayesian, vector, golden.params.hybridAlpha);

  for (const scoreSet of golden.scores) {
    const query = queries.find((q) => q.text === scoreSet.query)!;
    const emb = query.embedding!;

    it(`query "${scoreSet.query}" matches every doc score`, () => {
      for (const gd of scoreSet.perDoc) {
        const doc = corpus.getDocument(gd.id)!;
        expectClose(bm25.score(query.terms, doc), gd.bm25, `bm25[${gd.id}]`);
        expectClose(bayesian.score(query.terms, doc), gd.bayesian, `bayesian[${gd.id}]`);
        expectClose(vector.score(emb, doc), gd.vector, `vector[${gd.id}]`);
        expectClose(hybrid.scoreOr(query.terms, emb, doc), gd.hybridOr, `hybridOr[${gd.id}]`);
        expectClose(hybrid.scoreAnd(query.terms, emb, doc), gd.hybridAnd, `hybridAnd[${gd.id}]`);

        for (const gt of gd.terms) {
          expectClose(bm25.scoreTermStandard(gt.term, doc), gt.bm25Term, `bm25Term[${gd.id}][${gt.term}]`);
          expectClose(bayesian.scoreTerm(gt.term, doc), gt.bayesianTerm, `bayesTerm[${gd.id}][${gt.term}]`);
        }
      }
    });
  }
});
