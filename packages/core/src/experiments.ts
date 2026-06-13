/**
 * Validation harness (exp1..exp13).
 *
 * The pass/fail predicates and EPSILON comparisons are deterministic. The
 * `details` strings are summaries and are not part of golden parity checks.
 */
import { BayesianBM25Scorer } from "./bayesian.js";
import { BM25Scorer } from "./bm25.js";
import type { Corpus, Document } from "./corpus.js";
import { buildDefaultCorpus, buildDefaultQueries } from "./defaults.js";
import {
  balancedLogOddsFusion,
  cosineToProbability,
  Gating,
  logOddsConjunction,
  probAnd,
  probNot,
  probOr,
} from "./fusion.js";
import { HybridScorer } from "./hybrid.js";
import { EPSILON, safeLog, sigmoid } from "./mathUtils.js";
import { ParameterLearner } from "./parameterLearner.js";
import { VectorScorer } from "./vector.js";

export interface Query {
  text: string;
  terms: string[];
  embedding: number[] | null;
  relevant: string[];
}

export interface ExperimentResult {
  name: string;
  passed: boolean;
  details: string;
}

/** Descending numeric sort. */
function sortByScoreDesc<T>(items: T[], score: (item: T) => number): T[] {
  return items.slice().sort((a, b) => score(b) - score(a));
}

export class ExperimentRunner {
  private readonly corpus: Corpus;
  private readonly queries: Query[];
  private readonly bm25: BM25Scorer;
  private readonly bayesian: BayesianBM25Scorer;
  private readonly vector: VectorScorer;
  private readonly hybrid: HybridScorer;

  constructor(corpus: Corpus, queries: Query[], k1 = 1.2, b = 0.75) {
    this.corpus = corpus;
    this.queries = queries;
    this.bm25 = new BM25Scorer(corpus, k1, b);
    this.bayesian = new BayesianBM25Scorer(this.bm25, 1.0, 0.5, null);
    this.vector = new VectorScorer();
    this.hybrid = new HybridScorer(this.bayesian, this.vector, 0.5);
  }

  runAll(): ExperimentResult[] {
    const experiments: [string, () => [boolean, string]][] = [
      ["1. BM25 Formula Equivalence", () => this.exp1FormulaEquivalence()],
      ["2. Score Calibration", () => this.exp2ScoreCalibration()],
      ["3. Monotonicity Preservation", () => this.exp3Monotonicity()],
      ["4. Prior Bounds", () => this.exp4PriorBounds()],
      ["5. IDF Properties", () => this.exp5IdfProperties()],
      ["6. Hybrid Search Quality", () => this.exp6HybridQuality()],
      ["7. Naive vs RRF vs Bayesian", () => this.exp7MethodComparison()],
      ["8. Log-space Numerical Stability", () => this.exp8NumericalStability()],
      ["9. Parameter Learning Convergence", () => this.exp9ParameterLearning()],
      ["10. Conjunction/Disjunction Bounds", () => this.exp10ConjunctionDisjunction()],
      ["11. Base Rate Prior", () => this.exp11BaseRatePrior()],
      ["12. Log-Odds Conjunction Properties", () => this.exp12LogOddsConjunction()],
      ["13. Fusion Primitives", () => this.exp13FusionPrimitives()],
    ];

    const out: ExperimentResult[] = [];
    for (const [name, func] of experiments) {
      const [passed, details] = func();
      out.push({ name, passed, details });
    }
    return out;
  }

  private exp1FormulaEquivalence(): [boolean, string] {
    let maxDiff = 0.0;
    let comparisons = 0;

    for (const query of this.queries) {
      for (const doc of this.corpus.documents()) {
        for (const term of query.terms) {
          const s1 = this.bm25.scoreTermStandard(term, doc);
          const s2 = this.bm25.scoreTermRewritten(term, doc);
          const diff = Math.abs(s1 - s2);
          if (diff > maxDiff) {
            maxDiff = diff;
          }
          comparisons += 1;
        }
      }
    }

    const passed = maxDiff < 1e-10;
    const details = `max_diff=${maxDiff.toExponential(2)} across ${comparisons} comparisons`;
    return [passed, details];
  }

  private exp2ScoreCalibration(): [boolean, string] {
    let allInRange = true;
    let orderingPreserved = true;
    const violations: string[] = [];

    for (const query of this.queries) {
      const bm25Scores: [string, number][] = [];
      const bayesianScores: [string, number][] = [];

      for (const doc of this.corpus.documents()) {
        const raw = this.bm25.score(query.terms, doc);
        const calibrated = this.bayesian.score(query.terms, doc);

        bm25Scores.push([doc.id, raw]);
        bayesianScores.push([doc.id, calibrated]);

        if (calibrated < -EPSILON || calibrated > 1.0 + EPSILON) {
          allInRange = false;
          violations.push(`doc=${doc.id} calibrated=${calibrated.toFixed(6)}`);
        }
      }

      const sortedBm25 = sortByScoreDesc(bm25Scores, (p) => p[1]);
      const bayesianMap = new Map<string, number>();
      for (const [id, score] of bayesianScores) {
        bayesianMap.set(id, score);
      }

      for (let i = 0; i < Math.max(sortedBm25.length - 1, 0); i++) {
        const a = sortedBm25[i]!;
        const bPair = sortedBm25[i + 1]!;
        const idA = a[0];
        const scoreA = a[1];
        const idB = bPair[0];
        const scoreB = bPair[1];
        if (scoreA > scoreB + EPSILON) {
          const ba = bayesianMap.get(idA) ?? 0.0;
          const bb = bayesianMap.get(idB) ?? 0.0;
          if (ba < bb - EPSILON) {
            orderingPreserved = false;
            violations.push(
              `query=${query.text}: BM25(${idA})=${scoreA.toFixed(4)} > BM25(${idB})=${scoreB.toFixed(4)} but Bayesian(${idA})=${ba.toFixed(6)} < Bayesian(${idB})=${bb.toFixed(6)}`,
            );
          }
        }
      }
    }

    const parts = [`range=[0,1]: ${allInRange}`, `ordering: ${orderingPreserved}`];
    if (violations.length > 0) {
      parts.push(`violations: ${violations.slice(0, 3).join("; ")}`);
    }

    const passed = allInRange && orderingPreserved;
    return [passed, parts.join(", ")];
  }

  private exp3Monotonicity(): [boolean, string] {
    let passed = true;
    const detailsParts: string[] = [];
    let termsTested = 0;

    for (const term of this.corpus.df.keys()) {
      const matchingDocs: Document[] = [];
      for (const doc of this.corpus.documents()) {
        if ((doc.termFreq.get(term) ?? 0) > 0) {
          matchingDocs.push(doc);
        }
      }

      if (matchingDocs.length < 2) {
        continue;
      }

      // Stable ascending sort by term frequency.
      matchingDocs.sort((a, b) => (a.termFreq.get(term) ?? 0) - (b.termFreq.get(term) ?? 0));

      for (let i = 0; i < matchingDocs.length - 1; i++) {
        const d1 = matchingDocs[i]!;
        const d2 = matchingDocs[i + 1]!;
        const tf1 = d1.termFreq.get(term) ?? 0;
        const tf2 = d2.termFreq.get(term) ?? 0;
        if (tf1 === tf2) {
          continue;
        }
        if (Math.abs(d1.length - d2.length) <= 3) {
          const s1 = this.bayesian.scoreTerm(term, d1);
          const s2 = this.bayesian.scoreTerm(term, d2);
          termsTested += 1;
          if (s1 > s2 + EPSILON) {
            passed = false;
            detailsParts.push(
              `term=${term}: tf(${d1.id})=${tf1} > tf(${d2.id})=${tf2} but score ${s1.toFixed(4)} > ${s2.toFixed(4)}`,
            );
          }
        }
      }
    }

    let syntheticPassed = true;
    for (const rawScore of [0.1, 0.5, 1.0, 2.0, 5.0]) {
      for (const prior of [0.2, 0.5, 0.8]) {
        const p1 = this.bayesian.posterior(rawScore, prior);
        const p2 = this.bayesian.posterior(rawScore + 0.1, prior);
        if (p2 < p1 - EPSILON) {
          syntheticPassed = false;
        }
      }
    }

    passed = passed && syntheticPassed;
    let detail = `terms_tested=${termsTested}, synthetic_monotonic=${syntheticPassed}`;
    if (detailsParts.length > 0) {
      detail += `, violations: ${detailsParts.slice(0, 3).join("; ")}`;
    }

    return [passed, detail];
  }

  private exp4PriorBounds(): [boolean, string] {
    let allBounded = true;
    let minPrior = 1.0;
    let maxPrior = 0.0;
    const violations: string[] = [];

    for (const doc of this.corpus.documents()) {
      for (const [term, tf] of doc.termFreq) {
        const prior = this.bayesian.compositePrior(tf, doc.length, this.corpus.avgdl);
        minPrior = Math.min(minPrior, prior);
        maxPrior = Math.max(maxPrior, prior);
        if (prior < 0.1 - EPSILON || prior > 0.9 + EPSILON) {
          allBounded = false;
          violations.push(`doc=${doc.id} term=${term} prior=${prior.toFixed(6)}`);
        }
      }
    }

    let detail = `range=[${minPrior.toFixed(4)}, ${maxPrior.toFixed(4)}]`;
    if (violations.length > 0) {
      detail += `, violations: ${violations.slice(0, 3).join("; ")}`;
    }

    return [allBounded, detail];
  }

  private exp5IdfProperties(): [boolean, string] {
    const allTerms: string[] = [...this.corpus.df.keys()];
    allTerms.sort();

    const idfValues = new Map<string, number>();
    for (const t of allTerms) {
      idfValues.set(t, this.bm25.idf(t));
    }

    let nonNegOk = true;
    for (const term of allTerms) {
      const dfT = this.corpus.df.get(term) ?? 0;
      if (dfT <= this.corpus.n / 2.0) {
        if ((idfValues.get(term) ?? 0.0) < -EPSILON) {
          nonNegOk = false;
        }
      }
    }

    const dfIdfPairs: [number, number][] = allTerms.map((t) => [
      this.corpus.df.get(t) ?? 0,
      idfValues.get(t)!,
    ]);
    // Stable ascending sort by document frequency.
    dfIdfPairs.sort((a, b) => a[0] - b[0]);

    let monotonicOk = true;
    for (let i = 0; i < dfIdfPairs.length - 1; i++) {
      const [df1, idf1] = dfIdfPairs[i]!;
      const [df2, idf2] = dfIdfPairs[i + 1]!;
      if (df1 < df2 && idf1 < idf2 - EPSILON) {
        monotonicOk = false;
      }
    }

    let boundOk = true;
    for (const query of this.queries) {
      for (const term of query.terms) {
        const ub = this.bm25.upperBound(term);
        for (const doc of this.corpus.documents()) {
          const actual = this.bm25.scoreTermStandard(term, doc);
          if (actual > ub + EPSILON) {
            boundOk = false;
          }
        }
      }
    }

    const detail = `non_neg=${nonNegOk}, monotonic=${monotonicOk}, upper_bound=${boundOk}`;
    return [nonNegOk && monotonicOk && boundOk, detail];
  }

  private exp6HybridQuality(): [boolean, string] {
    let passed = true;
    let tests = 0;
    const violations: string[] = [];

    for (const query of this.queries) {
      const embedding = query.embedding;
      if (embedding === null) {
        continue;
      }
      for (const doc of this.corpus.documents()) {
        const bayesianP = this.bayesian.score(query.terms, doc);
        const vectorP = this.vector.score(embedding, doc);
        const probs = [bayesianP, vectorP];

        const andScore = this.hybrid.probabilisticAnd(probs);
        const orScore = this.hybrid.probabilisticOr(probs);

        tests += 1;

        if (andScore > orScore + EPSILON) {
          passed = false;
          violations.push(
            `AND=${andScore.toFixed(6)} > OR=${orScore.toFixed(6)} (doc=${doc.id})`,
          );
        }

        if (bayesianP > 0.5 && vectorP > 0.5) {
          const geoMean = Math.sqrt(bayesianP * vectorP);
          if (andScore < geoMean - EPSILON) {
            passed = false;
            violations.push(
              `no amplification: AND=${andScore.toFixed(6)} < geo_mean=${geoMean.toFixed(6)} (doc=${doc.id})`,
            );
          }
        }

        if (bayesianP < 0.5 && vectorP < 0.5) {
          if (andScore > 0.5 + EPSILON) {
            passed = false;
            violations.push(
              `irrelevance violated: AND=${andScore.toFixed(6)} > 0.5 (doc=${doc.id})`,
            );
          }
        }

        let maxP = Number.NEGATIVE_INFINITY;
        for (const p of probs) {
          maxP = Math.max(maxP, p);
        }
        if (orScore < maxP - EPSILON) {
          passed = false;
          violations.push(
            `OR=${orScore.toFixed(6)} < max=${maxP.toFixed(6)} (doc=${doc.id})`,
          );
        }
      }
    }

    let detail = `tests=${tests}`;
    if (violations.length > 0) {
      detail += `, violations: ${violations.slice(0, 3).join("; ")}`;
    }

    return [passed, detail];
  }

  private exp7MethodComparison(): [boolean, string] {
    const resultsTable: [string, Map<string, string[]>][] = [];

    for (const query of this.queries) {
      const embedding = query.embedding;
      if (embedding === null) {
        continue;
      }
      const docScores: Map<string, number>[] = [];
      const idList: string[] = [];

      for (const doc of this.corpus.documents()) {
        const bm25Raw = this.bm25.score(query.terms, doc);
        const bayesianP = this.bayesian.score(query.terms, doc);
        const vectorP = this.vector.score(embedding, doc);
        const hybridOr = this.hybrid.scoreOr(query.terms, embedding, doc);
        const hybridAnd = this.hybrid.scoreAnd(query.terms, embedding, doc);
        const naive = this.hybrid.naiveSum([bm25Raw, vectorP]);

        const scores = new Map<string, number>();
        scores.set("bm25", bm25Raw);
        scores.set("bayesian", bayesianP);
        scores.set("vector", vectorP);
        scores.set("hybrid_or", hybridOr);
        scores.set("hybrid_and", hybridAnd);
        scores.set("naive", naive);
        scores.set("rrf", 0.0);
        docScores.push(scores);
        idList.push(doc.id);
      }

      const bm25Ranked: [string, number][] = idList.map((id, i) => [
        id,
        docScores[i]!.get("bm25")!,
      ]);
      const sortedBm25 = sortByScoreDesc(bm25Ranked, (p) => p[1]);

      const vectorRanked: [string, number][] = idList.map((id, i) => [
        id,
        docScores[i]!.get("vector")!,
      ]);
      const sortedVector = sortByScoreDesc(vectorRanked, (p) => p[1]);

      const bm25Rank = new Map<string, number>();
      sortedBm25.forEach(([id], i) => bm25Rank.set(id, i + 1));
      const vectorRank = new Map<string, number>();
      sortedVector.forEach(([id], i) => vectorRank.set(id, i + 1));

      for (let idx = 0; idx < idList.length; idx++) {
        const id = idList[idx]!;
        const rrf = this.hybrid.rrfScore([bm25Rank.get(id)!, vectorRank.get(id)!], 60);
        docScores[idx]!.set("rrf", rrf);
      }

      const top5 = new Map<string, string[]>();
      for (const method of ["bm25", "bayesian", "hybrid_or", "hybrid_and", "naive", "rrf"]) {
        const ranked: [string, number][] = idList.map((id, i) => [
          id,
          docScores[i]!.get(method)!,
        ]);
        const sorted = sortByScoreDesc(ranked, (p) => p[1]);
        const ids = sorted.slice(0, 5).map(([id]) => id);
        top5.set(method, ids);
      }

      resultsTable.push([query.text, top5]);
    }

    const detailLines: string[] = [];
    for (const [queryText, top5] of resultsTable) {
      detailLines.push(`query='${queryText}':`);
      for (const method of ["bm25", "bayesian", "hybrid_or", "naive", "rrf"]) {
        const ids = top5.get(method) ?? [];
        detailLines.push(`  ${method} top5: [${ids.map((s) => `"${s}"`).join(", ")}]`);
      }
    }

    return [true, detailLines.join("\n")];
  }

  private exp8NumericalStability(): [boolean, string] {
    let passed = true;
    const tests: string[] = [];

    const extremeProbs = [1e-15, 1e-10, 1e-5, 0.001, 0.5, 0.999, 1.0 - 1e-10];
    for (const p of extremeProbs) {
      const andResult = this.hybrid.probabilisticAnd([p, p]);
      if (andResult < -EPSILON || andResult > 1.0 + EPSILON) {
        passed = false;
      }
      if (Number.isNaN(andResult) || !Number.isFinite(andResult)) {
        passed = false;
      }
      tests.push(`AND(${p.toExponential(2)}, ${p.toExponential(2)})=${andResult.toExponential(2)}`);

      const orResult = this.hybrid.probabilisticOr([p, p]);
      if (orResult < -EPSILON || orResult > 1.0 + EPSILON) {
        passed = false;
      }
      if (Number.isNaN(orResult) || !Number.isFinite(orResult)) {
        passed = false;
      }
      tests.push(`OR(${p.toExponential(2)}, ${p.toExponential(2)})=${orResult.toExponential(2)}`);
    }

    for (const x of [-700.0, -100.0, -1.0, 0.0, 1.0, 100.0, 700.0]) {
      const s = sigmoid(x);
      if (s < 0.0 || s > 1.0 || Number.isNaN(s) || !Number.isFinite(s)) {
        passed = false;
      }
      tests.push(`sigmoid(${x.toFixed(0)})=${s.toFixed(6)}`);
    }

    for (const p of [0.0, 1e-300, 1e-15, 0.5, 1.0]) {
      const result = safeLog(p);
      if (Number.isNaN(result) || !Number.isFinite(result)) {
        passed = false;
      }
      tests.push(`safe_log(${p.toExponential(2)})=${result.toFixed(2)}`);
    }

    const preview = tests.slice(0, 10).join("; ");
    const detail = `${preview} ... (${tests.length} total tests)`;
    return [passed, detail];
  }

  private exp9ParameterLearning(): [boolean, string] {
    const query = this.queries[0]!;
    const relevantIds = new Set<string>(query.relevant);

    const scores: number[] = [];
    const labels: number[] = [];
    for (const doc of this.corpus.documents()) {
      const rawScore = this.bm25.score(query.terms, doc);
      scores.push(rawScore);
      labels.push(relevantIds.has(doc.id) ? 1.0 : 0.0);
    }

    const learner = new ParameterLearner(0.1, 500, 1e-8);
    const result = learner.learn(scores, labels);

    const lossHistory = result.lossHistory;
    const last = lossHistory.length > 0 ? lossHistory[lossHistory.length - 1]! : 0.0;
    const lossDecreased = last < lossHistory[0]!;
    const alphaPositive = result.alpha > 0.0;
    let decreasingSteps = 0;
    for (let i = 0; i < lossHistory.length - 1; i++) {
      if (lossHistory[i + 1]! <= lossHistory[i]! + EPSILON) {
        decreasingSteps += 1;
      }
    }
    const mostlyDecreasing =
      decreasingSteps >= Math.floor(((lossHistory.length - 1) * 8) / 10);

    const passed = lossDecreased && alphaPositive && mostlyDecreasing;
    const detail = `alpha=${result.alpha.toFixed(4)}, beta=${result.beta.toFixed(4)}, initial_loss=${lossHistory[0]!.toFixed(4)}, final_loss=${last.toFixed(4)}, decreasing_steps=${decreasingSteps}/${Math.max(lossHistory.length - 1, 0)}, converged=${result.converged}`;

    return [passed, detail];
  }

  private exp10ConjunctionDisjunction(): [boolean, string] {
    let passed = true;
    let tests = 0;
    const violations: string[] = [];

    const testProbs = [0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99];
    for (const p1 of testProbs) {
      for (const p2 of testProbs) {
        const probs = [p1, p2];
        const andResult = this.hybrid.probabilisticAnd(probs);
        const orResult = this.hybrid.probabilisticOr(probs);
        tests += 1;

        if (andResult > orResult + EPSILON) {
          passed = false;
          violations.push(
            `AND(${p1.toFixed(2)},${p2.toFixed(2)})=${andResult.toFixed(6)} > OR=${orResult.toFixed(6)}`,
          );
        }

        if (p1 > 0.5 && p2 > 0.5) {
          const geoMean = Math.sqrt(p1 * p2);
          if (andResult < geoMean - EPSILON) {
            passed = false;
            violations.push(
              `AND(${p1.toFixed(2)},${p2.toFixed(2)})=${andResult.toFixed(6)} < geo_mean=${geoMean.toFixed(6)}`,
            );
          }
        }

        if (p1 < 0.5 && p2 < 0.5) {
          if (andResult > 0.5 + EPSILON) {
            passed = false;
            violations.push(
              `AND(${p1.toFixed(2)},${p2.toFixed(2)})=${andResult.toFixed(6)} > 0.5`,
            );
          }
        }

        const maxP = Math.max(p1, p2);
        if (orResult < maxP - EPSILON) {
          passed = false;
          violations.push(
            `OR(${p1.toFixed(2)},${p2.toFixed(2)})=${orResult.toFixed(6)} < max=${maxP.toFixed(2)}`,
          );
        }
      }
    }

    for (const p of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      const andSingle = this.hybrid.probabilisticAnd([p]);
      tests += 1;
      if (Math.abs(andSingle - p) > EPSILON) {
        passed = false;
        violations.push(`identity: AND(${p.toFixed(1)})=${andSingle.toFixed(6)} != ${p.toFixed(1)}`);
      }
    }

    for (const p1 of [0.1, 0.5, 0.9]) {
      for (const p2 of [0.2, 0.6]) {
        for (const p3 of [0.3, 0.8]) {
          const probs = [p1, p2, p3];
          const andResult = this.hybrid.probabilisticAnd(probs);
          const orResult = this.hybrid.probabilisticOr(probs);
          tests += 1;

          if (andResult > orResult + EPSILON) {
            passed = false;
          }

          let maxP = Number.NEGATIVE_INFINITY;
          for (const p of probs) {
            maxP = Math.max(maxP, p);
          }
          if (orResult < maxP - EPSILON) {
            passed = false;
          }

          if (probs.every((p) => p > 0.5)) {
            const geoMean = Math.pow(p1 * p2 * p3, 1.0 / 3.0);
            if (andResult < geoMean - EPSILON) {
              passed = false;
            }
          }
        }
      }
    }

    let detail = `tests=${tests}`;
    if (violations.length > 0) {
      detail += `, violations: ${violations.slice(0, 3).join("; ")}`;
    }

    return [passed, detail];
  }

  private exp11BaseRatePrior(): [boolean, string] {
    let passed = true;
    const details: string[] = [];

    const query = this.queries[0]!;

    const scorerNone = new BayesianBM25Scorer(this.bm25, 1.0, 0.5, null);
    const scorerLow = new BayesianBM25Scorer(this.bm25, 1.0, 0.5, 0.01);
    const scorerNeutral = new BayesianBM25Scorer(this.bm25, 1.0, 0.5, 0.5);

    let lowReduces = true;
    let neutralOk = true;
    let allInRange = true;

    for (const doc of this.corpus.documents()) {
      const scoreNone = scorerNone.score(query.terms, doc);
      const scoreLow = scorerLow.score(query.terms, doc);
      const scoreNeutral = scorerNeutral.score(query.terms, doc);

      if (scoreNone > EPSILON && scoreLow > scoreNone + EPSILON) {
        lowReduces = false;
        details.push(`doc=${doc.id}: low=${scoreLow.toFixed(6)} > none=${scoreNone.toFixed(6)}`);
      }

      if (Math.abs(scoreNeutral - scoreNone) > 1e-4) {
        neutralOk = false;
        details.push(`doc=${doc.id}: neutral=${scoreNeutral.toFixed(6)} != none=${scoreNone.toFixed(6)}`);
      }

      for (const s of [scoreNone, scoreLow, scoreNeutral]) {
        if (s < -EPSILON || s > 1.0 + EPSILON) {
          allInRange = false;
        }
      }
    }

    if (!lowReduces) {
      passed = false;
    }
    if (!neutralOk) {
      passed = false;
    }
    if (!allInRange) {
      passed = false;
    }

    let detail = `low_reduces=${lowReduces}, neutral_ok=${neutralOk}, all_in_range=${allInRange}`;
    if (details.length > 0) {
      detail += `, violations: ${details.slice(0, 3).join("; ")}`;
    }

    return [passed, detail];
  }

  private exp12LogOddsConjunction(): [boolean, string] {
    let passed = true;
    const violations: string[] = [];

    let result = logOddsConjunction([0.9, 0.9], null, null, Gating.None);
    if (result <= 0.9) {
      passed = false;
      violations.push(`agreement: conj([0.9,0.9])=${result.toFixed(6)} <= 0.9`);
    }

    result = logOddsConjunction([0.9, 0.1], null, null, Gating.None);
    if (Math.abs(result - 0.5) > 0.15) {
      passed = false;
      violations.push(`disagreement: conj([0.9,0.1])=${result.toFixed(6)} not near 0.5`);
    }

    result = logOddsConjunction([0.5, 0.5], null, null, Gating.None);
    if (Math.abs(result - 0.5) > EPSILON) {
      passed = false;
      violations.push(`neutral: conj([0.5,0.5])=${result.toFixed(6)} != 0.5`);
    }

    const conj2 = logOddsConjunction([0.8, 0.8], null, null, Gating.None);
    const conj3 = logOddsConjunction([0.8, 0.8, 0.8], null, null, Gating.None);
    if (conj3 <= conj2) {
      passed = false;
      violations.push(
        `amplification: conj([0.8]*3)=${conj3.toFixed(6)} <= conj([0.8]*2)=${conj2.toFixed(6)}`,
      );
    }

    const lowAlpha = logOddsConjunction([0.8, 0.8], 0.3, null, Gating.None);
    const highAlpha = logOddsConjunction([0.8, 0.8], 0.8, null, Gating.None);
    if (highAlpha <= lowAlpha) {
      passed = false;
      violations.push(`alpha: high_alpha=${highAlpha.toFixed(6)} <= low_alpha=${lowAlpha.toFixed(6)}`);
    }

    const uniformW = [0.5, 0.5];
    const probs = [0.7, 0.8];
    const weighted = logOddsConjunction(probs, 0.0, uniformW, Gating.None);
    const unweighted = logOddsConjunction(probs, 0.0, null, Gating.None);
    if (Math.abs(weighted - unweighted) > 1e-6) {
      passed = false;
      violations.push(
        `uniform_weights: weighted=${weighted.toFixed(6)} != unweighted=${unweighted.toFixed(6)}`,
      );
    }

    const detail =
      violations.length === 0
        ? "all properties verified"
        : `violations: ${violations.slice(0, 3).join("; ")}`;

    return [passed, detail];
  }

  private exp13FusionPrimitives(): [boolean, string] {
    let passed = true;
    const violations: string[] = [];

    for (const p of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      const roundtrip = probNot(probNot(p));
      if (Math.abs(roundtrip - p) > 1e-8) {
        passed = false;
        violations.push(`involution: not(not(${p.toFixed(1)}))=${roundtrip.toFixed(6)} != ${p.toFixed(1)}`);
      }
    }

    const notLow = probNot(0.01);
    const notHigh = probNot(0.99);
    if (notLow < 0.0 || notLow > 1.0 || notHigh < 0.0 || notHigh > 1.0) {
      passed = false;
      violations.push("prob_not out of bounds");
    }

    for (const [p1, p2] of [
      [0.3, 0.7],
      [0.1, 0.9],
      [0.5, 0.5],
    ] as [number, number][]) {
      const lhs = probNot(probAnd([p1, p2]));
      const rhs = probOr([probNot(p1), probNot(p2)]);
      if (Math.abs(lhs - rhs) > 1e-8) {
        passed = false;
        violations.push(
          `de_morgan: not(and(${p1.toFixed(1)},${p2.toFixed(1)}))=${lhs.toFixed(6)} != or(not,not)=${rhs.toFixed(6)}`,
        );
      }
    }

    const sparse = [0.3, 0.6, 0.8];
    const dense = [0.1, 0.5, 0.9];
    const fused = balancedLogOddsFusion(sparse, dense, 0.5);
    if (fused.length !== sparse.length) {
      passed = false;
      violations.push(`fusion dim: ${fused.length} != ${sparse.length}`);
    }

    const fusedDense = balancedLogOddsFusion(sparse, dense, 1.0);
    const fusedSparse = balancedLogOddsFusion(sparse, dense, 0.0);
    if (vecEq(fusedDense, fusedSparse)) {
      passed = false;
      violations.push("weight has no effect");
    }

    for (const s of [-1.0, -0.5, 0.0, 0.5, 1.0]) {
      const p = cosineToProbability(s);
      if (p <= 0.0 || p >= 1.0) {
        passed = false;
        violations.push(`cos_to_prob(${s.toFixed(1)})=${p.toFixed(6)} out of (0,1)`);
      }
    }

    let prev = cosineToProbability(-1.0);
    for (const s of [-0.5, 0.0, 0.5, 1.0]) {
      const curr = cosineToProbability(s);
      if (curr < prev - EPSILON) {
        passed = false;
        violations.push(`cos_to_prob not monotonic at ${s.toFixed(1)}`);
      }
      prev = curr;
    }

    const detail =
      violations.length === 0
        ? "all primitives verified"
        : `violations: ${violations.slice(0, 3).join("; ")}`;

    return [passed, detail];
  }
}

/** Element-wise equality for numeric vectors. */
function vecEq(a: number[], b: number[]): boolean {
  if (a.length !== b.length) {
    return false;
  }
  for (let i = 0; i < a.length; i++) {
    if (a[i] !== b[i]) {
      return false;
    }
  }
  return true;
}

export function runExperiments(): ExperimentResult[] {
  const corpus = buildDefaultCorpus();
  const queries = buildDefaultQueries();
  return new ExperimentRunner(corpus, queries, 1.2, 0.75).runAll();
}
