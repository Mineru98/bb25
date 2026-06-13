/**
 * BM25 scorer.
 */
import type { Corpus, Document } from "./corpus.js";

export class BM25Scorer {
  private readonly corpus: Corpus;
  private readonly k1: number;
  private readonly b: number;

  constructor(corpus: Corpus, k1 = 1.2, b = 0.75) {
    this.corpus = corpus;
    this.k1 = k1;
    this.b = b;
  }

  idf(term: string): number {
    const n = this.corpus.n;
    const dfT = this.corpus.df.get(term) ?? 0;
    return Math.log((n - dfT + 0.5) / (dfT + 0.5));
  }

  private lengthNorm(doc: Document): number {
    return 1.0 - this.b + (this.b * doc.length) / this.corpus.avgdl;
  }

  scoreTermStandard(term: string, doc: Document): number {
    const tf = doc.termFreq.get(term) ?? 0;
    if (tf === 0) {
      return 0.0;
    }
    const norm = this.lengthNorm(doc);
    const idfVal = this.idf(term);
    return (idfVal * (this.k1 + 1.0) * tf) / (this.k1 * norm + tf);
  }

  scoreTermRewritten(term: string, doc: Document): number {
    const tf = doc.termFreq.get(term) ?? 0;
    if (tf === 0) {
      return 0.0;
    }
    const norm = this.lengthNorm(doc);
    const boost = ((this.k1 + 1.0) * tf) / (this.k1 * norm + tf);
    const idfVal = this.idf(term);
    return idfVal * boost;
  }

  score(queryTerms: string[], doc: Document): number {
    // Reference: query_terms.iter().map(score_term_standard).sum() — left to right.
    let sum = 0.0;
    for (const term of queryTerms) {
      sum += this.scoreTermStandard(term, doc);
    }
    return sum;
  }

  upperBound(term: string): number {
    const idfVal = this.idf(term);
    if (idfVal <= 0.0) {
      return 0.0;
    }
    return (this.k1 + 1.0) * idfVal;
  }

  avgdl(): number {
    return this.corpus.avgdl;
  }
}
