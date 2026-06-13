import { BM25Scorer, type BM25Method } from "./bm25.js";
import { BayesianBM25Scorer } from "./bayesian.js";
import { Corpus } from "./corpus.js";
import { logOddsConjunction } from "./fusion.js";

export interface MultiFieldDocument {
  id: string;
  fields: Record<string, string[]>;
}

export interface MultiFieldScorerOptions {
  fields: string[];
  fieldWeights?: Record<string, number>;
  alpha?: number;
  beta?: number;
  fusionAlpha?: number;
  baseRate?: number | null;
  k1?: number;
  b?: number;
  method?: BM25Method;
}

export class MultiFieldScorer {
  private readonly docIds: string[];
  private readonly fields: string[];
  private readonly weights: number[];
  private readonly alpha: number;
  private readonly scorers = new Map<string, BayesianBM25Scorer>();
  private readonly corpora = new Map<string, Corpus>();
  private readonly docById = new Map<string, number>();

  constructor(documents: MultiFieldDocument[], options: MultiFieldScorerOptions) {
    if (options.fields.length === 0) {
      throw new Error("fields must be a non-empty list");
    }
    if (new Set(options.fields).size !== options.fields.length) {
      throw new Error("fields must not contain duplicates");
    }

    this.fields = options.fields.slice();
    this.weights = resolveFieldWeights(this.fields, options.fieldWeights);
    this.alpha = options.fusionAlpha ?? 0.5;
    this.docIds = documents.map((doc) => doc.id);
    this.docIds.forEach((id, idx) => this.docById.set(id, idx));

    for (const field of this.fields) {
      const corpus = new Corpus();
      for (const doc of documents) {
        const tokens = doc.fields[field];
        if (tokens === undefined) {
          throw new Error(`document ${doc.id} missing field "${field}"`);
        }
        corpus.addDocumentTokens(doc.id, "", tokens);
      }
      corpus.buildIndex();
      this.corpora.set(field, corpus);
      const bm25 = new BM25Scorer(corpus, options.k1 ?? 1.2, options.b ?? 0.75, options.method ?? "robertson");
      this.scorers.set(
        field,
        new BayesianBM25Scorer(bm25, options.alpha ?? 1.0, options.beta ?? 0.5, options.baseRate ?? null),
      );
    }
  }

  documentIds(): string[] {
    return this.docIds.slice();
  }

  score(queryTerms: string[], docId: string): number {
    const idx = this.docById.get(docId);
    if (idx === undefined) {
      throw new Error(`unknown document id "${docId}"`);
    }

    const probs: number[] = [];
    for (const field of this.fields) {
      const scorer = this.scorers.get(field)!;
      const fieldDoc = this.corpora.get(field)!.getDocument(this.docIds[idx]!);
      if (fieldDoc === undefined) {
        throw new Error(`unknown field document id "${this.docIds[idx]!}"`);
      }
      probs.push(scorer.score(queryTerms, fieldDoc));
    }
    return logOddsConjunction(probs, this.alpha, this.weights);
  }

  scores(queryTerms: string[]): [string, number][] {
    return this.docIds.map((docId) => [docId, this.score(queryTerms, docId)]);
  }
}

function resolveFieldWeights(fields: string[], fieldWeights: Record<string, number> | undefined): number[] {
  if (fieldWeights === undefined) {
    return fields.map(() => 1.0 / fields.length);
  }

  const weights = fields.map((field) => {
    const weight = fieldWeights[field];
    if (weight === undefined) {
      throw new Error(`fieldWeights missing key "${field}"`);
    }
    if (weight < 0.0) {
      throw new Error(`field weight for "${field}" must be non-negative`);
    }
    return weight;
  });
  const sum = weights.reduce((acc, weight) => acc + weight, 0.0);
  if (Math.abs(sum - 1.0) > 1e-6) {
    throw new Error(`fieldWeights must sum to 1.0, got ${sum}`);
  }
  return weights;
}
