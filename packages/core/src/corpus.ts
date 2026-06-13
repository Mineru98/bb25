/**
 * Corpus and Document. Direct port of `src/corpus.rs`.
 */
import { Tokenizer } from "./tokenizer.js";

/** An indexed document. Mirrors the reference `Document` struct. */
export interface Document {
  id: string;
  text: string;
  /** f64 embedding. May be empty when no embedding was supplied. */
  embedding: number[];
  tokens: string[];
  length: number;
  termFreq: Map<string, number>;
}

export interface CorpusStats {
  numDocs: number;
  avgDocLength: number;
}

export class Corpus {
  private readonly tokenizer: Tokenizer;
  private readonly docs: Document[] = [];
  private readonly docById = new Map<string, number>();

  /** Number of documents (reference: `n`). Valid after `buildIndex()`. */
  n = 0;
  /** Average document length (reference: `avgdl`). */
  avgdl = 0.0;
  /** Document frequency per term (reference: `df`). */
  df = new Map<string, number>();

  constructor(tokenizer: Tokenizer = new Tokenizer()) {
    this.tokenizer = tokenizer;
  }

  addDocument(docId: string, text: string, embedding: number[] = []): void {
    const tokens = this.tokenizer.tokenize(text);
    const termFreq = new Map<string, number>();
    for (const token of tokens) {
      termFreq.set(token, (termFreq.get(token) ?? 0) + 1);
    }

    const doc: Document = {
      id: docId,
      text,
      embedding,
      length: tokens.length,
      tokens,
      termFreq,
    };

    const idx = this.docs.length;
    this.docs.push(doc);
    this.docById.set(docId, idx);
  }

  buildIndex(): void {
    this.n = this.docs.length;
    this.df = new Map<string, number>();
    let totalLength = 0;

    for (const doc of this.docs) {
      totalLength += doc.length;
      for (const term of doc.termFreq.keys()) {
        this.df.set(term, (this.df.get(term) ?? 0) + 1);
      }
    }

    this.avgdl = this.n > 0 ? totalLength / this.n : 0.0;
  }

  getDocument(docId: string): Document | undefined {
    const idx = this.docById.get(docId);
    if (idx === undefined) {
      return undefined;
    }
    return this.docs[idx];
  }

  documents(): readonly Document[] {
    return this.docs;
  }

  get stats(): CorpusStats {
    return { numDocs: this.n, avgDocLength: this.avgdl };
  }
}
