/**
 * index.json schema + serialization helpers.
 */
import { readFileSync, writeFileSync } from "node:fs";
import { Corpus } from "@bb25/core";
import type { BM25Method } from "@bb25/core";

export interface IndexParams {
  k1: number;
  b: number;
  bm25Method: BM25Method;
  alpha: number;
  beta: number;
  baseRate: number | null;
  hybridAlpha: number;
}

export interface EmbedderMeta {
  model: string;
  dim: number;
  dtype: string;
  pooling: string;
  normalize: boolean;
  cacheDir?: string;
  localOnly?: boolean;
}

export interface IndexDoc {
  id: string;
  text: string;
  terms?: string[] | null;
  embedding: number[] | null;
}

export interface IndexFile {
  version: 1;
  params: IndexParams;
  embedder: EmbedderMeta | null;
  documents: IndexDoc[];
  stats: { n: number; avgdl: number };
}

export const DEFAULT_PARAMS: IndexParams = {
  k1: 1.2,
  b: 0.75,
  bm25Method: "robertson",
  alpha: 1.0,
  beta: 0.5,
  baseRate: null,
  hybridAlpha: 0.5,
};

export function saveIndex(path: string, index: IndexFile): void {
  writeFileSync(path, JSON.stringify(index), "utf8");
}

export function loadIndex(path: string): IndexFile {
  const raw = JSON.parse(readFileSync(path, "utf8")) as IndexFile;
  if (raw.version !== 1) {
    throw new Error(`unsupported index version: ${(raw as { version: unknown }).version}`);
  }
  return raw;
}

/** Rebuild a Corpus (df/avgdl recomputed via buildIndex) from an IndexFile. */
export function corpusFromIndex(index: IndexFile): Corpus {
  const corpus = new Corpus();
  for (const doc of index.documents) {
    if (doc.terms !== undefined && doc.terms !== null) {
      corpus.addDocumentTokens(doc.id, doc.text, doc.terms, doc.embedding ?? []);
    } else {
      corpus.addDocument(doc.id, doc.text, doc.embedding ?? []);
    }
  }
  corpus.buildIndex();
  return corpus;
}
