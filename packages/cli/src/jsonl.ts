/** JSONL loaders (docs / queries / qrels), mirroring benchmarks/run_benchmark.py. */
import { readFileSync } from "node:fs";

export function* loadJsonl(path: string): Generator<Record<string, unknown>> {
  const content = readFileSync(path, "utf8");
  for (const rawLine of content.split("\n")) {
    const line = rawLine.trim();
    if (line === "" || line.startsWith("#")) {
      continue;
    }
    yield JSON.parse(line) as Record<string, unknown>;
  }
}

export interface DocRecord {
  docId: string;
  text: string;
  embedding: number[];
}

export interface QueryRecord {
  queryId: string;
  text: string;
  terms: string[] | null;
  embedding: number[] | null;
}

export function loadDocs(
  path: string,
  idField = "doc_id",
  textField = "text",
  embeddingField: string | null = null,
): DocRecord[] {
  const docs: DocRecord[] = [];
  for (const row of loadJsonl(path)) {
    const embedding =
      embeddingField !== null && row[embeddingField] != null
        ? (row[embeddingField] as number[]).map(Number)
        : [];
    docs.push({ docId: String(row[idField]), text: String(row[textField]), embedding });
  }
  return docs;
}

export function loadQueries(
  path: string,
  idField = "query_id",
  textField = "text",
  termsField: string | null = null,
  embeddingField: string | null = null,
): QueryRecord[] {
  const queries: QueryRecord[] = [];
  for (const row of loadJsonl(path)) {
    const terms =
      termsField !== null && row[termsField] != null
        ? (row[termsField] as unknown[]).map(String)
        : null;
    const embedding =
      embeddingField !== null && row[embeddingField] != null
        ? (row[embeddingField] as number[]).map(Number)
        : null;
    queries.push({
      queryId: String(row[idField]),
      text: String(row[textField]),
      terms,
      embedding,
    });
  }
  return queries;
}

/** qrels: {qid: {docId: relevance}}. Supports .jsonl and TSV/whitespace. */
export function loadQrels(path: string): Map<string, Map<string, number>> {
  const qrels = new Map<string, Map<string, number>>();
  const add = (qid: string, did: string, rel: number): void => {
    let m = qrels.get(qid);
    if (m === undefined) {
      m = new Map();
      qrels.set(qid, m);
    }
    m.set(did, rel);
  };

  if (path.endsWith(".jsonl")) {
    for (const row of loadJsonl(path)) {
      add(String(row["query_id"]), String(row["doc_id"]), Number(row["relevance"] ?? 1.0));
    }
    return qrels;
  }

  const content = readFileSync(path, "utf8");
  let firstLine = true;
  for (const rawLine of content.split("\n")) {
    const line = rawLine.trim();
    if (line === "" || line.startsWith("#")) {
      continue;
    }
    const parts = line.includes("\t") ? line.split("\t") : line.split(/\s+/);
    if (parts.length < 3) {
      continue;
    }
    if (firstLine) {
      firstLine = false;
      if (Number.isNaN(Number(parts[2]))) {
        continue; // header row
      }
    }
    add(parts[0] as string, parts[1] as string, Number(parts[2]));
  }
  return qrels;
}
