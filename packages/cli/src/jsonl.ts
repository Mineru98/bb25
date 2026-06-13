/** JSONL loaders for docs, queries, and qrels. */
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
  terms: string[] | null;
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
  termsField: string | null = null,
): DocRecord[] {
  const docs: DocRecord[] = [];
  for (const row of loadJsonl(path)) {
    const terms = readTerms(row, termsField);
    const embedding =
      embeddingField !== null && row[embeddingField] != null
        ? (row[embeddingField] as number[]).map(Number)
        : [];
    docs.push({ docId: String(row[idField]), text: String(row[textField]), terms, embedding });
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
    const terms = readTerms(row, termsField);
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

function readTerms(row: Record<string, unknown>, termsField: string | null): string[] | null {
  if (termsField === null || row[termsField] == null) {
    return null;
  }
  const raw = row[termsField];
  if (Array.isArray(raw)) {
    return raw.map(String);
  }
  return String(raw)
    .trim()
    .split(/\s+/)
    .filter((term) => term.length > 0);
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
    const relColumn = parts.length >= 4 ? 3 : 2;
    const rel = Number(parts[relColumn]);
    if (firstLine) {
      firstLine = false;
      if (Number.isNaN(rel)) {
        continue; // header row
      }
    }
    if (Number.isNaN(rel)) {
      throw new Error(`invalid qrels relevance in ${path}: ${line}`);
    }
    add(parts[0] as string, parts.length >= 4 ? (parts[2] as string) : (parts[1] as string), rel);
  }
  return qrels;
}
