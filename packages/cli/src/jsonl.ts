/** JSONL loaders for docs, queries, and qrels. */
import { closeSync, openSync, readSync } from "node:fs";
import { StringDecoder } from "node:string_decoder";

function* loadTextLines(path: string): Generator<string> {
  const fd = openSync(path, "r");
  const decoder = new StringDecoder("utf8");
  const buffer = Buffer.allocUnsafe(1024 * 1024);
  let carry = "";
  try {
    for (;;) {
      const bytes = readSync(fd, buffer, 0, buffer.length, null);
      if (bytes === 0) break;
      const chunk = carry + decoder.write(buffer.subarray(0, bytes));
      const lines = chunk.split("\n");
      carry = lines.pop() ?? "";
      for (const line of lines) {
        yield line;
      }
    }
    const tail = carry + decoder.end();
    if (tail.length > 0) {
      yield tail;
    }
  } finally {
    closeSync(fd);
  }
}

export function* loadJsonl(path: string): Generator<Record<string, unknown>> {
  for (const rawLine of loadTextLines(path)) {
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
  fields: Record<string, string[]> | null;
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
  fieldTermsField: string | null = null,
): DocRecord[] {
  const docs: DocRecord[] = [];
  for (const row of loadJsonl(path)) {
    const terms = readTerms(row, termsField);
    const fields = readFieldTerms(row, fieldTermsField);
    const embedding =
      embeddingField !== null && row[embeddingField] != null
        ? (row[embeddingField] as number[]).map(Number)
        : [];
    docs.push({ docId: String(row[idField]), text: String(row[textField]), terms, fields, embedding });
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
  return readTermValue(row[termsField]);
}

function readTermValue(raw: unknown): string[] {
  if (Array.isArray(raw)) {
    return raw.map(String);
  }
  return String(raw)
    .trim()
    .split(/\s+/)
    .filter((term) => term.length > 0);
}

function readFieldTerms(row: Record<string, unknown>, fieldTermsField: string | null): Record<string, string[]> | null {
  if (fieldTermsField === null || row[fieldTermsField] == null) {
    return null;
  }
  const raw = row[fieldTermsField];
  if (raw === null || typeof raw !== "object" || Array.isArray(raw)) {
    throw new Error(`field terms field "${fieldTermsField}" must be an object`);
  }
  const out: Record<string, string[]> = {};
  for (const [field, value] of Object.entries(raw as Record<string, unknown>)) {
    out[field] = readTermValue(value);
  }
  return out;
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

  let firstLine = true;
  for (const rawLine of loadTextLines(path)) {
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
