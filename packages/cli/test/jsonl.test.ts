import { describe, expect, it } from "vitest";
import { mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { loadDocs, loadQrels, loadQueries } from "../src/jsonl.js";

function tempFile(name: string, content: string): string {
  const dir = mkdtempSync(join(tmpdir(), "bb25-jsonl-"));
  const path = join(dir, name);
  writeFileSync(path, content, "utf8");
  return path;
}

describe("jsonl loaders", () => {
  it("loads pretokenized doc and query terms from arrays or whitespace strings", () => {
    const docsPath = tempFile(
      "docs.jsonl",
      [
        JSON.stringify({ doc_id: "d1", text: "Original Text", terms: ["orig", "text"] }),
        JSON.stringify({ doc_id: "d2", text: "Other Text", terms: "other text" }),
      ].join("\n") + "\n",
    );
    const queriesPath = tempFile(
      "queries.jsonl",
      JSON.stringify({ query_id: "q1", text: "Original?", query_terms: ["orig"] }) + "\n",
    );

    const docs = loadDocs(docsPath, "doc_id", "text", null, "terms");
    const queries = loadQueries(queriesPath, "query_id", "text", "query_terms", null);

    expect(docs[0]!.terms).toEqual(["orig", "text"]);
    expect(docs[1]!.terms).toEqual(["other", "text"]);
    expect(queries[0]!.terms).toEqual(["orig"]);
  });

  it("loads both three-column and four-column qrels", () => {
    const path = tempFile(
      "qrels.tsv",
      ["query_id\tQ0\tdoc_id\trelevance", "q1\t0\td1\t2", "q1\td2\t1"].join("\n") + "\n",
    );
    const qrels = loadQrels(path);

    expect(qrels.get("q1")!.get("d1")).toBe(2);
    expect(qrels.get("q1")!.get("d2")).toBe(1);
  });
});
