import { describe, it, expect } from "vitest";
import { Tokenizer } from "../src/tokenizer.js";
import { golden } from "./_golden.js";

describe("Tokenizer parity (§14 Q1)", () => {
  const tok = new Tokenizer();

  for (const c of golden.tokenizer) {
    it(`tokenizes ${JSON.stringify(c.input)}`, () => {
      expect(tok.tokenize(c.input)).toEqual(c.tokens);
    });
  }

  it("lowercases ASCII only and drops non-ASCII (café -> caf)", () => {
    expect(tok.tokenize("café")).toEqual(["caf"]);
  });

  it("treats hangul as a delimiter", () => {
    expect(tok.tokenize("한글 text")).toEqual(["text"]);
  });

  it("splits on punctuation (TF-IDF -> tf, idf)", () => {
    expect(tok.tokenize("TF-IDF")).toEqual(["tf", "idf"]);
  });

  it("returns [] for empty input", () => {
    expect(tok.tokenize("")).toEqual([]);
  });
});
