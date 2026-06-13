import { describe, it, expect } from "vitest";
import { readFileSync, readdirSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const here = dirname(fileURLToPath(import.meta.url));
const pkg = JSON.parse(
  readFileSync(resolve(here, "../package.json"), "utf8"),
) as { dependencies?: Record<string, string>; sideEffects?: boolean };

describe("@bb25/core packaging invariants", () => {
  it("has zero runtime dependencies (core must not pull in ONNX/fs/etc.)", () => {
    const deps = Object.keys(pkg.dependencies ?? {});
    expect(deps).toEqual([]);
  });

  it("is marked side-effect free for tree-shaking", () => {
    expect(pkg.sideEffects).toBe(false);
  });

  it("source imports nothing outside @bb25/core (no node:/transformers/onnx/fs)", () => {
    const srcDir = resolve(here, "../src");
    const forbidden = /from\s+["'](node:|@huggingface\/|onnxruntime|fs|path|url)["']/;
    const offenders: string[] = [];
    for (const file of readdirSync(srcDir)) {
      if (!file.endsWith(".ts")) continue;
      const content = readFileSync(resolve(srcDir, file), "utf8");
      if (forbidden.test(content)) {
        offenders.push(file);
      }
    }
    expect(offenders).toEqual([]);
  });
});
