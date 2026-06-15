# bb25

TypeScript-native BM25 and Bayesian retrieval toolkit.

The workspace is a pnpm monorepo with a dependency-free core package, an optional
embeddings package backed by transformers.js, and a CLI for indexing, searching,
warmup, and benchmark runs.

## Origin and Credits

bb25 began as a port and experimental validation of the Bayesian BM25 work by
Jaepil Jeong (Cognica), based on the original Python reference implementation:
[cognica-io/bayesian-bm25](https://github.com/cognica-io/bayesian-bm25).
Earlier project history also identifies the Rust + Python bindings
implementation as [instructkr/bb25](https://github.com/instructkr/bb25).

The project has since been migrated into this TypeScript-native pnpm monorepo,
but the core Bayesian BM25 ideas, probability calibration approach, and hybrid
retrieval framing should be credited to the original Bayesian BM25 paper and
reference implementation, with implementation lineage from instructkr/bb25.

## Packages

| Package | Description |
| --- | --- |
| [`@bb25/core`](packages/core) | Tokenizer, Corpus, BM25, Bayesian BM25, vector and hybrid scorers, fusion, metrics, calibration, learnable weights, attention weights, block-max indexing, and experiment helpers. |
| [`@bb25/embeddings`](packages/embeddings) | BGE-M3 dense embeddings via `@huggingface/transformers`. |
| [`@bb25/cli`](packages/cli) | `bb25` CLI: `index`, `search`, `warmup`, and `bench`. |

`@bb25/core` has no runtime dependencies and does not import filesystem,
transformers.js, or ONNX APIs. It works with strings and numeric vectors only.

## Quick Start

```bash
corepack enable pnpm
pnpm install
pnpm -r build
pnpm -r test
```

## Library

```ts
import {
  BayesianBM25Scorer,
  BM25Scorer,
  Corpus,
  HybridScorer,
  VectorScorer,
} from "@bb25/core";

const corpus = new Corpus();
corpus.addDocument("d1", "machine learning from data", [0.1, 0.2, 0.3]);
corpus.buildIndex();

const bm25 = new BM25Scorer(corpus, 1.2, 0.75);
const bayes = new BayesianBM25Scorer(bm25, 1.0, 0.5);
const hybrid = new HybridScorer(bayes, new VectorScorer(), 0.5);

const doc = corpus.getDocument("d1")!;
const p = hybrid.scoreOr(["machine", "learning"], [0.1, 0.2, 0.3], doc);
console.log(p);
```

## CLI

```bash
bb25 index corpus.jsonl -o index.json
bb25 search "bayesian retrieval" --index index.json --top-k 10
bb25 bench --docs docs.jsonl --queries queries.jsonl --qrels qrels.tsv --embed
```

`corpus.jsonl` and `docs.jsonl` use one JSON object per line:

```json
{"doc_id":"d1","text":"machine learning from data","embedding":[0.1,0.2,0.3]}
```

`queries.jsonl` accepts `query_id`, `text`, optional `terms`, and optional
`embedding`. `qrels` can be TSV (`query_id<TAB>doc_id<TAB>relevance`) or JSONL.

## SQuAD Slice

```bash
corepack pnpm -r build
corepack pnpm bench:squad-smoke -- \
  --regenerated-embeddings \
  --embedding-cache-dir /tmp/bb25-embedding-cache/bge-m3-q8
# After the cache is populated, add:
#   --require-embedding-cache --hash-embedding-cache --embedding-local-only
```

## Benchmark Gates

Gate tiers:

- **PR/local**: `corepack pnpm typecheck`, `corepack pnpm test`, and
  `corepack pnpm bench:synthetic-smoke -- --out /tmp/bb25-synthetic-smoke.json --manifest-out /tmp/bb25-synthetic-smoke-manifest.json`.
  These are fixture/local checks; use explicit temporary smoke outputs in PR
  checks so `reference-results/**` is not rewritten. There is currently no
  `lint` package script, so lint evidence is **N/A** until a lint command is
  added.
- **Nightly/manual**: Python environment setup, BEIR JSONL preparation, sparse
  parity, baseline parity, and `bench:audit-readiness -- --profile all`.
- **Release/public claim**: strict dataset-level readiness plus Bayesian hybrid
  claim gate. Public numbers must point to result JSON, manifest JSON, command
  return codes, and input/artifact hashes. Treat any `reference-results/**`
  update as an explicit artifact-refresh change, not incidental verification
  output.

```bash
corepack pnpm bench:synthetic-smoke -- \
  --out /tmp/bb25-synthetic-smoke.json \
  --manifest-out /tmp/bb25-synthetic-smoke-manifest.json
corepack pnpm bench:setup-env -- \
  --python python3.12 \
  --venv .venv-bench \
  --require
corepack pnpm bench:sparse-parity -- \
  --python .venv-bench/bin/python \
  --root /tmp/beir-jsonl-sparse \
  --reference-ranking reference-results/python/sparse-benchmark.json \
  --reference-calibration reference-results/python/base-rate.json
corepack pnpm bench:baseline-parity -- \
  --python .venv-bench/bin/python \
  --root /tmp/beir-jsonl \
  --reference reference-results/python/hybrid-beir.json
corepack pnpm bench:audit-readiness -- \
  --profile all \
  --out reference-results/manifests/readiness-audit.json
# Release/public-claim parity is stricter: it enforces the same threshold per
# dataset instead of only on the five-dataset method average. The hybrid claim
# gate consumes this strict baseline-parity result and rejects missing metric
# provenance or a non-strict dataset gate.
corepack pnpm bench:baseline-parity -- \
  --python .venv-bench/bin/python \
  --root /tmp/beir-jsonl \
  --reference reference-results/python/hybrid-beir.json \
  --datasets arguana,fiqa,nfcorpus,scidocs,scifact \
  --methods BM25,Dense,Convex,RRF \
  --metric ndcg@10 \
  --tolerance-points 0.50 \
  --dataset-gate strict \
  --dataset-tolerance-points 0.50
corepack pnpm bench:audit-readiness -- \
  --profile release \
  --dataset-tolerance-points 0.50 \
  --out reference-results/manifests/readiness-audit-release.json
corepack pnpm bench:hybrid-claim-gate -- \
  --actual reference-results/ts/hybrid-beir-pytrec.json \
  --reference reference-results/python/hybrid-beir.json \
  --baseline-parity reference-results/ts/baseline-parity.json \
  --actual-manifest reference-results/manifests/ts-hybrid-beir-pytrec.json \
  --baseline-runner-manifest reference-results/manifests/baseline-parity-runner.json \
  --out reference-results/ts/hybrid-claim-gate.json \
  --manifest-out reference-results/manifests/hybrid-claim-gate.json
```

Known release caveat: the current stored BM25/SciFact baseline comparison has
been observed above the `0.50` per-dataset NDCG-point threshold while the
method-average development gate remains green. A release/public claim must
either root-cause and refresh that artifact or carry an explicit manifest-backed
waiver.
