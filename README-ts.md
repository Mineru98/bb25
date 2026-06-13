# bb25-ts — TypeScript port of bb25 (Bayesian BM25)

TypeScript-native port of the [bb25](https://github.com/instructkr/bb25) Rust core
(BM25 / Bayesian BM25 / Vector / Hybrid retrieval, plus probability calibration,
log-odds fusion, learnable/attention weights, and BMW indexing). Embeddings are
decoupled from the core and provided via [transformers.js](https://github.com/huggingface/transformers.js)
+ BGE-M3 (dense).

> Numeric parity with the Rust/Python reference is the top priority. Every core
> algorithm is validated against golden fixtures extracted from the reference
> (`fixtures/golden*.json`), within `|a-b| ≤ 1e-12 + 1e-9·|b|`.

## Packages (pnpm monorepo)

| Package | Description | Deps |
| --- | --- | --- |
| [`@bb25/core`](packages/core) | Pure-TS algorithms: tokenizer, Corpus, BM25, Bayesian BM25, Vector, Hybrid, fusion, metrics, ParameterLearner, probability transforms, calibrators, learnable/attention/multi-head weights, BlockMaxIndex, FusionDebugger, runExperiments. | **none** |
| [`@bb25/embeddings`](packages/embeddings) | `BgeM3Embedder` (dense) via `@huggingface/transformers`. The only package that depends on the ONNX runtime. | transformers.js |
| [`@bb25/cli`](packages/cli) | `bb25` CLI: `index` / `search` / `warmup` / `bench`. | core, embeddings |

**Core invariant:** `@bb25/core` never imports transformers.js / ONNX / fs. Inputs
are `string` and `number[]`/`Float32Array` only. (Enforced by a test.)

## Quick start

```bash
corepack enable pnpm
pnpm install
pnpm -r build
pnpm -r test            # core: golden-parity suite; cli: bench metrics
```

### CLI

```bash
# Build a lexical index (no embeddings)
bb25 index corpus.jsonl -o index.json

# Build a hybrid index (downloads BGE-M3 on first use)
bb25 index corpus.jsonl -o index.json --embed --dtype fp32

# Search
bb25 search "bayesian retrieval" --index index.json --top-k 10            # lexical
bb25 search "bayesian retrieval" --index index.json --mode and --embed    # hybrid

# Benchmark a dataset (NDCG/MAP/MRR over bm25/bayesian/hybrid/balanced/rrf)
bb25 bench --docs docs.jsonl --queries queries.jsonl --qrels qrels.tsv --embed
```

`corpus.jsonl` / `docs.jsonl`: `{"doc_id","text","embedding"?}` per line.
`queries.jsonl`: `{"query_id","text","terms"?,"embedding"?}`.
`qrels`: TSV `query_id<TAB>doc_id<TAB>relevance` or `.jsonl`.

### Library

```ts
import { Corpus, BM25Scorer, BayesianBM25Scorer, VectorScorer, HybridScorer } from "@bb25/core";

const corpus = new Corpus();
corpus.addDocument("d1", "machine learning from data", embeddingOrUndefined);
corpus.buildIndex();

const bm25 = new BM25Scorer(corpus, 1.2, 0.75);
const bayes = new BayesianBM25Scorer(bm25, 1.0, 0.5);          // alpha, beta (sigmoid likelihood)
const hybrid = new HybridScorer(bayes, new VectorScorer(), 0.5);
const p = hybrid.scoreOr(["machine", "learning"], queryEmbedding, corpus.getDocument("d1")!);
```

## Parity & verification

- **Golden fixtures** are extracted from the reference Rust via `cargo run --example extract_golden*`
  and committed under `fixtures/`. The TS test-suite asserts numeric/boolean parity.
- **Model output** (BGE-M3) is *not* bit-reproducible across runtimes (ONNX vs PyTorch);
  the embedder is validated on its own properties (dim=1024, L2-norm≈1, semantic ordering,
  determinism) rather than against a PyTorch reference.
- See [docs/ts-migration-design.md](docs/ts-migration-design.md) and
  [docs/ts-migration-roadmap.md](docs/ts-migration-roadmap.md).

## Benchmarks (SQuAD)

```bash
node scripts/prepare-squad.mjs --out /tmp/squad --max-questions 200
bb25 bench --docs /tmp/squad/docs.jsonl --queries /tmp/squad/queries.jsonl \
           --qrels /tmp/squad/qrels.tsv --embed --dtype fp32
```

Full-corpus target-metric reproduction is compute-bound (embedding thousands of
passages); the harness + prep script let you run it at any scale.
