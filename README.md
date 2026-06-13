# bb25

TypeScript-native BM25 and Bayesian retrieval toolkit.

The workspace is a pnpm monorepo with a dependency-free core package, an optional
embeddings package backed by transformers.js, and a CLI for indexing, searching,
warmup, and benchmark runs.

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
node scripts/prepare-squad.mjs --out /tmp/squad --max-questions 200
bb25 bench --docs /tmp/squad/docs.jsonl --queries /tmp/squad/queries.jsonl \
  --qrels /tmp/squad/qrels.tsv --embed --dtype fp32
```
