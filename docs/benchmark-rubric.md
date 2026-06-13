# Bayesian BM25 Benchmark Rubric

This document defines the benchmark plan for comparing this TypeScript port
against the original Bayesian BM25 work. The goal is not to make a flattering
demo; it is to detect migration defects. If a benchmark is run with the same
data, tokenizer, BM25 variant, embeddings, candidate protocol, and evaluator as
the reference, a material regression should be treated as an implementation
bug until proven otherwise.

## Sources

Primary sources used for this rubric:

- Paper: https://www.cognica.io/papers/bayesian-bm25.pdf
- Reference Python implementation: https://github.com/cognica-io/bayesian-bm25
- Reference TypeScript implementation: https://github.com/cognica-io/bayesian-bm25-js
- BEIR benchmark framework: https://github.com/beir-cellar/beir
- `ir_datasets` BEIR catalog: https://ir-datasets.com/beir.html
- Current local history: `950f8cf`, `414988c`, `93331c0`, `52df382`

The Python reference is the source of truth for the BEIR benchmark protocol. It
currently reports BEIR hybrid search over ArguAna, FiQA, NFCorpus, SciDocs, and
SciFact with top-1000 sparse retrieval, top-1000 dense retrieval, union
candidates, and `pytrec_eval`. Scores in the reference BEIR tables are reported
as NDCG points on a `0-100` scale; local CLI output is on a `0-1` scale unless
explicitly converted.

## Current Gap Assessment

The current repository can run a useful benchmark, but it is not yet the same
benchmark as `cognica-io/bayesian-bm25`.

Current implementation strengths:

- Core probability, fusion, calibration, attention, multi-head, and block-max
  modules are covered by golden fixtures extracted from the earlier Rust/Python
  port.
- CLI benchmark supports `docs.jsonl`, `queries.jsonl`, `qrels.tsv|jsonl`,
  optional embeddings, and reports `NDCG@k`, `MAP@k`, `MRR@k`, and `recall@k`.
- A SQuAD slice smoke benchmark is already recorded at
  `fixtures/bench/squad120-q8-results.txt`.

Critical gaps before claiming reference-equivalent BEIR performance:

- BM25 variant mismatch: the reference hybrid BEIR run uses Lucene BM25 via
  `bm25s` with Snowball English stemming and stop-word removal. The local
  `BM25Scorer.idf()` currently uses `log((N-df+0.5)/(df+0.5))`, which can be
  negative and is not the same as the Lucene-style positive IDF used by the
  reference benchmark.
- Tokenization mismatch: the local tokenizer is simple regex tokenization. The
  sparse calibration scripts use `lower().split()` over `ir_datasets` text,
  while the hybrid BEIR script uses Snowball English stemming and stop-word
  removal.
- Candidate protocol mismatch: the local `runBench()` scores every document.
  The reference hybrid benchmark retrieves top-1000 from sparse and dense
  independently, fuses only their union, then evaluates.
- Embedding mismatch: local CLI defaults to BGE-M3 via transformers.js. The
  reference BEIR benchmark uses `all-MiniLM-L6-v2` via sentence-transformers.
- Missing reference features: the local port does not yet expose
  `VectorProbabilityTransform` or the full VPT row set.

These are not small details. If the current benchmark underperforms the
reference while any of the above differs, the result is not an apples-to-apples
implementation verdict.

## Execution Status

| Tier | Status | Intended Cadence | Current Verdict |
| --- | --- | --- | --- |
| Tier 0 | Partly implemented | PR CI | Golden fixtures exist; seeded property tests are required. |
| Tier 1 | Partly implemented | PR/manual smoke | SQuAD result is recorded; manifest and automatic comparison are required. |
| Tier 2 | Target | PR/nightly | Reference synthetic scripts must be ported or fixture-backed. |
| Tier 3 | Target | Nightly/manual | Needs BEIR loader, Lucene BM25, protocol-matched tokenizer, and evaluator parity. |
| Tier 4 | Target | Manual/release | Needs sparse+dense top-1000 union, `all-MiniLM-L6-v2`, embedding cache, and `pytrec_eval` parity. |
| Tier 5 | Target | Release audit | Run only after Tier 0-4 are stable. |

Only Tier 0 and the non-embedding portion of Tier 1 should block normal pull
requests. Embedding-backed SQuAD and BEIR runs are runtime- and model-cache
sensitive and should run as manual or scheduled jobs with saved manifests.

## Selected Benchmark Tiers

### Tier 0: Mathematical Parity

Purpose: catch direct migration bugs without dataset noise.

Source basis:

- `bayesian-bm25-js/tests/paper_theorems.test.ts`
- `bayesian-bm25-js/tests/probability.test.ts`
- `bayesian-bm25-js/tests/fusion.test.ts`
- Local `fixtures/golden*.json`

Current local coverage:

- Golden fixtures already cover deterministic math/scoring/probability module
  parity.
- Seeded property tests are still required to catch broad monotonicity,
  round-trip, and upper-bound failures outside the fixed fixture points.

Required checks:

- `sigmoid` and `logit` round-trip, symmetry, monotonicity, and finite extreme
  handling.
- Posterior equivalence between direct Bayes formula and log-odds addition.
- Composite prior bounds and monotonic score-to-probability behavior.
- Log-odds conjunction agreement amplification, disagreement moderation,
  uniform weighted equivalence, and weight validation.
- `cosineToProbability`, `probAnd`, `probOr`, `probNot`.
- WAND/BMW upper bounds never below actual probabilities.
- Calibration metrics exactness on known examples.
- Learnable and attention weights recover seeded synthetic behavior.

Pass criteria:

- Exact numeric fixture parity within `abs <= 1e-12 + 1e-9 * abs(expected)`.
- Property tests must pass for seeded random cases.
- Any failure in this tier is a correctness bug, not a benchmark variance issue.

### Tier 1: Local IR Smoke

Purpose: fast regression test for the current CLI and embedding path.

Datasets:

- Existing default corpus and `fixtures/golden.json`.
- Existing SQuAD slice: `scripts/prepare-squad.mjs` and
  `fixtures/bench/squad120-q8-results.txt`.

Current status:

- The recorded SQuAD table is a useful smoke target but is not yet a fully
  reproducible gate until a manifest, fixed CLI options, and automatic
  comparison script exist.
- Embedding rows must record model, dtype, runtime, and embedding cache identity.

Current recorded SQuAD 120-query q8 result:

| Method | NDCG@10 | MRR@10 |
| --- | ---: | ---: |
| bm25 | 0.8322 | 0.7995 |
| bayesian | 0.8582 | 0.8263 |
| bayesian_fitted | 0.8699 | 0.8417 |
| hybrid_or | 0.8635 | 0.8329 |
| hybrid_and | 0.8695 | 0.8386 |
| balanced_fusion | 0.9258 | 0.9067 |
| rrf | 0.9072 | 0.8797 |

Pass criteria:

- With the same model, dtype, selected questions, and generated vectors, each
  row should stay within `0.005` absolute NDCG/MRR.
- If embeddings are regenerated on a different runtime, allow `0.015` absolute
  tolerance but require the ordering:
  `balanced_fusion > rrf > bayesian_fitted >= bayesian > bm25`.
- If `bm25` itself moves materially, inspect tokenization or corpus preparation
  before debugging Bayesian code.

### Tier 2: Synthetic Reference Benchmarks

Purpose: cover reference benchmark modules that require no external IR corpus.

Selected scripts from the Python reference. These scripts are not present in
this repository today; they are port targets or fixture-generation sources.

- `benchmarks/weighted_fusion.py`
- `benchmarks/learnable_weights.py`
- `benchmarks/gating_functions.py`
- `benchmarks/wand_upper_bound.py`
- `benchmarks/bmw_upper_bound.py`
- `benchmarks/neural_calibration.py`
- `benchmarks/multi_head_attention.py`

Pass criteria:

- Seed must be fixed at `42` unless a multi-seed report is requested.
- Learned weights must move toward oracle weights in the same direction as the
  reference.
- Weighted log-odds must beat uniform fusion when signal noise is asymmetric.
- WAND/BMW pruning must produce zero false-prune cases.
- Calibration methods must reduce Brier/ECE relative to uncalibrated synthetic
  scores in the same scenarios as the reference.

### Tier 3: Sparse BEIR Calibration

Purpose: validate BM25-to-probability calibration without dense embeddings.

Reference script:

- `benchmarks/benchmark.py`
- `benchmarks/base_rate.py`

Selected datasets:

- Required: BEIR NFCorpus and SciFact.
- Protocol: `ir_datasets`, split queries 50/50 using RNG seed `42`, `k=10`.
- Tokenization: Python `lower().split()` in the sparse reference scripts unless
  a new reference run says otherwise.
- BM25: `k1=1.2`, `b=0.75`, Lucene variant.

Reference sparse ranking targets:

| Method | NFCorpus NDCG@10 | NFCorpus MAP | SciFact NDCG@10 | SciFact MAP |
| --- | ---: | ---: | ---: | ---: |
| Raw BM25 | 0.5023 | 0.4395 | 0.5900 | 0.5426 |
| Bayesian auto | 0.5050 | 0.4403 | 0.5791 | 0.5283 |
| Bayesian auto + base rate | 0.5050 | 0.4403 | 0.5791 | 0.5283 |
| Bayesian batch fit | 0.5041 | 0.4400 | 0.5826 | 0.5305 |

Reference calibration targets:

| Method | NFCorpus ECE | NFCorpus Brier | SciFact ECE | SciFact Brier |
| --- | ---: | ---: | ---: | ---: |
| Bayesian no base rate | 0.6519 | 0.4667 | 0.7989 | 0.6635 |
| Bayesian base_rate=auto | 0.1461 | 0.0619 | 0.2577 | 0.1308 |
| Batch fit + base_rate=auto | 0.0085 | 0.0096 | 0.0021 | 0.0013 |

Pass criteria:

- Raw BM25 must reproduce within `0.005` absolute NDCG/MAP before Bayesian rows
  are judged. Otherwise the tokenizer/BM25 path is not equivalent.
- Base-rate calibration must not change ranking order for monotonic transforms.
- `base_rate=auto` must reduce ECE by at least 50% versus no-base-rate on both
  datasets; the reference reduction is about 68-78%.
- Batch fit plus base rate should produce ECE below `0.02` on both selected
  datasets when the same split and labels are used.

### Tier 4: Full BEIR Hybrid Search

Purpose: reproduce the main reference hybrid retrieval claim.

Reference script:

- `benchmarks/hybrid_beir.py`

Required protocol:

- Datasets: ArguAna, FiQA, NFCorpus, SciDocs, SciFact.
- BM25: `k1=1.2`, `b=0.75`, Lucene variant.
- Sparse tokenizer: Snowball English stemmer plus English stop-word removal.
- Dense encoder: `all-MiniLM-L6-v2`.
- Candidate protocol: retrieve top `R=1000` from sparse and dense separately,
  fuse the union candidates, evaluate at `k=10`.
- Evaluator: `pytrec_eval` semantics for `ndcg_cut_10`, `map_cut_10`,
  `recall_10`.
- Cache embeddings by dataset/model to make repeat runs comparable.

BEIR harness target methods:

- `BM25`
- `Dense`
- `Convex`
- `RRF`
- `Bayesian-OR`
- `Bayesian-LogOdds`
- `Bayesian-LogOdds-BR`
- `Bayesian-Balanced`
- `Bayesian-Gated-ReLU`
- `Bayesian-Gated-Swish`
- `Bayesian-Gated-GELU`
- `Bayesian-Attention`
- `Bayesian-MultiHead`

Full parity methods after missing features are implemented:

- `Bayesian-Balanced-Mix`
- `Bayesian-Balanced-Elbow`
- `Bayesian-Gated-Swish-B2`
- `Bayesian-Gated-Softplus`
- `Bayesian-Attn-Norm`
- `Bayesian-Attn-Norm-CV`
- `Bayesian-MultiHead-Norm`
- `Bayesian-MultiField`
- `Bayesian-MultiField-Bal`
- `Bayesian-Vector-Balanced`
- `Bayesian-Vector-Softplus`
- `Bayesian-Vector-Attn`
- Dense calibration and VPT ablations.

Reference average NDCG@10:

| Method | Avg NDCG@10 |
| --- | ---: |
| BM25 | 35.38 |
| Dense | 38.32 |
| Convex | 41.15 |
| RRF | 40.49 |
| Bayesian-Balanced | 41.50 |
| Bayesian-Attn-Norm | 41.67 |

Training and leakage taxonomy:

- Zero-shot rows: `BM25`, `Dense`, `Convex`, `RRF`, basic Bayesian fusion rows
  with fixed parameters and no qrels-derived calibration.
- Calibration rows: rows using `base_rate="auto"`, Platt/isotonic fitting, or
  batch fit must record the qrels split used to estimate parameters.
- Tuned rows: attention normalization, cross-validation, dense Platt, and any
  dataset-specific parameter search are not comparable to zero-shot rows unless
  train/test separation is explicit in the manifest.

Pass criteria:

- First gate: `BM25`, `Dense`, `Convex`, and `RRF` must reproduce within
  `0.50` NDCG points. If they do not, the benchmark environment is not aligned.
- Same embeddings/tokenizer/evaluator: Bayesian methods should reproduce within
  `0.50` NDCG points.
- Different embedding runtime but same model: allow up to `1.50` NDCG points,
  but require stable ordering and deltas.
- `Bayesian-Balanced` should be at least `BM25 + 4.0` NDCG points on the
  five-dataset average and should not trail `RRF` by more than `0.50` points.
- If baseline rows match but `Bayesian-Balanced` or log-odds rows regress,
  investigate probability transform, base-rate logic, logit normalization, or
  candidate fusion before tuning parameters.

### Tier 5: External Leaderboard / MTEB

Purpose: outside validation after local parity is achieved.

The reference README states that Bayesian BM25 is included as a retrieval model
in MTEB. This should be treated as a release-level audit, not a day-to-day CI
test. Run only after Tier 0-4 are stable.

## Failure Triage Rules

Use these rules before changing thresholds:

1. If Tier 0 fails, fix implementation. Do not inspect IR metrics first.
2. If raw BM25 is off, fix BM25 variant, tokenization, stemming, stop-word
   handling, qrels parsing, or candidate protocol.
3. If dense baseline is off, fix model choice, embedding normalization,
   quantization, or cache invalidation.
4. If `BM25`, `Dense`, `Convex`, and `RRF` match but Bayesian methods regress,
   treat it as a Bayesian implementation defect.
5. If tuned methods improve but zero-shot methods regress, do not hide the
   regression with tuning. The reference BEIR table is primarily zero-shot.
6. If a method underperforms in the reference too, preserve that failure mode.
   For example, `Bayesian-OR` is expected to fail badly on ArguAna; making it
   look good can indicate an accidental protocol change.

## Implementation Backlog

To make this repository capable of running the same benchmark:

1. Add a BM25 method option compatible with the reference Lucene variant.
2. Add a BEIR tokenizer/pretokenized corpus path. The safest path is to store
   reference-produced tokens in JSONL arrays and bypass local tokenization.
3. Add `softplus` gating and generalized gating beta CLI exposure.
4. Add `VectorProbabilityTransform`, `ivfDensityPrior`, and `knnDensityPrior`
   before attempting Paper 3/VPT rows.
5. Add `MultiFieldScorer` before claiming MultiField parity.
6. Add a BEIR harness that can either call Python `pytrec_eval` or write TREC
   run files and qrels for external evaluation.
7. Add embedding cache support and a fixed `all-MiniLM-L6-v2` path for reference
   BEIR runs. BGE-M3 can remain a separate product benchmark, not the reference
   benchmark.
8. Store benchmark manifests and result JSON with commit SHA, package versions,
   dataset checksums, model name, dtype, tokenizer, BM25 method, and evaluator.

Current implementation support added for this migration audit:

- `BM25Scorer(..., method="lucene")` and CLI `--bm25-method lucene`.
- Pretokenized `terms` arrays for docs and queries via `--doc-terms` and
  `--query-terms`.
- TREC-style four-column qrels loading.
- `recall@k` and JSON output for `bb25 bench`.
- Hybrid sparse+dense top-R union via `--candidate-depth`.
- Reference baseline rows `dense`, `convex`, and `rrf`; `convex` uses
  per-query min-max normalization of raw BM25 scores and dense cosine scores,
  and `rrf` gives no contribution to signals outside their top-R retrieval set.
- Bayesian hybrid rows `bayesian_logodds` and `bayesian_logodds_br`.
  The BR row is emitted when a base rate is configured and is marked as a
  calibration row in JSON scorer metadata.
- Gated Bayesian hybrid diagnostic rows `bayesian_gated_relu`,
  `bayesian_gated_swish`, `bayesian_gated_gelu`,
  `bayesian_gated_swish_b2`, and `bayesian_gated_softplus`.
- Split-aware tuned attention rows `bayesian_attention_split`,
  `bayesian_attn_norm_split`, `bayesian_multihead_split`, and
  `bayesian_multihead_norm_split`, emitted only with `--fit-split` and
  accompanied by `attentionSplits` train/eval/head-count metadata.
- Multi-field Bayesian rows `bayesian_multifield` and
  `bayesian_multifield_bal` when docs provide `field_terms`.
- TREC run export via `bb25 bench --trec-run-dir` and a Python
  `pytrec_eval` wrapper in `scripts/evaluate-trec-run.py`.
- JSON baseline comparison gate in `scripts/check-bench-json.mjs`.
- `base_rate="auto"` via `bb25 bench --base-rate auto`, with percentile,
  mixture, and elbow estimators recorded in JSON output as a resolved numeric
  base rate with method/seed/sample metadata.
- Sparse calibration diagnostics via `bb25 bench --calibration`, reporting ECE
  and Brier separately from ranking metrics.
- Split-aware batch fit via `bb25 bench --fit-split`, emitted as
  `bayesian_fitted_split` with train/eval query ids, split seed, train ratio,
  training pair count, and fitted alpha/beta metadata.
- BEIR JSONL export and runner scripts:
  `scripts/prepare-beir-jsonl.py` and `scripts/run-beir-jsonl-bench.mjs`.

## Recommended Commands

Fast local smoke:

```bash
corepack pnpm -r build
corepack pnpm -r test
node scripts/prepare-squad.mjs --out /tmp/squad --max-questions 120
corepack pnpm --filter @bb25/cli exec bb25 bench \
  --docs /tmp/squad/docs.jsonl \
  --queries /tmp/squad/queries.jsonl \
  --qrels /tmp/squad/qrels.tsv \
  --embed --dtype q8 --cutoffs 5,10
```

Reference Python sparse calibration:

```bash
python benchmarks/benchmark.py -o /tmp/bayesian-bm25-sparse.json
python benchmarks/base_rate.py -o /tmp/bayesian-bm25-base-rate.json
```

Reference Python BEIR hybrid:

```bash
python benchmarks/hybrid_beir.py \
  -d /tmp/beir \
  --download \
  --datasets arguana fiqa nfcorpus scidocs scifact \
  --model all-MiniLM-L6-v2 \
  -R 1000 -k 10 \
  -o /tmp/bayesian-bm25-beir-hybrid.json
```

TypeScript sparse BEIR harness after exporting JSONL:

```bash
python scripts/prepare-beir-jsonl.py \
  --dataset nfcorpus \
  --out /tmp/beir/nfcorpus \
  --tokenizer split
corepack pnpm --filter @bb25/cli build
node scripts/run-beir-jsonl-bench.mjs \
  --root /tmp/beir \
  --datasets nfcorpus \
  --bm25-method lucene \
  --base-rate auto \
  --base-rate-method percentile \
  --fit-split \
  --fit-train-ratio 0.5 \
  --fit-split-seed 42 \
  --calibration \
  --cutoffs 10 \
  --out /tmp/bb25-beir-sparse.json
```

TypeScript hybrid BEIR harness after exporting embeddings:

```bash
python scripts/prepare-beir-jsonl.py \
  --dataset scifact \
  --out /tmp/beir/scifact \
  --tokenizer snowball \
  --embed-model all-MiniLM-L6-v2
corepack pnpm --filter @bb25/cli build
node scripts/run-beir-jsonl-bench.mjs \
  --root /tmp/beir \
  --datasets scifact \
  --doc-embedding embedding \
  --query-embedding embedding \
  --doc-fields title,body \
  --doc-field-terms field_terms \
  --candidate-depth 1000 \
  --base-rate auto \
  --base-rate-method percentile \
  --bm25-method lucene \
  --cutoffs 10 \
  --out /tmp/bb25-beir-hybrid.json
```

Export TREC runs and judge with `pytrec_eval`:

```bash
node scripts/run-beir-jsonl-bench.mjs \
  --root /tmp/beir \
  --datasets scifact \
  --doc-embedding embedding \
  --query-embedding embedding \
  --doc-fields title,body \
  --doc-field-terms field_terms \
  --candidate-depth 1000 \
  --trec-run-dir /tmp/bb25-beir-runs \
  --trec-run-depth 1000 \
  --bm25-method lucene \
  --cutoffs 10 \
  --out /tmp/bb25-beir-hybrid-internal.json

python scripts/evaluate-trec-run.py \
  --root /tmp/beir \
  --datasets scifact \
  --runs /tmp/bb25-beir-runs \
  --cutoffs 10 \
  --out /tmp/bb25-beir-hybrid-pytrec.json
```

Compare baseline parity against the stored Python reference:

```bash
node scripts/check-bench-json.mjs \
  --reference reference-results/python/hybrid-beir.json \
  --actual /tmp/bb25-beir-hybrid-pytrec.json \
  --methods BM25,Dense,Convex,RRF \
  --metric ndcg@10 \
  --tolerance-points 0.50
```

When the TypeScript BEIR harness exists, its output should be compared against
the Python output row-by-row before any new model or dataset is introduced.
