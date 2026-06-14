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

Critical controls before claiming reference-equivalent BEIR performance:

- BM25 variant: the reference hybrid BEIR run uses Lucene BM25 via `bm25s`;
  local parity runs must use `--bm25-method lucene`.
- Tokenization: sparse calibration uses `lower().split()`, while hybrid BEIR
  uses Snowball English stemming and stop-word removal. Local JSONL exports
  must choose the matching tokenizer for the benchmark being judged.
- Candidate protocol: reference hybrid retrieves top-1000 from sparse and
  dense independently, fuses only their union, then evaluates. Local hybrid
  runs must use `--candidate-depth 1000`.
- Embeddings: reference BEIR uses `all-MiniLM-L6-v2` via sentence-transformers.
  Local parity runs should use embeddings exported by
  `scripts/prepare-beir-jsonl.py --embed-model all-MiniLM-L6-v2
  --embed-cache-dir <cache>`, not the default CLI embedder. Once the cache is
  populated, `--embed-local-files-only` should be used for offline
  reproducibility checks.
- Missing reference evidence: cross-validated attention,
  `VectorProbabilityTransform`, VPT ablation rows, and reference all-qrels
  attention smoke rows are now exposed under lowercase CLI names, but stored
  Python reference outputs and manifests are still required before claiming
  parity.

These are not small details. If the current benchmark underperforms the
reference while any of the above differs, the result is not an apples-to-apples
implementation verdict.

## Execution Status

| Tier | Status | Intended Cadence | Current Verdict |
| --- | --- | --- | --- |
| Tier 0 | Implemented | PR CI | Golden fixtures and seeded property tests cover fixed parity points plus broad probability/pruning invariants. |
| Tier 1 | Mostly implemented | PR/manual smoke | SQuAD runner records fixed cache path/local-only mode and can hash cache contents; actual embedding cache population remains runtime-dependent. |
| Tier 2 | Partly implemented | PR/nightly | Golden fixture-backed synthetic smoke runner exists; direct Python-script parity fixtures are still future work. |
| Tier 3 | Target | Nightly/manual | Needs BEIR loader, Lucene BM25, protocol-matched tokenizer, and evaluator parity. |
| Tier 4 | Target | Manual/release | Needs stored sparse+dense top-1000 union runs with `all-MiniLM-L6-v2` and `pytrec_eval` parity; export cache paths are now manifest-backed. |
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
- Seeded property tests cover broad monotonicity, round-trip, composite-prior,
  fusion identity, WAND, and block-max upper-bound failures outside the fixed
  fixture points.

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
  reproducible gate unless it is run through `scripts/run-squad-smoke.mjs`,
  which records the manifest, fixed CLI options, command logs, input hashes,
  output hash, table comparison result, model, dtype, runtime, and relevant
  cache environment variables.
- `bb25 bench --embed` accepts `--cache-dir` and `--local-only`, and the SQuAD
  smoke runner forwards these through `--embedding-cache-dir` and
  `--embedding-local-only`. `--require-embedding-cache` validates that the cache
  exists before running; `--hash-embedding-cache` records a content hash for
  stricter identity checks.

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
this repository today; they are port targets or fixture-generation sources. The
current local gate is fixture-backed through `scripts/run-synthetic-smoke.mjs`
and targeted core module tests.

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
- Cache embeddings by dataset/model with `--embed-cache-dir`; record the cache
  path and local-only mode in the export manifest.

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
- Tuned rows: attention normalization, cross-validation, and any
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
   - Implemented in core and exported from `@bb25/core`.
5. Add `MultiFieldScorer` before claiming MultiField parity.
6. Add a BEIR harness that can either call Python `pytrec_eval` or write TREC
   run files and qrels for external evaluation.
7. Add embedding cache support and a fixed `all-MiniLM-L6-v2` path for reference
   BEIR runs. Implemented in `prepare-beir-jsonl.py` and
   `prepare-beir-jsonl-suite.py` via `--embed-cache-dir` and
   `--embed-local-files-only`; stored reference runs are still required. BGE-M3
   can remain a separate product benchmark, not the reference benchmark.
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
  `bayesian_attn_norm_split`, `bayesian_attn_norm_cv`,
  `bayesian_multihead_split`, and `bayesian_multihead_norm_split`, emitted only
  with `--fit-split` and accompanied by `attentionSplits`
  train/eval/head-count metadata. The CV row also records fold-level
  train/eval query ids.
- Reference all-qrels attention smoke rows `bayesian_attention`,
  `bayesian_attn_norm`, `bayesian_multihead`, and
  `bayesian_multihead_norm`, accompanied by `attentionSplits`
  `protocol="all-qrels"` metadata.
- Split-aware dense calibration rows `dense_platt_split` and
  `dense_isotonic_split`, emitted only with `--fit-split` and accompanied by
  `denseCalibrationSplits` train/eval/training-pair metadata.
- Core vector calibration support via `VectorProbabilityTransform`,
  `ivfDensityPrior`, and `knnDensityPrior`.
- VPT ablation rows `bayesian_vector_balanced`,
  `bayesian_vector_softplus`, and split-aware tuned
  `bayesian_vector_attn_split`.
- Multi-field Bayesian rows `bayesian_multifield` and
  `bayesian_multifield_bal` when docs provide `field_terms`.
- TREC run export via `bb25 bench --trec-run-dir` and a Python
  `pytrec_eval` wrapper in `scripts/evaluate-trec-run.py`.
- Reproducibility manifests from `scripts/run-beir-jsonl-bench.mjs` and
  `scripts/evaluate-trec-run.py` via `--manifest-out`, including command
  records with explicit return codes.
- One-command Tier 1 SQuAD smoke runner
  `scripts/run-squad-smoke.mjs`, exposed as `pnpm bench:squad-smoke`, with
  embedding cache preflight and cache-tree manifest support.
- Fixture-backed Tier 2 synthetic smoke runner
  `scripts/run-synthetic-smoke.mjs`, exposed as `pnpm bench:synthetic-smoke`.
- Python benchmark environment manifest writer
  `scripts/write-benchmark-env-manifest.py`, recording Python/platform,
  package import/version status, pip freeze, and git state.
- Python benchmark environment setup runner `scripts/setup-benchmark-env.py`,
  exposed as `pnpm bench:setup-env`, to create the benchmark venv, install
  `requirements-bench.txt`, and record setup command logs plus env/freeze
  manifests.
- Python reference benchmark runner `scripts/run-reference-benchmarks.py`,
  recording sparse/base-rate/hybrid reference commands, local/reference git
  state, command logs, environment manifest, and output hashes.
- Python reference sparse/base-rate results for NFCorpus and SciFact and full
  hybrid BEIR reference results for ArguAna, FIQA, NFCorpus, SCIDOCS, and
  SciFact are stored under `reference-results/python/`.
- `bb25 bench --metric-style python-reference` matches the Python sparse
  benchmark helper semantics: linear top-k NDCG, AP averaged over retrieved
  relevant hits, and stable corpus-order tie handling. The default remains
  pytrec-oriented.
- In `python-reference` mode, sparse Bayesian scoring follows the Python
  reference protocol: full-query BM25 score, unique query/document term-overlap
  count as the prior `tf`, all nonzero probabilities for calibration with
  unjudged documents treated as negative, and Python batch-fit hyperparameters.
- `BM25Scorer(..., method="lucene")` uses the same score scale as `bm25s`
  Lucene BM25, which matters for BM25-to-probability calibration even when raw
  ranking is unchanged by a constant score factor.
- JSON baseline comparison gate in `scripts/check-bench-json.mjs`, including
  required dataset checks, row-level failure classification, result JSON, and
  manifest output. It also compares calibration rows such as ECE/Brier with
  unit-scale tolerances via `--metric ece|brier`.
- Baseline parity orchestration in `scripts/run-baseline-parity.mjs`, exposed
  as `pnpm bench:baseline-parity`, to run fresh TS BEIR, pytrec judgment, and
  JSON parity check in one command.
- `bb25 bench --scorers` and the baseline-only fast path keep full BEIR baseline
  parity focused on `bm25`, `dense`, `convex`, and `rrf`, reusing per-query
  sparse/dense scores instead of running all diagnostic Bayesian rows.
- JSONL loading is streaming/chunked, avoiding single-string limits on large
  exported BEIR files such as FIQA docs with embedded vectors.
- Benchmark readiness audit in `scripts/audit-benchmark-readiness.mjs`, exposed
  as `pnpm bench:audit-readiness`, to verify required result JSON, manifests,
  command records with return codes, dataset coverage, Python dependency
  status, baseline parity before claiming Phase 1 complete, and sparse
  calibration parity artifacts with `--profile sparse` or `--profile all`.
- `base_rate="auto"` via `bb25 bench --base-rate auto`, with percentile,
  mixture, and elbow estimators recorded in JSON output as a resolved numeric
  base rate with method/seed/sample metadata.
- Sparse calibration diagnostics via `bb25 bench --calibration`, reporting ECE
  and Brier separately from ranking metrics.
- `bayesian_no_base_rate` is emitted when a base rate is configured, giving the
  sparse calibration gate a same-run baseline for ECE/Brier reduction checks.
- Sparse calibration gate `scripts/check-calibration-gate.mjs`, exposed as
  `pnpm bench:calibration-gate`, verifies minimum ECE/Brier reduction and fitted
  maximum thresholds.
- Sparse calibration parity runner
  `scripts/run-sparse-calibration-parity.mjs`, exposed as
  `pnpm bench:sparse-parity`, runs the Phase 2 TS sparse benchmark, NDCG/MAP
  parity checks, ECE/Brier parity checks, the calibration reduction gate, and a
  runner manifest in one command.
- Split-aware batch fit via `bb25 bench --fit-split`, emitted as
  `bayesian_fitted_split` with train/eval query ids, split seed, train ratio,
  training pair count, and fitted alpha/beta metadata.
- Sparse parity writes NumPy `default_rng(seed).shuffle` query split files and
  feeds them to `bb25 bench --fit-split-file`, so split-aware fitted rows use
  the same train/eval query ids as the Python sparse reference.
- Current stored sparse parity passes Raw BM25 average NDCG/MAP, zero-shot
  Bayesian average NDCG/MAP, split-fitted Bayesian average NDCG/MAP, sparse
  ECE/Brier parity, and the calibration reduction/fitted-ECE gate.
- Current stored hybrid baseline parity passes `BM25`, `Dense`, `Convex`, and
  `RRF` average NDCG@10 within `0.50` points against the Python reference using
  `pytrec_eval` over the five BEIR datasets.
- BEIR JSONL export and runner scripts:
  `scripts/prepare-beir-jsonl.py`, `scripts/prepare-beir-jsonl-suite.py`, and
  `scripts/run-beir-jsonl-bench.mjs`; export manifests include
  tokenizer/model settings, Python/package versions, command logs, and file
  hashes. Short BEIR dataset names resolve to the registered `ir_datasets`
  identifiers, using explicit split ids where available, such as
  `beir/nfcorpus/test`, and unsplit ids where required, such as `beir/arguana`.

## Recommended Commands

Fast local smoke:

```bash
corepack pnpm -r build
corepack pnpm -r test
corepack pnpm bench:squad-smoke -- \
  --regenerated-embeddings \
  --embedding-cache-dir /tmp/bb25-embedding-cache/bge-m3-q8 \
  --require-embedding-cache \
  --hash-embedding-cache
corepack pnpm bench:synthetic-smoke
```

Reference Python sparse calibration and BEIR hybrid:

```bash
corepack pnpm bench:setup-env -- \
  --python python3.12 \
  --venv .venv-bench \
  --require
. .venv-bench/bin/activate
python scripts/run-reference-benchmarks.py \
  --reference-repo /tmp/cognica-bayesian-bm25 \
  --beir-dir /tmp/beir \
  --download \
  --datasets arguana fiqa nfcorpus scidocs scifact \
  --model all-MiniLM-L6-v2 \
  --retrieve-k 1000 \
  --top-k 10 \
  --require-env
```

TypeScript sparse BEIR harness after exporting JSONL:

```bash
corepack pnpm bench:prepare-beir -- \
  --out-root /tmp/beir-sparse \
  --datasets nfcorpus scifact \
  --tokenizer split \
  --manifest-out /tmp/bb25-beir-sparse-export-manifest.json
corepack pnpm bench:sparse-parity -- \
  --root /tmp/beir-sparse \
  --datasets nfcorpus,scifact \
  --reference-ranking reference-results/python/sparse-benchmark.json \
  --reference-calibration reference-results/python/base-rate.json
```

TypeScript hybrid BEIR harness after exporting embeddings:

```bash
corepack pnpm bench:prepare-beir -- \
  --out-root /tmp/beir \
  --datasets arguana fiqa nfcorpus scidocs scifact \
  --tokenizer snowball \
  --embed-model all-MiniLM-L6-v2 \
  --embed-cache-dir /tmp/bb25-embedding-cache/all-MiniLM-L6-v2 \
  --manifest-out /tmp/bb25-beir-hybrid-export-manifest.json
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
  --out /tmp/bb25-beir-hybrid.json \
  --manifest-out /tmp/bb25-beir-hybrid-manifest.json
```

Compare baseline parity against the stored Python reference:

```bash
corepack pnpm bench:baseline-parity -- \
  --python .venv-bench/bin/python \
  --root /tmp/beir \
  --reference reference-results/python/hybrid-beir.json \
  --datasets arguana,fiqa,nfcorpus,scidocs,scifact \
  --methods BM25,Dense,Convex,RRF \
  --metric ndcg@10 \
  --tolerance-points 0.50

corepack pnpm bench:audit-readiness -- \
  --profile all \
  --root reference-results \
  --out reference-results/manifests/readiness-audit.json
```

Keep the stored manifests with the reported results; a green score without the
matching command records, input hashes, and dependency manifests is not a valid
parity claim.
