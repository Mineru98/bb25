# Benchmark Implementation Plan

This plan turns the benchmark rubric into executable work. The priority is:

1. Make benchmark results judgeable.
2. Reproduce reference baselines.
3. Improve Bayesian and hybrid rows only after the baseline protocol is aligned.

Do not tune for higher numbers until the benchmark can explain whether a change
is an implementation improvement, a protocol drift, or data leakage.

## Current State

Implemented in this repository:

- Tier 0 golden fixtures and seeded property tests, including composite-prior
  invariants and WAND/block-max upper-bound safety checks.
- CLI benchmark over JSONL docs, queries, qrels, and optional embeddings.
- `recall@k`, JSON output, TREC-style four-column qrels.
- Lucene-style BM25 IDF option.
- Pretokenized doc/query term paths.
- Hybrid sparse+dense top-R union via `--candidate-depth`.
- Reference hybrid baseline rows: `dense`, `convex`, and `rrf`.
- TREC run export via `bb25 bench --trec-run-dir`.
- `pytrec_eval` wrapper for scorer run files.
- JSON comparison gate for Python-vs-TS baseline parity, with required-dataset
  checks, row-level failure classification, result JSON, and reproducibility
  manifest output. The same gate can compare calibration rows via `--metric ece`
  or `--metric brier` using unit-scale tolerances.
- TS BEIR runner and pytrec wrapper can write reproducibility manifests with
  command records, return codes, git/package metadata, input hashes, split
  metadata, and scorer metadata.
- Baseline parity runner orchestrates the fresh TS BEIR run, optional pytrec
  judgment, JSON parity check, and runner manifest in one command.
- Benchmark readiness audit checks required `reference-results` JSON/manifests,
  command success records with explicit return codes, dataset coverage, Python
  dependency status, baseline parity status, and profile-specific sparse
  calibration parity artifacts via `--profile sparse` or `--profile all`.
- Python benchmark environment manifest writer records Python/platform,
  package import/version status, pip freeze output, and git state.
- Python benchmark environment setup runner `pnpm bench:setup-env` creates the
  benchmark venv, installs `requirements-bench.txt`, writes the Python env
  manifest/freeze file, and records setup command logs.
- Python reference benchmark runner records the reference commands, command
  stdout/stderr/return codes, local/reference git state, environment manifest,
  and output hashes.
- Python reference sparse/base-rate outputs for NFCorpus and SciFact have been
  generated under `reference-results/python/`, with command manifests under
  `reference-results/manifests/`.
- `bb25 bench --metric-style python-reference` reproduces the reference
  benchmark's linear top-k NDCG, top-k AP, and corpus-order tie handling without
  changing the default pytrec-oriented metrics.
- In `python-reference` benchmark mode, sparse Bayesian rows use the Python
  reference scoring protocol: one full-query BM25 score, unique query/document
  term-overlap count as `tf`, all nonzero scores for calibration/training with
  unjudged documents labeled negative, and the Python batch-fit parameters
  `learning_rate=0.05`, `max_iterations=3000`.
- Lucene BM25 score scale now matches `bm25s` for reference parity while keeping
  the default non-Lucene scorer behavior separate.
- Sparse JSONL export now uses BEIR split ids such as `beir/nfcorpus/test` for
  short dataset names and records the split in export manifests.
- `--base-rate auto` with percentile, mixture, and elbow estimators for
  Bayesian sparse rows.
- Optional sparse calibration report via `--calibration` with ECE and Brier.
- When a base rate is configured, `bayesian_no_base_rate` is emitted as a
  calibration baseline so ECE reduction can be measured in the same run.
- Sparse calibration parity runner `pnpm bench:sparse-parity` runs the Phase 2
  TS sparse benchmark, ranking parity checks for NDCG/MAP, ECE/Brier parity
  checks, the calibration reduction gate, and a runner manifest with command
  records.
- Sparse parity writes NumPy `default_rng(seed).shuffle` query split files under
  `reference-results/manifests/fit-splits/` and passes them to
  `bb25 bench --fit-split-file`, so split-aware fitted rows use the same
  train/eval query ids as the Python sparse reference.
- Split-aware batch fit via `--fit-split`, emitted as `bayesian_fitted_split`
  with train/eval query ids and learned alpha/beta metadata.
- Split-aware dense calibration rows behind `--fit-split`:
  `dense_platt_split` and `dense_isotonic_split`, with train/eval query ids,
  training pair count, trained flag, and Platt parameters when applicable.
- Hybrid Bayesian LogOdds rows: `bayesian_logodds` and, when a base rate is
  configured, `bayesian_logodds_br`. The BR row is marked as calibration
  metadata.
- Gated Bayesian hybrid diagnostic rows: `bayesian_gated_relu`,
  `bayesian_gated_swish`, `bayesian_gated_gelu`,
  `bayesian_gated_swish_b2`, and `bayesian_gated_softplus`.
- Split-aware tuned attention rows behind `--fit-split`:
  `bayesian_attention_split`, `bayesian_attn_norm_split`,
  `bayesian_attn_norm_cv`, `bayesian_multihead_split`, and
  `bayesian_multihead_norm_split`, with train/eval query ids, feature set,
  normalization flag, head count, and training pair count metadata. The CV row
  records per-fold train/eval query ids.
- Reference all-qrels attention smoke rows:
  `bayesian_attention`, `bayesian_attn_norm`, `bayesian_multihead`, and
  `bayesian_multihead_norm`, with `attentionSplits` metadata marked
  `protocol="all-qrels"` so they are not treated as leakage-safe improvements.
- Multi-field Bayesian rows when field terms are supplied:
  `bayesian_multifield` and `bayesian_multifield_bal`.
- Vector probability transform support with VPT ablation rows:
  `bayesian_vector_balanced`, `bayesian_vector_softplus`, and split-aware
  `bayesian_vector_attn_split`.
- SQuAD smoke manifest, table comparison script, and one-command smoke runner
  with embedding cache path/local-only forwarding, cache preflight, and optional
  cache-tree hashing.
- Tier 2 synthetic smoke runner backed by golden fixtures and targeted core
  module tests.
- BEIR JSONL export, suite export, and TS runner scripts; export manifests
  include tokenizer/model settings, Python/package versions, command logs, and
  input file hashes.
- `bb25 bench --scorers` can restrict expensive full-BEIR runs to the rows being
  judged, and the baseline-only path reuses per-query BM25/dense scores for
  `bm25`, `dense`, `convex`, and `rrf`.
- JSONL loaders stream large files instead of reading the whole file into a
  single string, so FIQA-scale exported embeddings can be loaded by the CLI.
- Stored full hybrid BEIR Python reference output and TS baseline parity
  artifacts now exist under `reference-results/`, including internal TS,
  `pytrec_eval`, parity result, and manifests.
- The full readiness audit currently passes for `--profile all`: hybrid and
  sparse artifacts, command records, dependency manifests, dataset coverage, and
  parity gates are all present.

Known caveats and protocol boundaries:

- The Python benchmark environment manifest now imports the required reference
  packages, including `pytrec_eval`, when created through `pnpm bench:setup-env`.
- TS emits reference-oriented hybrid rows using lowercase CLI names; the
  comparison gate maps them to Python reference names such as `BM25`, `Dense`,
  `Convex`, and `RRF`.
- The legacy `bayesian_fitted` row remains an all-qrels smoke row. Use
  `bayesian_fitted_split` for reference calibration claims.
- Sparse BM25, zero-shot Bayesian ranking, split-fitted Bayesian ranking,
  sparse ECE/Brier parity, and the sparse calibration gate now pass on the
  NFCorpus/SciFact averages when using the stored Python reference and generated
  NumPy split files.
- Full hybrid baseline parity currently covers `BM25`, `Dense`, `Convex`, and
  `RRF`. Bayesian hybrid rows should still be judged only after the baseline
  gate remains green for the stored input/export manifests.

## Phase 1: Make Results Judgeable

Goal: produce a benchmark run where a regression can be attributed to a specific
layer.

Deliverables:

- `reference-results/python/*.json`
- `reference-results/ts/*.json`
- `reference-results/manifests/*.json`
- A single comparison command that fails when baseline parity thresholds are not
  met.

Implementation tasks:

1. Create a Python benchmark environment.
   - Install `requirements-bench.txt`.
   - Record exact versions with `pip freeze`.
   - Store the Python version, platform, package import status, and local git
     state in the manifest.

2. Run the reference Python benchmarks.
   - Use `scripts/run-reference-benchmarks.py` so commands, stdout/stderr,
     return codes, reference git state, environment manifest, and output hashes
     are recorded together.
   - Sparse calibration:
     - `benchmarks/benchmark.py`
     - `benchmarks/base_rate.py`
   - Hybrid BEIR:
     - `benchmarks/hybrid_beir.py`
     - datasets: `arguana`, `fiqa`, `nfcorpus`, `scidocs`, `scifact`
     - model: `all-MiniLM-L6-v2`
     - retrieval depth: `R=1000`
     - cutoff: `k=10`

3. Export shared BEIR inputs for TS.
   - Sparse calibration export uses `--tokenizer split`.
   - Hybrid export uses `--tokenizer snowball`.
   - Multi-field export writes `field_terms` for `title` and `body`.
   - First parity run should use Python-generated `all-MiniLM-L6-v2`
     embeddings stored in JSONL so TS does not introduce embedding runtime
     variance.
   - Hybrid exports should pin `--embed-cache-dir` and save the generated
     export manifest. After the cache is populated, add
     `--embed-local-files-only` for offline reproducibility checks.

4. Use `pytrec_eval` for final BEIR judgment.
   - TS writes scorer-level TREC run files with `--trec-run-dir`.
   - `scripts/evaluate-trec-run.py` evaluates those files with `pytrec_eval`.
   - Internal TS metrics remain useful for smoke tests, but BEIR parity is judged
     by `pytrec_eval`.

5. Add comparison gates.
   - Baseline gate:
     - `BM25`, `Dense`, `Convex`, `RRF` must match the Python reference within
       `0.50` NDCG points on the five-dataset average.
     - If embeddings are regenerated in a different runtime, allow `1.50`
       points, but require stable ordering and deltas.
   - Bayesian gate:
     - Judge Bayesian rows only after the baseline gate passes.

Exit criteria:

- A fresh TS run can be compared against a stored Python reference run with one
  command.
- A failing row is classified as one of: BM25/tokenizer, dense embedding,
  evaluator, candidate protocol, or Bayesian fusion.

## Phase 2: Reproduce Sparse BEIR Calibration

Goal: match the Python sparse benchmark on NFCorpus and SciFact before touching
hybrid fusion.

Reference targets:

| Method | NFCorpus NDCG@10 | NFCorpus MAP | SciFact NDCG@10 | SciFact MAP |
| --- | ---: | ---: | ---: | ---: |
| Raw BM25 | 0.5023 | 0.4395 | 0.5900 | 0.5426 |
| Bayesian auto | 0.5050 | 0.4403 | 0.5791 | 0.5283 |
| Bayesian batch fit | 0.5041 | 0.4400 | 0.5826 | 0.5305 |

Implementation tasks:

1. Verify `Raw BM25`.
   - Use `--bm25-method lucene`.
   - Use pretokenized `lower().split()` terms.
   - Use the same query split seed, `42`.
   - Do not judge Bayesian rows until Raw BM25 is within `0.005` absolute
     NDCG/MAP.

2. Implement `base_rate="auto"`.
   - Percentile, mixture, and elbow estimators are implemented.
   - Store the chosen estimator in the result manifest.

3. Implement sparse calibration report.
   - ECE and Brier are reported when `--calibration` is passed.
   - Ranking rows and calibration rows should be separate because monotonic base
     rate transforms may not change rank order.

4. Add split-aware batch fit.
   - Training labels and evaluation labels are separated when `--fit-split`
     is passed.
   - JSON output includes train/test query ids, split seed, train ratio,
     training pair count, and fitted alpha/beta.

Exit criteria:

- Raw BM25 reproduces both selected datasets within `0.005`.
- `base_rate="auto"` reduces ECE by at least 50% versus no-base-rate on both
  datasets.
- Batch fit plus base rate reaches ECE below `0.02` on both datasets when using
  the same split as the reference.

## Phase 3: Reproduce Full BEIR Hybrid Baselines

Goal: make the hybrid benchmark judgeable before implementing more Bayesian
variants.

Reference average NDCG@10:

| Method | Avg NDCG@10 |
| --- | ---: |
| BM25 | 35.38 |
| Dense | 38.32 |
| Convex | 41.15 |
| RRF | 40.49 |

Implementation tasks:

1. Add `Dense` row.
   - Score by cosine similarity or the exact probability mapping used by the
     reference row.
   - Confirm embedding normalization is identical.

2. Add `Convex` row.
   - Match the reference convex blend formula and weight.
   - Do not tune the weight until the zero-shot row reproduces.

3. Confirm `RRF` row.
   - RRF must use the same rank depth and `k` constant.
   - RRF should run only over the sparse+dense top-1000 union for BEIR parity.

4. Confirm candidate protocol.
   - Sparse top `R=1000`.
   - Dense top `R=1000`.
   - Fusion only over the union candidates.
   - Evaluation at `k=10`.

Exit criteria:

- `BM25`, `Dense`, `Convex`, and `RRF` average NDCG@10 are within `0.50` points
  of the Python reference when using the same embeddings.
- Dataset-level deviations are printed so outliers can be debugged.

## Phase 4: Improve Bayesian Hybrid Rows

Goal: reproduce the original Bayesian advantage, then improve carefully.

Reference average NDCG@10:

| Method | Avg NDCG@10 | Delta vs BM25 |
| --- | ---: | ---: |
| BM25 | 35.38 | 0.00 |
| RRF | 40.49 | +5.11 |
| Bayesian-Balanced | 41.50 | +6.12 |
| Bayesian-Attn-Norm | 41.67 | +6.29 |

Implementation tasks:

1. Add `Bayesian-LogOdds`.
   - The same logit-space sparse+dense fusion as the Python reference is
     implemented as `bayesian_logodds`.
   - Tests cover agreement, disagreement, base-rate correction, dense-only
     fallback, and finite probability bounds.

2. Add `Bayesian-LogOdds-BR`.
   - Reuses `base_rate="auto"` or an explicit numeric base rate from Phase 2.
   - JSON scorer metadata marks `bayesian_logodds_br` as a calibration row, not
     a zero-shot row.

3. Stabilize `Bayesian-Balanced`.
   - Use logit-space min-max normalization per query over the union candidates.
   - Match the reference handling for constant or near-constant score ranges.
   - Target: `Bayesian-Balanced >= BM25 + 4.0` average NDCG points and no more
     than `0.50` behind RRF.

4. Add gating rows.
   - Implemented as `bayesian_gated_relu`, `bayesian_gated_swish`, and
     `bayesian_gated_gelu`.
   - Additional diagnostic rows `bayesian_gated_swish_b2` and
     `bayesian_gated_softplus` cover the reference generalized Swish beta and
     softplus gates.
   - Each row uses the active Bayesian sparse probability, base-rate corrected
     when `--base-rate` is configured, plus dense cosine mapped to probability.
   - Keep these as diagnostic rows. The reference shows they are not expected to
     beat `Bayesian-Balanced` on average.

5. Add attention rows.
   - All-qrels reference smoke rows are implemented as `bayesian_attention`,
     `bayesian_attn_norm`, `bayesian_multihead`, and
     `bayesian_multihead_norm`. They are marked as `smoke` and record
     `protocol="all-qrels"` because they train and evaluate on the same qrels.
   - Split-aware `bayesian_attention_split` is implemented behind
     `--fit-split`.
   - Split-aware `bayesian_attn_norm_split` is implemented with seven query
     features and per-query logit normalization.
   - Cross-validated `bayesian_attn_norm_cv` is implemented with five folds
     when enough queries are available, using rich features and per-query logit
     normalization.
   - Split-aware `bayesian_multihead_split` and
     `bayesian_multihead_norm_split` are implemented with four heads.
   - JSON scorer metadata marks these attention rows as `tuned`, and
     `attentionSplits` records train/eval query ids, head count, and training
     pair counts so they are not compared as zero-shot improvements. The CV row
     also records fold-level train/eval query ids.

Exit criteria:

- Baselines pass first.
- `Bayesian-Balanced` reproduces the Python reference within `0.50` NDCG points
  when using the same inputs and evaluator.
- Any attempted improvement is reported as both absolute score and delta versus
  `BM25`, `RRF`, and the stored Python reference.

## Phase 5: Add Advanced Reference Rows

Goal: cover the remaining paper/reference rows without mixing them into baseline
parity.

Implementation tasks:

1. Add `softplus` gating.
   - Implemented as the diagnostic row `bayesian_gated_softplus`.
2. Add `VectorProbabilityTransform`.
   - Implemented in core with likelihood-ratio vector calibration,
     weighted KDE/GMM density estimates, and `ivfDensityPrior` /
     `knnDensityPrior` helpers.
3. Add dense Platt/isotonic calibration rows.
   - Implemented as split-aware calibration rows `dense_platt_split` and
     `dense_isotonic_split`, emitted only when `--fit-split` is configured.
   - JSON metadata records train/eval query ids, training pair count, trained
     flag, and Platt `a`/`b` parameters.
4. Add `MultiFieldScorer`.
   - Implemented in core and exported from `@bb25/core`.
5. Add `Bayesian-MultiField` and `Bayesian-MultiField-Bal`.
   - Implemented as `bayesian_multifield` and `bayesian_multifield_bal` when
     `bb25 bench` receives `--doc-fields title,body --doc-field-terms field_terms`.
6. Add VPT ablations:
   - `Bayesian-Vector-Balanced`
     - Implemented as zero-shot row `bayesian_vector_balanced`.
   - `Bayesian-Vector-Softplus`
     - Implemented as diagnostic row `bayesian_vector_softplus`.
   - `Bayesian-Vector-Attn`
     - Implemented as tuned split-aware row `bayesian_vector_attn_split` with
       `attentionSplits` metadata.

Exit criteria:

- Each advanced row declares whether it is zero-shot, calibration, or tuned.
- Tuned rows must include split metadata and must not be compared as zero-shot
  improvements.

## Failure Classification

Use this table before changing thresholds or tuning parameters.

| Symptom | Likely Cause | First Check |
| --- | --- | --- |
| BM25 differs | tokenizer, Lucene IDF, qrels, corpus text | Dump term stats and top-10 sparse results |
| Dense differs | embedding model, normalization, cache | Compare vector norms and dense top-10 |
| RRF differs but BM25/Dense match | rank tie-break, candidate union, RRF `k` | Compare sparse/dense ranks per query |
| Convex differs but BM25/Dense match | normalization or blend weight | Compare per-query normalized score ranges |
| Bayesian differs but baselines match | probability transform or fusion | Compare sparse probability, dense probability, logit values |
| Calibration improves train but fails test | leakage or overfitting | Check split manifest and tuned parameters |

## Recommended Command Flow

Reference Python:

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

TypeScript sparse export:

```bash
corepack pnpm bench:prepare-beir -- \
  --out-root /tmp/beir-jsonl-sparse \
  --datasets nfcorpus scifact \
  --tokenizer split \
  --manifest-out reference-results/manifests/beir-jsonl-sparse-export.json
```

TypeScript hybrid export:

```bash
corepack pnpm bench:prepare-beir -- \
  --out-root /tmp/beir-jsonl \
  --datasets arguana fiqa nfcorpus scidocs scifact \
  --tokenizer snowball \
  --embed-model all-MiniLM-L6-v2 \
  --embed-cache-dir /tmp/bb25-embedding-cache/all-MiniLM-L6-v2 \
  --manifest-out reference-results/manifests/beir-jsonl-hybrid-export.json
```

TypeScript sparse benchmark:

```bash
corepack pnpm bench:sparse-parity -- \
  --python .venv-bench/bin/python \
  --root /tmp/beir-jsonl-sparse \
  --datasets nfcorpus,scifact \
  --reference-ranking reference-results/python/sparse-benchmark.json \
  --reference-calibration reference-results/python/base-rate.json
```

The runner writes:

- `reference-results/ts/sparse-calibration-ts.json`
- `reference-results/ts/sparse-ranking-ndcg_10-parity.json`
- `reference-results/ts/sparse-ranking-map_10-parity.json`
- `reference-results/ts/sparse-calibration-ece-parity.json`
- `reference-results/ts/sparse-calibration-brier-parity.json`
- `reference-results/ts/sparse-calibration-gate.json`
- `reference-results/manifests/sparse-calibration-parity-runner.json`

The individual commands remain useful for debugging a single failure:

```bash
node scripts/check-bench-json.mjs \
  --reference reference-results/python/base-rate.json \
  --actual reference-results/ts/sparse-calibration-ts.json \
  --methods bayesian,bayesian_fitted_split \
  --metric ece \
  --metric-scale unit \
  --tolerance 0.005 \
  --datasets nfcorpus \
  --out reference-results/ts/sparse-calibration-parity.json \
  --manifest-out reference-results/manifests/sparse-calibration-parity.json

corepack pnpm bench:calibration-gate -- \
  --actual reference-results/ts/sparse-calibration-ts.json \
  --datasets nfcorpus \
  --baseline bayesian_no_base_rate \
  --calibrated bayesian \
  --fitted bayesian_fitted_split \
  --metric ece \
  --min-reduction 0.50 \
  --fitted-max 0.02 \
  --out reference-results/ts/sparse-calibration-gate.json \
  --manifest-out reference-results/manifests/sparse-calibration-gate.json
```

Local SQuAD smoke gate:

```bash
corepack pnpm -r build
corepack pnpm bench:squad-smoke -- \
  --regenerated-embeddings \
  --embedding-cache-dir /tmp/bb25-embedding-cache/bge-m3-q8 \
  --require-embedding-cache \
  --hash-embedding-cache
```

Synthetic fixture-backed gate:

```bash
corepack pnpm bench:synthetic-smoke
```

TypeScript baseline parity against stored Python reference:

```bash
corepack pnpm bench:baseline-parity -- \
  --python .venv-bench/bin/python \
  --root /tmp/beir-jsonl \
  --reference reference-results/python/hybrid-beir.json \
  --datasets arguana,fiqa,nfcorpus,scidocs,scifact \
  --methods BM25,Dense,Convex,RRF \
  --metric ndcg@10 \
  --tolerance-points 0.50
```

Readiness audit:

```bash
corepack pnpm bench:audit-readiness -- \
  --profile all \
  --root reference-results \
  --out reference-results/manifests/readiness-audit.json
```

## Milestone Order

| Milestone | Outcome | Blocks |
| --- | --- | --- |
| M1 | Python reference results and manifests saved | All BEIR judgment |
| M2 | TS BM25 sparse parity on NFCorpus/SciFact | Sparse Bayesian judgment |
| M3 | `pytrec_eval` wrapper or TREC export comparison | BEIR parity gate |
| M4 | Dense/Convex/RRF reproduce hybrid reference | Bayesian hybrid judgment |
| M5 | Bayesian-Balanced reproduces reference | Performance improvement work |
| M6 | Base-rate and calibration rows reproduce reference | Calibration improvements |
| M7 | Attention/MultiField/VPT rows added | Advanced paper parity |
