# Benchmark Implementation Plan

This plan turns the benchmark rubric into executable work. The priority is:

1. Make benchmark results judgeable.
2. Reproduce reference baselines.
3. Improve Bayesian and hybrid rows only after the baseline protocol is aligned.

Do not tune for higher numbers until the benchmark can explain whether a change
is an implementation improvement, a protocol drift, or data leakage.

## Current State

Implemented in this repository:

- Tier 0 golden fixtures and seeded property tests.
- CLI benchmark over JSONL docs, queries, qrels, and optional embeddings.
- `recall@k`, JSON output, TREC-style four-column qrels.
- Lucene-style BM25 IDF option.
- Pretokenized doc/query term paths.
- Hybrid sparse+dense top-R union via `--candidate-depth`.
- Reference hybrid baseline rows: `dense`, `convex`, and `rrf`.
- TREC run export via `bb25 bench --trec-run-dir`.
- `pytrec_eval` wrapper for scorer run files.
- JSON comparison gate for Python-vs-TS baseline parity.
- `--base-rate auto` with percentile, mixture, and elbow estimators for
  Bayesian sparse rows.
- Optional sparse calibration report via `--calibration` with ECE and Brier.
- Split-aware batch fit via `--fit-split`, emitted as `bayesian_fitted_split`
  with train/eval query ids and learned alpha/beta metadata.
- Hybrid Bayesian LogOdds rows: `bayesian_logodds` and, when a base rate is
  configured, `bayesian_logodds_br`. The BR row is marked as calibration
  metadata.
- Gated Bayesian hybrid diagnostic rows: `bayesian_gated_relu`,
  `bayesian_gated_swish`, `bayesian_gated_gelu`,
  `bayesian_gated_swish_b2`, and `bayesian_gated_softplus`.
- Split-aware tuned attention rows behind `--fit-split`:
  `bayesian_attention_split`, `bayesian_attn_norm_split`,
  `bayesian_multihead_split`, and `bayesian_multihead_norm_split`, with
  train/eval query ids, feature set, normalization flag, head count, and
  training pair count metadata.
- Multi-field Bayesian rows when field terms are supplied:
  `bayesian_multifield` and `bayesian_multifield_bal`.
- SQuAD smoke manifest and table comparison script.
- BEIR JSONL export and TS runner scripts.

Not yet judgeable against the Python reference:

- The local environment currently lacks Python BEIR dependencies:
  `ir_datasets`, `sentence_transformers`, and `snowballstemmer`.
- The local environment currently lacks `pytrec_eval`; the wrapper exists but
  requires the benchmark Python environment.
- TS does not yet emit all reference hybrid rows: reference all-qrels/CV
  attention names and VPT rows are incomplete or missing. The implemented
  baseline rows are emitted using lowercase CLI names: `bm25`, `dense`,
  `convex`, and `rrf`.
- The legacy `bayesian_fitted` row remains an all-qrels smoke row. Use
  `bayesian_fitted_split` for reference calibration claims.
- Reference Python benchmark outputs are not yet checked into a stable local
  result directory with manifests.

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
   - Install `ir_datasets`, `sentence-transformers`, `pytrec_eval`, and
     `snowballstemmer`.
   - Record exact versions with `pip freeze`.
   - Store the Python version and platform in the manifest.

2. Run the reference Python benchmarks.
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
   - Split-aware `bayesian_attention_split` is implemented behind
     `--fit-split`.
   - Split-aware `bayesian_attn_norm_split` is implemented with seven query
     features and per-query logit normalization.
   - Split-aware `bayesian_multihead_split` and
     `bayesian_multihead_norm_split` are implemented with four heads.
   - JSON scorer metadata marks these attention rows as `tuned`, and
     `attentionSplits` records train/eval query ids, head count, and training
     pair counts so they are not compared as zero-shot improvements.

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
3. Add dense Platt/isotonic calibration rows.
4. Add `MultiFieldScorer`.
   - Implemented in core and exported from `@bb25/core`.
5. Add `Bayesian-MultiField` and `Bayesian-MultiField-Bal`.
   - Implemented as `bayesian_multifield` and `bayesian_multifield_bal` when
     `bb25 bench` receives `--doc-fields title,body --doc-field-terms field_terms`.
6. Add VPT ablations:
   - `Bayesian-Vector-Balanced`
   - `Bayesian-Vector-Softplus`
   - `Bayesian-Vector-Attn`

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
python -m venv .venv-bench
. .venv-bench/bin/activate
pip install ir_datasets sentence-transformers pytrec_eval snowballstemmer
pip freeze > reference-results/manifests/python-freeze.txt

python /tmp/cognica-bayesian-bm25/benchmarks/benchmark.py \
  -o reference-results/python/sparse-benchmark.json
python /tmp/cognica-bayesian-bm25/benchmarks/base_rate.py \
  -o reference-results/python/base-rate.json
python /tmp/cognica-bayesian-bm25/benchmarks/hybrid_beir.py \
  -d /tmp/beir \
  --download \
  --datasets arguana fiqa nfcorpus scidocs scifact \
  --model all-MiniLM-L6-v2 \
  -R 1000 \
  -k 10 \
  -o reference-results/python/hybrid-beir.json
```

TypeScript sparse:

```bash
python scripts/prepare-beir-jsonl.py \
  --dataset nfcorpus \
  --out /tmp/beir-jsonl/nfcorpus \
  --tokenizer split
corepack pnpm --filter @bb25/cli build
node scripts/run-beir-jsonl-bench.mjs \
  --root /tmp/beir-jsonl \
  --datasets nfcorpus \
  --bm25-method lucene \
  --base-rate auto \
  --base-rate-method percentile \
  --fit-split \
  --fit-train-ratio 0.5 \
  --fit-split-seed 42 \
  --calibration \
  --cutoffs 10 \
  --out reference-results/ts/sparse-nfcorpus.json
```

TypeScript TREC/pytrec judgment:

```bash
node scripts/run-beir-jsonl-bench.mjs \
  --root /tmp/beir-jsonl \
  --datasets arguana,fiqa,nfcorpus,scidocs,scifact \
  --doc-embedding embedding \
  --query-embedding embedding \
  --doc-fields title,body \
  --doc-field-terms field_terms \
  --candidate-depth 1000 \
  --trec-run-dir reference-results/ts/runs \
  --trec-run-depth 1000 \
  --bm25-method lucene \
  --cutoffs 10 \
  --out reference-results/ts/hybrid-beir-internal.json

python scripts/evaluate-trec-run.py \
  --root /tmp/beir-jsonl \
  --datasets arguana,fiqa,nfcorpus,scidocs,scifact \
  --runs reference-results/ts/runs \
  --cutoffs 10 \
  --out reference-results/ts/hybrid-beir-pytrec.json
```

TypeScript hybrid:

```bash
python scripts/prepare-beir-jsonl.py \
  --dataset scifact \
  --out /tmp/beir-jsonl/scifact \
  --tokenizer snowball \
  --embed-model all-MiniLM-L6-v2
corepack pnpm --filter @bb25/cli build
node scripts/run-beir-jsonl-bench.mjs \
  --root /tmp/beir-jsonl \
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
  --out reference-results/ts/hybrid-scifact.json
```

Baseline parity gate:

```bash
node scripts/check-bench-json.mjs \
  --reference reference-results/python/hybrid-beir.json \
  --actual reference-results/ts/hybrid-beir-pytrec.json \
  --methods BM25,Dense,Convex,RRF \
  --metric ndcg@10 \
  --tolerance-points 0.50
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
