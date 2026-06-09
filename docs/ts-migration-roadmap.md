# bb25 TypeScript 마이그레이션 — 잔여 작업 명세 & 구현 계획

대상: `feat/ts-migration` 브랜치. 본 문서는 **P5~P8 + 부가 코어 모듈 전체 패리티**의 명세와 구현 순서를 정의한다.
[ts-migration-design.md](./ts-migration-design.md)(설계 + §14 답)와 짝을 이룬다.

## 0. 현재 상태 (완료, 커밋 `5ab2ccd`)
- pnpm 모노레포 + `@bb25/core`(순수 TS, 의존성 0).
- P1~P4: Tokenizer / Corpus / BM25 / BayesianBM25 / Vector / Hybrid(OR·AND) / fusion 프리미티브.
- 골든 추출기 `examples/extract_golden.rs` → `fixtures/golden.json`. 코어 테스트 32/32 통과,
  점수 700건 중 676건 비트 일치, 최대 오차 1.11e-16.

## 1. 확정된 결정 (사용자 합의)
1. **범위: 전체 기능 패리티** — 레퍼런스 `src/` 전 모듈을 TS로 포팅.
2. **벤치마크: 목표 NDCG 재현 시도** — 레퍼런스 임베딩 캡처 + 실제 SQuAD 실행으로 설계 목표치(WS 0.9149 등)를 ±밴드 내 재현 검증.
3. **임베딩 정합성: 레퍼런스 캡처 + 주입(권장)** — 알고리즘 패리티와 모델 출력 재현을 분리.

## 2. 정합성 검증 2-트랙 원칙 (전 구간 공통)
- **트랙 A — 알고리즘 정확성 (정확 일치)**: 결정론적. 입력을 고정(임베딩도 고정값 주입)하고 레퍼런스에서
  골든값 추출 → TS가 `|a−b| ≤ 1e-12 + 1e-9·|b|`로 통과. 학습형 모듈(fit/update)도 결정론적이므로
  동일하게 골든 일치 가능. **RNG는 BigInt LCG로 비트 복제(§4.3).**
- **트랙 B — 모델 출력 재현 (밴드)**: transformers.js(ONNX) ≠ sentence-transformers(PyTorch)이므로
  임베딩은 비트 동일 불가. (a) TS 임베더는 캡처본 대비 코사인 유사도 ≥ 0.99, L2 norm ≈ 1, dim=1024로 검증.
  (b) end-to-end 랭킹은 **캡처 임베딩을 양쪽에 주입**해 알고리즘만 정확 비교, 그 뒤 TS 임베더 임베딩으로
  목표 NDCG를 ±0.5%p 밴드 내 재현.

## 3. 디렉터리 / 패키지 최종 형태
```
packages/
  core/         @bb25/core      (순수 TS, 전 알고리즘 — 부가 모듈 포함)
  embeddings/   @bb25/embeddings(@huggingface/transformers, BgeM3Embedder; ONNX는 여기서만)
  cli/          @bb25/cli       (bin "bb25": index/search/bench/warmup)
fixtures/
  golden.json            (코어 스칼라 골든 — 확장)
  golden_modules.json    (부가 모듈 골든 — 신규)
  embeddings/*.json       (sentence-transformers BGE-M3 캡처본)
  bench/squad/*           (docs/queries/qrels + reference 벤치 결과)
scripts/
  extract_golden (examples/extract_golden.rs 확장)
  capture_embeddings.py   (sentence-transformers 캡처)
  prepare_squad.py        (SQuAD → JSONL)
```

## 4. 부가 코어 모듈 포팅 (Phase A — P5 이전 선행)

레퍼런스 매핑(`src/lib.rs`, `src/pybindings.rs` 기준). 각 모듈은 **포팅 + 골든 추출 + 테스트**가 1세트.

### 4.1 의존 순서 (병렬 가능 그룹 표시)
1. `mathUtils`에 **`softmaxRows`** 추가(attention/multihead가 사용). *(현재 누락)*
2. **metrics** (`metrics.rs` → `metrics.ts`): `expectedCalibrationError`, `brierScore`, `reliabilityDiagram`, `calibrationReport`. 의존 없음, 단독.
3. **ParameterLearner** (`parameter_learner.rs`): `crossEntropyLoss`, `learn`(GD, 기본 lr=0.01/iter=1000/tol=1e-6; pybinding 기본 lr=0.1/iter=500/tol=1e-8 주의 — exp9는 후자). 결과 `{alpha,beta,lossHistory,converged}`.
4. **runExperiments** (`experiments.rs`): `Query`, `ExperimentRunner`(k1,b 기본 1.2/0.75; 내부 Bayesian (1.0,0.5,None), Hybrid alpha=0.5), `runAll()` → exp1~13의 `{name,passed,details}`. ParameterLearner(0.1,500,1e-8) 사용.
5. **probability** (`probability.rs`): `BayesianProbabilityTransform`(alpha/beta/baseRate, `likelihood`/`tfPrior`/`normPrior`/`compositePrior`/`posterior`(여기는 결과 safeProb 적용 — 코어 BayesianBM25와 다름)/`scoreToProbability`/`wandUpperBound`/`fit`/`update`), `TrainingMode`(balanced/priorAware/priorFree), `TemporalBayesianTransform`(decayHalfLife, 가중 fit/update). pybinding 기본 beta=0.0 주의(코어 스코어러는 0.5).
6. **calibration** (`calibration.rs`): `PlattCalibrator`(a,b; fit GD; calibrate/calibrateBatch), `IsotonicCalibrator`(PAVA fit; calibrate/calibrateBatch).
7. **learnable_weights** (`learnable_weights.rs`): `LearnableLogOddsWeights`(nSignals, alpha=0, baseRate?; `combine`/`fit`/`update`, weights/averagedWeights).
8. **attention_weights** (`attention_weights.rs`): `AttentionLogOddsWeights`(§4.3 RNG). `combine`/`fit`/`update`/`computeUpperBounds`/`prune`.
9. **multi_head_attention** (`multi_head_attention.rs`): `MultiHeadAttentionLogOddsWeights`(nHeads, …). 동일 RNG 스킴(헤드별 시드) 확인 후 복제.
10. **block_max_index** (`block_max_index.rs`): `BlockMaxIndex`(blockSize=128; `build`/`blockUpperBound`/`bayesianBlockUpperBound`).
11. **debug** (`debug.rs`): `FusionDebugger` + 트레이스 구조체들. 최후순위(진단용).

### 4.2 파라미터 기본값 표 (반드시 레퍼런스 그대로)
| 클래스 | 생성 기본값 | 비고 |
|---|---|---|
| BM25Scorer | k1=1.2, b=0.75 | ✅완료 |
| BayesianBM25Scorer | alpha=1.0, beta=0.5, baseRate=None | ✅완료. posterior 결과 **non-clamped** |
| HybridScorer | alpha=0.5 | ✅완료 |
| BayesianProbabilityTransform | alpha=1.0, **beta=0.0**, baseRate=None | posterior **clamped** |
| ParameterLearner | lr=0.01, iter=1000, tol=1e-6 | exp9는 (0.1,500,1e-8) |
| LearnableLogOddsWeights | alpha=0.0 | weighted 경로(alpha 기본 0.0) |
| AttentionLogOddsWeights | alpha=0.5, normalize=false, seed=0 | RNG 복제 필수 |
| MultiHeadAttention | alpha=0.5, normalize=false | |
| BlockMaxIndex | blockSize=128 | |
| fit/update 공통 | lr=0.01, momentum=0.9, decayTau=1000, maxGradNorm=1.0, avgDecay=0.995 | online |

### 4.3 RNG 비트 복제 (정합성 핵심)
`simple_normal_init`(LCG + Box–Muller)을 TS에서 **BigInt**로 복제:
```ts
let state = (BigInt(seed) + 1n) & MASK64;            // MASK64 = (1n<<64n)-1n
state = (state * 6364136223846793005n + 1442695040888963407n) & MASK64;
const u1 = Math.max(Number(state >> 11n) / 2**53, 1e-15);
// ... 두 번째 draw로 u2, Box–Muller: r=sqrt(-2 ln u1), θ=2π u2
// push r*cos(θ)*scale, r*sin(θ)*scale ; n개로 truncate
```
`Number(2^53)` 정밀도, `wrapping_mul/add`=`& MASK64`. multi_head도 동일 스킴인지 소스 확인 후 적용.

### 4.4 골든 확장
- `examples/extract_golden.rs`에 신규 섹션 추가 → 별도 파일 `fixtures/golden_modules.json` 권장.
- 결정론 추출 대상: metrics(고정 prob/label), ParameterLearner.learn(exp9 입력), runExperiments 전체
  `{name,passed,details}`, ProbabilityTransform.{scoreToProbability,posterior,fit후 alpha/beta},
  calibrators.{fit후 파라미터, calibrate batch}, learnable/attention/multihead.{초기 weightsMatrix(시드),
  combine, fit후 weights, prune 결과}, blockMaxIndex.{blockUpperBound, bayesianBlockUpperBound}.
- 테스트: `packages/core/test/modules/*.test.ts` (모듈별), 트랙 A 허용오차.

## 5. P5 — `@bb25/embeddings` (BgeM3Embedder)
### 명세
- `BgeM3Embedder implements Embedder`. opts: `model`(기본 `Xenova/bge-m3`), `dtype`(fp32|fp16|q8|q4, 기본 fp32),
  `pooling`(기본 `cls`), `normalize`(기본 true), `device`(cpu|wasm|webgpu), `cacheDir`, `localOnly`. `dim=1024`.
- 내부: `pipeline("feature-extraction", model)` + `{ pooling:"cls", normalize:true }`. dense-only.
- `embed(texts): Promise<Float32Array[]>` 배치 처리.
### 검증 (트랙 B)
- 단위: dim=1024, `normalize:true`일 때 ‖v‖₂≈1(±1e-4), 동일 입력 반복 시 동일 출력(결정론).
- 의미 검증: 알려진 관련/무관 문장쌍에서 코사인 순서(관련>무관).
- 캡처 대조: `fixtures/embeddings/`의 sentence-transformers BGE-M3 캡처본과 코사인 유사도 ≥ 0.99
  (dtype=fp32 기준). 미달 시 pooling/normalize/정규화 위치 점검.
### 캡처 스크립트
- `scripts/capture_embeddings.py`: `SentenceTransformer("BAAI/bge-m3")`로 SQuAD 샘플 doc/query 임베딩 → JSON.
  (벤치 러너의 `encode_embeddings`와 동일 모델·풀링; sentence-transformers BGE-M3 = cls 풀링 + 정규화.)

## 6. P6 — `@bb25/cli` (bin `bb25`)
### 명령 명세
```
bb25 index  <corpus.jsonl> -o <index.json> [--embed] [--dtype fp32] [--model Xenova/bge-m3]
bb25 search "<query>" --index <index.json> [--top-k 10] [--mode or|and|bm25|bayesian]
bb25 warmup [--dtype fp32]          # 모델 프리페치/캐시 워밍
bb25 bench  <dir|--docs --queries --qrels> [--method bm25|bayesian|hybrid_or|hybrid_and|balanced|rrf|ws] [--cutoffs 5,10,20,100]
```
- `index`: corpus.jsonl(`{doc_id,text,embedding?}`) 로드 → Corpus 빌드 → (`--embed` 시 @bb25/embeddings로 임베딩) → index.json 저장.
- `search`: index.json 로드 → 질의 토큰화(코어 Tokenizer) → (`--embed` 임베딩 or 인덱스 임베딩) → 점수 → `(-score, id)` 정렬 top-k.
- CLI는 `@bb25/core` + `@bb25/embeddings`만 의존. 파서는 의존성 최소(`node:util parseArgs`).
### index.json 스키마 (신규 — §14 Q6, 레퍼런스 포맷 없음)
```jsonc
{
  "version": 1,
  "params": { "k1":1.2, "b":0.75, "alpha":1.0, "beta":0.5, "baseRate":null, "hybridAlpha":0.5 },
  "embedder": { "model":"Xenova/bge-m3", "dim":1024, "dtype":"fp32", "pooling":"cls", "normalize":true } | null,
  "documents": [ { "id":"d01", "text":"…", "embedding":[…] | null } ],
  "stats": { "n":20, "avgdl":10.8 }   // df는 로드 시 재계산(buildIndex)으로 충분
}
```
(df를 직렬화할지 여부는 구현 시 결정 — 재계산이 단순하고 작은 코퍼스엔 충분.)

## 7. P7 — 벤치마크 + SQuAD
### 7.1 하니스 (코어/별도 `packages/cli` 또는 `@bb25/bench`)
`run_benchmark.py`를 TS로 충실 포팅:
- 메트릭: `averagePrecisionAtK`, `dcgAtK`(게인 `2^rel−1`, `log2(idx+1)`), `ndcgAtK`, `mrrAtK`.
- **`rankDocs`: `sort by (-score, doc_id 오름차순)`** — 동점 타이브레이크까지 정확 복제(랭킹 정합성 필수).
- 로더: `loadJsonl`/`loadDocs`/`loadQueries`/`loadQrels`(tsv·jsonl), `parseCutoffs`.
- 스코어러 평가 함수 전부: `evaluate`(bm25/bayesian), `evaluateFittedBayesian`(ProbabilityTransform.fit),
  `evaluateHybrid`(or/and), `evaluateBalancedFusion`, `evaluateGatedFusion`(relu/swish/gelu),
  `evaluateLearnedWeightsFusion`, `evaluateAttentionFusion`, `evaluateMultiHeadAttentionFusion`,
  `collectCalibrationData`(ECE/Brier/reliability). → 4.x 모듈 전부 선행 필요.
- CLI `--method ws|rrf`: ws=balanced/weighted log-odds, rrf=RRF(`hybrid.rrfScore`) 기반 랭킹.
### 7.2 SQuAD 준비
- `scripts/prepare_squad.py`: SQuAD(v1.1/v2) → `docs.jsonl`(context 단락=doc), `queries.jsonl`(question),
  `qrels`(question→정답 포함 단락 rel=1). 규모/버전은 실행 시 인자.
- 임베딩: `scripts/capture_embeddings.py`로 BGE-M3 임베딩을 docs/queries에 부여(또는 벤치 러너 `--embedding-model`).
### 7.3 패리티 검증
- **트랙 A (정확)**: 동일 SQuAD 슬라이스 + **동일 캡처 임베딩**을 Python `run_benchmark.py`와 TS 하니스에
  주입 → 스코어러별 `ndcg@k/map@k/mrr@k`가 트랙 A 허용오차 내 일치(러너 산술이 동일하므로 정확 일치 기대).
  → `fixtures/bench/squad/reference_results.json` 골든화.
- **트랙 B (밴드)**: TS 임베더(transformers.js)로 생성한 임베딩으로 end-to-end 실행 →
  설계 목표(WS BB25+Dense NDCG@10 0.9149/MRR@10 0.8850, WS BM25+Dense 0.9051/0.8717,
  RRF 0.8874/0.8483)를 ±0.5%p 밴드 내 재현. 미달 시 풀링/정규화/dtype·결합 가중치 점검 후 밴드 재조정.

## 8. P8 — 패키징 / 배포
- 빌드: `tsc`(현행). 필요시 `tsup`로 ESM+d.ts 번들 일원화 검토.
- 각 패키지 `exports`/`types`/`sideEffects:false`/`files:["dist"]` 정비. 트리셰이킹 확인.
- `@bb25/core` 0-dep 불변식 CI 검사(번들 그래프에 onnx/fs 없음). `@bb25/embeddings`만 peerDeps `@huggingface/transformers`.
- CLI: `bin` 등록, `npx bb25` 동작, shebang.
- 1차 배포 대상 Node ≥20. 브라우저 E2E(코어/임베딩 wasm·webgpu)는 옵션, 후속.
- README(루트 + 패키지별), CHANGELOG, 버전(`0.x`).

## 9. 작업 순서 (권장 실행 시퀀스)
1. **A1** `softmaxRows` 추가 + metrics + ParameterLearner + runExperiments(빠른 골든 승리).
2. **A2** probability(Transform/Temporal) + calibration.
3. **A3** learnable → attention(RNG) → multi_head → block_max_index → debug. 골든 `golden_modules.json`.
4. **P5** embeddings 패키지 + 캡처 스크립트 + 트랙 B 단위/대조 테스트.
5. **P6** CLI + index.json + 명령 e2e(소형 코퍼스).
6. **P7** 벤치 하니스 포팅 → SQuAD 준비 → 트랙 A 정확 패리티 → 트랙 B 목표 재현.
7. **P8** 패키징/배포 정비.

각 단계는 **직전 골든/테스트 통과 전제**. 모듈 추가 시 `extract_golden.rs`를 함께 확장하고 `pnpm -r test` 녹색 유지.

## 10. 리스크 & 완화
- **RNG 정합성**(attention/multihead): u64 wrapping → **BigInt LCG**로 해결(§4.3). 초기 weights 골든으로 즉시 검증.
- **부동소수 누적 순서**: 행렬곱/그래디언트 합산을 레퍼런스 루프 순서대로(좌→우, row/col/k 순) 고정.
- **임베딩 모델 불일치**: 트랙 분리로 흡수. sentence-transformers BGE-M3 vs Xenova/bge-m3 풀링 차이가 있으면
  캡처 대조에서 즉시 드러남 → pooling/normalize 옵션 조정.
- **SQuAD 규모/시간**: 캡처·실행 비용 큼. 슬라이스(예: 1k 질의)로 트랙 A 검증 후, 전체는 사용자 실행 단계로 분리.
- **pybinding vs 코어 기본값 차이**(beta 0.0 vs 0.5 등): 4.2 표로 고정, 테스트에서 양쪽 경로 모두 커버.

## 11. 완료 기준 (Definition of Done)
- `@bb25/core`가 레퍼런스 전 모듈 API를 camelCase로 노출, 모든 모듈 골든 트랙 A 통과.
- `@bb25/embeddings`가 dim=1024·norm≈1·캡처 코사인 ≥0.99 통과.
- `@bb25/cli` index/search/warmup/bench 동작, index.json 라운드트립.
- 벤치 하니스가 트랙 A에서 Python 러너와 정확 일치(주입 임베딩), 트랙 B에서 목표 NDCG ±0.5%p.
- `pnpm -r build && pnpm -r test` 녹색, 0-dep 코어 불변식 유지.
</content>
