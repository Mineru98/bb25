# bb25 TypeScript 마이그레이션 설계 문서

대상: instructkr/bb25 (Bayesian BM25). 현재 구현: Rust 코어 + Python 바인딩(PyO3/maturin), Pyodide(WASM) 빌드. 목표: TypeScript 네이티브 라이브러리로 이식, 백엔드와 독립 CLI 공용. 임베딩: transformers.js v4(@huggingface/transformers) + BGE-M3(Xenova/bge-m3, dense).

## 목표
- 코어 알고리즘(BM25 / Bayesian BM25 / Vector / Hybrid)을 TS로 이식
- 임베딩 생성을 코어에서 분리, transformers.js + BGE-M3 기반 교체 가능 구현 제공
- 동일 코어를 백엔드와 독립 CLI 양쪽에서 재사용하게 패키징
- 레퍼런스(Rust/Python) 대비 수치 정합성 검증 테스트 스위트 구축
- ESM 우선, 트리셰이킹 가능한 npm 배포 형태

## 비목표
- 원저자(cognica-io/bayesian-bm25) 프로덕션 레퍼런스 대체 아님
- BGE-M3 sparse·ColBERT 멀티벡터 검색은 1차 제외(dense-only)
- Python 패키지와 ABI/바이너리 호환 추구 안 함(동작 동등성만)
- 대규모 분산 색인/디스크 인덱스 영속화 표준화는 후속

## 아키텍처 (pnpm 모노레포)
bb25-ts/ packages/ core(@bb25/core, 순수 TS 의존성 0: Corpus/Index/스코어러/Embedder 인터페이스/픽스처/실험 러너), embeddings(@bb25/embeddings, @huggingface/transformers 의존, BgeM3Embedder, ONNX는 여기서만), cli(@bb25/cli, bin "bb25": index/search/bench/warmup). fixtures/(레퍼런스 골든값). pnpm-workspace.yaml.

핵심 불변식: @bb25/core는 transformers.js/ONNX/파일시스템에 의존하지 않는다. 입력은 string과 Float32Array/number[]뿐. ONNX 런타임은 @bb25/embeddings에만.

## 코어 API (TS)
```ts
type Vector = Float32Array | number[]
interface Document { id; text; embedding? }
interface CorpusStats { numDocs; avgDocLength }
class Corpus { addDocument(id,text,embedding?); buildIndex(); documents(); get stats }
class BM25Scorer { constructor(corpus, k1=1.2, b=0.75); idf(term); score(queryTerms, doc) }
class BayesianBM25Scorer { constructor(bm25, alpha=1.0, beta=0.5, baseRate?); score(queryTerms, doc): [0,1] }
class VectorScorer { score(queryEmbedding, doc) }
class HybridScorer { constructor(bayes, vector, alpha=0.5); scoreOr(...); scoreAnd(...) }
interface Embedder { dim; embed(texts: string[]): Promise<Vector[]> }
```
score류는 동기 함수. 내부 누적은 number(f64), 레퍼런스의 누적 순서를 그대로 따른다.

## 임베딩 (@bb25/embeddings)
BgeM3Embedder implements Embedder. options: model(기본 Xenova/bge-m3), dtype(fp32/fp16/q8/q4), pooling(기본 cls), normalize(기본 true), device(cpu/wasm/webgpu), cacheDir, localOnly. dim=1024. 내부적으로 pipeline("feature-extraction","Xenova/bge-m3") + {pooling:"cls", normalize:true}. dense-only.

## CLI
```
bb25 index <corpus.jsonl> -o <index.json> [--embed] [--dtype fp16]
bb25 search "<query>" --index <index.json> [--top-k 10] [--mode or|and]
bb25 warmup [--dtype fp16]
bb25 bench <dataset> [--method ws|rrf]
```

## Python ↔ TS 매핑
snake_case → camelCase. 파라미터 순서/기본값은 레퍼런스 그대로. (Corpus, add_document→addDocument, build_index→buildIndex, BM25Scorer(corpus,1.2,0.75), idf, BayesianBM25Scorer(bm25,1.0,0.5), VectorScorer, HybridScorer(bayes,vector,0.5), score_or→scoreOr, score_and→scoreAnd, build_default_corpus→buildDefaultCorpus, build_default_queries→buildDefaultQueries, run_experiments→runExperiments)

## 수치 정합성 검증 (핵심)
"알고리즘 정확성"과 "모델 출력 재현"을 분리.
- 알고리즘: 고정 입력 벡터 주입, 레퍼런스에서 IDF/BM25/Bayesian확률/Hybrid OR·AND/run_experiments 골든값 추출 → fixtures/ 저장. 허용 오차 1e-9~1e-6.
- 토큰화 일치 최우선: Rust 토크나이저 동작 그대로 복제, 토큰화 전용 골든 테스트.
- 부동소수점: f64 동일 표준, 합산 순서 레퍼런스 따름.
- 모델 출력: end-to-end(SQuAD)는 허용 밴드로. 레퍼런스 목표치 WS(BB25+Dense) NDCG@10 0.9149/MRR@10 0.8850, WS(BM25+Dense) 0.9051/0.8717, RRF(BM25+Dense) 0.8874/0.8483. dtype fp32 고정, ±0.5%p 수준 허용(실측 후 조정).

## 로드맵
P0 모노레포·툴체인, 골든 픽스처 추출 / P1 코어 데이터모델+토큰화 / P2 BM25+IDF / P3 BayesianBM25 / P4 Vector+Hybrid(OR/AND) / P5 embeddings(BGE-M3) / P6 cli / P7 벤치마크+SQuAD / P8 패키징·배포. P1~P4가 핵심 경로, 각 단계는 직전 골든 테스트 통과 전제.

---

## 14. 미해결 질문 — 레퍼런스 src/ 확인 결과 (확정)

레퍼런스 소스(`src/tokenizer.rs`, `corpus.rs`, `bm25_scorer.rs`, `bayesian_scorer.rs`,
`vector_scorer.rs`, `hybrid_scorer.rs`, `fusion.rs`, `math_utils.rs`, `defaults.rs`,
`experiments.rs`, `pybindings.rs`)를 직접 읽고 확정한 내용. **추정 없이 코드 그대로 복제.**

### Q1. 토크나이저 동작 (`src/tokenizer.rs`)
```rust
for ch in text.chars() {
    if ch.is_ascii_alphanumeric() { current.push(ch.to_ascii_lowercase()); }
    else if !current.is_empty() { tokens.push(take(current)); }
}
if !current.is_empty() { tokens.push(current); }
```
- **소문자화**: ASCII 전용(`to_ascii_lowercase`). `A–Z`만 소문자로. 비ASCII 문자(é, 한글, 전각 등)는
  `is_ascii_alphanumeric()==false`이므로 **구분자로 취급되어 버려진다**.
- **분절**: 토큰 = `[A-Za-z0-9]`의 최대 연속 런. 그 외 모든 문자(공백, 구두점, 비ASCII)는 구분자.
  예) `"BM25!!"` → `["bm25"]`, `"TF-IDF"` → `["tf","idf"]`, `"café"` → `["caf"]`(é는 구분자), `"한글"` → `[]`.
- **불용어 제거 없음. 스테밍 없음.** 단순 정규화 + 분절뿐.
- **TS 구현 주의**: JS의 `toLowerCase()`/`\w`/`\p{...}`는 유니코드 인지라 동작이 다르다.
  반드시 char code 검사(`0-9`:48–57, `A-Z`:65–90, `a-z`:97–122)로 ASCII 전용 의미를 복제할 것.
  `String.prototype.toLowerCase`를 쓰면 안 되고, `A-Z`만 +32 하는 방식으로 직접 처리.

### Q2. Bayesian 파라미터 `(1.0, 0.5)`의 의미와 점수→확률 변환식 (`src/bayesian_scorer.rs`)
- 설계 초안의 `priorA/priorB`라는 이름은 **오해**. 실제 시그니처는
  `BayesianBM25Scorer::new(bm25, alpha=1.0, beta=0.5, base_rate=None)`이며 **alpha/beta는 시그모이드
  우도(likelihood)의 파라미터**다. `experiments.rs`에서 `(1.0, 0.5, None)`으로 생성.
- 변환 파이프라인 (`score_term` → `score`):
  1. `raw = bm25.score_term_standard(term, doc)`. `raw == 0`이면 그 항은 0(이후 OR에서 제외).
  2. `likelihood(raw) = sigmoid(alpha * (raw - beta))`.
  3. `prior = composite_prior(tf, docLen, avgdl)`:
     - `tf_prior(tf) = 0.2 + 0.7 * min(1, tf/10)`
     - `norm_prior(docLen, avgdl) = avgdl<1 ? 0.5 : 0.3 + 0.6*(1 - min(1, |docLen/avgdl - 0.5|*2))`
     - `composite = clamp(0.7*tf_prior + 0.3*norm_prior, 0.1, 0.9)`
  4. `posterior(raw, prior)`: `lik = safe_prob(likelihood(raw))`, `p = safe_prob(prior)`,
     `p1 = lik*p / (lik*p + (1-lik)*(1-p))`. `base_rate` 있으면 2단계 업데이트
     `p2 = p1*br / (p1*br + (1-p1)*(1-br))`, 없으면 `p1`.
     **주의: `BayesianBM25Scorer::posterior`는 결과에 `safe_prob`를 적용하지 않는다**
     (별도 클래스 `BayesianProbabilityTransform::posterior`는 적용함 — 혼동 금지).
  5. `score(queryTerms, doc)`: 각 term의 `score_term`을 term 순서대로 구해 `> 0`인 것만 모아
     `fusion::prob_or(posteriors)`로 결합. 비면 0.
- `safe_prob(p) = clamp(p, 1e-10, 1-1e-10)`, `EPSILON = 1e-10`.
- `sigmoid`는 분기형(수치 안정): `x>=0 ? 1/(1+e^-x) : e^x/(1+e^x)`.

### Q3. Hybrid `score_or`/`score_and` 결합 수식 (`src/hybrid_scorer.rs`, `src/fusion.rs`)
- `HybridScorer::new(bayesian, vector, alpha=0.5)`.
- `score_or(terms, emb, doc)`:
  `b = bayesian.score(terms, doc)`, `v = vector.score(emb, doc)`,
  `return prob_or([b, v]) = 1 - exp( Σ ln(1 - safe_prob(p_i)) )`.
- `score_and(terms, emb, doc)`:
  `b, v` 계산 → `b < EPSILON && v < EPSILON`이면 **0** 반환(단락).
  아니면 `probabilistic_and([b, v])`.
- `probabilistic_and = log_odds_conjunction(probs, alpha=Some(0.5), weights=None, NoGating)`
  = `sigmoid( mean_i(logit(safe_prob(p_i))) * n^0.5 )`, 여기서 `n = probs.length`.
  (즉 **곱셈 규칙 `prob_and`가 아니라 로그-오즈 결합**. `prob_and`(곱)는 De Morgan 테스트 등에만 사용.)
- `logit(p) = ln(p/(1-p))` with `p = clamp(p, 1e-10, 1-1e-10)`.
- `rrf_score(ranks, k=60) = Σ 1/(k + rank_i)`. `naive_sum = Σ scores`.

### Q4. 기본 코퍼스/쿼리 임베딩 차원 & VectorScorer 차원 비의존성 (`src/vector_scorer.rs`, `math_utils.rs`)
- 기본 코퍼스/쿼리는 **8차원** 임베딩(`defaults.rs`). BGE-M3는 1024차원(P5).
- `VectorScorer::score(q, doc) = score_to_probability(cosine_similarity(q, doc.embedding))`.
  `score_to_probability(sim) = clamp((1+sim)/2, 0, 1)`.
- `cosine_similarity(a, b)`: `dot = Σ a_i*b_i` (**`zip`이므로 길이가 다르면 짧은 쪽까지만**),
  `mag = sqrt(Σ v_i^2)` (각 벡터 전체), `mag_a<EPSILON || mag_b<EPSILON`이면 0,
  아니면 `dot/(mag_a*mag_b)`.
- **차원 비의존**: dim에 대한 assert 없음. q와 doc 임베딩 차원이 같으면 정상, 다르면 zip이 짧은 쪽으로
  잘리는 동작까지 그대로 복제해야 함. 8차원이든 1024차원이든 동일 코드로 동작.
- **f64 정합성 주의**: 기본 코퍼스 임베딩은 Rust `Vec<f64>` 리터럴. TS에서는 `number[]`(f64)로 보관해야
  골든과 일치. `Float32Array`(f32 반올림)를 쓰면 값이 달라진다 — BGE-M3 경로에서만 Float32Array 허용.

### Q5. 배포 범위
- 코드상 결정 아님(설계 선택). 코어(@bb25/core)는 fs/ONNX 비의존 → 브라우저 안전.
- **1차: Node/CLI 중심**으로 패키징·검증. 코어는 순수 TS라 브라우저 번들도 가능하지만
  브라우저 E2E는 후속(P8). embeddings는 device(cpu/wasm/webgpu) 옵션으로 브라우저 확장 여지 둠.

### Q6. 인덱스 영속화 포맷
- **레퍼런스에 인덱스 직렬화 포맷이 존재하지 않음** (Rust는 메모리상 `Corpus`만 빌드, 디스크 저장 없음).
- 따라서 `index.json`은 **신규 스키마**로 정의(P6). 코어 정합성과 무관하므로 P0~P4 범위 밖.
  잠정 스키마: `{ version, params:{k1,b,alpha,beta,baseRate}, documents:[{id,text,embedding?}], stats:{n,avgdl,df} }`.

### 보충: 누적 순서 / 부동소수점
- 모든 합산은 레퍼런스 iterator 순서(좌→우)를 그대로. TS는 명시적 for-loop로 순서 고정.
- JS `number` == IEEE754 f64 == Rust `f64`. 동일 연산·동일 순서면 비트 단위 동일 결과 기대.
- 허용 오차: 알고리즘 골든 1e-12(사실상 정확), 안전 마진 1e-9.
</content>
</invoke>
