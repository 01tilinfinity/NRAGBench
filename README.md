# NRAGBench

`NRAGBench`는 부정 제약 조건이 포함된 질의에서 RAG 검색 병목을 분석하기 위한 벤치마크 프로젝트입니다.
현재 저장소의 중심 실험은 `Track 1: Retrieval Bottleneck`이며, `exclusion`과 `explicit_negation`을 분리해 평가할 수 있도록 구성되어 있습니다.

## 1. 현재 실험 목표

이 실험은 Dense Retriever가 부정 신호를 제대로 반영하지 못해 발생하는 `semantic collapse`를 측정합니다.

핵심 실패 유형은 두 가지입니다.

- `Misunderstanding`: `Q_neg`에 대해 `D_gold`보다 `D_distractor`를 더 높게 랭크하는 경우
- `Absence`: `D_gold`를 제거했을 때 `D_distractor`가 top-1이 되어, 답이 없는 상황에서도 잘못된 문서를 정답처럼 고르는 경우

## 2. 데이터 구조

각 샘플은 다음 구조를 갖습니다.

```json
{
  "sample_id": "34300-neg-exclusion",
  "query_pos": "What is cost of sales?",
  "query_neg": "What is cost of sales excluding the cost of goods sold?",
  "doc_gold": {
    "doc_id": "34300_gold",
    "text": "..."
  },
  "doc_distractor": {
    "doc_id": "34300_dist",
    "text": "..."
  },
  "answer_gold": "...",
  "excluded_target": "cost of goods sold",
  "negation_type": "exclusion",
  "domain": "factual",
  "corpus": [
    {"doc_id": "34300_gold", "text": "...", "label": "gold"},
    {"doc_id": "34300_dist", "text": "...", "label": "distractor"},
    {"doc_id": "other_doc", "text": "...", "label": "background"}
  ]
}
```

필드 의미:

- `query_pos`: positive query
- `query_neg`: negation-aware query
- `doc_gold`: 부정 제약을 만족하는 정답 문서
- `doc_distractor`: 어휘적으로는 비슷하지만 부정 제약을 위반하는 방해 문서
- `corpus`: `gold`, `distractor`, `background`를 포함한 retrieval 대상 문서 집합
- `negation_type`: 현재는 `exclusion` 또는 `explicit_negation`

## 3. 환경 설정

```bash
conda env create -f environment.yml
conda activate negrag-msmarco
```

프로젝트 루트에 `.env` 파일을 두고 OpenAI 정보를 채웁니다.

```bash
cp .env.example .env
```

```env
OPENAI_API_KEY=your_api_key
OPENAI_MODEL=gpt-4.1-mini
```

## 4. Track 1 Benchmark 생성

MS MARCO에서 샘플을 뽑아 OpenAI로 `query_neg`, `doc_gold`, `doc_distractor`를 생성합니다.

### 4-1. 기본 생성

```bash
conda run -n negrag-msmarco python src/build_track1_benchmark.py \
  --sample-size 100 \
  --background-docs 18 \
  --output-path outputs/track1_benchmark_100.jsonl
```

### 4-2. `exclusion : explicit_negation = 50:50` balanced 생성

```bash
conda run -n negrag-msmarco python src/build_track1_benchmark.py \
  --sample-size 100 \
  --background-docs 18 \
  --candidate-multiplier 8 \
  --target-types exclusion explicit_negation \
  --output-path outputs/track1_benchmark_100_balanced.jsonl
```

주의:

- `sample-size`는 `target-types` 개수로 나누어떨어져야 합니다.
- 현재 balanced 설정은 `exclusion`과 `explicit_negation`만 강제합니다.
- 생성 프롬프트는 [track1_triplet_generation.txt](/Users/canho/Desktop/codes/ai/NegRag/prompts/track1_triplet_generation.txt)에서 수정할 수 있습니다.

## 5. Retrieval 평가

비교 대상:

- Dense: `BAAI/bge-small-en-v1.5`
- Dense: `intfloat/e5-small-v2`
- Sparse: BM25
- Hybrid: Dense + BM25 RRF

실행:

```bash
conda run -n negrag-msmarco python src/run_retrieval_benchmark.py \
  --input-path outputs/track1_benchmark_100_balanced.jsonl \
  --output-dir outputs/retrieval_benchmark_100_balanced
```

평가 지표:

- `SRN`: `Score(Q_neg, D_gold) > Score(Q_neg, D_dist)` 비율
- `MRR`: gold document의 reciprocal rank
- `Distance Gap`: `distance(Q_neg, D_dist) - distance(Q_neg, D_gold)` 평균
- `Absence Top-1`: gold 제거 후 distractor가 top-1이 되는 비율
- `Semantic Collapse Rate`: dense distance gap이 음수인 비율

## 6. 출력 구조

`run_retrieval_benchmark.py`는 전체와 유형별 결과를 따로 저장합니다.

예시:

```text
outputs/retrieval_benchmark_100_balanced/
  metrics_all.csv
  metrics_all.json
  overall/
    metrics.csv
    metrics.json
    tsne_*.png
    collapse_examples_*.jsonl
  exclusion/
    metrics.csv
    metrics.json
    tsne_*.png
    collapse_examples_*.jsonl
  explicit_negation/
    metrics.csv
    metrics.json
    tsne_*.png
    collapse_examples_*.jsonl
```

즉 다음 세 수준으로 결과가 나옵니다.

- `overall`: 전체 샘플 기준
- `exclusion`: exclusion 샘플만 기준
- `explicit_negation`: explicit negation 샘플만 기준

## 7. 현재 완료된 실험

현재 저장소에는 balanced 100개 실험 결과가 포함되어 있습니다.

벤치마크 파일:

- [track1_benchmark_100_balanced.jsonl](/Users/canho/Desktop/codes/ai/NegRag/outputs/track1_benchmark_100_balanced.jsonl)

통합 결과:

- [metrics_all.csv](/Users/canho/Desktop/codes/ai/NegRag/outputs/retrieval_benchmark_100_balanced/metrics_all.csv)

요약:

```text
balanced benchmark: exclusion 50 / explicit_negation 50

overall
- BM25: SRN 0.72, MRR 0.8162
- BGE-small dense: SRN 0.79, MRR 0.8950, Collapse 0.21
- E5-small dense: SRN 0.74, MRR 0.8700, Collapse 0.26

type-wise
- exclusion이 explicit_negation보다 더 어려운 경향
- 유형별 t-SNE와 collapse examples를 별도 저장
```

## 8. 관련 스크립트

- [build_track1_benchmark.py](/Users/canho/Desktop/codes/ai/NegRag/src/build_track1_benchmark.py)
  MS MARCO에서 Track 1 benchmark 생성
- [run_retrieval_benchmark.py](/Users/canho/Desktop/codes/ai/NegRag/src/run_retrieval_benchmark.py)
  Dense / BM25 / Hybrid 평가
- [generate_dataset.py](/Users/canho/Desktop/codes/ai/NegRag/src/generate_dataset.py)
  초기 negation dataset 생성용 스크립트
- [rerun_failed_records.py](/Users/canho/Desktop/codes/ai/NegRag/src/rerun_failed_records.py)
  OpenAI 출력 실패 레코드 재생성

## 9. 참고

- [MS MARCO dataset page](https://huggingface.co/datasets/microsoft/ms_marco)
