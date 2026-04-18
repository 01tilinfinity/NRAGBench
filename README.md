# NegRag MS MARCO Baseline

`microsoft/ms_marco`를 불러와서 OpenAI API로 초벌 데이터셋을 생성하는 최소 실험 환경입니다.

현재 기본 설정은 다음 실험을 가정합니다:

- 입력: `query`, `gold answer`, `selected positive passage`
- 출력: 질의와 주제는 비슷하지만 정답 근거는 되지 않는 `negative passage`

## 1. Conda 환경 만들기

```bash
conda env create -f environment.yml
conda activate negrag-msmarco
```

## 2. OpenAI API 키 넣는 위치

프로젝트 루트에 `.env` 파일을 만들고 아래처럼 넣으면 됩니다.

```bash
cp .env.example .env
```

그 다음 `.env` 파일에서 아래 두 줄을 직접 채워 주세요.

```env
OPENAI_API_KEY=여기에_본인_API_KEY
OPENAI_MODEL=여기에_사용할_모델명
```

예를 들어 모델명은 본인이 실험에 쓸 OpenAI 모델 이름으로 넣으면 됩니다.

## 3. 실행

기본 실행:

```bash
python src/generate_dataset.py --sample-size 20
```

다른 subset / split 예시:

```bash
python src/generate_dataset.py --subset v2.1 --split validation --sample-size 50
```

출력 파일 기본 경로:

```bash
outputs/msmarco_negative_rag.jsonl
```

## 4. 출력 포맷

각 줄은 JSON 하나이며 대략 아래 구조입니다.

```json
{
  "source_dataset": "microsoft/ms_marco",
  "subset": "v1.1",
  "split": "train",
  "query_id": 123,
  "query_type": "description",
  "query": "what is rba",
  "gold_answers": ["..."],
  "positive_passages": ["..."],
  "model": "your-model",
  "generated": {
    "negative_passage": "...",
    "misleading_reason": "...",
    "difficulty": "hard"
  }
}
```

## 5. 수정 포인트

- 프롬프트는 `prompts/negative_rag_generation.txt`에서 바꾸면 됩니다.
- 생성 형식을 더 늘리고 싶으면 `src/generate_dataset.py`의 `build_user_payload()`와 `record` 구성을 수정하면 됩니다.

## 6. 참고

MS MARCO는 Hugging Face의 `microsoft/ms_marco`를 `datasets` 라이브러리로 불러오도록 구성했습니다:

- [MS MARCO dataset page](https://huggingface.co/datasets/microsoft/ms_marco)

## 7. Track 1 Retrieval Benchmark

최신 실험 설계 기준으로 `query_neg`, `doc_gold`, `doc_distractor`, `corpus`를 직접 생성한 뒤,
Dense / BM25 / Hybrid 검색 성능을 비교할 수 있습니다.

### 7-1. Track 1 benchmark 생성

MS MARCO에서 샘플을 뽑아 OpenAI로 triplet benchmark를 생성:

```bash
python src/build_track1_benchmark.py \
  --sample-size 1000 \
  --output-path outputs/track1_benchmark_1k.jsonl
```

출력 구조:

```json
{
  "sample_id": "34300",
  "query_pos": "What is cost of sales?",
  "query_neg": "What is not cost of sales?",
  "doc_gold": {"doc_id": "34300_gold", "text": "..."},
  "doc_distractor": {"doc_id": "34300_dist", "text": "..."},
  "answer_gold": "...",
  "excluded_target": "...",
  "negation_type": "exclusion",
  "domain": "factual",
  "corpus": [
    {"doc_id": "34300_gold", "text": "...", "label": "gold"},
    {"doc_id": "34300_dist", "text": "...", "label": "distractor"},
    {"doc_id": "etc", "text": "...", "label": "background"}
  ]
}
```

### 7-2. Retrieval Bottleneck 평가

```bash
python src/run_retrieval_benchmark.py \
  --input-path outputs/track1_benchmark_1k.jsonl \
  --output-dir outputs/retrieval_benchmark_1k
```

평가 항목:

- `SRN`: `Score(Q_neg, D_gold) > Score(Q_neg, D_dist)` 비율
- `MRR`: gold doc의 reciprocal rank
- `Distance Gap`: `distance(Q_neg, D_dist) - distance(Q_neg, D_gold)` 평균
- `Absence Top-1`: gold 제거 후 distractor가 top-1이 되는 비율
- `Semantic Collapse Rate`: dense distance gap이 음수인 비율

출력:

- `metrics.csv`
- `metrics.json`
- `tsne_*.png`
- `collapse_examples_*.jsonl`
