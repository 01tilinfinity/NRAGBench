import argparse
import json
import random
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from generate_dataset import ROOT_DIR, extract_json_object, read_prompt, repair_json_output, require_env


DEFAULT_PROMPT_PATH = ROOT_DIR / "prompts" / "track1_triplet_generation.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a Track 1 negation-retrieval benchmark from MS MARCO."
    )
    parser.add_argument("--subset", default="v1.1", choices=["v1.1", "v2.1"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--target-types",
        nargs="*",
        default=["exclusion", "explicit_negation"],
        help="Negation types to balance across.",
    )
    parser.add_argument(
        "--candidate-multiplier",
        type=int,
        default=6,
        help="Multiplier for sampling MS MARCO candidates before filtering by target type.",
    )
    parser.add_argument(
        "--background-docs",
        type=int,
        default=18,
        help="Number of extra background passages added to each query corpus.",
    )
    parser.add_argument(
        "--output-path",
        default=str(ROOT_DIR / "outputs" / "track1_benchmark_1k.jsonl"),
    )
    parser.add_argument(
        "--prompt-path",
        default=str(DEFAULT_PROMPT_PATH),
    )
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of overwriting it.",
    )
    return parser.parse_args()


def extract_selected_passage(example: dict[str, Any]) -> str:
    passages = example.get("passages") or {}
    texts = passages.get("passage_text") or []
    selected = passages.get("is_selected") or []

    for text, is_selected in zip(texts, selected):
        normalized = (text or "").strip()
        if normalized and int(is_selected) == 1:
            return normalized

    for text in texts:
        normalized = (text or "").strip()
        if normalized:
            return normalized

    return ""


def build_prompt_payload(example: dict[str, Any]) -> dict[str, Any]:
    passage = extract_selected_passage(example)
    answers = [answer.strip() for answer in example.get("answers", []) if answer.strip()]
    well_formed = [
        answer.strip()
        for answer in example.get("wellFormedAnswers", [])
        if isinstance(answer, str) and answer.strip()
    ]
    return {
        "query_id": example.get("query_id"),
        "query_type": example.get("query_type"),
        "original_query": example.get("query"),
        "gold_answers": well_formed or answers,
        "passage": passage,
    }


def normalize_type(value: str | None) -> str | None:
    if value is None:
        return None
    return value.strip().lower()


def generate_triplet(
    client: OpenAI,
    model: str,
    system_prompt: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": system_prompt}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(payload, ensure_ascii=False, indent=2),
                    }
                ],
            },
        ],
    )

    try:
        parsed = extract_json_object(response.output_text)
    except Exception:
        parsed = repair_json_output(client, model, response.output_text)

    return {
        "model": model,
        "generated": parsed,
        "raw_response_text": response.output_text,
    }


def build_background_pool(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    pool = []
    seen = set()
    for row in rows:
        passage = extract_selected_passage(row)
        if not passage:
            continue
        doc_id = f"{row['query_id']}_msmarco"
        key = (doc_id, passage)
        if key in seen:
            continue
        seen.add(key)
        pool.append({"doc_id": doc_id, "text": passage})
    return pool


def sample_examples(dataset: Any, sample_size: int, seed: int) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    sample_size = min(sample_size, len(dataset))
    sampled_indices = rng.sample(range(len(dataset)), k=sample_size)
    return [dataset[int(index)] for index in sampled_indices]


def build_corpus(
    query_id: str,
    doc_gold: str,
    doc_distractor: str,
    background_pool: list[dict[str, str]],
    background_docs: int,
    rng: random.Random,
) -> list[dict[str, str]]:
    blocked_ids = {f"{query_id}_gold", f"{query_id}_dist", f"{query_id}_msmarco"}
    candidates = [doc for doc in background_pool if doc["doc_id"] not in blocked_ids]
    sampled = rng.sample(candidates, k=min(background_docs, len(candidates)))

    corpus = [
        {"doc_id": f"{query_id}_gold", "text": doc_gold, "label": "gold"},
        {"doc_id": f"{query_id}_dist", "text": doc_distractor, "label": "distractor"},
    ]
    corpus.extend(
        {"doc_id": doc["doc_id"], "text": doc["text"], "label": "background"}
        for doc in sampled
    )
    rng.shuffle(corpus)
    return corpus


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT_DIR / ".env")

    api_key = require_env("OPENAI_API_KEY")
    model = require_env("OPENAI_MODEL")
    prompt_text = read_prompt(args.prompt_path)
    target_types = [normalize_type(item) for item in args.target_types]
    if not target_types:
        raise ValueError("At least one target negation type must be provided.")
    if args.sample_size % len(target_types) != 0:
        raise ValueError(
            f"sample-size must be divisible by the number of target types ({len(target_types)})."
        )
    quota_per_type = args.sample_size // len(target_types)
    quotas = {item: quota_per_type for item in target_types}

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_mode = "a" if args.append else "w"

    dataset = load_dataset("microsoft/ms_marco", args.subset, split=args.split)
    candidate_size = min(
        len(dataset),
        max(args.sample_size, args.sample_size * args.candidate_multiplier),
    )
    sampled_examples = sample_examples(dataset, candidate_size, args.seed)
    background_pool = build_background_pool(sampled_examples)
    rng = random.Random(args.seed)
    client = OpenAI(api_key=api_key)
    accepted = 0
    accepted_counts = {item: 0 for item in target_types}
    seen_query_ids = set()

    with output_path.open(file_mode, encoding="utf-8") as fout:
        progress = tqdm(total=args.sample_size, desc="Building Track 1 benchmark")
        for example in sampled_examples:
            if accepted >= args.sample_size:
                break

            remaining_types = [item for item in target_types if accepted_counts[item] < quotas[item]]
            if not remaining_types:
                break

            target_type = min(remaining_types, key=lambda item: accepted_counts[item])
            payload = build_prompt_payload(example)
            if not payload["passage"]:
                continue
            if payload["query_id"] in seen_query_ids:
                continue
            payload["target_negation_type"] = target_type

            result = None

            for attempt in range(1, args.max_retries + 1):
                try:
                    result = generate_triplet(client, model, prompt_text, payload)
                    break
                except Exception as exc:
                    if attempt < args.max_retries:
                        time.sleep(1.5 * attempt)

            if result is None:
                continue

            generated = result["generated"]
            generated_type = normalize_type(generated.get("negation_type"))
            if generated_type != target_type:
                continue
            query_id = str(payload["query_id"])
            corpus = build_corpus(
                query_id=query_id,
                doc_gold=generated["doc_gold"].strip(),
                doc_distractor=generated["doc_distractor"].strip(),
                background_pool=background_pool,
                background_docs=args.background_docs,
                rng=rng,
            )

            record = {
                "sample_id": generated.get("id", query_id),
                "source_dataset": "microsoft/ms_marco",
                "subset": args.subset,
                "split": args.split,
                "query_id": payload["query_id"],
                "query_type": payload["query_type"],
                "original_query": payload["original_query"],
                "original_gold_answers": payload["gold_answers"],
                "model": model,
                "query_pos": generated["query_pos"].strip(),
                "query_neg": generated["query_neg"].strip(),
                "doc_gold": {
                    "doc_id": f"{query_id}_gold",
                    "text": generated["doc_gold"].strip(),
                },
                "doc_distractor": {
                    "doc_id": f"{query_id}_dist",
                    "text": generated["doc_distractor"].strip(),
                },
                "answer_gold": generated["answer_gold"].strip(),
                "excluded_target": generated["excluded_target"].strip(),
                "negation_type": generated_type,
                "domain": generated["domain"],
                "corpus": corpus,
                "raw_response_text": result["raw_response_text"],
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            accepted += 1
            accepted_counts[target_type] += 1
            seen_query_ids.add(payload["query_id"])
            progress.update(1)
            progress.set_postfix(accepted_counts)

        progress.close()

    if accepted < args.sample_size:
        raise RuntimeError(
            f"Only built {accepted} samples. Current counts: {accepted_counts}. "
            "Increase candidate-multiplier or rerun with another seed."
        )

    print(f"Saved Track 1 benchmark to: {output_path}")
    print(f"Accepted counts: {accepted_counts}")


if __name__ == "__main__":
    main()
