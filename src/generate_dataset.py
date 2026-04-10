import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from datasets import load_dataset
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROMPT_PATH = ROOT_DIR / "prompts" / "negative_rag_generation.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a negative-RAG style dataset from MS MARCO with OpenAI."
    )
    parser.add_argument("--subset", default="v1.1", choices=["v1.1", "v2.1"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-size", type=int, default=20)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--random-sample",
        action="store_true",
        help="Sample examples randomly instead of taking a contiguous slice.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --random-sample is enabled.",
    )
    parser.add_argument(
        "--output-path",
        default=str(ROOT_DIR / "outputs" / "msmarco_negative_rag.jsonl"),
    )
    parser.add_argument(
        "--prompt-path",
        default=str(DEFAULT_PROMPT_PATH),
        help="Text file containing the generation instructions.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Retries per sample if the API call or JSON parsing fails.",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to the output file instead of overwriting it.",
    )
    return parser.parse_args()


def read_prompt(prompt_path: str) -> str:
    return Path(prompt_path).read_text(encoding="utf-8").strip()


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"Missing environment variable: {name}. "
            f"Set it in a .env file or export it in your shell."
        )
    return value


def extract_positive_passages(example: dict[str, Any]) -> list[str]:
    passages = example.get("passages") or {}
    texts = passages.get("passage_text") or []
    selected = passages.get("is_selected") or []

    positives = [
        text.strip()
        for text, is_selected in zip(texts, selected)
        if int(is_selected) == 1 and text and text.strip()
    ]

    if positives:
        return positives

    return [text.strip() for text in texts if text and text.strip()]


def build_user_payload(example: dict[str, Any]) -> dict[str, Any]:
    positives = extract_positive_passages(example)
    answers = [answer.strip() for answer in example.get("answers", []) if answer.strip()]
    well_formed = [
        answer.strip()
        for answer in example.get("wellFormedAnswers", [])
        if isinstance(answer, str) and answer.strip()
    ]

    gold_answers = well_formed or answers
    if not gold_answers:
        gold_answers = [""]

    return {
        "query_id": example.get("query_id"),
        "query_type": example.get("query_type"),
        "query": example.get("query", ""),
        "gold_answers": gold_answers,
        "positive_passages": positives[:3],
    }


def extract_json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model output.")
    return json.loads(cleaned[start : end + 1])


def repair_json_output(
    client: OpenAI,
    model: str,
    broken_text: str,
) -> dict[str, Any]:
    repair_prompt = (
        "Convert the following content into strict valid JSON. "
        "Return JSON only. Do not add markdown fences or commentary."
    )
    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "system",
                "content": [{"type": "input_text", "text": repair_prompt}],
            },
            {
                "role": "user",
                "content": [{"type": "input_text", "text": broken_text}],
            },
        ],
    )
    return extract_json_object(response.output_text)


def generate_one(
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


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT_DIR / ".env")

    api_key = require_env("OPENAI_API_KEY")
    model = require_env("OPENAI_MODEL")
    prompt_text = read_prompt(args.prompt_path)

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("microsoft/ms_marco", args.subset, split=args.split)
    if args.random_sample:
        sample_size = min(args.sample_size, len(dataset))
        rng = random.Random(args.seed)
        sampled_indices = rng.sample(range(len(dataset)), k=sample_size)
        subset = dataset.select(sampled_indices)
    else:
        end_index = min(args.start_index + args.sample_size, len(dataset))
        subset = dataset.select(range(args.start_index, end_index))

    client = OpenAI(api_key=api_key)

    file_mode = "a" if args.append else "w"

    with output_path.open(file_mode, encoding="utf-8") as fout:
        for example in tqdm(subset, desc="Generating"):
            payload = build_user_payload(example)

            if not payload["positive_passages"]:
                continue

            result = None
            error_message = None

            for attempt in range(1, args.max_retries + 1):
                try:
                    result = generate_one(client, model, prompt_text, payload)
                    break
                except Exception as exc:
                    error_message = str(exc)
                    if attempt < args.max_retries:
                        time.sleep(1.5 * attempt)

            record = {
                "source_dataset": "microsoft/ms_marco",
                "subset": args.subset,
                "split": args.split,
                "query_id": payload["query_id"],
                "query_type": payload["query_type"],
                "query": payload["query"],
                "gold_answers": payload["gold_answers"],
                "positive_passages": payload["positive_passages"],
            }

            if result is not None:
                record.update(result)
            else:
                record["error"] = error_message

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Saved results to: {output_path}")


if __name__ == "__main__":
    main()
