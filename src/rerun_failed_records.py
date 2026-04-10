import argparse
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm

from generate_dataset import ROOT_DIR, generate_one, read_prompt, require_env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Re-run only failed records in an existing JSONL output file."
    )
    parser.add_argument(
        "--input-path",
        default=str(ROOT_DIR / "outputs" / "msmarco_negative_rag_random50.jsonl"),
    )
    parser.add_argument(
        "--prompt-path",
        default=str(ROOT_DIR / "prompts" / "negative_rag_generation.txt"),
    )
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    load_dotenv(ROOT_DIR / ".env")

    api_key = require_env("OPENAI_API_KEY")
    model = require_env("OPENAI_MODEL")
    prompt_text = read_prompt(args.prompt_path)

    input_path = Path(args.input_path)
    rows = [
        json.loads(line)
        for line in input_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failed_indices = [idx for idx, row in enumerate(rows) if "error" in row]

    client = OpenAI(api_key=api_key)

    for idx in tqdm(failed_indices, desc="Re-running failed rows"):
        row = rows[idx]
        payload = {
            "query_id": row["query_id"],
            "query_type": row.get("query_type"),
            "query": row["query"],
            "gold_answers": row.get("gold_answers", []),
            "positive_passages": row.get("positive_passages", []),
        }

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

        updated = {
            "source_dataset": row["source_dataset"],
            "subset": row["subset"],
            "split": row["split"],
            "query_id": row["query_id"],
            "query_type": row.get("query_type"),
            "query": row["query"],
            "gold_answers": row.get("gold_answers", []),
            "positive_passages": row.get("positive_passages", []),
        }

        if result is not None:
            updated.update(result)
        else:
            updated["error"] = error_message

        rows[idx] = updated

    with input_path.open("w", encoding="utf-8") as fout:
        for row in rows:
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Updated file: {input_path}")
    print(f"Retried rows: {len(failed_indices)}")


if __name__ == "__main__":
    main()
